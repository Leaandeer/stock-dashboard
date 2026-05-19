"""Pull fundamentals, call Claude, cache result in SQLite."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from analyst.prompts import SYSTEM, USER_TEMPLATE
from utils.data_fetch import fetch_financials

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cache.db"
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute(
        """CREATE TABLE IF NOT EXISTS analyst_cache (
            ticker TEXT NOT NULL,
            quarter_end TEXT NOT NULL,
            model TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (ticker, quarter_end, model)
        )"""
    )
    return c


def _row_or_none(income: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    if income is None or income.empty:
        return None
    for cand in candidates:
        for idx in income.index:
            if isinstance(idx, str) and idx.strip().lower() == cand.lower():
                return income.loc[idx]
    return None


def _series_last_n(s: pd.Series | None, n: int = 4) -> list[float | None]:
    if s is None or s.empty:
        return [None] * n
    s = s.dropna()
    s = s.sort_index(ascending=False)  # most recent first
    out: list[float | None] = []
    for v in s.iloc[:n].tolist():
        try:
            out.append(float(v))
        except Exception:
            out.append(None)
    while len(out) < n:
        out.append(None)
    return out


def _fmt(values: list[Any]) -> str:
    return ", ".join("n/a" if v is None else f"{v:,.0f}" for v in values)


def _quarter_end(income: pd.DataFrame) -> str:
    if income is None or income.empty:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        cols = sorted(income.columns)
        return pd.to_datetime(cols[-1]).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def build_prompt_payload(ticker: str) -> tuple[str, dict, str]:
    """Return (user_prompt, financial_table, quarter_end_iso)."""
    fin = fetch_financials(ticker)
    income = fin.get("income", pd.DataFrame())
    cashflow = fin.get("cashflow", pd.DataFrame())
    balance = fin.get("balance", pd.DataFrame())
    info = fin.get("info", {})

    revenue = _row_or_none(income, ["Total Revenue", "Revenue", "TotalRevenue"])
    net_income = _row_or_none(income, ["Net Income", "Net Income Common Stockholders"])
    gross_profit = _row_or_none(income, ["Gross Profit"])
    operating_income = _row_or_none(income, ["Operating Income", "Operating Income or Loss"])
    ocf = _row_or_none(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex = _row_or_none(cashflow, ["Capital Expenditure", "Capital Expenditures"])
    total_debt = _row_or_none(balance, ["Total Debt"])
    total_equity = _row_or_none(balance, ["Total Equity Gross Minority Interest", "Stockholders Equity", "Total Stockholder Equity"])
    accounts_receivable = _row_or_none(balance, ["Accounts Receivable", "Net Receivables"])

    rev_vals = _series_last_n(revenue)
    ni_vals = _series_last_n(net_income)
    ocf_vals = _series_last_n(ocf)
    capex_vals = _series_last_n(capex)
    fcf_vals = [
        (o - abs(c)) if (o is not None and c is not None) else None
        for o, c in zip(ocf_vals, capex_vals)
    ]
    gp_vals = _series_last_n(gross_profit)
    op_vals = _series_last_n(operating_income)
    gm_vals = [round(g / r * 100, 1) if (g and r) else None for g, r in zip(gp_vals, rev_vals)]
    om_vals = [round(o / r * 100, 1) if (o and r) else None for o, r in zip(op_vals, rev_vals)]

    de_vals = []
    debt_vals = _series_last_n(total_debt)
    eq_vals = _series_last_n(total_equity)
    for d, e in zip(debt_vals, eq_vals):
        de_vals.append(round(d / e, 2) if (d and e) else None)
    roe_vals = []
    for n, e in zip(ni_vals, eq_vals):
        roe_vals.append(round(n / e * 100, 1) if (n and e) else None)

    cfo_ni_ratio = None
    if ocf_vals[0] is not None and ni_vals[0]:
        cfo_ni_ratio = round(ocf_vals[0] / ni_vals[0], 2)

    ar_growth_pct = None
    rev_growth_pct = None
    ar_vals = _series_last_n(accounts_receivable)
    if ar_vals[0] and ar_vals[-1]:
        ar_growth_pct = round((ar_vals[0] / ar_vals[-1] - 1) * 100, 1)
    if rev_vals[0] and rev_vals[-1]:
        rev_growth_pct = round((rev_vals[0] / rev_vals[-1] - 1) * 100, 1)

    user_prompt = USER_TEMPLATE.format(
        ticker=ticker,
        revenue=_fmt(rev_vals),
        net_income=_fmt(ni_vals),
        ocf=_fmt(ocf_vals),
        fcf=_fmt(fcf_vals),
        gross_margin=", ".join("n/a" if v is None else f"{v}" for v in gm_vals),
        op_margin=", ".join("n/a" if v is None else f"{v}" for v in om_vals),
        de=", ".join("n/a" if v is None else f"{v}" for v in de_vals),
        roe=", ".join("n/a" if v is None else f"{v}" for v in roe_vals),
        cfo_ni_ratio="n/a" if cfo_ni_ratio is None else cfo_ni_ratio,
        ar_growth_pct="n/a" if ar_growth_pct is None else ar_growth_pct,
        rev_growth_pct="n/a" if rev_growth_pct is None else rev_growth_pct,
    )

    table = {
        "Revenue": rev_vals,
        "Net Income": ni_vals,
        "OCF": ocf_vals,
        "FCF": fcf_vals,
        "Gross Margin %": gm_vals,
        "Operating Margin %": om_vals,
        "Debt/Equity": de_vals,
        "ROE %": roe_vals,
    }
    return user_prompt, table, _quarter_end(income)


def _call_claude(user_prompt: str, model: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    from anthropic import Anthropic  # local import keeps startup fast

    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", "text") == "text")
    text = text.strip()
    # strip code fences if present
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lower().startswith("json\n"):
            text = text[5:]
    return json.loads(text)


@dataclass
class AnalystResult:
    ticker: str
    quarter_end: str
    model: str
    payload: dict
    cached: bool


def analyze(ticker: str, model: str = DEFAULT_MODEL, force: bool = False) -> AnalystResult:
    user_prompt, table, quarter_end = build_prompt_payload(ticker)
    conn = _conn()
    try:
        if not force:
            row = conn.execute(
                "SELECT payload FROM analyst_cache WHERE ticker=? AND quarter_end=? AND model=?",
                (ticker, quarter_end, model),
            ).fetchone()
            if row:
                cached = json.loads(row[0])
                cached["_financial_table"] = table
                return AnalystResult(ticker, quarter_end, model, cached, cached=True)
        payload = _call_claude(user_prompt, model)
        payload["_financial_table"] = table
        conn.execute(
            "INSERT OR REPLACE INTO analyst_cache(ticker, quarter_end, model, payload, created_at) VALUES (?,?,?,?,?)",
            (ticker, quarter_end, model, json.dumps(payload), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return AnalystResult(ticker, quarter_end, model, payload, cached=False)
    finally:
        conn.close()
