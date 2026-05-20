# Stock Analysis Dashboard

A 3-layer market deployment and stock-ranking dashboard. Streamlit frontend, Python backend, yfinance data, Anthropic API for the qualitative layer.

It answers three nested questions:

1. **Should I be deploying capital today, and how aggressively?** → Macro Gate
2. **If yes, which tickers look best by the numbers?** → Quantitative Scanner
3. **Do the fundamentals back up what the numbers say?** → Claude Analyst

Each layer gates the next: a DEFENSIVE macro reading disables the scanner; scanner output is the input universe for the analyst.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY

# Refresh data (optional — pages will compute on first load)
python run_macro_gate.py
python run_scanner.py
python run_analysis.py --scan-and-analyze
python run_performance.py            # mature the forward-return track record

# Launch dashboard
streamlit run streamlit_app.py
```

### Nightly cron (builds the track record)

```bash
# crontab -e — runs after the US close
30 22 * * 1-5  cd /path/to/stock-dashboard && \
  python run_macro_gate.py && python run_scanner.py && python run_performance.py && \
  git add data/track_record.db && git commit -m "nightly track record" && git push
```

Each scanner run snapshots its top 20 picks; `run_performance.py` computes the
1d/5d/20d forward return of every snapshot ever taken. Committing
`data/track_record.db` is what makes the record survive Streamlit Cloud
restarts (the rest of `data/` is ephemeral on the free tier).

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. Visit https://streamlit.io/cloud and create a new app pointing at `streamlit_app.py`.
3. In **Settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ANTHROPIC_MODEL = "claude-sonnet-4-5"
   ```
4. Deploy. On first load each page will fetch data and persist caches into `data/`.

## Layout

- `signals/` — six macro signals + composite blender
- `scanner/` — universe, `factors/` package (6 factors), `fundamentals.py`, ranker (no longer macro-gated — Phase 2 Part C)
- `analyst/` — fundamentals → Claude prompt → SQLite cache → 60/40 blender
- `backtest/` — `panel.py` (shared signal panel), `deployment_backtest.py` (zone overlay), `walk_forward.py` (10y macro validation), `scanner_backtest.py` (scanner walk-forward, v1/v2 factor sets)
- `tracking/` — forward-performance track record (snapshot + nightly returns)
- `tests/` — factor unit tests (`pytest tests/`)
- `pages/` — Streamlit multipage UI (macro gate, scanner, analyst, validation)
- `utils/` — yfinance + parquet cache, theme tokens

## Validation layer

The validation page answers the only question that matters: **is the composite
actually predictive, or are we tuning to noise?**

- **Forward performance** — every scanner run snapshots its top 20 picks into
  `data/track_record.db`. `run_performance.py` computes their 1d/5d/20d forward
  return vs. SPY. The page buckets picks by composite score — a predictive
  score shows excess return rising monotonically across buckets.
- **Walk-forward backtest** — refits the macro composite weights on a rolling
  5-year window, freezes them, tests on the next 6 months, rolls forward over
  all available history (~2013-present out-of-sample: spans 2018 vol, COVID,
  the 2022 bear, the 2023-24 recovery). A fitted-vs-equal-weight comparison
  shows whether refitting earns its keep out-of-sample — and, just as
  usefully, when it does not.
- **Scanner walk-forward** (`backtest/scanner_backtest.py`, `run_scanner_backtest.py`)
  — reconstructs the scanner weekly over 5 years with no look-ahead and reports
  whether the top-10 picks beat the universe forward, plus a per-factor decile
  diagnostic. Two factor sets: **v1** (original 5 factors) and **v2** (the
  Phase 2 refactor). Only price factors are point-in-time backtestable;
  fundamental factors (short interest in v1; quality / value / earnings
  surprise in v2) are flagged "not backtestable" — yfinance exposes no
  historical fundamentals. Survivorship caveat: uses the current S&P 500.

## Scanner factors (Phase 2)

The scanner ranks on 5 sector-neutral factors, **momentum-tilted** (weights
driven by the Part A re-validation, not equal):

| Factor | Weight | Definition |
|---|---|---|
| 12-1 Momentum | 0.35 | return from month t-12 to t-1 (Jegadeesh-Titman) |
| Relative Strength | 0.20 | 20-day return spread vs SPY |
| Quality | 0.15 | gross profit (TTM) / total assets (Novy-Marx) |
| Value | 0.15 | inverted EV/EBITDA; negative-EBITDA names excluded |
| Earnings Surprise | 0.15 | weighted EPS surprise, last 2 quarters (PEAD) |

Why the tilt: the Part A re-validation found 12-1 Momentum was by far the
strongest factor (+95 bps per 20d) and Low Volatility was non-predictive
(-7 bps) — so Low Volatility was dropped and 12-1 Momentum overweighted.
Quality / Value / Earnings Surprise can't be point-in-time backtested with
yfinance, so they are kept as fundamental ballast at a smaller, untested
weight.

Each factor is percentile-ranked **within sector** so the scanner can't pile
into one hot sector. A ticker missing any factor is dropped, never imputed.
The macro gate no longer gates or resizes the scanner — it failed walk-forward
validation and is now informational context only.

## Notes

- Analyst results are cached in `data/cache.db` keyed by `(ticker, quarter_end, model)` — re-runs on the same trading day hit cache and don't burn API credits.
- yfinance is unofficial; bulk downloads occasionally fail. The data layer caches per-ticker parquet files in `data/prices/` so re-runs don't re-fetch everything.
- Both backtests re-weight to 4 daily-computable signals (excludes breadth + crowding, both of which need 500-ticker recomputation per day).
