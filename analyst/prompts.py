"""System + user prompts for the Claude analyst."""

SYSTEM = (
    "You are a senior equity research analyst. Score 1-10 on each of: "
    "Earnings Quality, Growth Trajectory, Balance Sheet Health, Margin Trends, Red Flags. "
    "Reply with valid JSON only, no preamble."
)

USER_TEMPLATE = """Analyze {ticker} based on the following 4 quarters of fundamentals.

Revenue (most recent first): {revenue}
Net Income: {net_income}
Operating Cash Flow: {ocf}
Free Cash Flow: {fcf}
Gross Margin %: {gross_margin}
Operating Margin %: {op_margin}
Debt/Equity: {de}
Return on Equity %: {roe}

Derived signals:
- CFO/NI ratio (most recent): {cfo_ni_ratio}
- AR growth vs revenue growth (most recent YoY): AR {ar_growth_pct}% vs revenue {rev_growth_pct}%

Score 1-10 on each dimension. Higher Red Flags score = fewer red flags (i.e. 10 = clean, 1 = many warning signs).
Compute "overall" as the simple average of the 5 sub-scores, rounded to 1 decimal.

Reply with JSON only, in exactly this shape:
{{
  "ticker": "{ticker}",
  "scores": {{
    "earnings_quality": <1-10>,
    "growth_trajectory": <1-10>,
    "balance_sheet_health": <1-10>,
    "margin_trends": <1-10>,
    "red_flags": <1-10>
  }},
  "overall": <number 1-10>,
  "summary": "<two-sentence summary>",
  "key_observations": ["<bullet>", "<bullet>", "<bullet>"]
}}"""
