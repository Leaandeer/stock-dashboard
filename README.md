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

# Launch dashboard
streamlit run streamlit_app.py
```

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
- `scanner/` — S&P 500 universe, five factors, ranker (gated by macro)
- `analyst/` — fundamentals → Claude prompt → SQLite cache → 60/40 blender
- `backtest/` — 2y zone overlay (re-weighted 4-signal composite for speed)
- `pages/` — Streamlit multipage UI (macro gate, scanner, analyst)
- `utils/` — yfinance + parquet cache, theme tokens

## Notes

- Analyst results are cached in `data/cache.db` keyed by `(ticker, quarter_end, model)` — re-runs on the same trading day hit cache and don't burn API credits.
- yfinance is unofficial; bulk downloads occasionally fail. The data layer caches per-ticker parquet files in `data/prices/` so re-runs don't re-fetch everything.
- The backtest re-weights to 4 daily-computable signals (excludes breadth + crowding, both of which need 500-ticker recomputation per day).
