# Momentum Quality Scanner MVP — Phase 2 Live Data

A Streamlit app for screening US and UK shares using momentum, volume, fundamentals, alpha, beta, Sharpe ratio, volatility, RSI and 52-week-high proximity.

## Phase 2 upgrade

This version uses a free-first live data approach:

- Yahoo Finance via `yfinance`
- US and UK stock support
- S&P 500 benchmark for US stocks: `^GSPC`
- FTSE 100 benchmark for UK stocks: `^FTSE`
- Streamlit caching to reduce rate-limit problems

## Files

- `app.py` — Streamlit app
- `data_provider.py` — Yahoo Finance live data connector
- `scoring.py` — scoring and risk-adjusted calculations
- `sample_data.py` — fallback demo data
- `requirements.txt` — Python packages
- `Momentum_Quality_App_MVP.ipynb` — GitHub notebook overview

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

Use:

```text
Main file path: app.py
```

## Notes

Free APIs can be delayed, rate-limited or incomplete. This app uses caching and smaller stock universes to keep the MVP practical.

This is a decision-support screener only. It does not guarantee investment returns or provide personal financial advice.
