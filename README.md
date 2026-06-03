# Momentum Quality Scanner MVP

A fast MVP for screening shares using:

- Consecutive up days
- Relative volume
- 52-week high proximity
- RSI
- EPS positivity and EPS growth
- Positive P/E
- Positive and rising shareholder equity / net assets
- Multi-timeframe returns
- Alpha vs benchmark
- Win rate
- Beta vs benchmark
- Sharpe ratio
- Annualized volatility
- News sentiment score

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` — Streamlit user interface
- `scoring.py` — all calculations and scoring logic, including beta and Sharpe ratio
- `sample_data.py` — realistic sample stock universe
- `requirements.txt` — dependencies
- `Momentum_Quality_App_MVP.ipynb` — notebook version for GitHub preview

## Key formulas

- Beta = covariance(stock daily returns, benchmark daily returns) / variance(benchmark daily returns)
- Sharpe ratio = annualized excess return / annualized volatility
- Alpha = stock return - benchmark return
- Relative volume = latest volume / 30-day average volume

## Next development step

Replace `sample_data.py` with real data connectors:

1. Daily prices and volume
2. Fundamentals: EPS, P/E, shareholder equity
3. Benchmark index prices
4. News and sentiment

## Suggested production architecture

- Frontend: React / Next.js or Streamlit for early users
- Backend: FastAPI
- Database: PostgreSQL
- Jobs: Cron, Celery, or Airflow
- Hosting: Render, AWS, Railway, or Azure
- Alerts: email, Telegram, WhatsApp, or push notifications

## Disclaimer

This tool is for screening and research only. It is not financial advice.
