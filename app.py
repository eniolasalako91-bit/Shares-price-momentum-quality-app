from __future__ import annotations

import pandas as pd
import streamlit as st

from sample_data import load_sample_universe
from scoring import compute_metrics

st.set_page_config(page_title="Momentum Quality Scanner", layout="wide")

st.title("Momentum Quality Scanner")
st.caption("Find shares with technical momentum, strong volume, improving fundamentals, positive alpha, and supportive sentiment.")

with st.sidebar:
    st.header("User Parameters")
    min_up_days = st.slider("Minimum consecutive up days", 0, 10, 3)
    max_gap = st.slider("Maximum gap from 52-week high (%)", 0.0, 100.0, 5.0, 0.1)
    min_rel_volume = st.slider("Minimum relative volume", 0.0, 10.0, 1.5, 0.1)
    require_positive_eps = st.checkbox("Require positive EPS", value=True)
    require_rising_eps = st.checkbox("Require rising EPS", value=True)
    require_positive_pe = st.checkbox("Require positive P/E", value=True)
    require_positive_equity = st.checkbox("Require positive shareholder equity", value=True)
    require_rising_equity = st.checkbox("Require rising shareholder equity", value=False)
    min_alpha = st.slider("Minimum 1-year alpha vs benchmark (%)", -100.0, 200.0, 0.0, 1.0)
    min_score = st.slider("Minimum Momentum Quality Score", 0, 100, 60)
    rsi_min, rsi_max = st.slider("Acceptable RSI range", 0, 100, (45, 85))

universe, benchmark = load_sample_universe()
rows = []
for symbol, data in universe.items():
    rows.append(
        compute_metrics(
            symbol=symbol,
            company=data["company"],
            prices=data["prices"],
            benchmark_prices=benchmark,
            fundamentals=data["fundamentals"],
            sentiment_score=data["sentiment_score"],
        )
    )

results = pd.DataFrame(rows)

filtered = results.copy()
filtered = filtered[filtered["consecutive_up_days"] >= min_up_days]
filtered = filtered[filtered["gap_to_52w_high_pct"] <= max_gap]
filtered = filtered[filtered["relative_volume"] >= min_rel_volume]
filtered = filtered[filtered["alpha_1Y"] >= min_alpha]
filtered = filtered[filtered["momentum_quality_score"] >= min_score]
filtered = filtered[(filtered["rsi"] >= rsi_min) & (filtered["rsi"] <= rsi_max)]

if require_positive_eps:
    filtered = filtered[filtered["eps"] > 0]
if require_rising_eps:
    filtered = filtered[filtered["eps_growth_pct"] > 0]
if require_positive_pe:
    filtered = filtered[filtered["pe_ratio"] > 0]
if require_positive_equity:
    filtered = filtered[filtered["shareholder_equity"] > 0]
if require_rising_equity:
    filtered = filtered[filtered["equity_growth_pct"] > 0]

filtered = filtered.sort_values("momentum_quality_score", ascending=False)

summary_cols = [
    "symbol", "company", "momentum_quality_score", "consecutive_up_days",
    "latest_price", "high_52w", "gap_to_52w_high_pct", "relative_volume",
    "rsi", "eps", "eps_growth_pct", "pe_ratio", "equity_growth_pct",
    "return_5D", "return_1M", "return_3M", "return_1Y", "alpha_1Y",
    "win_rate_252D", "sentiment_score"
]

st.subheader("Ranked Matches")
st.dataframe(
    filtered[summary_cols].style.format({
        "momentum_quality_score": "{:.1f}",
        "latest_price": "{:.2f}",
        "high_52w": "{:.2f}",
        "gap_to_52w_high_pct": "{:.2f}%",
        "relative_volume": "{:.2f}x",
        "rsi": "{:.1f}",
        "eps": "{:.2f}",
        "eps_growth_pct": "{:.1f}%",
        "pe_ratio": "{:.1f}",
        "equity_growth_pct": "{:.1f}%",
        "return_5D": "{:.1f}%",
        "return_1M": "{:.1f}%",
        "return_3M": "{:.1f}%",
        "return_1Y": "{:.1f}%",
        "alpha_1Y": "{:.1f}%",
        "win_rate_252D": "{:.1f}%",
        "sentiment_score": "{:.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.subheader("All Stocks Scored")
st.dataframe(results[summary_cols].sort_values("momentum_quality_score", ascending=False), use_container_width=True, hide_index=True)

st.subheader("Score Breakdown")
selected = st.selectbox("Select a stock", results["symbol"].tolist())
row = results[results["symbol"] == selected].iloc[0]
score_cols = [
    "price_momentum_score", "volume_score", "high_gap_score", "rsi_score",
    "fundamental_score", "equity_score", "alpha_quality_score", "sentiment_component_score"
]
st.bar_chart(row[score_cols])

st.markdown("""
### How to connect real data
This MVP currently runs with sample data so you can test the product immediately.
Replace `sample_data.py` with connectors for:

- Market prices and volume: Polygon, Tiingo, Alpha Vantage, Twelve Data, Yahoo Finance-style feeds
- Fundamentals: Financial Modeling Prep, Intrinio, EODHD, IEX Cloud, SEC/company filings
- News sentiment: Benzinga, Finnhub, NewsAPI, RavenPack-style sentiment, or your own AI classifier
- Benchmark: S&P 500, Nasdaq, FTSE 100, TSX, or user-selected index

### Important
This app is a decision-support screener. It should show users which shares match their selected rules, not guarantee profit or provide personal financial advice.
""")
