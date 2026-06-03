from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from data_provider import BENCHMARKS, DEFAULT_TICKERS, clean_tickers, load_live_universe
from sample_data import load_sample_universe
from scoring import compute_metrics

st.set_page_config(page_title="Momentum Quality Scanner", layout="wide")

st.title("Momentum Quality Scanner")
st.caption(
    "Live US/UK momentum, quality, alpha, beta, Sharpe, volatility and fundamentals scanner. "
    "Free-first Phase 2 uses Yahoo Finance via yfinance."
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_live_universe(tickers_tuple, benchmark_symbol: str, include_fundamentals: bool):
    return load_live_universe(
        tickers=list(tickers_tuple),
        benchmark_symbol=benchmark_symbol,
        period="18mo",
        include_fundamentals=include_fundamentals,
    )


with st.sidebar:
    st.header("Data Source")
    data_mode = st.radio("Choose data mode", ["Live Yahoo Finance", "Sample demo data"], index=0)
    market = st.selectbox("Market", ["US", "UK"], index=0)
    benchmark_symbol = st.text_input("Benchmark symbol", value=BENCHMARKS[market])
    include_fundamentals = st.checkbox(
        "Fetch fundamentals from Yahoo Finance",
        value=True,
        help="This may be slower and can be incomplete for some stocks, especially UK shares.",
    )

    default_text = ", ".join(DEFAULT_TICKERS[market])
    ticker_text = st.text_area(
        "Tickers to scan",
        value=default_text,
        height=120,
        help="Separate tickers with commas. UK tickers normally use .L, e.g. BARC.L. If you type BARC in UK mode, the app converts it to BARC.L.",
    )
    max_tickers = st.slider("Maximum tickers per scan", 5, 40, 25, 5)

    st.header("User Parameters")
    min_up_days = st.slider("Minimum consecutive up days", 0, 10, 3)
    max_gap = st.slider("Maximum gap from 52-week high (%)", 0.0, 100.0, 5.0, 0.1)
    min_rel_volume = st.slider("Minimum relative volume", 0.0, 10.0, 1.5, 0.1)
    require_positive_eps = st.checkbox("Require positive EPS", value=False)
    require_rising_eps = st.checkbox("Require rising EPS", value=False)
    require_positive_pe = st.checkbox("Require positive P/E", value=False)
    require_positive_equity = st.checkbox("Require positive shareholder equity", value=False)
    require_rising_equity = st.checkbox("Require rising shareholder equity", value=False)
    min_alpha = st.slider("Minimum 1-year alpha vs benchmark (%)", -100.0, 200.0, 0.0, 1.0)
    min_beta, max_beta = st.slider("Acceptable beta range", -1.0, 5.0, (0.5, 2.5), 0.1)
    min_sharpe = st.slider("Minimum 1-year Sharpe ratio", -3.0, 5.0, 0.0, 0.1)
    max_volatility = st.slider("Maximum annualized volatility (%)", 5.0, 200.0, 120.0, 1.0)
    min_score = st.slider("Minimum Momentum Quality Score", 0, 100, 50)
    rsi_min, rsi_max = st.slider("Acceptable RSI range", 0, 100, (40, 90))
    risk_free_rate = st.slider("Risk-free rate for Sharpe calculation (%)", 0.0, 10.0, 4.0, 0.25) / 100


if data_mode == "Sample demo data":
    universe, benchmark = load_sample_universe()
    data_errors = []
    scanned_tickers = list(universe.keys())
else:
    scanned_tickers = clean_tickers(ticker_text, market=market, max_tickers=max_tickers)
    with st.spinner(f"Fetching Yahoo Finance data for {len(scanned_tickers)} tickers..."):
        universe, benchmark, data_errors = cached_live_universe(
            tuple(scanned_tickers), benchmark_symbol.strip().upper(), include_fundamentals
        )

if data_errors:
    with st.expander("Data warnings", expanded=False):
        for msg in data_errors[:25]:
            st.warning(msg)
        if len(data_errors) > 25:
            st.info(f"{len(data_errors) - 25} additional warnings hidden.")

if not universe or benchmark is None or benchmark.empty:
    st.error("No usable data was returned. Try fewer tickers, switch to sample demo data, or refresh later.")
    st.stop()

rows = []
for symbol, data in universe.items():
    try:
        rows.append(
            compute_metrics(
                symbol=symbol,
                company=data["company"],
                prices=data["prices"],
                benchmark_prices=benchmark,
                fundamentals=data["fundamentals"],
                sentiment_score=data.get("sentiment_score", 0.0),
                risk_free_rate=risk_free_rate,
            )
        )
    except Exception as exc:
        st.warning(f"Could not score {symbol}: {exc}")

if not rows:
    st.error("Data downloaded, but no stocks could be scored.")
    st.stop()

results = pd.DataFrame(rows)

# Filters. Missing fundamentals should not accidentally pass strict fundamental filters.
filtered = results.copy()
filtered = filtered[filtered["consecutive_up_days"] >= min_up_days]
filtered = filtered[filtered["gap_to_52w_high_pct"] <= max_gap]
filtered = filtered[filtered["relative_volume"] >= min_rel_volume]
filtered = filtered[filtered["alpha_1Y"] >= min_alpha]
filtered = filtered[filtered["beta_1Y"].between(min_beta, max_beta)]
filtered = filtered[filtered["sharpe_1Y"] >= min_sharpe]
filtered = filtered[filtered["volatility_1Y_pct"] <= max_volatility]
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
results = results.sort_values("momentum_quality_score", ascending=False)

summary_cols = [
    "symbol", "company", "momentum_quality_score", "consecutive_up_days",
    "latest_price", "high_52w", "gap_to_52w_high_pct", "relative_volume",
    "rsi", "beta_1Y", "sharpe_1Y", "volatility_1Y_pct",
    "eps", "eps_growth_pct", "pe_ratio", "equity_growth_pct",
    "return_5D", "return_1M", "return_3M", "return_6M", "return_1Y", "alpha_1Y",
    "win_rate_252D", "sentiment_score"
]
summary_cols = [c for c in summary_cols if c in results.columns]

format_map = {
    "momentum_quality_score": "{:.1f}",
    "latest_price": "{:.2f}",
    "high_52w": "{:.2f}",
    "gap_to_52w_high_pct": "{:.2f}%",
    "relative_volume": "{:.2f}x",
    "rsi": "{:.1f}",
    "beta_1Y": "{:.2f}",
    "sharpe_1Y": "{:.2f}",
    "volatility_1Y_pct": "{:.1f}%",
    "eps": "{:.2f}",
    "eps_growth_pct": "{:.1f}%",
    "pe_ratio": "{:.1f}",
    "equity_growth_pct": "{:.1f}%",
    "return_5D": "{:.1f}%",
    "return_1M": "{:.1f}%",
    "return_3M": "{:.1f}%",
    "return_6M": "{:.1f}%",
    "return_1Y": "{:.1f}%",
    "alpha_1Y": "{:.1f}%",
    "win_rate_252D": "{:.1f}%",
    "sentiment_score": "{:.2f}",
}

st.subheader("Scan Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers requested", len(scanned_tickers))
c2.metric("Stocks scored", len(results))
c3.metric("Matches", len(filtered))
c4.metric("Benchmark", benchmark_symbol.strip().upper() if data_mode != "Sample demo data" else "Sample")

st.subheader("Ranked Matches")
if filtered.empty:
    st.info("No stocks match the current filters. Try relaxing the sidebar filters.")
else:
    st.dataframe(
        filtered[summary_cols].style.format(format_map, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("All Stocks Scored")
st.dataframe(
    results[summary_cols].style.format(format_map, na_rep="—"),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Score Breakdown")
selected = st.selectbox("Select a stock", results["symbol"].tolist())
row = results[results["symbol"] == selected].iloc[0]
score_cols = [
    "price_momentum_score", "volume_score", "high_gap_score", "rsi_score",
    "fundamental_score", "equity_score", "alpha_quality_score", "risk_quality_score",
    "sentiment_component_score"
]
st.bar_chart(row[score_cols].astype(float))

st.markdown("""
### Phase 2 live-data notes

- This version uses **Yahoo Finance through `yfinance`** for a free-first MVP.
- US benchmark default: **S&P 500** (`^GSPC`).
- UK benchmark default: **FTSE 100** (`^FTSE`).
- UK shares normally require the `.L` suffix on Yahoo Finance, for example `BARC.L`, `LLOY.L`, `BP.L`.
- Data is cached for 1 hour to reduce rate-limit problems.
- Fundamentals from free sources may be missing or inconsistent; strict EPS/P/E/equity filters are therefore optional.

### Important
This app is a decision-support screener. It identifies shares matching selected rules; it does not guarantee profit or provide personal financial advice.
""")
