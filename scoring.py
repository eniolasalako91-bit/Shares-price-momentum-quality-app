from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


LOOKBACKS = {
    "5D": 5,
    "1W": 5,
    "2W": 10,
    "3W": 15,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "9M": 189,
    "1Y": 252,
}

TRADING_DAYS = 252


def pct_return(series: pd.Series, periods: int) -> float:
    if len(series) <= periods or series.iloc[-periods - 1] == 0:
        return np.nan
    return (series.iloc[-1] / series.iloc[-periods - 1] - 1) * 100


def consecutive_up_days(close: pd.Series, max_days: int = 10) -> int:
    diffs = close.diff().dropna()
    count = 0
    for value in reversed(diffs.tail(max_days).tolist()):
        if value > 0:
            count += 1
        else:
            break
    return count


def rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return np.nan
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs.iloc[-1]))
    if np.isnan(value):
        return 100.0 if gain.iloc[-1] > 0 else np.nan
    return float(value)


def win_rate(close: pd.Series, periods: int = 252) -> float:
    changes = close.diff().dropna().tail(periods)
    if len(changes) == 0:
        return np.nan
    return (changes.gt(0).sum() / len(changes)) * 100


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def beta_vs_benchmark(close: pd.Series, benchmark_close: pd.Series, periods: int = 252) -> float:
    """Beta = covariance(stock returns, benchmark returns) / variance(benchmark returns)."""
    aligned = pd.concat(
        [daily_returns(close).rename("stock"), daily_returns(benchmark_close).rename("benchmark")],
        axis=1,
    ).dropna().tail(periods)
    if len(aligned) < 30:
        return np.nan
    benchmark_variance = aligned["benchmark"].var()
    if benchmark_variance == 0 or np.isnan(benchmark_variance):
        return np.nan
    return float(aligned["stock"].cov(aligned["benchmark"]) / benchmark_variance)


def annualized_volatility(close: pd.Series, periods: int = 252) -> float:
    returns = daily_returns(close).tail(periods)
    if len(returns) < 30:
        return np.nan
    return float(returns.std() * np.sqrt(TRADING_DAYS) * 100)


def sharpe_ratio(close: pd.Series, risk_free_rate: float = 0.04, periods: int = 252) -> float:
    """Annualized Sharpe ratio using daily returns and a default 4% risk-free rate."""
    returns = daily_returns(close).tail(periods)
    if len(returns) < 30 or returns.std() == 0:
        return np.nan
    daily_risk_free = risk_free_rate / TRADING_DAYS
    excess_daily = returns - daily_risk_free
    return float((excess_daily.mean() / returns.std()) * np.sqrt(TRADING_DAYS))


def normalize(value: float, low: float, high: float, inverse: bool = False) -> float:
    if value is None or np.isnan(value):
        return 0.0
    if high == low:
        return 0.0
    score = (value - low) / (high - low) * 100
    score = max(0.0, min(100.0, score))
    return 100 - score if inverse else score


def beta_score(beta_value: float) -> float:
    """Reward useful momentum risk: not too defensive, not dangerously volatile."""
    if beta_value is None or np.isnan(beta_value):
        return 0.0
    if 0.8 <= beta_value <= 1.8:
        return 100.0
    if 0.5 <= beta_value < 0.8:
        return 75.0
    if 1.8 < beta_value <= 2.5:
        return 70.0
    if 0.0 <= beta_value < 0.5:
        return 45.0
    if 2.5 < beta_value <= 3.5:
        return 35.0
    return 15.0


def score_row(row: Dict) -> Dict:
    """Create component scores and final Momentum Quality Score."""
    up_days_score = normalize(row.get("consecutive_up_days", 0), 0, 10)
    returns_score = np.nanmean([
        normalize(row.get("return_5D", np.nan), -10, 20),
        normalize(row.get("return_1M", np.nan), -20, 60),
        normalize(row.get("return_3M", np.nan), -30, 100),
        normalize(row.get("return_1Y", np.nan), -50, 200),
    ])
    price_momentum = np.nanmean([up_days_score, returns_score])

    relative_volume_score = normalize(row.get("relative_volume", np.nan), 0.5, 5.0)
    high_gap_score = normalize(row.get("gap_to_52w_high_pct", np.nan), 0, 15, inverse=True)

    rsi_value = row.get("rsi", np.nan)
    if np.isnan(rsi_value):
        rsi_component = 0
    elif 55 <= rsi_value <= 75:
        rsi_component = 100
    elif 45 <= rsi_value < 55:
        rsi_component = 65
    elif 75 < rsi_value <= 85:
        rsi_component = 70
    elif rsi_value > 85:
        rsi_component = 45
    else:
        rsi_component = 25

    eps_score = 0
    if row.get("eps", np.nan) > 0:
        eps_score += 45
    if row.get("eps_growth_pct", np.nan) > 0:
        eps_score += 55

    pe = row.get("pe_ratio", np.nan)
    if np.isnan(pe) or pe <= 0:
        pe_score = 0
    elif pe <= 40:
        pe_score = 100
    elif pe <= 80:
        pe_score = 65
    else:
        pe_score = 35

    fundamentals = np.nanmean([eps_score, pe_score])

    equity_component = 0
    if row.get("shareholder_equity", np.nan) > 0:
        equity_component += 50
    if row.get("equity_growth_pct", np.nan) > 0:
        equity_component += 50

    alpha_score = normalize(row.get("alpha_1Y", np.nan), -50, 100)
    winrate_score = normalize(row.get("win_rate_252D", np.nan), 40, 70)
    alpha_quality = np.nanmean([alpha_score, winrate_score])

    beta_component = beta_score(row.get("beta_1Y", np.nan))
    sharpe_component = normalize(row.get("sharpe_1Y", np.nan), -1, 3)
    volatility_component = normalize(row.get("volatility_1Y_pct", np.nan), 20, 120, inverse=True)
    risk_quality = np.nanmean([beta_component, sharpe_component, volatility_component])

    sentiment_component = normalize(row.get("sentiment_score", 0), -1, 1)

    final = (
        price_momentum * 0.18
        + relative_volume_score * 0.12
        + high_gap_score * 0.08
        + rsi_component * 0.08
        + fundamentals * 0.14
        + equity_component * 0.10
        + alpha_quality * 0.10
        + risk_quality * 0.10
        + sentiment_component * 0.10
    )

    return {
        "price_momentum_score": round(price_momentum, 1),
        "volume_score": round(relative_volume_score, 1),
        "high_gap_score": round(high_gap_score, 1),
        "rsi_score": round(rsi_component, 1),
        "fundamental_score": round(fundamentals, 1),
        "equity_score": round(equity_component, 1),
        "alpha_quality_score": round(alpha_quality, 1),
        "risk_quality_score": round(risk_quality, 1),
        "sentiment_component_score": round(sentiment_component, 1),
        "momentum_quality_score": round(final, 1),
    }


def compute_metrics(
    symbol: str,
    company: str,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    fundamentals: Dict,
    sentiment_score: float = 0.0,
    risk_free_rate: float = 0.04,
) -> Dict:
    df = prices.copy().sort_index()
    close = df["close"]
    volume = df["volume"]
    benchmark_close = benchmark_prices["close"].sort_index()

    latest_price = float(close.iloc[-1])
    high_52w = float(close.tail(252).max())
    gap = ((high_52w - latest_price) / high_52w) * 100 if high_52w else np.nan
    rel_vol = float(volume.iloc[-1] / volume.tail(30).mean()) if len(volume) >= 30 else np.nan

    row = {
        "symbol": symbol,
        "company": company,
        "latest_price": latest_price,
        "high_52w": high_52w,
        "gap_to_52w_high_pct": gap,
        "latest_volume": int(volume.iloc[-1]),
        "avg_volume_30D": int(volume.tail(30).mean()),
        "relative_volume": rel_vol,
        "consecutive_up_days": consecutive_up_days(close),
        "rsi": rsi(close),
        "win_rate_252D": win_rate(close, 252),
        "beta_1Y": beta_vs_benchmark(close, benchmark_close, 252),
        "sharpe_1Y": sharpe_ratio(close, risk_free_rate, 252),
        "volatility_1Y_pct": annualized_volatility(close, 252),
        "sentiment_score": sentiment_score,
        **fundamentals,
    }

    for label, days in LOOKBACKS.items():
        row[f"return_{label}"] = pct_return(close, days)
        stock_return = row[f"return_{label}"]
        bench_return = pct_return(benchmark_close, days)
        row[f"benchmark_return_{label}"] = bench_return
        row[f"alpha_{label}"] = stock_return - bench_return if not np.isnan(stock_return) and not np.isnan(bench_return) else np.nan

    row.update(score_row(row))
    return row
