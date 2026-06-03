from __future__ import annotations

import numpy as np
import pandas as pd


def make_price_series(seed: int, start_price: float, trend: float, volatility: float, force_up_days: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=280)
    returns = rng.normal(trend / 252, volatility / np.sqrt(252), size=len(dates))
    prices = start_price * np.exp(np.cumsum(returns))

    if force_up_days > 0:
        base = prices[-force_up_days - 1]
        increments = np.linspace(0.004, 0.025, force_up_days)
        for i in range(force_up_days):
            base *= 1 + increments[i]
            prices[-force_up_days + i] = base

    volumes = rng.integers(500_000, 8_000_000, len(dates))
    if force_up_days > 0:
        volumes[-force_up_days:] = volumes[-force_up_days:] * rng.integers(2, 5)

    return pd.DataFrame({"close": prices, "volume": volumes}, index=dates)


def load_sample_universe():
    benchmark = make_price_series(999, 500, 0.15, 0.12, 0)
    universe = {
        "BB": {
            "company": "BlackBerry",
            "prices": make_price_series(1, 4.2, 1.15, 0.45, 7),
            "fundamentals": {
                "eps": 0.42,
                "previous_eps": 0.18,
                "eps_growth_pct": 133.3,
                "pe_ratio": 26.7,
                "shareholder_equity": 2_100_000_000,
                "previous_shareholder_equity": 1_880_000_000,
                "equity_growth_pct": 11.7,
            },
            "sentiment_score": 0.72,
        },
        "NVDA": {
            "company": "Nvidia",
            "prices": make_price_series(2, 85, 0.75, 0.32, 4),
            "fundamentals": {
                "eps": 2.9,
                "previous_eps": 2.2,
                "eps_growth_pct": 31.8,
                "pe_ratio": 38.0,
                "shareholder_equity": 55_000_000_000,
                "previous_shareholder_equity": 48_000_000_000,
                "equity_growth_pct": 14.6,
            },
            "sentiment_score": 0.55,
        },
        "XYZ": {
            "company": "Example Recovery Plc",
            "prices": make_price_series(3, 20, -0.25, 0.55, 6),
            "fundamentals": {
                "eps": -0.25,
                "previous_eps": -0.40,
                "eps_growth_pct": 37.5,
                "pe_ratio": -12.0,
                "shareholder_equity": -500_000_000,
                "previous_shareholder_equity": -450_000_000,
                "equity_growth_pct": -11.1,
            },
            "sentiment_score": -0.35,
        },
        "ABC": {
            "company": "Example Quality Corp",
            "prices": make_price_series(4, 35, 0.32, 0.20, 3),
            "fundamentals": {
                "eps": 1.85,
                "previous_eps": 1.65,
                "eps_growth_pct": 12.1,
                "pe_ratio": 18.5,
                "shareholder_equity": 4_200_000_000,
                "previous_shareholder_equity": 3_900_000_000,
                "equity_growth_pct": 7.7,
            },
            "sentiment_score": 0.22,
        },
        "HOT": {
            "company": "Example Hot Momentum Inc",
            "prices": make_price_series(5, 9, 0.95, 0.75, 10),
            "fundamentals": {
                "eps": 0.05,
                "previous_eps": -0.05,
                "eps_growth_pct": 200.0,
                "pe_ratio": 180.0,
                "shareholder_equity": 120_000_000,
                "previous_shareholder_equity": 80_000_000,
                "equity_growth_pct": 50.0,
            },
            "sentiment_score": 0.85,
        },
    }
    return universe, benchmark
