from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


BENCHMARKS = {
    "US": "^GSPC",   # S&P 500
    "UK": "^FTSE",   # FTSE 100
}

DEFAULT_TICKERS = {
    "US": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
        "LLY", "UNH", "XOM", "COST", "MA", "NFLX", "AMD", "CRM", "BB", "PLTR",
    ],
    "UK": [
        "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "DGE.L",
        "LLOY.L", "BARC.L", "VOD.L", "NWG.L", "RR.L", "GLEN.L", "TSCO.L", "PRU.L",
    ],
}


def clean_tickers(raw: str | Iterable[str], market: str = "US", max_tickers: int = 40) -> List[str]:
    """Clean ticker input. UK tickers should normally end in .L for Yahoo Finance."""
    if isinstance(raw, str):
        parts = raw.replace("\n", ",").replace(";", ",").split(",")
    else:
        parts = list(raw)

    tickers: List[str] = []
    for part in parts:
        ticker = str(part).strip().upper()
        if not ticker:
            continue
        if market == "UK" and ticker.startswith("^") is False and "." not in ticker:
            ticker = f"{ticker}.L"
        if ticker not in tickers:
            tickers.append(ticker)
    return tickers[:max_tickers]


def _standardize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a lowercase close/volume DataFrame for the scoring engine."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["close", "volume"])

    out = pd.DataFrame(index=pd.to_datetime(df.index))
    close_col = "Close" if "Close" in df.columns else "Adj Close" if "Adj Close" in df.columns else None
    if close_col is None:
        return pd.DataFrame(columns=["close", "volume"])
    out["close"] = pd.to_numeric(df[close_col], errors="coerce")
    out["volume"] = pd.to_numeric(df.get("Volume", np.nan), errors="coerce").fillna(0)
    out = out.dropna(subset=["close"])
    return out


def download_prices(tickers: List[str], period: str = "18mo") -> Dict[str, pd.DataFrame]:
    """Batch-download price and volume history from Yahoo Finance."""
    if not tickers:
        return {}

    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    prices: Dict[str, pd.DataFrame] = {}

    if len(tickers) == 1:
        prices[tickers[0]] = _standardize_price_frame(raw)
        return prices

    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
                prices[ticker] = _standardize_price_frame(raw[ticker])
            else:
                prices[ticker] = pd.DataFrame(columns=["close", "volume"])
        except Exception:
            prices[ticker] = pd.DataFrame(columns=["close", "volume"])

    return prices


def _safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if np.isinf(value):
            return default
        return value
    except Exception:
        return default


def get_fundamentals(ticker: str) -> Dict[str, float]:
    """Fetch basic fundamentals from yfinance.

    Notes:
    - yfinance fundamentals can be incomplete for some UK stocks.
    - This function is intentionally defensive so missing data does not break the app.
    """
    eps = previous_eps = pe_ratio = shareholder_equity = previous_shareholder_equity = np.nan
    eps_growth_pct = equity_growth_pct = np.nan
    company_name = ticker

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        company_name = info.get("shortName") or info.get("longName") or ticker
        eps = _safe_float(info.get("trailingEps"))
        pe_ratio = _safe_float(info.get("trailingPE"))

        # Yahoo sometimes provides earnings growth as a decimal, e.g. 0.25 = 25%.
        earnings_growth = _safe_float(info.get("earningsQuarterlyGrowth"))
        if not np.isnan(earnings_growth):
            eps_growth_pct = earnings_growth * 100
            if not np.isnan(eps) and earnings_growth != -1:
                previous_eps = eps / (1 + earnings_growth)

        # Balance sheet: latest and previous shareholder equity when available.
        try:
            bs = tk.balance_sheet
            if bs is not None and not bs.empty:
                possible_rows = [
                    "Stockholders Equity",
                    "Total Stockholder Equity",
                    "Common Stock Equity",
                    "Total Equity Gross Minority Interest",
                ]
                row_name = next((r for r in possible_rows if r in bs.index), None)
                if row_name is not None:
                    values = pd.to_numeric(bs.loc[row_name], errors="coerce").dropna()
                    if len(values) >= 1:
                        shareholder_equity = _safe_float(values.iloc[0])
                    if len(values) >= 2:
                        previous_shareholder_equity = _safe_float(values.iloc[1])
                    if previous_shareholder_equity and not np.isnan(previous_shareholder_equity) and previous_shareholder_equity != 0:
                        equity_growth_pct = ((shareholder_equity - previous_shareholder_equity) / abs(previous_shareholder_equity)) * 100
        except Exception:
            pass
    except Exception:
        pass

    return {
        "company": company_name,
        "eps": eps,
        "previous_eps": previous_eps,
        "eps_growth_pct": eps_growth_pct,
        "pe_ratio": pe_ratio,
        "shareholder_equity": shareholder_equity,
        "previous_shareholder_equity": previous_shareholder_equity,
        "equity_growth_pct": equity_growth_pct,
    }


def load_live_universe(
    tickers: List[str],
    benchmark_symbol: str,
    period: str = "18mo",
    include_fundamentals: bool = True,
) -> Tuple[Dict[str, Dict], pd.DataFrame, List[str]]:
    """Return universe dict, benchmark DataFrame, and error messages."""
    errors: List[str] = []
    all_symbols = list(dict.fromkeys(tickers + [benchmark_symbol]))
    price_map = download_prices(all_symbols, period=period)

    benchmark = price_map.get(benchmark_symbol, pd.DataFrame())
    if benchmark.empty:
        errors.append(f"Could not download benchmark data for {benchmark_symbol}.")

    universe: Dict[str, Dict] = {}
    for ticker in tickers:
        price_df = price_map.get(ticker, pd.DataFrame())
        if price_df is None or price_df.empty or len(price_df) < 60:
            errors.append(f"Skipped {ticker}: not enough price history returned.")
            continue

        fundamentals = get_fundamentals(ticker) if include_fundamentals else {
            "company": ticker,
            "eps": np.nan,
            "previous_eps": np.nan,
            "eps_growth_pct": np.nan,
            "pe_ratio": np.nan,
            "shareholder_equity": np.nan,
            "previous_shareholder_equity": np.nan,
            "equity_growth_pct": np.nan,
        }
        company = fundamentals.pop("company", ticker)

        universe[ticker] = {
            "company": company,
            "prices": price_df,
            "fundamentals": fundamentals,
            # Free-first Phase 2 does not yet include news sentiment.
            "sentiment_score": 0.0,
        }

    return universe, benchmark, errors
