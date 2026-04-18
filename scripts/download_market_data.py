from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import yaml


def download_from_stooq(ticker: str) -> pd.DataFrame:
    symbol = f"{ticker.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    if "Close" in df.columns and "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    return df


def generate_synthetic_ohlcv(start: str, end: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    if len(dates) == 0:
        raise ValueError("No business days available for synthetic market generation.")

    regime_noise = rng.normal(0.0002, 0.012, size=len(dates))
    regime_trend = 0.00015 * np.sin(np.linspace(0, 8 * np.pi, len(dates)))
    returns = regime_noise + regime_trend
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.002, size=len(dates)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.01, size=len(dates)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.01, size=len(dates)))
    volume = rng.integers(5_000_000, 20_000_000, size=len(dates))

    return pd.DataFrame(
        {
            "Date": dates,
            "Adj Close": close,
            "Close": close,
            "High": high,
            "Low": low,
            "Open": open_,
            "Volume": volume,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download market data for RL trading experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    md = config["market_data"]
    ticker = md["ticker"]
    df = yf.download(
        ticker,
        start=md["start"],
        end=md["end"],
        interval=md["interval"],
        auto_adjust=False,
        progress=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if df.empty:
        print(f"[data] yahoo download returned no rows for ticker={ticker}; falling back to stooq")
        try:
            df = download_from_stooq(ticker)
        except Exception as exc:
            print(f"[data] stooq fallback failed: {exc}")
            df = pd.DataFrame()

        if df.empty:
            print(f"[data] external sources unavailable; generating synthetic OHLCV for ticker={ticker}")
            df = generate_synthetic_ohlcv(md["start"], md["end"], seed=int(config.get("seed", 42)))

    out = Path(md["output_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(out, index=False)
    print(f"[data] saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
