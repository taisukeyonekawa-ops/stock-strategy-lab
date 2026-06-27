import argparse
import os
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "reports"
CACHE_DIR = ROOT_DIR / ".cache"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))

import matplotlib.pyplot as plt


def fetch_prices(ticker: str, period: str) -> pd.DataFrame:
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No price data found for ticker: {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index.name = "date"
    return data


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    close = result["Close"]

    result["daily_return"] = close.pct_change()
    result["cumulative_return"] = (1 + result["daily_return"]).cumprod() - 1
    result["ma_20"] = close.rolling(window=20).mean()
    result["ma_50"] = close.rolling(window=50).mean()

    return result


def save_chart(data: pd.DataFrame, ticker: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / f"{ticker.replace('.', '_')}_chart.png"

    plt.figure(figsize=(12, 7))
    plt.plot(data.index, data["Close"], label="Close", linewidth=1.8)
    plt.plot(data.index, data["ma_20"], label="MA 20", linewidth=1.2)
    plt.plot(data.index, data["ma_50"], label="MA 50", linewidth=1.2)
    plt.title(f"{ticker} Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path


def save_summary(data: pd.DataFrame, ticker: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / f"{ticker.replace('.', '_')}_analysis.csv"
    data.to_csv(output_path)
    return output_path


def print_summary(data: pd.DataFrame, ticker: str) -> None:
    latest = data.dropna().iloc[-1]
    start_close = data["Close"].dropna().iloc[0]
    latest_close = latest["Close"]
    total_return = latest_close / start_close - 1

    print(f"Ticker: {ticker}")
    print(f"Latest close: {latest_close:.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"20-day moving average: {latest['ma_20']:.2f}")
    print(f"50-day moving average: {latest['ma_50']:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze stock price trends.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, e.g. AAPL or 7203.T")
    parser.add_argument("--period", default="1y", help="Data period, e.g. 6mo, 1y, 5y")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = fetch_prices(args.ticker, args.period)
    analysis = add_indicators(prices)

    csv_path = save_summary(analysis, args.ticker)
    chart_path = save_chart(analysis, args.ticker)

    print_summary(analysis, args.ticker)
    print(f"CSV saved: {csv_path}")
    print(f"Chart saved: {chart_path}")


if __name__ == "__main__":
    main()
