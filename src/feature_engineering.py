"""
feature_engineering.py
Transforms raw OHLCV data into a feature-rich dataset for modeling:
moving averages, RSI, MACD, Bollinger Bands, lag features, and rolling stats.

Usage (from project root):
    python -m src.feature_engineering
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    LAG_DAYS,
    ROLLING_WINDOWS,
    TARGET_COLUMN,
    PROCESSED_DATA_DIR,
    LOG_LEVEL,
)
from src.data_loader import fetch_stock_data, validate_data

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def add_moving_averages(df: pd.DataFrame, windows: list = ROLLING_WINDOWS) -> pd.DataFrame:
    """Simple and exponential moving averages over the closing price."""
    for w in windows:
        df[f"SMA_{w}"] = df[TARGET_COLUMN].rolling(window=w).mean()
        df[f"EMA_{w}"] = df[TARGET_COLUMN].ewm(span=w, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index — momentum oscillator (0-100).
    >70 conventionally read as overbought, <30 as oversold.
    """
    delta = df[TARGET_COLUMN].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence) — trend-following momentum indicator.
    Returns the MACD line, signal line, and histogram (their difference).
    """
    ema_fast = df[TARGET_COLUMN].ewm(span=fast, adjust=False).mean()
    ema_slow = df[TARGET_COLUMN].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands — volatility bands around a moving average."""
    sma = df[TARGET_COLUMN].rolling(window=window).mean()
    std = df[TARGET_COLUMN].rolling(window=window).std()

    df["BB_middle"] = sma
    df["BB_upper"] = sma + (num_std * std)
    df["BB_lower"] = sma - (num_std * std)
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]
    return df


def add_lag_features(df: pd.DataFrame, lags: list = LAG_DAYS) -> pd.DataFrame:
    """Lagged closing prices — lets tree-based models 'see' recent history without sequences."""
    for lag in lags:
        df[f"{TARGET_COLUMN}_lag_{lag}"] = df[TARGET_COLUMN].shift(lag)
    return df


def add_rolling_stats(df: pd.DataFrame, windows: list = ROLLING_WINDOWS) -> pd.DataFrame:
    """Rolling volatility and range features."""
    for w in windows:
        df[f"rolling_std_{w}"] = df[TARGET_COLUMN].rolling(window=w).std()
        df[f"rolling_max_{w}"] = df[TARGET_COLUMN].rolling(window=w).max()
        df[f"rolling_min_{w}"] = df[TARGET_COLUMN].rolling(window=w).min()
    return df


def add_volume_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Volume-based features — sudden volume spikes often precede price moves."""
    df["volume_change_pct"] = df["Volume"].pct_change()
    df[f"volume_avg_{window}"] = df["Volume"].rolling(window=window).mean()
    return df


def add_target_variable(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Prediction target: next-day closing price (default horizon=1).
    Kept separate and explicit so it's obvious what the models are trained to predict.
    """
    df["target"] = df[TARGET_COLUMN].shift(-horizon)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline in sequence."""
    logger.info("Building technical indicators and features...")
    df = df.copy()

    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_lag_features(df)
    df = add_rolling_stats(df)
    df = add_volume_features(df)
    df = add_target_variable(df)

    before = len(df)
    df.dropna(inplace=True)   # rolling windows/lags create NaNs at the start of the series
    after = len(df)
    logger.info(f"Dropped {before - after} rows with NaNs from rolling/lag windows. {after} rows remain.")

    return df


if __name__ == "__main__":
    raw_df = fetch_stock_data()
    validate_data(raw_df)

    featured_df = build_features(raw_df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "featured_data.csv"
    featured_df.to_csv(out_path)

    logger.info(f"Saved featured dataset to {out_path}")
    logger.info(f"Final shape: {featured_df.shape}")
    print(featured_df.tail())