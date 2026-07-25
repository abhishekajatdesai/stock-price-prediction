
"""
data_loader.py
Handles fetching raw stock price data from Yahoo Finance and caching it locally.
 
Usage (from project root):
    python -m src.data_loader
"""
 
import logging
import sys
from pathlib import Path
 
import pandas as pd
import yfinance as yf
 
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TICKER, START_DATE, END_DATE, INTERVAL, RAW_DATA_DIR, LOG_LEVEL
 
# ---------- Logging setup ----------
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
 
 
def get_raw_data_path(ticker: str = TICKER) -> Path:
    """Return the expected local cache path for a given ticker's raw data."""
    safe_name = ticker.replace(".", "_")
    return RAW_DATA_DIR / f"{safe_name}_raw.csv"
 
 
def fetch_stock_data(
    ticker: str = TICKER,
    start: str = START_DATE,
    end: str = END_DATE,
    interval: str = INTERVAL,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a ticker, using a local cache when available.
 
    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol, e.g. "RELIANCE.NS".
    start, end : str
        Date range in "YYYY-MM-DD" format.
    interval : str
        Candle interval, e.g. "1d", "1wk".
    force_refresh : bool
        If True, ignore any local cache and re-download from Yahoo Finance.
 
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Date with columns: Open, High, Low, Close, Adj Close, Volume.
    """
    cache_path = get_raw_data_path(ticker)
 
    if cache_path.exists() and not force_refresh:
        logger.info(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df
 
    logger.info(f"Downloading {ticker} data from Yahoo Finance ({start} to {end})...")
 
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    except Exception as e:
        logger.error(f"Failed to download data for {ticker}: {e}")
        raise
 
    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Check the symbol is correct (NSE tickers need a '.NS' suffix)."
        )
 
    # yfinance sometimes returns MultiIndex columns when multiple tickers are passed —
    # flatten just in case, so downstream code always sees plain column names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
 
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    logger.info(f"Saved {len(df)} rows to {cache_path}")
 
    return df
 
 
def validate_data(df: pd.DataFrame) -> None:
    """Basic sanity checks so silent data issues don't propagate downstream."""
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
 
    null_counts = df[list(required_cols)].isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Found missing values:\n{null_counts[null_counts > 0]}")
 
    if not df.index.is_monotonic_increasing:
        logger.warning("Date index is not sorted ascending — sorting now.")
        df.sort_index(inplace=True)
 
    logger.info(f"Validation passed. Date range: {df.index.min()} to {df.index.max()}, rows: {len(df)}")
 
 
if __name__ == "__main__":
    data = fetch_stock_data(force_refresh=False)
    validate_data(data)
    print(data.tail())