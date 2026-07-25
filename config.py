"""
Central configuration for the Stock Price Prediction project.
Keeping all tunable values here means changing the ticker, date range,
or model settings never requires touching the actual logic files.
"""
 
from pathlib import Path
 
# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
 
# ---------- Data settings ----------
TICKER = "RELIANCE.NS"          # NSE-listed Reliance Industries
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
INTERVAL = "1d"                 # daily candles
 
# ---------- Feature engineering ----------
LAG_DAYS = [1, 2, 3, 5, 10]          # lag features to generate
ROLLING_WINDOWS = [5, 10, 20, 50]    # moving average / rolling stat windows
TARGET_COLUMN = "Close"
 
# ---------- Train/test split ----------
TEST_SIZE = 0.2                 # last 20% of the timeline held out (no shuffling — time series!)
 
# ---------- LSTM settings ----------
LSTM_LOOKBACK = 60               # days of history fed into each LSTM prediction
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
 
# ---------- Logging ----------
LOG_LEVEL = "INFO"
 