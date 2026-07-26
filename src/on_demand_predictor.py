"""
on_demand_predictor.py
Runs the full pipeline (fetch/load -> feature engineer -> train 3 models ->
predict) for ANY ticker or user-uploaded CSV, without touching the saved
RELIANCE.NS models used elsewhere in this project.

Because this trains fresh models on every request, it is intentionally slower
than the main app's predictions (LSTM training in particular takes real time,
even with reduced epochs below). That trade-off was a deliberate choice to
keep results consistent with the project's main analysis, rather than
skipping LSTM for speed.

Usage:
    from src.on_demand_predictor import run_on_demand_prediction
    result = run_on_demand_prediction(ticker="TCS.NS")
    result = run_on_demand_prediction(uploaded_csv_path="/path/to/file.csv")
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TEST_SIZE, LSTM_LOOKBACK, LOG_LEVEL
from src.feature_engineering import build_features
from src.train_lr import time_series_split
from src.evaluate import evaluate_predictions

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Reduced epochs for on-demand LSTM training — full 50 epochs would make a
# single web request take several minutes. Early stopping still applies.
ON_DEMAND_LSTM_EPOCHS = 25
MIN_ROWS_REQUIRED = LSTM_LOOKBACK + 100  # need enough data for sequences + a meaningful test set


class OnDemandError(Exception):
    """Raised for user-facing errors (bad ticker, insufficient data, bad CSV) — caller shows the message directly."""
    pass


def _load_raw_data(ticker: str = None, uploaded_csv_path: str = None) -> pd.DataFrame:
    if uploaded_csv_path:
        try:
            df = pd.read_csv(uploaded_csv_path, index_col=0, parse_dates=True)
        except Exception as e:
            raise OnDemandError(f"Could not read the uploaded CSV: {e}")

        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise OnDemandError(
                f"Uploaded CSV is missing required columns: {missing}. "
                "Expected format: Date index + Open, High, Low, Close, Volume columns "
                "(standard Yahoo Finance export format)."
            )
        df.sort_index(inplace=True)
        return df

    if ticker:
        logger.info(f"Downloading data for {ticker}...")
        try:
            df = yf.download(ticker, period="10y", interval="1d", progress=False)
        except Exception as e:
            raise OnDemandError(f"Failed to download data for '{ticker}': {e}")

        if df.empty:
            raise OnDemandError(
                f"No data found for ticker '{ticker}'. Check the symbol is correct "
                "(NSE tickers need a '.NS' suffix, e.g. 'TCS.NS')."
            )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    raise OnDemandError("Either a ticker or an uploaded CSV must be provided.")


def _train_lr_quick(train_df, test_df, feature_cols):
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = evaluate_predictions(y_test.values, y_pred, model_name="Linear Regression")

    latest_scaled = scaler.transform(test_df[feature_cols].iloc[[-1]])
    next_day = float(model.predict(latest_scaled)[0])

    chart_data = {
        "dates": [d.strftime("%Y-%m-%d") for d in test_df.index],
        "actual": [round(float(v), 2) for v in y_test.values],
        "predicted": [round(float(v), 2) for v in y_pred],
    }
    return next_day, metrics, chart_data


def _train_gb_quick(train_df, test_df, feature_cols):
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_predictions(y_test.values, y_pred, model_name="Gradient Boosting")

    next_day = float(model.predict(test_df[feature_cols].iloc[[-1]])[0])

    chart_data = {
        "dates": [d.strftime("%Y-%m-%d") for d in test_df.index],
        "actual": [round(float(v), 2) for v in y_test.values],
        "predicted": [round(float(v), 2) for v in y_pred],
    }
    return next_day, metrics, chart_data


def _train_lstm_quick(df, feature_cols, split_idx):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_train_raw = feature_scaler.fit_transform(df[feature_cols].iloc[:split_idx])
    X_test_raw = feature_scaler.transform(df[feature_cols].iloc[split_idx:])
    y_train_raw = target_scaler.fit_transform(df[["target"]].iloc[:split_idx]).flatten()
    y_test_raw = target_scaler.transform(df[["target"]].iloc[split_idx:]).flatten()

    def make_sequences(X, y, lookback):
        Xs, ys = [], []
        for i in range(lookback, len(X)):
            Xs.append(X[i - lookback:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    X_train, y_train = make_sequences(X_train_raw, y_train_raw, LSTM_LOOKBACK)
    X_test, y_test = make_sequences(X_test_raw, y_test_raw, LSTM_LOOKBACK)

    if len(X_train) < 30 or len(X_test) < 5:
        raise OnDemandError(
            "Not enough historical data for LSTM training after sequence windowing. "
            "Try a ticker/CSV with more history."
        )

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train, validation_split=0.1,
        epochs=ON_DEMAND_LSTM_EPOCHS, batch_size=32,
        callbacks=[early_stop], verbose=0,
    )

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    metrics = evaluate_predictions(y_true, y_pred, model_name="LSTM")

    latest_seq = feature_scaler.transform(df[feature_cols].iloc[-LSTM_LOOKBACK:])
    latest_seq = latest_seq.reshape(1, LSTM_LOOKBACK, len(feature_cols))
    next_scaled = model.predict(latest_seq, verbose=0)
    next_day = float(target_scaler.inverse_transform(next_scaled)[0][0])

    # LSTM's test predictions start LSTM_LOOKBACK rows into the test split,
    # since the first `lookback` rows are consumed building the first sequence.
    test_dates = df.index[split_idx + LSTM_LOOKBACK:]
    chart_data = {
        "dates": [d.strftime("%Y-%m-%d") for d in test_dates],
        "actual": [round(float(v), 2) for v in y_true],
        "predicted": [round(float(v), 2) for v in y_pred],
    }

    return next_day, metrics, chart_data


def run_on_demand_prediction(ticker: str = None, uploaded_csv_path: str = None,
                              company_label: str = None) -> dict:
    """
    Full on-demand pipeline: fetch/load data, engineer features, train all 3
    models fresh, and return next-day predictions + evaluation metrics.

    Raises OnDemandError with a user-facing message for any recoverable failure
    (bad ticker, insufficient data, malformed upload).
    """
    raw_df = _load_raw_data(ticker=ticker, uploaded_csv_path=uploaded_csv_path)

    if len(raw_df) < MIN_ROWS_REQUIRED:
        raise OnDemandError(
            f"Not enough historical data ({len(raw_df)} rows found, "
            f"{MIN_ROWS_REQUIRED} required) to train reliably. "
            "Try a ticker with a longer trading history."
        )

    df = build_features(raw_df)
    feature_cols = [c for c in df.columns if c != "target"]

    train_df, test_df = time_series_split(df, test_size=TEST_SIZE)
    split_idx = len(train_df)

    latest_close = float(df.iloc[-1]["Close"])
    latest_date = df.index[-1].strftime("%Y-%m-%d")

    predictions = {"Naive Persistence": round(latest_close, 2)}
    metrics_all = []
    chart_data_all = {}

    logger.info("Training Linear Regression...")
    lr_next, lr_metrics, lr_chart = _train_lr_quick(train_df, test_df, feature_cols)
    predictions["Linear Regression"] = round(lr_next, 2)
    metrics_all.append(lr_metrics)
    chart_data_all["Linear Regression"] = lr_chart

    logger.info("Training Gradient Boosting...")
    gb_next, gb_metrics, gb_chart = _train_gb_quick(train_df, test_df, feature_cols)
    predictions["Gradient Boosting"] = round(gb_next, 2)
    metrics_all.append(gb_metrics)
    chart_data_all["Gradient Boosting"] = gb_chart

    logger.info("Training LSTM (this is the slow part — please wait)...")
    lstm_next, lstm_metrics, lstm_chart = _train_lstm_quick(df, feature_cols, split_idx)
    predictions["LSTM"] = round(lstm_next, 2)
    metrics_all.append(lstm_metrics)
    chart_data_all["LSTM"] = lstm_chart

    naive_metrics = evaluate_predictions(
        test_df["target"].values, test_df["Close"].values, model_name="Naive Persistence"
    )
    metrics_all.insert(0, naive_metrics)
    chart_data_all["Naive Persistence"] = {
        "dates": [d.strftime("%Y-%m-%d") for d in test_df.index],
        "actual": [round(float(v), 2) for v in test_df["target"].values],
        "predicted": [round(float(v), 2) for v in test_df["Close"].values],
    }

    return {
        "company_label": company_label or ticker or "Uploaded Data",
        "ticker": ticker or "Uploaded CSV",
        "latest_date": latest_date,
        "latest_close": round(latest_close, 2),
        "predictions": predictions,
        "metrics": metrics_all,
        "chart_data": chart_data_all,
    }