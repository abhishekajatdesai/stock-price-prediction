"""
train_lstm.py
LSTM (Long Short-Term Memory) model — the only one of the three that sees data
as an actual sequence, rather than independent rows. It looks at the last
LSTM_LOOKBACK days of all features to predict the next day's close.

Usage (from project root):
    python -m src.train_lstm
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TEST_SIZE,
    LSTM_LOOKBACK,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LOG_LEVEL,
)
from src.evaluate import evaluate_predictions, save_results
from src.train_lr import load_featured_data

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------- TensorFlow install check ----------
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    logger.error(
        "TensorFlow is not installed in this environment.\n"
        "Install it with:\n"
        "    pip install tensorflow\n"
        "(On Apple Silicon Macs, if plain `tensorflow` fails, use:\n"
        "    pip install tensorflow-macos tensorflow-metal\n"
        ")\n"
        "Then re-run: python -m src.train_lstm"
    )
    sys.exit(1)


def create_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """
    Convert flat feature rows into overlapping sequences of length `lookback`.
    Each sequence of `lookback` days' features maps to the target on the day
    immediately after the sequence window.
    """
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def build_lstm_model(input_shape: tuple) -> "Sequential":
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_lstm():
    df = load_featured_data()
    feature_cols = [c for c in df.columns if c != "target"]

    # Scale BEFORE splitting into sequences — LSTM is very sensitive to feature scale.
    # Fit scaler on train portion only to avoid leaking test-set information.
    split_idx = int(len(df) * (1 - TEST_SIZE))

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_train_raw = feature_scaler.fit_transform(df[feature_cols].iloc[:split_idx])
    X_test_raw = feature_scaler.transform(df[feature_cols].iloc[split_idx:])

    y_train_raw = target_scaler.fit_transform(df[["target"]].iloc[:split_idx]).flatten()
    y_test_raw = target_scaler.transform(df[["target"]].iloc[split_idx:]).flatten()

    X_train, y_train = create_sequences(X_train_raw, y_train_raw, LSTM_LOOKBACK)
    X_test, y_test = create_sequences(X_test_raw, y_test_raw, LSTM_LOOKBACK)

    logger.info(
        f"Sequences built — X_train: {X_train.shape}, X_test: {X_test.shape} "
        f"(lookback={LSTM_LOOKBACK} days, {len(feature_cols)} features)"
    )

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    logger.info(model.summary())

    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    logger.info("Training LSTM model...")
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )

    y_pred_scaled = model.predict(X_test).flatten()

    # Inverse-transform back to actual price scale before computing metrics —
    # otherwise RMSE/MAE would be in meaningless 0-1 scaled units, not rupees.
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    results = evaluate_predictions(y_true, y_pred, model_name="LSTM")
    save_results(results)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "lstm_model.keras")
    joblib.dump(feature_scaler, MODELS_DIR / "lstm_feature_scaler.pkl")
    joblib.dump(target_scaler, MODELS_DIR / "lstm_target_scaler.pkl")
    logger.info(f"Model and scalers saved to {MODELS_DIR}")

    # Dates for the test predictions — offset by lookback since the first
    # `lookback` test rows are consumed building the first sequence.
    test_dates = df.index[split_idx + LSTM_LOOKBACK:]
    pred_df = pd.DataFrame({
        "date": test_dates,
        "actual": y_true,
        "predicted": y_pred,
    })
    pred_df.to_csv(MODELS_DIR / "lstm_predictions.csv", index=False)

    return model, results


if __name__ == "__main__":
    train_lstm()