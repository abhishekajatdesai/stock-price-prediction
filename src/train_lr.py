"""
train_lr.py
Baseline model: Linear Regression on engineered features.

This exists as a baseline so the LSTM and Gradient Boosting results mean something —
"our LSTM beats a Linear Regression baseline by X%" is a real claim; "we built an LSTM"
alone is not.

Usage (from project root):
    python -m src.train_lr
"""

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR, MODELS_DIR, TEST_SIZE, LOG_LEVEL
from src.evaluate import evaluate_predictions, save_results

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def load_featured_data() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "featured_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.feature_engineering` first."
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def time_series_split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """
    Time-ordered split — NEVER shuffle time series data. The test set must be
    the most recent chunk of the timeline, simulating "predicting the future"
    rather than leaking future information into training.
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(
        f"Train: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()}) | "
        f"Test: {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})"
    )
    return train_df, test_df


def train_linear_regression():
    df = load_featured_data()
    train_df, test_df = time_series_split(df)

    feature_cols = [c for c in df.columns if c != "target"]

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    # Linear Regression is sensitive to feature scale — standardize inputs.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    results = evaluate_predictions(y_test.values, y_pred, model_name="Linear Regression")
    save_results(results)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "linear_regression.pkl")
    joblib.dump(scaler, MODELS_DIR / "lr_scaler.pkl")
    logger.info(f"Model and scaler saved to {MODELS_DIR}")

    # Save predictions vs actuals for plotting/reporting later
    pred_df = pd.DataFrame({
        "date": test_df.index,
        "actual": y_test.values,
        "predicted": y_pred,
    })
    pred_df.to_csv(MODELS_DIR / "lr_predictions.csv", index=False)

    return model, results


if __name__ == "__main__":
    train_linear_regression()