"""
train_gb.py
Gradient Boosting model (XGBoost) on engineered features.

Unlike Linear Regression, tree-based models don't need feature scaling and can
capture non-linear relationships between indicators (e.g. RSI + MACD interactions).

Usage (from project root):
    python -m src.train_gb
"""

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR, MODELS_DIR, TEST_SIZE, LOG_LEVEL
from src.evaluate import evaluate_predictions, save_results
from src.train_lr import load_featured_data, time_series_split  # reuse, don't duplicate

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def train_gradient_boosting():
    df = load_featured_data()
    train_df, test_df = time_series_split(df)

    feature_cols = [c for c in df.columns if c != "target"]

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    logger.info("Training Gradient Boosting (XGBoost) model...")
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results = evaluate_predictions(y_test.values, y_pred, model_name="Gradient Boosting")
    save_results(results)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "gradient_boosting.pkl")
    logger.info(f"Model saved to {MODELS_DIR}")

    # Feature importance — genuinely useful for your report/interview talking points:
    # "which indicators actually mattered" is a much stronger story than a metric alone.
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(MODELS_DIR / "gb_feature_importance.csv", index=False)
    logger.info("Top 10 most important features:\n" + importance.head(10).to_string(index=False))

    pred_df = pd.DataFrame({
        "date": test_df.index,
        "actual": y_test.values,
        "predicted": y_pred,
    })
    pred_df.to_csv(MODELS_DIR / "gb_predictions.csv", index=False)

    return model, results


if __name__ == "__main__":
    train_gradient_boosting()