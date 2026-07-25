"""
naive_baseline.py
The simplest possible forecast: tomorrow's close = today's close ("naive persistence").

This is the real benchmark for a stock prediction project. If a sophisticated model
can't beat this, it isn't adding value — this script makes that comparison explicit
and honest instead of letting a low MAPE speak for itself.

Usage (from project root):
    python -m src.naive_baseline
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import LOG_LEVEL
from src.evaluate import evaluate_predictions, save_results, print_comparison
from src.train_lr import load_featured_data, time_series_split

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def run_naive_baseline():
    df = load_featured_data()
    train_df, test_df = time_series_split(df)

    # Naive prediction: predict today's Close as tomorrow's Close.
    # This is exactly what "target" already is one step ahead of — so the naive
    # prediction for each row is simply that row's current Close value.
    y_true = test_df["target"].values
    y_pred_naive = test_df["Close"].values

    results = evaluate_predictions(y_true, y_pred_naive, model_name="Naive Persistence")
    save_results(results)

    pred_df = pd.DataFrame({
        "date": test_df.index,
        "actual": y_true,
        "predicted": y_pred_naive,
    })
    pred_df.to_csv("models/naive_predictions.csv", index=False)

    logger.info(
        "Naive baseline saved. Compare this against Linear Regression, GB, and LSTM "
        "to see how much (if any) real predictive value each model adds beyond "
        "'tomorrow looks like today'."
    )

    print_comparison()
    return results


if __name__ == "__main__":
    run_naive_baseline()