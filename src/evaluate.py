"""
evaluate.py
Shared evaluation metrics and comparison utilities used by all three models
(Linear Regression, LSTM, Gradient Boosting) so results are computed identically
and can be fairly compared.

Usage:
    from src.evaluate import evaluate_predictions, save_results, print_comparison
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import REPORTS_DIR, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = REPORTS_DIR / "model_comparison.json"


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """
    Compute RMSE, MAE, and MAPE for a set of predictions.

    Returns a dict so results can be logged, saved, and compared across models
    using identical logic — critical for a fair Linear Regression vs LSTM vs
    Gradient Boosting comparison.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    results = {
        "model": model_name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape_pct": round(mape, 4),
    }

    logger.info(f"[{model_name}] RMSE={results['rmse']} | MAE={results['mae']} | MAPE={results['mape_pct']}%")
    return results


def save_results(results: dict) -> None:
    """
    Append a model's results to the shared comparison file (reports/model_comparison.json).
    Re-running a model's training script updates its entry rather than duplicating it.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r") as f:
            all_results = json.load(f)

    all_results[results["model"]] = results

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results saved to {RESULTS_PATH}")


def print_comparison() -> None:
    """Print a clean side-by-side comparison table of all models evaluated so far."""
    if not RESULTS_PATH.exists():
        logger.warning("No results found yet. Train at least one model first.")
        return

    with open(RESULTS_PATH, "r") as f:
        all_results = json.load(f)

    df = pd.DataFrame(all_results.values())
    df = df.sort_values("rmse")
    print("\n" + "=" * 50)
    print("MODEL COMPARISON (lower RMSE/MAE/MAPE = better)")
    print("=" * 50)
    print(df.to_string(index=False))
    print("=" * 50 + "\n")


if __name__ == "__main__":
    print_comparison()