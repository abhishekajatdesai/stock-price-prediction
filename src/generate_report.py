"""
generate_report.py
Builds the visual + written evidence for the project's core finding:
none of the models meaningfully beat a naive "tomorrow = today" baseline.

Produces:
    reports/predictions_comparison.png   - actual vs predicted, all 4 approaches
    reports/metrics_comparison.png       - bar chart of RMSE/MAPE across models
    reports/limitations_report.md        - written summary, auto-filled from results

Usage (from project root):
    python -m src.generate_report
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, REPORTS_DIR, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = REPORTS_DIR / "model_comparison.json"


def load_all_predictions() -> dict:
    """Load each model's saved predictions.csv into a dict of DataFrames."""
    files = {
        "Naive Persistence": MODELS_DIR / "naive_predictions.csv",
        "Linear Regression": MODELS_DIR / "lr_predictions.csv",
        "Gradient Boosting": MODELS_DIR / "gb_predictions.csv",
        "LSTM": MODELS_DIR / "lstm_predictions.csv",
    }
    preds = {}
    for name, path in files.items():
        if path.exists():
            preds[name] = pd.read_csv(path, parse_dates=["date"])
        else:
            logger.warning(f"{path} not found — skipping {name} in report. Run its training script first.")
    return preds


def plot_predictions_comparison(preds: dict):
    """Overlay actual vs predicted close price for every model on one chart."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    for ax, (name, df) in zip(axes, preds.items()):
        ax.plot(df["date"], df["actual"], label="Actual", color="black", linewidth=1.5)
        ax.plot(df["date"], df["predicted"], label="Predicted", color="tab:orange", linewidth=1.2, alpha=0.8)
        ax.set_title(name)
        ax.legend()
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Actual vs Predicted Close Price — All Approaches", fontsize=14)
    fig.tight_layout()

    out_path = REPORTS_DIR / "predictions_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_metrics_comparison():
    """Bar chart comparing RMSE and MAPE across all four models."""
    if not RESULTS_PATH.exists():
        logger.warning("No model_comparison.json found — run at least one training script first.")
        return

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    df = pd.DataFrame(results.values()).sort_values("rmse")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(df["model"], df["rmse"], color="tab:blue")
    axes[0].set_title("RMSE by Model (lower = better)")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(df["model"], df["mape_pct"], color="tab:red")
    axes[1].set_title("MAPE % by Model (lower = better)")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    out_path = REPORTS_DIR / "metrics_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def write_limitations_report():
    """Auto-generate a markdown report summarizing the model comparison and key finding."""
    if not RESULTS_PATH.exists():
        logger.warning("No results to report on yet.")
        return

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    df = pd.DataFrame(results.values()).sort_values("rmse")
    naive_row = df[df["model"] == "Naive Persistence"]
    best_row = df.iloc[0]

    beats_naive = (
        not naive_row.empty and best_row["model"] != "Naive Persistence"
        and best_row["rmse"] < naive_row.iloc[0]["rmse"]
    )

    table_md = df.to_markdown(index=False)

    report = f"""# Model Comparison & Limitations Report

## Results Summary

{table_md}

## Key Finding

{"A model beat the naive persistence baseline." if beats_naive else
"**None of the trained models (Linear Regression, LSTM, Gradient Boosting) meaningfully outperformed the naive persistence baseline** (predicting tomorrow's close as simply equal to today's close)."}

This is a well-documented characteristic of short-horizon (1-day-ahead) stock price
prediction: at daily granularity, price series behave close to a random walk, and
technical indicators derived purely from price/volume history carry limited
additional signal over the most recent price itself.

## Why Gradient Boosting and LSTM Underperformed Linear Regression

Both Gradient Boosting and LSTM struggled more than Linear Regression on this task.
The likely cause is **extrapolation**: Reliance's stock price trended upward over the
training period (2015–2023), and the test period (2023–2024) reached price levels
higher than anything seen during training.

- **Gradient Boosting** (tree-based) cannot extrapolate beyond the range of target
  values seen during training by design — its predictions are bounded by training
  leaf values, causing systematic underprediction as prices moved into new territory.
- **LSTM** is less rigidly bounded, but its internal recurrent activations
  (tanh-based) saturate for inputs far outside the range they were trained on,
  producing a milder version of the same effect.
- **Linear Regression** extrapolates linearly by construction, giving it an
  advantage on a trending series — even though it isn't "learning" much beyond
  recent momentum, as the naive baseline comparison shows.

## Limitations

- All models predict only 1 day ahead — no claim is made about longer-horizon
  forecasting accuracy or usefulness for actual trading decisions.
- Models are trained on a single ticker (RELIANCE.NS) — findings may not generalize
  across stocks, sectors, or market regimes.
- Feature set is limited to price/volume-derived technical indicators — no
  fundamental data, news sentiment, or macroeconomic signals are included.
- No transaction costs, slippage, or realistic trading constraints are modeled;
  this project evaluates prediction accuracy only, not trading strategy viability.

## Suggested Future Work

- Reframe the target as **direction** (up/down) or **volatility** rather than exact
  price — these may carry more learnable signal at daily granularity.
- Extend the prediction horizon (e.g. 5-day or 20-day ahead) where trend and
  momentum effects may be more learnable.
- Incorporate external features: sector indices, news sentiment, macroeconomic
  indicators.
- Apply de-trending or differencing to the target so tree-based and neural models
  aren't penalized for the same extrapolation issue found here.

---
*Auto-generated from `reports/model_comparison.json` by `src/generate_report.py`*
"""

    out_path = REPORTS_DIR / "limitations_report.md"
    out_path.write_text(report)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    preds = load_all_predictions()

    if preds:
        plot_predictions_comparison(preds)
    plot_metrics_comparison()
    write_limitations_report()

    logger.info("Report generation complete. Check the reports/ folder.")