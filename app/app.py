"""
app.py
Flask app that loads all 4 trained approaches (Naive, Linear Regression,
Gradient Boosting, LSTM) and serves a next-day prediction comparison,
plus the evaluation plots generated in Phase 5.

Usage (from project root):
    python -m app.app
Then open http://127.0.0.1:5000 in your browser.
"""

import logging
import sys
from pathlib import Path

import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, send_from_directory, request

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, LSTM_LOOKBACK, TICKER, LOG_LEVEL
from src.company_lookup import get_company_list, get_ticker
from src.on_demand_predictor import run_on_demand_prediction, OnDemandError

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = Path(__file__).resolve().parent / "uploads"
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)

# ---------- Load everything once at startup, not per-request ----------
_featured_df = None
_lr_model = _lr_scaler = None
_gb_model = None
_lstm_model = _lstm_feature_scaler = _lstm_target_scaler = None


def load_all_models():
    """Load the dataset and all trained models/scalers into memory once."""
    global _featured_df, _lr_model, _lr_scaler, _gb_model
    global _lstm_model, _lstm_feature_scaler, _lstm_target_scaler

    data_path = PROCESSED_DATA_DIR / "featured_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run `python -m src.feature_engineering` first."
        )
    _featured_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded featured data: {_featured_df.shape}")

    lr_path = MODELS_DIR / "linear_regression.pkl"
    if lr_path.exists():
        _lr_model = joblib.load(lr_path)
        _lr_scaler = joblib.load(MODELS_DIR / "lr_scaler.pkl")
        logger.info("Loaded Linear Regression model.")
    else:
        logger.warning("Linear Regression model not found — run `python -m src.train_lr` first.")

    gb_path = MODELS_DIR / "gradient_boosting.pkl"
    if gb_path.exists():
        _gb_model = joblib.load(gb_path)
        logger.info("Loaded Gradient Boosting model.")
    else:
        logger.warning("Gradient Boosting model not found — run `python -m src.train_gb` first.")

    lstm_path = MODELS_DIR / "lstm_model.keras"
    if lstm_path.exists():
        # Imported lazily — TensorFlow is slow to import and only needed if LSTM exists.
        from tensorflow.keras.models import load_model
        _lstm_model = load_model(lstm_path)
        _lstm_feature_scaler = joblib.load(MODELS_DIR / "lstm_feature_scaler.pkl")
        _lstm_target_scaler = joblib.load(MODELS_DIR / "lstm_target_scaler.pkl")
        logger.info("Loaded LSTM model.")
    else:
        logger.warning("LSTM model not found — run `python -m src.train_lstm` first.")


def predict_next_day() -> dict:
    """
    Generate a next-day closing price prediction from each available approach,
    using the most recent data as input. Returns a dict keyed by model name.
    """
    feature_cols = [c for c in _featured_df.columns if c != "target"]
    latest_row = _featured_df.iloc[[-1]][feature_cols]
    latest_close = float(_featured_df.iloc[-1]["Close"])
    latest_date = _featured_df.index[-1]

    predictions = {
        "Naive Persistence": round(latest_close, 2),  # tomorrow = today, by definition
    }

    if _lr_model is not None:
        X_scaled = _lr_scaler.transform(latest_row)
        predictions["Linear Regression"] = round(float(_lr_model.predict(X_scaled)[0]), 2)

    if _gb_model is not None:
        predictions["Gradient Boosting"] = round(float(_gb_model.predict(latest_row)[0]), 2)

    if _lstm_model is not None:
        lookback_rows = _featured_df.iloc[-LSTM_LOOKBACK:][feature_cols]
        if len(lookback_rows) == LSTM_LOOKBACK:
            X_scaled = _lstm_feature_scaler.transform(lookback_rows)
            X_seq = X_scaled.reshape(1, LSTM_LOOKBACK, len(feature_cols))
            y_scaled = _lstm_model.predict(X_seq, verbose=0)
            y_pred = _lstm_target_scaler.inverse_transform(y_scaled)[0][0]
            predictions["LSTM"] = round(float(y_pred), 2)
        else:
            logger.warning("Not enough rows for LSTM lookback window — skipping LSTM prediction.")

    return {
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "latest_close": round(latest_close, 2),
        "ticker": TICKER,
        "predictions": predictions,
    }


def generate_narrative_summary(result: dict) -> str:
    """
    Plain-English summary of the prediction comparison, generated with simple rules
    rather than an LLM call — keeps the app free to run with no API dependency.

    Structured deliberately so this function's return value could later be swapped
    for a real Claude API call without touching any other part of the app: just
    replace the body with an API request that returns a string, same signature.
    """
    preds = result["predictions"]
    latest_close = result["latest_close"]

    naive = preds.get("Naive Persistence")
    lr = preds.get("Linear Regression")
    gb = preds.get("Gradient Boosting")
    lstm = preds.get("LSTM")

    lines = []

    # Spread across models — how much do they actually disagree?
    values = [v for v in [naive, lr, gb, lstm] if v is not None]
    spread = max(values) - min(values)
    spread_pct = (spread / latest_close) * 100

    lines.append(
        f"The four approaches predict a next-day close ranging from "
        f"₹{min(values):.2f} to ₹{max(values):.2f} — a spread of "
        f"₹{spread:.2f} ({spread_pct:.2f}% of the current price)."
    )

    if lr is not None and naive is not None:
        lr_vs_naive = abs(lr - naive)
        if lr_vs_naive < 0.5 * (latest_close * 0.001):  # within a very tight band
            lines.append(
                "Linear Regression's prediction is nearly identical to the naive "
                "'tomorrow equals today' baseline, consistent with this project's "
                "finding that it adds little signal beyond recent price momentum."
            )
        else:
            direction = "above" if lr > naive else "below"
            lines.append(
                f"Linear Regression predicts {direction} the naive baseline by "
                f"₹{lr_vs_naive:.2f}."
            )

    if gb is not None and naive is not None:
        if gb < naive - (latest_close * 0.01):
            lines.append(
                "Gradient Boosting predicts notably below both the naive baseline and "
                "the current price — consistent with this model's known tendency to "
                "underpredict when prices move beyond the range it saw during training."
            )

    if lstm is not None and naive is not None:
        if lstm < naive - (latest_close * 0.01):
            lines.append(
                "LSTM also predicts below the naive baseline, showing a similar "
                "(though less extreme) pattern to Gradient Boosting."
            )

    lines.append(
        "As shown in this project's evaluation, none of the trained models "
        "meaningfully outperformed the naive baseline on held-out test data — "
        "these predictions should be read as illustrative, not as trading signals."
    )

    return " ".join(lines)


def load_default_chart_data() -> dict:
    """Load the saved prediction CSVs for the main RELIANCE.NS view into chart-ready JSON."""
    files = {
        "Naive Persistence": MODELS_DIR / "naive_predictions.csv",
        "Linear Regression": MODELS_DIR / "lr_predictions.csv",
        "Gradient Boosting": MODELS_DIR / "gb_predictions.csv",
        "LSTM": MODELS_DIR / "lstm_predictions.csv",
    }
    chart_data = {}
    for name, path in files.items():
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            chart_data[name] = {
                "dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
                "actual": [round(float(v), 2) for v in df["actual"]],
                "predicted": [round(float(v), 2) for v in df["predicted"]],
            }
    return chart_data


def load_default_metrics() -> list:
    """Load saved metrics from reports/model_comparison.json for the metrics bar chart."""
    path = REPORTS_DIR / "model_comparison.json"
    if not path.exists():
        return []
    with open(path) as f:
        results = json.load(f)
    order = ["Naive Persistence", "Linear Regression", "Gradient Boosting", "LSTM"]
    return [results[m] for m in order if m in results]


@app.route("/")
def index():
    result = predict_next_day()
    result["narrative"] = generate_narrative_summary(result)
    result["company_label"] = TICKER
    result["metrics"] = load_default_metrics()
    result["chart_data"] = load_default_chart_data()
    return render_template("index.html", result=result, companies=get_company_list())


@app.route("/search", methods=["POST"])
def search():
    """Handle company search from the dropdown or free-text ticker field."""
    company_choice = request.form.get("company_dropdown", "").strip()
    manual_ticker = request.form.get("manual_ticker", "").strip()

    ticker = None
    company_label = None

    if manual_ticker:
        ticker = manual_ticker.upper()
        company_label = ticker
    elif company_choice and company_choice != "-- Select a company --":
        ticker = get_ticker(company_choice)
        company_label = company_choice

    if not ticker:
        return render_template(
            "index.html",
            error="Please select a company or enter a ticker symbol.",
            companies=get_company_list(),
        )

    try:
        result = run_on_demand_prediction(ticker=ticker, company_label=company_label)
        result["narrative"] = generate_narrative_summary(result)
        return render_template("index.html", result=result, companies=get_company_list())
    except OnDemandError as e:
        logger.warning(f"On-demand prediction failed for {ticker}: {e}")
        return render_template("index.html", error=str(e), companies=get_company_list())


@app.route("/upload", methods=["POST"])
def upload():
    """Handle manual CSV upload as an alternative to fetching from Yahoo Finance."""
    file = request.files.get("csv_file")

    if not file or file.filename == "":
        return render_template(
            "index.html",
            error="Please choose a CSV file to upload.",
            companies=get_company_list(),
        )

    if not file.filename.lower().endswith(".csv"):
        return render_template(
            "index.html",
            error="Only .csv files are supported.",
            companies=get_company_list(),
        )

    save_path = app.config["UPLOAD_FOLDER"] / file.filename
    file.save(save_path)

    try:
        result = run_on_demand_prediction(
            uploaded_csv_path=save_path,
            company_label=f"Uploaded: {file.filename}",
        )
        result["narrative"] = generate_narrative_summary(result)
        return render_template("index.html", result=result, companies=get_company_list())
    except OnDemandError as e:
        logger.warning(f"On-demand prediction failed for uploaded file: {e}")
        return render_template("index.html", error=str(e), companies=get_company_list())
    finally:
        save_path.unlink(missing_ok=True)  # don't retain uploaded files after processing


@app.route("/reports/<path:filename>")
def serve_report(filename):
    """Serve the plots generated by src/generate_report.py directly from reports/."""
    return send_from_directory(REPORTS_DIR, filename)


if __name__ == "__main__":
    load_all_models()
    app.run(debug=True, port=8000, threaded=True)