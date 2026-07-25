# Model Comparison & Limitations Report

## Results Summary

| model             |     rmse |      mae |   mape_pct |
|:------------------|---------:|---------:|-----------:|
| Naive Persistence |  16.8775 |  11.9413 |     0.9254 |
| Linear Regression |  17.3183 |  12.3858 |     0.9609 |
| LSTM              | 108.492  |  85.1641 |     6.0614 |
| Gradient Boosting | 153.785  | 107.979  |     7.6124 |

## Key Finding

**None of the trained models (Linear Regression, LSTM, Gradient Boosting) meaningfully outperformed the naive persistence baseline** (predicting tomorrow's close as simply equal to today's close).

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
