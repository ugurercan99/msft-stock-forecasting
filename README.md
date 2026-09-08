# MSFT Stock Price Forecasting

Comparing four approaches to forecasting Microsoft (MSFT) daily closing
price — a naive persistence baseline, RNN-family models (LSTM, GRU), and a
transformer-based time-series model (PatchTST).

## Context

This was the artefact for **Assessment 2 (CA2)** of the **Deep Learning
module** in the MSc in Artificial Intelligence at Dublin Business School.
The brief asked for an individual report covering the architecture of three
selected deep learning forecasting models (scope, type, mathematical
foundations, modelling assumptions), a documented and error-free dataset
implementation (preprocessing, feature engineering, train/val/test split
with temporal ordering preserved), and a critical evaluation of each
model's design, hyperparameters, and performance (MAE, MSE, RMSE, MAPE),
alongside conclusions and supporting plots.

The original assignment brief (a DBS-internal assessment document) is not
included in this repository; the summary above covers its scope.
`report/Report.docx` is the full CRISP-DM write-up;
`report/Individual_Contribution_Statement.docx` documents the individual
development process, as required by the brief.

## Data

Daily MSFT OHLCV data (2010–2026, per `notebooks/01_data_prep.ipynb`'s
`START_DATE`) is pulled live via the `yfinance` API (see
`scripts/test_yf.py` for a minimal, unrelated connectivity check using its
own date range). The preprocessed,
windowed train/val/test tensors used for training are included under
`data/` (`X_*.npy`, `y_*.npy`, plus the corresponding date indices) so the
notebooks can be re-run without needing to regenerate the feature windows
from scratch.

## Approach

- **`notebooks/01_data_prep.ipynb`** — fetches and preprocesses the data,
  engineers features, and produces the windowed train/val/test splits.
- **`notebooks/02_rnn_models.ipynb`** — trains and tunes LSTM and GRU
  models; best checkpoints are saved to `models/`.
- **`notebooks/03_patchtst_model.ipynb`** — trains and tunes a PatchTST
  (patch-based transformer) forecaster via `neuralforecast`.
- **`notebooks/04_comparison.ipynb`** — brings all models (plus a naive
  persistence baseline) together for a final head-to-head comparison.
- **`scripts/create_*.py`** — the scripts originally used to generate the
  notebooks above programmatically.

## Results

| Model | MSE | RMSE | MAE | MAPE (%) | R² | Directional accuracy (%) |
|---|---|---|---|---|---|---|
| Naive (persistence) | 39.6 | 6.29 | 4.39 | 1.03 | 0.981 | – |
| **PatchTST** | 40.5 | 6.36 | 4.51 | 1.06 | 0.981 | 47.6 |
| GRU | 1195.6 | 34.58 | 30.75 | 6.89 | 0.435 | 47.4 |
| LSTM | 8556.4 | 92.50 | 86.77 | 19.61 | −3.04 | 46.9 |

The naive "predict tomorrow = today's close" baseline is extremely
competitive on one-step-ahead error metrics — a well-known property of
near-random-walk financial series — and PatchTST is the only model that
matches it. The RNN-family models (GRU, and especially LSTM) show much
higher error, and directional accuracy for every model sits close to chance
(50%), meaning none of these approaches reliably predicts the *direction*
of tomorrow's move even where absolute error looks small. Full
metrics-by-model, predictions, and tuning results are in
[`outputs/tables/`](outputs/tables/); learning curves, residuals, and
forecast plots are in [`outputs/plots/`](outputs/plots/).

## Repository structure

```
.
├── notebooks/            # 01 data prep -> 02 RNNs -> 03 PatchTST -> 04 comparison
├── scripts/               # notebook-generation scripts + a minimal yfinance fetch test
├── data/                  # preprocessed train/val/test tensors (.npy)
├── models/                # best trained checkpoints (GRU, LSTM) + fitted scalers
├── outputs/plots/         # learning curves, residuals, forecast plots
├── outputs/tables/        # per-model metrics, predictions, tuning results
└── report/                # full CRISP-DM report + individual contribution statement
```

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_data_prep.ipynb
```

Run the four notebooks in order (`01` → `02` → `03` → `04`). `01` will
re-fetch and re-window the data. `02` (RNNs) needs only the provided `.npy`
files in `data/`, so you can start directly there and skip `01`; `03`
(PatchTST) re-downloads its own raw OHLCV via `yfinance` regardless of
`data/`, so it always needs live network access. `02` and `03` will
overwrite the checkpoints in `models/` and the tables/plots in `outputs/`
with a fresh run.

## License

Code is shared for portfolio/educational purposes. MSFT price data is
sourced from Yahoo Finance via `yfinance` at run time and is not
redistributed here beyond the preprocessed arrays needed to reproduce the
reported results.
