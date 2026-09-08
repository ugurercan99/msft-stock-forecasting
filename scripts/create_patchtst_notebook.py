import nbformat as nbf

nb = nbf.v4.new_notebook()

code_1 = """import numpy as np
import pandas as pd
import torch
import yfinance as yf
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
import matplotlib.pyplot as plt

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

print("Libraries imported successfully.")"""

code_2 = """# 1. Download & Format Data for NeuralForecast
df = yf.download('MSFT', start='2011-01-01', end='2026-04-23', auto_adjust=True)
df.columns = df.columns.get_level_values(0)

df = df[['Close']] # PatchTST in neuralforecast is Univariate only
df.dropna(inplace=True)

# Format for NeuralForecast
nf_df = df.reset_index()
nf_df = nf_df.rename(columns={'Date': 'ds', 'Close': 'y'})
nf_df['unique_id'] = 'MSFT'

print(f"Data shape: {nf_df.shape}")
print(nf_df.head())"""

code_3 = """# 2. Train / Val / Test Splits
n = len(nf_df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_df = nf_df.iloc[:train_end].copy()
val_df = nf_df.iloc[train_end:val_end].copy()
test_df = nf_df.iloc[val_end:].copy()

print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")"""

code_4 = """# 3. PatchTST Model Configuration
window_size = 60
horizon = 1

model = PatchTST(
    h=horizon,
    input_size=window_size,
    patch_len=12,
    stride=12,
    hidden_size=64,
    n_heads=4,
    scaler_type='minmax',
    max_steps=500,     # Lightweight training
    learning_rate=1e-3,
    batch_size=64
)

nf = NeuralForecast(models=[model], freq='B')"""

code_5 = """# 4. Train Model
print("Training PatchTST...")
nf.fit(df=train_df)
print("Training complete.")"""

code_6 = """# 5. Inference / Predictions
print("Performing cross-validation to get predictions for Test set...")

step_size = 1
n_windows = len(test_df)

cv_df = nf.cross_validation(df=nf_df, n_windows=n_windows, step_size=step_size)
print(cv_df.head())"""

code_7 = """# 6. Save Predictions for Final Comparison
cv_df.to_csv('../outputs/tables/patchtst_predictions.csv', index=False)
print("Saved PatchTST predictions to outputs/tables/patchtst_predictions.csv")"""

nb.cells = [
    nbf.v4.new_markdown_cell("# PatchTST Transformer Model"),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_code_cell(code_7)
]

with open('notebooks/03_patchtst_model.ipynb', 'w') as f:
    nbf.write(nb, f)
