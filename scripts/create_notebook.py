import nbformat as nbf

nb = nbf.v4.new_notebook()

code_1 = """import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import torch

np.random.seed(42)
torch.manual_seed(42)

print("Libraries imported successfully.")"""

code_2 = """# 1. Download MSFT data
df = yf.download('MSFT', start='2011-01-01', end='2026-04-23', auto_adjust=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)

print(f"Data shape: {df.shape}")
df.head()"""

code_3 = """# Plot full dataset
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Close'], label='MSFT Close Price')
plt.title('MSFT Daily Close Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.savefig('../outputs/plots/dataset_overview.png')
plt.show()"""

code_4 = """# 2. Chronological Train/Val/Test Split (70/15/15)
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")"""

code_5 = """# 3. Scaling
# Fit scaler ONLY on train data to prevent data leakage
scaler_all = MinMaxScaler(feature_range=(0, 1))
scaler_all.fit(train_df)

# Separate scaler for Close only (useful for inverse transformation later)
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_close.fit(train_df[['Close']])

# Scale the data
train_scaled = scaler_all.transform(train_df)
val_scaled = scaler_all.transform(val_df)
test_scaled = scaler_all.transform(test_df)

# Note: We keep the scaled data as numpy arrays for window generation
print(f"Scaled Train shape: {train_scaled.shape}")"""

code_6 = """# 4. Window Generation Function
def create_sequences(data_scaled, target_col_idx=3, window_size=60):
    '''
    Create overlapping sequences.
    data_scaled: numpy array of all features
    target_col_idx: 3 is for 'Close' (Open=0, High=1, Low=2, Close=3, Volume=4)
    window_size: default 60
    '''
    X = []
    y = []
    for i in range(window_size, len(data_scaled)):
        # X is the past window_size days
        X.append(data_scaled[i-window_size:i])
        # y is the target (Close) at the current day
        y.append(data_scaled[i, target_col_idx])
    
    return np.array(X), np.array(y)

# 5. Apply Window Generation
# To properly utilize historical context without missing predictions at the start of val/test:
# We prepend the last 'window_size' days of the previous set to the current set.

window_size = 60
target_idx = df.columns.get_loc('Close')

# Train sequences
X_train, y_train = create_sequences(train_scaled, target_idx, window_size)

# Val sequences (prepend last 'window_size' from train)
val_input = np.vstack((train_scaled[-window_size:], val_scaled))
X_val, y_val = create_sequences(val_input, target_idx, window_size)

# Test sequences (prepend last 'window_size' from val)
test_input = np.vstack((val_scaled[-window_size:], test_scaled))
X_test, y_test = create_sequences(test_input, target_idx, window_size)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")"""

code_7 = """# Save processed data for modeling
np.save('../data/X_train.npy', X_train)
np.save('../data/y_train.npy', y_train)
np.save('../data/X_val.npy', X_val)
np.save('../data/y_val.npy', y_val)
np.save('../data/X_test.npy', X_test)
np.save('../data/y_test.npy', y_test)

# Also save scalers (we'll need them later, we can use joblib)
import joblib
joblib.dump(scaler_all, '../models/scaler_all.pkl')
joblib.dump(scaler_close, '../models/scaler_close.pkl')

print("Data preprocessing complete and saved.")"""

nb.cells = [
    nbf.v4.new_markdown_cell("# Data Preparation & Preprocessing"),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_code_cell(code_7)
]

with open('notebooks/01_data_prep.ipynb', 'w') as f:
    nbf.write(nb, f)
