import nbformat as nbf

nb = nbf.v4.new_notebook()

code_1 = """import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import joblib
import shutil
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print("Libraries imported successfully.")"""

code_2 = """# 1. Load Data and Scaler
X_test = np.load('../data/X_test.npy')
y_test = np.load('../data/y_test.npy')
scaler_close = joblib.load('../models/scaler_close.pkl')

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")"""

code_3 = """# 2. Define RNN Models to load weights
class LSTMModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])

lstm_model = LSTMModel().to(device)
lstm_model.load_state_dict(torch.load('../models/best_lstm.pth', map_location=device, weights_only=True))
lstm_model.eval()

gru_model = GRUModel().to(device)
gru_model.load_state_dict(torch.load('../models/best_gru.pth', map_location=device, weights_only=True))
gru_model.eval()"""

code_4 = """# 3. Generate RNN Predictions
batch_size = 64
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

lstm_preds = []
gru_preds = []

with torch.no_grad():
    for (X_batch,) in test_loader:
        X_batch = X_batch.to(device)
        lstm_preds.append(lstm_model(X_batch).cpu().numpy())
        gru_preds.append(gru_model(X_batch).cpu().numpy())
        
lstm_preds = np.concatenate(lstm_preds, axis=0)
gru_preds = np.concatenate(gru_preds, axis=0)"""

code_5 = """# 4. Inverse Transform all predictions to Dollar values
y_test_dollar = scaler_close.inverse_transform(y_test.reshape(-1, 1)).flatten()
lstm_preds_dollar = scaler_close.inverse_transform(lstm_preds).flatten()
gru_preds_dollar = scaler_close.inverse_transform(gru_preds).flatten()

# Naive Baseline (using today's close as tomorrow's)
naive_preds_scaled = X_test[:, -1, 3] # Index 3 is Close
naive_preds_dollar = scaler_close.inverse_transform(naive_preds_scaled.reshape(-1, 1)).flatten()

# Load PatchTST predictions
patchtst_df = pd.read_csv('../outputs/tables/patchtst_predictions.csv')
# PatchTST predictions are in the column named 'PatchTST'
patchtst_preds_dollar = patchtst_df['PatchTST'].values"""

code_6 = """# 5. Calculate Metrics
def calculate_metrics(y_true, y_pred, name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # Directional Accuracy
    # Did the model correctly predict if tomorrow's price goes up or down compared to today?
    # True direction = sign(y_true_tomorrow - y_true_today)
    # Predicted direction = sign(y_pred_tomorrow - y_true_today)
    # "y_true_today" is our naive baseline!
    true_direction = np.sign(y_true - naive_preds_dollar)
    pred_direction = np.sign(y_pred - naive_preds_dollar)
    
    # Remove cases where price stayed exactly the same to avoid zero signs
    valid_mask = true_direction != 0
    directional_acc = np.mean(true_direction[valid_mask] == pred_direction[valid_mask]) * 100
    
    return {
        'Model': name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape,
        'Directional Acc (%)': directional_acc
    }

results = []
results.append(calculate_metrics(y_test_dollar, naive_preds_dollar, 'Naive Baseline'))
results.append(calculate_metrics(y_test_dollar, lstm_preds_dollar, 'LSTM'))
results.append(calculate_metrics(y_test_dollar, gru_preds_dollar, 'GRU'))
results.append(calculate_metrics(y_test_dollar, patchtst_preds_dollar, 'PatchTST'))

results_df = pd.DataFrame(results).set_index('Model')
results_df.to_csv('../outputs/tables/final_metrics_comparison.csv')
results_df.round(4)"""

code_7 = """# 6. Plotting
# Align dates for plotting
dates = patchtst_df['ds'].values

plt.figure(figsize=(15, 7))
plt.plot(dates, y_test_dollar, label='Actual Close', color='black', linewidth=2)
plt.plot(dates, naive_preds_dollar, label='Naive Baseline', linestyle=':', color='gray')
plt.plot(dates, lstm_preds_dollar, label='LSTM', alpha=0.8)
plt.plot(dates, gru_preds_dollar, label='GRU', alpha=0.8)
plt.plot(dates, patchtst_preds_dollar, label='PatchTST', alpha=0.8)
plt.title('Test Set Predictions vs Actual')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.savefig('../outputs/plots/test_predictions_all.png')
plt.show()"""

code_8 = """# 7. Zoomed-in 30-day window
zoom_window = 30
dates_zoom = dates[-zoom_window:]

plt.figure(figsize=(15, 7))
plt.plot(dates_zoom, y_test_dollar[-zoom_window:], label='Actual Close', color='black', linewidth=2, marker='o')
plt.plot(dates_zoom, lstm_preds_dollar[-zoom_window:], label='LSTM', marker='x')
plt.plot(dates_zoom, gru_preds_dollar[-zoom_window:], label='GRU', marker='s')
plt.plot(dates_zoom, patchtst_preds_dollar[-zoom_window:], label='PatchTST', marker='^')
plt.title('Zoomed 30-Day Window: Notice the "Lagging Behavior"')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.savefig('../outputs/plots/zoomed_30d_predictions.png')
plt.show()"""

code_9 = """# 8. Compile Report Assets
# Copy all useful plots and tables to report_assets folder
assets = [
    '../outputs/plots/dataset_overview.png',
    '../outputs/plots/lstm_learning_curves.png',
    '../outputs/plots/gru_learning_curves.png',
    '../outputs/plots/test_predictions_all.png',
    '../outputs/plots/zoomed_30d_predictions.png',
    '../outputs/tables/final_metrics_comparison.csv'
]

for asset in assets:
    if os.path.exists(asset):
        shutil.copy(asset, '../report_assets/')
        print(f"Copied {os.path.basename(asset)} to report_assets/")
    else:
        print(f"File not found: {asset}")"""

nb.cells = [
    nbf.v4.new_markdown_cell("# Deep Learning Design and Evaluation (Comparison)"),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_code_cell(code_7),
    nbf.v4.new_code_cell(code_8),
    nbf.v4.new_code_cell(code_9)
]

with open('notebooks/04_comparison.ipynb', 'w') as f:
    nbf.write(nb, f)
