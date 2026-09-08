import nbformat as nbf

nb = nbf.v4.new_notebook()

code_1 = """import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")"""

code_2 = """# 1. Load Preprocessed Data
X_train = np.load('../data/X_train.npy')
y_train = np.load('../data/y_train.npy')
X_val = np.load('../data/X_val.npy')
y_val = np.load('../data/y_val.npy')
X_test = np.load('../data/X_test.npy')
y_test = np.load('../data/y_test.npy')

scaler_close = joblib.load('../models/scaler_close.pkl')

print(f"X_train shape: {X_train.shape}")
print(f"y_test shape: {y_test.shape}")"""

code_3 = """# 2. Naive Baseline
# The naive baseline predicts that tomorrow's close price will be exactly today's close price.
# In our sequenced data, today's close price is the last element of the X sequence for the 'Close' feature.
# We know 'Close' is at index 3.

def calculate_metrics(y_true, y_pred, name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Avoid division by zero for MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    print(f"{name} Metrics:")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.2f}%\\n")
    return mse, rmse, mae, mape

# Predict 'today' as 'tomorrow'
y_test_naive_scaled = X_test[:, -1, 3] # -1 is the last day in the 60-day window, 3 is the Close index

# Inverse transform to get real dollar values
y_test_dollar = scaler_close.inverse_transform(y_test.reshape(-1, 1))
y_test_naive_dollar = scaler_close.inverse_transform(y_test_naive_scaled.reshape(-1, 1))

naive_metrics = calculate_metrics(y_test_dollar, y_test_naive_dollar, "Naive Baseline (Test Set)")"""

code_4 = """# 3. PyTorch Dataloaders
batch_size = 64

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)"""

code_5 = """# 4. LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        return out"""

code_6 = """# 5. Training Loop Function (with Early Stopping)
def train_model(model, train_loader, val_loader, num_epochs=100, patience=10, lr=1e-3, model_path='../models/best_model.pth'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * X_batch.size(0)
            
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation Phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                epoch_val_loss += loss.item() * X_batch.size(0)
                
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Early Stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}')
            
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
            
    # Load best model
    model.load_state_dict(torch.load(model_path))
    return train_losses, val_losses"""

code_7 = """# 6. Train LSTM
lstm_model = LSTMModel().to(device)
print("Training LSTM...")
lstm_train_losses, lstm_val_losses = train_model(lstm_model, train_loader, val_loader, model_path='../models/best_lstm.pth')

plt.figure(figsize=(10, 5))
plt.plot(lstm_train_losses, label='Train Loss')
plt.plot(lstm_val_losses, label='Val Loss')
plt.title('LSTM Learning Curves')
plt.legend()
plt.grid(True)
plt.savefig('../outputs/plots/lstm_learning_curves.png')
plt.show()"""

code_8 = """# 7. GRU Model Definition
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
        
        out = self.fc(out[:, -1, :])
        return out"""

code_9 = """# 8. Train GRU
gru_model = GRUModel().to(device)
print("Training GRU...")
gru_train_losses, gru_val_losses = train_model(gru_model, train_loader, val_loader, model_path='../models/best_gru.pth')

plt.figure(figsize=(10, 5))
plt.plot(gru_train_losses, label='Train Loss')
plt.plot(gru_val_losses, label='Val Loss')
plt.title('GRU Learning Curves')
plt.legend()
plt.grid(True)
plt.savefig('../outputs/plots/gru_learning_curves.png')
plt.show()"""

nb.cells = [
    nbf.v4.new_markdown_cell("# RNN Models (LSTM & GRU) & Baseline"),
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

with open('notebooks/02_rnn_models.ipynb', 'w') as f:
    nbf.write(nb, f)
