import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error

# ==============================
# Dataset Class
# ==============================
class IEMOCAPRegressionDataset(Dataset):
    def __init__(self, csv_path, max_len=None):
        self.data = pd.read_csv(csv_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        mfcc = np.load(row['mfcc_path'])  # (T, 13)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)

        if self.max_len:
            T = mfcc.shape[0]
            if T < self.max_len:
                pad = torch.zeros((self.max_len - T, mfcc.shape[1]))
                mfcc = torch.cat([mfcc, pad], dim=0)
            else:
                mfcc = mfcc[:self.max_len]

        target = torch.tensor([row['valence'], row['arousal'], row['dominance']], dtype=torch.float32)
        return mfcc, target

# ==============================
# Updated Model with [1, 5] constraint
# ==============================
class EmotionVADRegressor(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()  # output ∈ (0, 1)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.regressor(hn[-1])  # shape: (batch, 3)
        return out * 4 + 1  # scale to [1, 5]

# ==============================
# Evaluation
# ==============================
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)

    mae = mean_absolute_error(targets_np, preds_np)
    acc = np.mean(np.round(preds_np) == np.round(targets_np))  # classification-style
    return mae, acc

# ==============================
# Training Loop
# ==============================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ==============================
# Main
# ==============================
def main():
    csv_path = "iemocap_regression_metadata.csv"
    batch_size = 16
    num_epochs = 5
    max_len = 128
    lr = 0.1
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = IEMOCAPRegressionDataset(csv_path, max_len=max_len)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = EmotionVADRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        mae, acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | MAE: {mae:.4f} | Rounded Acc: {acc:.4f}")

    torch.save(model.state_dict(), "vad_regressor.pt")
    print("✅ Model saved to vad_regressor.pt")

if __name__ == "__main__":
    main()
