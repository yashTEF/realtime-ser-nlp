import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# ==============================
# Global Configuration
# ==============================
DATA_DIR = "/kaggle/input/"
CSV_PATH = os.path.join(DATA_DIR, "iemocap/iemocap_regression_metadata.csv")
DATASET = os.path.join(DATA_DIR, "iemocapfullrelease")
MFCC_PATH = os.path.join(DATA_DIR, "mfcc-regression")

MODEL_SAVE_PATH = "vad_regressor.pt"

BATCH_SIZE = 16
NUM_EPOCHS = 10
MAX_LEN = 128
LR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 3

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
        mfcc = np.load(os.path.join(MFCC_PATH, row['mfcc_path']))
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
# Model
# ==============================
class EmotionVADRegressor(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.regressor(hn[-1])
        return out * 4 + 1  # Rescale to [1, 5]

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
    acc = np.mean(np.round(preds_np) == np.round(targets_np))
    return mae, acc

# ==============================
# Training Loop
# ==============================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, desc="Training", leave=False)
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)

# ==============================
# Main
# ==============================
def main():
    print("Files in dataset directory:", os.listdir(DATA_DIR))

    dataset = IEMOCAPRegressionDataset(CSV_PATH, max_len=MAX_LEN)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = EmotionVADRegressor().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_mae = float("inf")
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        mae, acc = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | MAE: {mae:.4f} | Rounded Acc: {acc:.4f}")

        if mae < best_mae:
            best_mae = mae
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"📈 New best MAE: {mae:.4f}. Model saved.")
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("🛑 Early stopping triggered.")
                break

    print(f"✅ Best MAE: {best_mae:.4f}. Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
