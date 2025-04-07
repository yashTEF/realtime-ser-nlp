import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WAV2VEC_MODEL = "facebook/wav2vec2-base"
EMOTION2VEC_MODEL = "iic/emotion2vec_base"
SAMPLE_RATE = 16000
HIDDEN_SIZE = 256
NUM_DIMENSIONS = 3  # For val, act, dom
NUM_LEVELS = 5  # Each dimension ranges from 0 to 4
BATCH_SIZE = 8
EPOCHS = 5

# Baseline SER Model (Dimensional Attributes)
class BaselineSER(nn.Module):
    def __init__(self, num_dimensions=NUM_DIMENSIONS, num_levels=NUM_LEVELS):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL)
        self.emotion2vec = Wav2Vec2Model.from_pretrained(EMOTION2VEC_MODEL)
        self.processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)
        
        self.audio_linear = nn.Linear(768 * 2, HIDDEN_SIZE)
        self.classifier = nn.Linear(HIDDEN_SIZE, num_dimensions * num_levels)
        
    def forward(self, audio):
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        wav2vec_out = self.wav2vec(**inputs).last_hidden_state.mean(dim=1)
        emotion2vec_out = self.emotion2vec(**inputs).last_hidden_state.mean(dim=1)
        combined = torch.cat((wav2vec_out, emotion2vec_out), dim=1)
        
        audio_enc = self.audio_linear(combined)
        logits = self.classifier(audio_enc)
        return logits.view(-1, NUM_DIMENSIONS, NUM_LEVELS)

# Collate function to pad variable-length audio sequences
def collate_fn(batch):
    audios, labels = zip(*batch)
    lengths = [audio.size(0) for audio in audios]
    max_len = max(lengths)
    padded_audios = [torch.nn.functional.pad(audio, (0, max_len - len(audio))) for audio in audios]
    padded_audios = torch.stack(padded_audios)
    labels = torch.stack([torch.tensor([label['val'], label['act'], label['dom']], dtype=torch.long) for label in labels])
    return padded_audios, labels

# IEMOCAP Dataset Class (Attributes)
class IEMOCAPDataset(Dataset):
    def __init__(self, csv_file, audio_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['file_path'])
        label = {
            'val': int(row['val']),
            'act': int(row['act']),
            'dom': int(row['dom'])
        }
        
        waveform, sr = torchaudio.load(file_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            orphanage = resampler(waveform)
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label

# Training function
def train_model(model, dataloader, criterion, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for audios, labels in dataloader:
            audios, labels = audios.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(audios)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

# Inference function for attribute labels
def predict_attributes(model, audio):
    model.eval()
    with torch.no_grad():
        audio = audio.to(DEVICE)
        logits = model(audio.unsqueeze(0))
        preds = torch.argmax(logits, dim=2).squeeze(0)
        return {'val': preds[0].item() + 1, 'act': preds[1].item() + 1, 'dom': preds[2].item() + 1}  # Shift back to 1-5

# Main Execution
if __name__ == "__main__":
    # Paths to your preprocessed CSV and IEMOCAP root directory
    csv_file = "data/iemocap_annotations.csv"
    audio_dir = "IEMOCAP_full_release"  # Update this to your IEMOCAP root directory
    
    # Verify paths exist
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}. Please run preprocess.py first.")
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    # Load dataset
    dataset = IEMOCAPDataset(csv_file, audio_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # Initialize model, loss, and optimizer
    model = BaselineSER().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # Train the model
    print("Training Baseline SER Model (Attributes)...")
    train_model(model, dataloader, criterion, optimizer)
    
    # Example inference on a single audio sample from the dataset
    sample_audio, _ = dataset[0]
    prediction = predict_attributes(model, sample_audio)
    print(f"\nPredicted Attributes: val={prediction['val']}, act={prediction['act']}, dom={prediction['dom']}")
    
    print("Baseline model execution completed!")