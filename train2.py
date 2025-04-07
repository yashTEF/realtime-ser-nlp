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
EMOTION2VEC_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
SAMPLE_RATE = 16000
HIDDEN_SIZE = 256
NUM_DIMENSIONS = 3
NUM_LEVELS = 5
BATCH_SIZE = 8
EPOCHS = 1

# Initialize processor
PROCESSOR = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)

class BaselineSER(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL).to(DEVICE)
        self.emotion2vec = Wav2Vec2Model.from_pretrained(EMOTION2VEC_MODEL).to(DEVICE)
        
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        for param in self.emotion2vec.parameters():
            param.requires_grad = False
            
        self.audio_linear = nn.Linear(1024 * 2, HIDDEN_SIZE)
        self.classifier = nn.Linear(HIDDEN_SIZE, NUM_DIMENSIONS * NUM_LEVELS)

    def forward(self, input_values, attention_mask=None):
        inputs = {
            "input_values": input_values,
            "attention_mask": attention_mask
        }
        
        with torch.no_grad():
            wav2vec_out = self.wav2vec(**inputs).last_hidden_state.mean(dim=1)
            emotion2vec_out = self.emotion2vec(**inputs).last_hidden_state.mean(dim=1)
        
        combined = torch.cat((wav2vec_out, emotion2vec_out), dim=1)
        audio_enc = torch.relu(self.audio_linear(combined))
        logits = self.classifier(audio_enc)
        return logits.view(-1, NUM_DIMENSIONS, NUM_LEVELS)

def collate_fn(batch):
    audios, labels = zip(*batch)
    
    processed = PROCESSOR(
        [audio.cpu().numpy() for audio in audios],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    
    label_tensor = torch.stack([
        torch.tensor([label['val']-1, label['act']-1, label['dom']-1], dtype=torch.long)
        for label in labels
    ])
    
    return (
        processed.input_values.to(DEVICE),
        processed.attention_mask.to(DEVICE),
        label_tensor.to(DEVICE)
    )

class IEMOCAPDataset(Dataset):
    def __init__(self, csv_file, audio_dir):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        
        if not os.path.exists(audio_dir):
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        if not {'file_path', 'val', 'act', 'dom'}.issubset(self.data.columns):
            raise ValueError("CSV missing required columns: file_path, val, act, dom")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['file_path'])
        
        try:
            waveform, sr = torchaudio.load(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            
        return waveform.squeeze(), {
            'val': int(row['val']),
            'act': int(row['act']),
            'dom': int(row['dom'])
        }

def train_model(model, dataloader, criterion, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for input_values, attention_mask, labels in dataloader:
            optimizer.zero_grad()
            logits = model(input_values, attention_mask)
            
            loss = sum(
                criterion(logits[:, i, :], labels[:, i])
                for i in range(NUM_DIMENSIONS)
            ) / NUM_DIMENSIONS
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

def predict_attributes(model, audio_waveform):
    model.eval()
    with torch.no_grad():
        processed = PROCESSOR(
            audio_waveform.cpu().numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).to(DEVICE)
        
        logits = model(processed.input_values, processed.attention_mask)
        preds = torch.argmax(logits, dim=2).squeeze(0)
        
    return {
        'val': preds[0].item() + 1,
        'act': preds[1].item() + 1,
        'dom': preds[2].item() + 1
    }

if __name__ == "__main__":
    model = BaselineSER().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': model.audio_linear.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=3e-4)
    
    dataset = IEMOCAPDataset("data/iemocap_annotations.csv", "IEMOCAP_full_release")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    print("Starting training...")
    train_model(model, dataloader, criterion, optimizer)
    
    test_audio, test_label = dataset[0]
    prediction = predict_attributes(model, test_audio.to(DEVICE))
    
    print("\nTest Sample Results:")
    print(f"Original: Val={test_label['val']}, Act={test_label['act']}, Dom={test_label['dom']}")
    print(f"Predicted: Val={prediction['val']}, Act={prediction['act']}, Dom={prediction['dom']}")
    
    torch.save(model.state_dict(), "ser_model.pth")
    print("\nModel saved successfully!")
