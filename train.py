import warnings
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")

import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoConfig
import transformers.modeling_utils as modeling_utils
from contextlib import nullcontext
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import time

# Updated import: use the new SpeechBrain inference module.
from speechbrain.inference import EncoderClassifier
from tqdm import tqdm

# Patch: define init_empty_weights if not present
if not hasattr(modeling_utils, "init_empty_weights"):
    modeling_utils.init_empty_weights = nullcontext

# Patch: define find_tied_parameters if not present
if not hasattr(modeling_utils, "find_tied_parameters"):
    def find_tied_parameters(model):
        return []
    modeling_utils.find_tied_parameters = find_tied_parameters

# Hyperparameters
DEVICE = torch.device('cpu')
print("CUDA is not available. Using CPU for training.")
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
WAV2VEC_MODEL = "facebook/wav2vec2-base"
EMOTION2VEC_MODEL = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
SAMPLE_RATE = 16000
HIDDEN_SIZE = 256
NUM_DIMENSIONS = 3   # For val, act, dom
NUM_LEVELS = 5       # Each dimension ranges from 0 to 4
BATCH_SIZE = 1       # Reduced batch size for memory concerns
EPOCHS = 5

# Maximum audio length in samples (e.g., 30 seconds)
MAX_AUDIO_LENGTH = SAMPLE_RATE * 2

# Baseline SER Model (Dimensional Attributes)
class BaselineSER(nn.Module):
    def __init__(self, num_dimensions=NUM_DIMENSIONS, num_levels=NUM_LEVELS):
        super().__init__()
        
        print("Loading wav2vec2 model...")
        self.wav2vec = self._load_model_safely(WAV2VEC_MODEL, model_class=Wav2Vec2Model)
        # Optionally disable gradient checkpointing if available.
        if hasattr(self.wav2vec, "gradient_checkpointing_disable"):
            self.wav2vec.gradient_checkpointing_disable()
        
        print("Loading emotion2vec model via SpeechBrain...")
        self.emotion2vec = EncoderClassifier.from_hparams(
            source=EMOTION2VEC_MODEL,
            savedir="pretrained_models/emotion2vec",
            run_opts={"device": DEVICE}
        )
        # Patch: if compute_features is missing, define it based on the wav2vec2 submodule.
        if not hasattr(self.emotion2vec.mods, "compute_features"):
            self.emotion2vec.mods.compute_features = lambda wavs: self.emotion2vec.mods.wav2vec2.extract_features(wavs)
        
        print("Loading processor...")
        self.processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL, cache_dir="./model_cache")
        
        # We assume each model produces embeddings of size 768.
        # After concatenation, the linear layer receives 768*2 features.
        self.audio_linear = nn.Linear(768 * 2, HIDDEN_SIZE)
        self.classifier = nn.Linear(HIDDEN_SIZE, num_dimensions * num_levels)
    
    def _load_model_safely(self, model_name, model_class, max_retries=3):
        for attempt in range(max_retries):
            try:
                return model_class.from_pretrained(
                    model_name,
                    cache_dir="./model_cache",
                    local_files_only=False,
                    token=None,
                )
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to load model {model_name} after {max_retries} attempts")
                    raise
        
    def forward(self, audio):
        # 'audio' is a tensor of shape [batch, sequence_length].
        # For wav2vec2, convert each sample to a NumPy array.
        audio_list = [a.cpu().numpy() for a in audio]
        inputs = self.processor(audio_list, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Get wav2vec2 embeddings and average over the time dimension.
        wav2vec_out = self.wav2vec(**inputs).last_hidden_state.mean(dim=1)
        assert wav2vec_out.dim() == 2 and wav2vec_out.size(1) == 768, \
            f"Expected wav2vec2 output shape (batch, 768), got {wav2vec_out.shape}"
        
        # For SpeechBrain, ensure audio is [batch, time]
        if audio.dim() == 3:  # [batch, channels, time]
            sb_audio = audio.squeeze(1)
        else:  # Already [batch, time]
            sb_audio = audio
                
        with torch.no_grad():
            # Get embeddings from the wav2vec2 component of emotion2vec
            embeddings = self.emotion2vec.mods.wav2vec2(sb_audio.to(DEVICE))
            emotion2vec_out = embeddings.mean(dim=1)
        
        assert emotion2vec_out.dim() == 2 and emotion2vec_out.size(1) == 768, \
            f"Expected emotion2vec output shape (batch, 768), got {emotion2vec_out.shape}"
        
        # Concatenate the two embeddings.
        combined = torch.cat((wav2vec_out, emotion2vec_out), dim=1)
        audio_enc = self.audio_linear(combined)
        assert audio_enc.dim() == 2 and audio_enc.size(1) == HIDDEN_SIZE, \
            f"Expected audio linear output shape (batch, {HIDDEN_SIZE}), got {audio_enc.shape}"
        
        logits = self.classifier(audio_enc)
        assert logits.dim() == 2 and logits.size(1) == NUM_DIMENSIONS * NUM_LEVELS, \
            f"Expected classifier output shape (batch, {NUM_DIMENSIONS * NUM_LEVELS}), got {logits.shape}"
        
        return logits.view(-1, NUM_DIMENSIONS, NUM_LEVELS)

# Collate function to pad variable-length audio sequences
def collate_fn(batch):
    audios, labels = zip(*batch)
    lengths = [audio.size(0) for audio in audios]
    max_len = max(lengths)
    padded_audios = [torch.nn.functional.pad(audio, (0, max_len - audio.size(0))) for audio in audios]
    padded_audios = torch.stack(padded_audios)
    labels = torch.stack([torch.tensor([label['val'], label['act'], label['dom']], dtype=torch.long) for label in labels])
    return padded_audios, labels

# IEMOCAP Dataset Class (Attributes)
class IEMOCAPDataset(Dataset):
    def __init__(self, csv_file, audio_dir, transform=None, sample_ratio=0.01):
        self.data = pd.read_csv(csv_file)
        if sample_ratio < 1.0:
            self.data = self.data.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
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
            waveform = resampler(waveform)
        if waveform.size(0) > MAX_AUDIO_LENGTH:
            waveform = waveform[:MAX_AUDIO_LENGTH]
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label

# Training function
def train_model(model, dataloader, criterion, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (audios, labels) in progress_bar:
            audios, labels = audios.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(audios)
            loss = 0
            for dim in range(NUM_DIMENSIONS):
                dim_logits = logits[:, dim, :]
                dim_labels = labels[:, dim]
                loss += criterion(dim_logits, dim_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

# Inference function for attribute labels
def predict_attributes(model, audio):
    model.eval()
    with torch.no_grad():
        audio = audio.to(DEVICE)
        logits = model(audio.unsqueeze(0))
        preds = torch.argmax(logits, dim=2).squeeze(0)
        return {'val': preds[0].item() + 1, 'act': preds[1].item() + 1, 'dom': preds[2].item() + 1}

# Main Execution
if __name__ == "__main__":
    csv_file = "data/iemocap_annotations.csv"
    audio_dir = "data/IEMOCAP_full_release"
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}. Please run preprocess.py first.")
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    dataset = IEMOCAPDataset(csv_file, audio_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    model = BaselineSER().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    print("Training Baseline SER Model (Attributes)...")
    train_model(model, dataloader, criterion, optimizer)
    
    sample_audio, _ = dataset[0]
    prediction = predict_attributes(model, sample_audio)
    print(f"\nPredicted Attributes: val={prediction['val']}, act={prediction['act']}, dom={prediction['dom']}")
    
    print("Baseline model execution completed!")
