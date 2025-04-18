import torch
import numpy as np
import argparse
import os
import librosa
import torch.nn as nn

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_MFCC = 13  # Number of MFCC features

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

def extract_mfcc_features(file_path, start, end, max_len=128):
    """
    Loads an audio file, extracts the segment from start to end (in seconds),
    and computes MFCC features.
    """
    # Load audio segment
    y, sr = librosa.load(file_path, sr=None, offset=start, duration=end-start)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T  # Transpose to get time steps as first dimension
    
    # Handle padding/truncation similar to training
    if max_len:
        T = mfcc.shape[0]
        if T < max_len:
            # Pad with zeros
            pad = np.zeros((max_len - T, mfcc.shape[1]))
            mfcc = np.concatenate([mfcc, pad], axis=0)
        else:
            # Truncate
            mfcc = mfcc[:max_len]
    
    # Convert to tensor
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)
    return mfcc_tensor

def predict_attributes(model, mfcc_features):
    """
    Runs inference on a single MFCC feature set.
    Returns a dictionary with predicted valence, arousal, and dominance.
    """
    model.eval()
    with torch.no_grad():
        mfcc_features = mfcc_features.to(DEVICE).unsqueeze(0)  # Add batch dimension
        predictions = model(mfcc_features)
        predictions = predictions.cpu().numpy()[0]  # Remove batch dimension
        
        return {
            'valence': predictions[0],
            'arousal': predictions[1],
            'dominance': predictions[2]
        }

def main(args):
    # Load the trained model checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file {args.checkpoint} not found!")
    
    model = EmotionVADRegressor().to(DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    
    # Load and process the audio segment
    if not os.path.exists(args.audio_file):
        raise FileNotFoundError(f"Audio file {args.audio_file} not found!")
    
    # Extract MFCC features
    mfcc_features = extract_mfcc_features(args.audio_file, args.start, args.end)
    
    # Predict attributes
    prediction = predict_attributes(model, mfcc_features)
    
    print(f"Predicted Attributes for window [{args.start:.2f}s - {args.end:.2f}s]:")
    print(f"Valence: {prediction['valence']:.2f}, Arousal: {prediction['arousal']:.2f}, Dominance: {prediction['dominance']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for VAD emotion predictions")
    parser.add_argument('--audio_file', type=str, required=True, help="Path to the audio file")
    parser.add_argument('--start', type=float, required=True, help="Start time of the window (in seconds)")
    parser.add_argument('--end', type=float, required=True, help="End time of the window (in seconds)")
    parser.add_argument('--checkpoint', type=str, default='vad_regressor.pt', help="Path to the trained model checkpoint")
    args = parser.parse_args()
    
    main(args)
