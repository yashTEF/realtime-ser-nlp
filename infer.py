import torch
import torchaudio
import argparse
import os
from train import BaselineSER  # Ensure that train.py is in the same directory or PYTHONPATH

# Constants
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio_segment(file_path, start, end):
    """
    Loads an audio file, resamples it to SAMPLE_RATE, converts it to mono,
    and extracts the segment from start to end (in seconds).
    Returns the audio segment and its duration.
    """
    waveform, sr = torchaudio.load(file_path)
    # Convert to mono if necessary
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    # Calculate sample indices based on start and end times
    start_sample = int(start * SAMPLE_RATE)
    end_sample = int(end * SAMPLE_RATE)
    segment = waveform[start_sample:end_sample]
    duration = end - start
    return segment, duration

def predict_attributes(model, audio, duration):
    """
    Runs inference on a single audio segment with its duration.
    Returns a dictionary with predicted valence, arousal, and dominance.
    """
    model.eval()
    with torch.no_grad():
        audio = audio.to(DEVICE)
        # Normalize duration (assuming max segment length is 15s, adjust if needed)
        duration_tensor = torch.tensor([duration], dtype=torch.float, device=DEVICE)
        logits = model(audio.unsqueeze(0), duration_tensor)
        preds = torch.argmax(logits, dim=2).squeeze(0)
        # Convert from 0-indexed (0-4) to ratings (1-5)
        return {
            'val': preds[0].item() + 1,
            'act': preds[1].item() + 1,
            'dom': preds[2].item() + 1
        }

def main(args):
    # Load the trained model checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file {args.checkpoint} not found!")
    model = BaselineSER().to(DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    
    # Load and slice the audio segment using provided window (start and end times)
    if not os.path.exists(args.audio_file):
        raise FileNotFoundError(f"Audio file {args.audio_file} not found!")
    segment, duration = load_audio_segment(args.audio_file, args.start, args.end)
    
    # Predict attributes for the given segment
    prediction = predict_attributes(model, segment, duration)
    print(f"Predicted Attributes for window [{args.start} - {args.end}] (duration: {duration:.2f}s):")
    print(f"Valence: {prediction['val']}, Arousal: {prediction['act']}, Dominance: {prediction['dom']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for real-time SER predictions")
    parser.add_argument('--audio_file', type=str, required=True, help="Path to the audio file")
    parser.add_argument('--start', type=float, required=True, help="Start time of the window (in seconds)")
    parser.add_argument('--end', type=float, required=True, help="End time of the window (in seconds)")
    parser.add_argument('--checkpoint', type=str, default='model.pt', help="Path to the trained model checkpoint")
    args = parser.parse_args()
    
    main(args)
