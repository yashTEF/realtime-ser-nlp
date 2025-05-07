import os
import re
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# CONFIG
ROOT = "IEMOCAP_full_release"  # Update to your IEMOCAP root path, e.g., "/kaggle/input/iemocapfullrelease/IEMOCAP_full_release"
SESSIONS = ["Session1", "Session2", "Session3", "Session4", "Session5"]
SR = 16000
N_MFCC = 13
OUTPUT_DIR = "mfcc_regression"
CSV_OUT = "iemocap_regression_metadata.csv"
TIME_EXTENSION = 10.0  # Seconds to extend start and end times

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
metadata = []   # List to hold metadata for all segments

def parse_eval_line(line):
    """
    Parse a line from the EmoEvaluation file.
    Returns: (start, end, utt_id, emotion, valence, arousal, dominance) or None if parsing fails.
    """
    pattern = r"\[(\d+\.\d+) - (\d+\.\d+)\]\s+(\S+)\s+(\S+)\s+\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]"
    match = re.match(pattern, line.strip())
    if match:
        return match.groups()
    return None

for session in SESSIONS:
    eval_dir = os.path.join(ROOT, session, "dialog", "EmoEvaluation")
    wav_dir = os.path.join(ROOT, session, "dialog", "wav")
    
    for file in tqdm(os.listdir(eval_dir), desc=f"Processing {session}"):
        if not file.endswith(".txt"):
            continue
        
        # Extract dialogue name from evaluation file (e.g., "Ses01F_impro01.txt" → "Ses01F_impro01")
        dialogue_name = file.replace(".txt", "")
        # Path to full dialogue .wav file (e.g., IEMOCAP_full_release/Session1/dialog/wav/Ses01F_impro01.wav)
        wav_path = os.path.join(wav_dir, f"{dialogue_name}.wav")
        
        if not os.path.exists(wav_path):
            print(f"Warning: Dialogue file {wav_path} does not exist.")
            continue
        
        # Load the full dialogue audio to get its duration
        try:
            y_full, sr = librosa.load(wav_path, sr=SR)
            audio_duration = librosa.get_duration(y=y_full, sr=SR)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            continue
        
        with open(os.path.join(eval_dir, file)) as f:
            for line in f:
                if not line.startswith("["):
                    continue

                parsed = parse_eval_line(line)
                if not parsed:
                    continue

                start, end, utt_id, emotion, val, act, dom = parsed

                if emotion == "xxx":  # Skip unclear utterances
                    continue
                
                try:
                    print(f"Processing utterance: {utt_id}")
                    # Convert timestamps to float
                    start = float(start)
                    end = float(end)
                    
                    # Extend timestamps by 10 seconds, ensuring they stay within audio bounds
                    extended_start = max(0.0, start - TIME_EXTENSION)
                    extended_end = min(audio_duration, end + TIME_EXTENSION)
                    
                    # Extract the segment from the full audio
                    start_sample = int(extended_start * SR)
                    end_sample = int(extended_end * SR)
                    y_segment = y_full[start_sample:end_sample]
                    
                    # Extract MFCC features
                    mfcc = librosa.feature.mfcc(y=y_segment, sr=SR, n_mfcc=N_MFCC)
                    mfcc = mfcc.T  # (time, features)

                    # Save MFCC as .npy
                    save_name = f"{utt_id}.npy"
                    save_path = os.path.join(OUTPUT_DIR, save_name)
                    np.save(save_path, mfcc)

                    # Append metadata
                    metadata.append({
                        "utt_id": utt_id,
                        "session": session,
                        "emotion": emotion,
                        "valence": float(val),
                        "arousal": float(act),
                        "dominance": float(dom),
                        "start": extended_start,
                        "end": extended_end,
                        "original_start": start,
                        "original_end": end,
                        "mfcc_path": save_path
                    })

                except Exception as e:
                    print(f"Error processing {utt_id}: {e}")
                    continue

# Save metadata
df = pd.DataFrame(metadata)
df.to_csv(CSV_OUT, index=False)
print(f"Preprocessing complete. Metadata saved to {CSV_OUT}")