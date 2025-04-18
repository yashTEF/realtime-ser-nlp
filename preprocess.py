import os
import re
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# CONFIG
ROOT = "IEMOCAP_full_release"
SESSIONS = ["Session1", "Session2", "Session3", "Session4", "Session5"]
SR = 16000
N_MFCC = 13
OUTPUT_DIR = "mfcc_regression"
CSV_OUT = "iemocap_regression_metadata.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
metadata = []

def parse_eval_line(line):
    pattern = r"\[(\d+\.\d+) - (\d+\.\d+)\]\s+(\S+)\s+(\S+)\s+\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]"
    match = re.match(pattern, line.strip())
    if match:
        return match.groups()
    return None

for session in SESSIONS:
    eval_dir = os.path.join(ROOT, session, "dialog", "EmoEvaluation")
    wav_dir = os.path.join(ROOT, session, "dialog", "wav")

    for file in os.listdir(eval_dir):
        if not file.endswith(".txt"):
            continue
        wav_name = file.replace(".txt", ".wav")
        wav_path = os.path.join(wav_dir, wav_name)
        with open(os.path.join(eval_dir, file)) as f:
            for line in f:
                if not line.startswith("["):
                    continue

                parsed = parse_eval_line(line)
                if not parsed:
                    continue

                start, end, utt_id, emotion, val, act, dom = parsed

                if emotion == "xxx":  # skip unclear utterances
                    continue
                
                
                try:
                    print("debugging", wav_path)
                    y, sr = librosa.load(wav_path, sr=SR)
                    start_s = int(float(start) * SR)
                    end_s = int(float(end) * SR)
                    segment = y[start_s:end_s]

                    mfcc = librosa.feature.mfcc(y=segment, sr=SR, n_mfcc=N_MFCC)
                    mfcc = mfcc.T  # (time, features)

                    save_name = f"{utt_id}.npy"
                    save_path = os.path.join(OUTPUT_DIR, save_name)
                    np.save(save_path, mfcc)

                    metadata.append({
                        "utt_id": utt_id,
                        "session": session,
                        "emotion": emotion,
                        "valence": float(val),
                        "arousal": float(act),
                        "dominance": float(dom),
                        "start": float(start),
                        "end": float(end),
                        "mfcc_path": save_path
                    })

                except Exception as e:
                    print(f"Error processing {utt_id}: {e}")

# Save metadata
df = pd.DataFrame(metadata)
df.to_csv(CSV_OUT, index=False)
print(f"Preprocessing complete. Metadata saved to {CSV_OUT}")
