import os
import pandas as pd
import torchaudio
import argparse
import re
from collections import defaultdict
import logging

# Constants
SAMPLE_RATE = 16000

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocess_iemocap.log'),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

def preprocess_iemocap(iemocap_root, output_csv):
    """
    Preprocess IEMOCAP dataset to create a CSV file with file paths and averaged attribute labels (val, act, dom).
    
    Args:
        iemocap_root (str): Root directory of IEMOCAP dataset
        output_csv (str): Path to save the output CSV file
    """
    data_dict = defaultdict(list)
    
    logger.info(f"Starting preprocessing of IEMOCAP dataset at {iemocap_root}")
    
    # Traverse all sessions (Session1 to Session5)
    for session in range(1, 6):
        session_dir = os.path.join(iemocap_root, f'Session{session}')
        if not os.path.exists(session_dir):
            logger.warning(f"Session directory {session_dir} not found, skipping...")
            continue
        
        # Path to evaluation and audio files
        eval_dir = os.path.join(session_dir, 'dialog', 'EmoEvaluation')
        audio_dir = os.path.join(session_dir, 'dialog', 'wav')
        
        if not os.path.exists(eval_dir):
            logger.warning(f"Evaluation directory {eval_dir} not found, skipping session {session}...")
            continue
        if not os.path.exists(audio_dir):
            logger.warning(f"Audio directory {audio_dir} not found, skipping session {session}...")
            continue
        
        logger.info(f"Processing session {session}")
        
        # Process each evaluation file
        for eval_file in os.listdir(eval_dir):
            if not eval_file.endswith('.txt'):
                continue
                
            eval_path = os.path.join(eval_dir, eval_file)
            logger.debug(f"Reading evaluation file: {eval_path}")
            
            try:
                with open(eval_path, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                logger.error(f"Failed to read {eval_path}: {e}")
                continue
                
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('['):
                    # Extract TURN_NAME (e.g., Ses01F_impro01_F000)
                    match = re.search(r'\[[\d\.\s-]+\]\s+(\S+)\s', line)
                    if not match:
                        logger.debug(f"No match found in line: {line}")
                        i += 1
                        continue
                    wav_id = match.group(1)
                    
                    # Collect attribute labels from subsequent lines
                    attrs = {'val': [], 'act': [], 'dom': []}
                    i += 1
                    while i < len(lines) and not lines[i].startswith('['):
                        sub_line = lines[i].strip()
                        if sub_line.startswith('A-'):
                            attr_match = re.search(r'val (\d); act (\d); dom\s*(\d);', sub_line)
                            if attr_match:
                                val, act, dom = map(int, attr_match.groups())
                                attrs['val'].append(val - 1)  # Shift to 0-4 range
                                attrs['act'].append(act - 1)
                                attrs['dom'].append(dom - 1)
                        i += 1
                    
                    # Average the attribute ratings
                    if attrs['val']:  # Only process if we have attribute ratings
                        avg_val = round(sum(attrs['val']) / len(attrs['val']))
                        avg_act = round(sum(attrs['act']) / len(attrs['act']))
                        avg_dom = round(sum(attrs['dom']) / len(attrs['dom']))
                        
                        # Construct audio file path (e.g., Ses01F_impro01.wav)
                        dialog_name = '_'.join(wav_id.split('_')[:-1])
                        audio_path = os.path.join(audio_dir, f'{dialog_name}.wav')
                        
                        if not os.path.exists(audio_path):
                            logger.warning(f"Audio file {audio_path} not found, skipping...")
                            continue
                        
                        # Verify audio
                        try:
                            backends = torchaudio.list_audio_backends()
                            logger.debug(f"Available audio backends: {backends}")
                            waveform, sr = torchaudio.load(audio_path)
                            logger.debug(f"Loaded {audio_path} with sample rate {sr}")
                            if sr != SAMPLE_RATE:
                                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                                waveform = resampler(waveform)
                                logger.debug(f"Resampled {audio_path} to {SAMPLE_RATE} Hz")
                            if waveform.size(0) > 1:
                                waveform = waveform.mean(dim=0, keepdim=True)
                                logger.debug(f"Converted {audio_path} to mono")
                        except Exception as e:
                            logger.error(f"Error loading {audio_path}: {e}")
                            continue
                        
                        rel_path = os.path.relpath(audio_path, iemocap_root)
                        data_dict[wav_id].append({
                            'file_path': rel_path,
                            'val': avg_val,
                            'act': avg_act,
                            'dom': avg_dom
                        })
                        logger.debug(f"Processed {wav_id} with val={avg_val}, act={avg_act}, dom={avg_dom}")
                    else:
                        logger.debug(f"No attribute ratings found for {wav_id}")
                else:
                    i += 1
    
    # Convert to list, ensuring one entry per wav_id
    data = [d[0] for d in data_dict.values()]
    
    # Save to CSV
    try:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        logger.info(f"Preprocessed IEMOCAP dataset saved to {output_csv}. Total samples: {len(df)}")
    except Exception as e:
        logger.error(f"Failed to save CSV to {output_csv}: {e}")
        raise
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess IEMOCAP dataset")
    parser.add_argument('--iemocap_root', type=str, required=True, help="Root directory of IEMOCAP dataset")
    parser.add_argument('--output_csv', type=str, default='data/iemocap_annotations.csv', help="Output CSV file path")
    args = parser.parse_args()
    
    preprocess_iemocap(args.iemocap_root, args.output_csv)