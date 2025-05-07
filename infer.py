import torch
import numpy as np
import pandas as pd
import librosa
import whisper
import nltk
from tqdm import tqdm
from pathlib import Path

def streaming_inference(model, audio_path, word2idx, embedding_matrix, mfcc_mean, mfcc_std, emotion_classes, 
                       chunk_duration=3.0, max_length=128, n_mfcc=13, max_audio_length=300, device='cuda', 
                       output_csv='/kaggle/working/streaming_inference_results.csv'):
    """
    Perform streaming inference on a WAV file using a multimodal emotion classifier.
    
    Args:
        model: Trained Multimodal_LSTMEmotionClassifier model.
        audio_path (str): Path to the input WAV file.
        word2idx (dict): Vocabulary mapping words to indices.
        embedding_matrix (torch.Tensor): Pretrained embedding matrix.
        mfcc_mean (torch.Tensor): Mean for MFCC normalization.
        mfcc_std (torch.Tensor): Standard deviation for MFCC normalization.
        emotion_classes (list): List of emotion labels (e.g., ['angry', 'happy', 'sad', 'neutral']).
        chunk_duration (float): Duration of each audio chunk in seconds (default: 3.0).
        max_length (int): Maximum text sequence length (default: 128).
        n_mfcc (int): Number of MFCC coefficients (default: 13).
        max_audio_length (int): Maximum audio sequence length in frames (default: 300).
        device (str): Device to run inference on (default: 'cuda').
        output_csv (str): Path to save inference results (default: '/kaggle/working/streaming_inference_results.csv').
    
    Returns:
        list: List of dictionaries containing inference results for each chunk.
    """
    # Initialize
    model.eval()
    model.to(device)
    idx_to_emotion = {idx: emo for idx, emo in enumerate(emotion_classes)}
    results = []
    context_texts = []  # Store up to 5 previous transcriptions
    
    # Load Whisper model
    try:
        whisper_model = whisper.load_model("tiny")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        raise
    
    # Load audio
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        raise
    
    # Calculate chunk parameters
    chunk_samples = int(chunk_duration * sr)
    total_duration = len(audio) / sr
    num_chunks = int(np.ceil(total_duration / chunk_duration))
    
    print(f"Processing audio: {audio_path}")
    print(f"Total duration: {total_duration:.2f}s, Chunks: {num_chunks}, Chunk duration: {chunk_duration}s")
    
    for chunk_idx in tqdm(range(num_chunks), desc="Processing audio chunks"):
        # Extract chunk
        start_sample = chunk_idx * chunk_samples
        end_sample = min((chunk_idx + 1) * chunk_samples, len(audio))
        chunk_audio = audio[start_sample:end_sample]
        chunk_start = start_sample / sr
        chunk_end = end_sample / sr
        
        # Extract MFCC
        try:
            mfcc = librosa.feature.mfcc(y=chunk_audio, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
            mfcc = torch.tensor(mfcc.T, dtype=torch.float32)  # [time_frames, n_mfcc]
        except Exception as e:
            print(f"Error extracting MFCC for chunk {chunk_idx}: {e}")
            continue
        
        # Normalize MFCC
        mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)
        
        # Compute sequence length
        seq_len = min(mfcc.shape[0], max_audio_length)
        
        # Pad or truncate MFCC
        if mfcc.shape[0] < max_audio_length:
            pad = torch.zeros((max_audio_length - mfcc.shape[0], mfcc.shape[1]))
            mfcc = torch.cat([mfcc, pad], dim=0)
        else:
            mfcc = mfcc[:max_audio_length]
        
        # Transcribe chunk
        try:
            # Save chunk temporarily for Whisper
            temp_wav = f"/tmp/chunk_{chunk_idx}.wav"
            librosa.output.write_wav(temp_wav, chunk_audio, sr)
            result = whisper_model.transcribe(temp_wav, language='en')
            transcription = result['text'].strip()
            Path(temp_wav).unlink(missing_ok=True)
        except Exception as e:
            print(f"Error transcribing chunk {chunk_idx}: {e}")
            transcription = ""
        
        # Update context
        if transcription:
            context_texts.append(transcription)
            if len(context_texts) > 5:
                context_texts.pop(0)
        else:
            context_texts.append("")  # Empty transcription for failed chunks
        
        # Prepare text input
        context_text = " ".join(context_texts)
        tokens = nltk.word_tokenize(context_text.lower())[:max_length]
        input_ids = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
        if len(input_ids) < max_length:
            input_ids += [word2idx['<PAD>']] * (max_length - len(input_ids))
        input_ids = torch.tensor([input_ids], dtype=torch.long)  # [1, max_length]
        
        # Prepare model inputs
        mfcc = mfcc.unsqueeze(0)  # [1, max_audio_length, n_mfcc]
        seq_len = torch.tensor([seq_len], dtype=torch.long)  # [1]
        
        # Move to device
        input_ids = input_ids.to(device)
        mfcc = mfcc.to(device)
        seq_len = seq_len.to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_ids, mfcc, seq_len)  # [1, num_classes]
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_emotion = idx_to_emotion[pred_idx]
            confidence = probs[0, pred_idx].item()
        
        # Print results
        print(f"\nChunk {chunk_idx + 1}: {chunk_start:.2f}s - {chunk_end:.2f}s")
        print(f"Transcription: {transcription}")
        print(f"Predicted Emotion: {pred_emotion} (Confidence: {confidence:.4f})")
        
        # Store results
        result = {
            'chunk_start': chunk_start,
            'chunk_end': chunk_end,
            'transcription': transcription,
            'predicted_emotion': pred_emotion,
            'confidence': confidence
        }
        results.append(result)
    
    # Save results to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)
    print(f"\nInference results saved to {output_csv}")
    
    return results