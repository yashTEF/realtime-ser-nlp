"""
Simplified audio processing for Speech Emotion Recognition.
Handles complete audio files only - no streaming/chunking.
"""

import numpy as np
import librosa
import soundfile as sf
import whisper
import torch
import tempfile
import os
import warnings
from pydub import AudioSegment

from config import (
    SR, N_MFCC, MAX_AUDIO_LENGTH, WHISPER_MODEL_SIZE, DEVICE
)

# Suppress warnings
warnings.filterwarnings("ignore")


class AudioProcessor:
    """Simplified audio processing for complete audio files only."""
    
    def __init__(self):
        self.whisper_model = None
        self.mfcc_mean = None
        self.mfcc_std = None
    
    def load_whisper_model(self):
        """Load the Whisper model for speech-to-text conversion."""
        if self.whisper_model is None:
            try:
                print(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
                self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
                print("Whisper model loaded successfully")
            except Exception as e:
                print(f"Failed to load Whisper model: {e}")
                self.whisper_model = None
    
    def set_mfcc_stats(self, mean, std):
        """Set MFCC normalization statistics."""
        self.mfcc_mean = mean
        self.mfcc_std = std
    
    def speech_to_text(self, audio_file_path):
        """Convert speech to text using Whisper."""
        try:
            self.load_whisper_model()
            
            if self.whisper_model is None:
                return "Unable to transcribe audio"
            
            # Use Whisper to transcribe
            result = self.whisper_model.transcribe(audio_file_path)
            transcribed_text = result["text"].strip()
            print(f"Transcribed: {transcribed_text}")
            return transcribed_text
            
        except Exception as e:
            print(f"Error in speech-to-text: {e}")
            return "Unable to transcribe audio"
    
    def extract_mfcc_features(self, audio_file_path):
        """Extract MFCC features from audio file."""
        try:
            # Load audio with librosa
            y, sr = librosa.load(audio_file_path, sr=SR)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            mfcc = mfcc.T  # (time, features)
            
            # Limit length
            if mfcc.shape[0] > MAX_AUDIO_LENGTH:
                mfcc = mfcc[:MAX_AUDIO_LENGTH]
            
            print(f"Extracted MFCC features: {mfcc.shape}")
            return mfcc
            
        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            # Return dummy features
            return np.random.randn(100, N_MFCC)
    
    def preprocess_mfcc(self, mfcc_features):
        """Preprocess MFCC features for model input."""
        try:
            # Convert input to numpy array if it isn't already
            if not isinstance(mfcc_features, np.ndarray):
                mfcc_features = np.array(mfcc_features)
            
            # Ensure it's 2D (time_frames, features)
            if mfcc_features.ndim == 1:
                # If 1D, assume it's a single time frame
                mfcc_features = mfcc_features.reshape(1, -1)
            
            # Convert to tensor
            mfcc = torch.tensor(mfcc_features, dtype=torch.float32)
            
            # Normalize if statistics are available
            if self.mfcc_mean is not None and self.mfcc_std is not None:
                mfcc = (mfcc - self.mfcc_mean) / (self.mfcc_std + 1e-8)
            
            # Pad or truncate to max length
            if mfcc.shape[0] < MAX_AUDIO_LENGTH:
                pad = torch.zeros((MAX_AUDIO_LENGTH - mfcc.shape[0], mfcc.shape[1]))
                mfcc = torch.cat([mfcc, pad], dim=0)
            else:
                mfcc = mfcc[:MAX_AUDIO_LENGTH]
            
            # Compute actual sequence length
            seq_len = min(mfcc_features.shape[0], MAX_AUDIO_LENGTH)
            
            return mfcc.unsqueeze(0), torch.tensor([seq_len], dtype=torch.long)
            
        except Exception as e:
            print(f"Error preprocessing MFCC: {e}")
            # Return dummy tensors
            mfcc = torch.zeros(1, MAX_AUDIO_LENGTH, N_MFCC)
            seq_len = torch.tensor([MAX_AUDIO_LENGTH], dtype=torch.long)
            return mfcc, seq_len
    
    def process_uploaded_file(self, file_path):
        """
        Process a complete uploaded audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with transcription and features
        """
        try:
            # Convert to standard format if needed
            processed_path = self._convert_to_wav(file_path)
            
            # Extract transcription
            transcription = self.speech_to_text(processed_path)
            
            # Extract MFCC features
            mfcc_features = self.extract_mfcc_features(processed_path)
            
            # Clean up temporary file if created
            if processed_path != file_path and os.path.exists(processed_path):
                os.unlink(processed_path)
            
            return {
                'transcription': transcription,
                'mfcc_features': mfcc_features,
                'success': True
            }
            
        except Exception as e:
            print(f"Error processing uploaded file: {e}")
            return {
                'transcription': "Unable to transcribe audio",
                'mfcc_features': np.random.randn(100, N_MFCC),
                'success': False,
                'error': str(e)
            }
    
    def _convert_to_wav(self, input_path):
        """Convert audio file to WAV format if needed."""
        try:
            # If already WAV, return as-is
            if input_path.lower().endswith('.wav'):
                return input_path
            
            # Use pydub to convert
            audio = AudioSegment.from_file(input_path)
            
            # Convert to mono and standard sample rate
            audio = audio.set_channels(1).set_frame_rate(SR)
            
            # Save as temporary WAV file
            wav_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
            audio.export(wav_path, format="wav")
            
            return wav_path
            
        except Exception as e:
            print(f"Audio conversion failed: {e}")
            # Return original path and hope for the best
            return input_path


# Global audio processor instance
audio_processor = AudioProcessor()