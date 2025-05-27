"""
Configuration file for the Emotion Recognition Web App.
Contains all constants, settings, and configuration parameters.
"""

import torch
from pathlib import Path

# Emotion classes
EMOTION_CLASSES = ["angry", "happy", "sad", "neutral"]

# Model configuration - matches training configuration from notebook
MAX_LENGTH = 128
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3

# Audio processing configuration
N_MFCC = 13
MAX_AUDIO_LENGTH = 300
SR = 16000  # Sample rate

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR.parent / "models"
TEMPLATES_DIR = BASE_DIR / "templates"

# Flask configuration
FLASK_CONFIG = {
    'debug': True,
    'host': '0.0.0.0',
    'port': 5000
}

# Whisper model configuration
WHISPER_MODEL_SIZE = "tiny"