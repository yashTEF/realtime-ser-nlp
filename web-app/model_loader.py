"""
Model loading utilities for the emotion recognition system.
Handles loading trained models, vocabularies, and preprocessing components.
"""

import torch
from pathlib import Path
from models import create_model
from audio_processing import audio_processor
from text_processing import text_processor
from prediction import emotion_predictor
from config import (
    MODELS_DIR, DEVICE, EMBEDDING_DIM, N_MFCC, EMOTION_CLASSES
)


class ModelLoader:
    """Handles loading and initialization of all model components."""
    
    def __init__(self):
        self.model_loaded = False
    
    def load_model_and_preprocessing(self):
        """
        Load the trained model and preprocessing components.
        
        Returns:
            Boolean indicating successful loading
        """
        try:
            print("Loading multimodal emotion classifier...")
            model_path = MODELS_DIR / "multimodal_emotion_classifier.pt"
            
            if model_path.exists():
                return self._load_trained_model(model_path)
            else:
                print(f"Model file not found at {model_path}")
                print("ERROR: Real trained model is required for this application.")
                print("Please ensure the trained model from nlp-ser-v4.ipynb is available.")
                return False
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("ERROR: Failed to load the real trained model.")
            return False
    
    def _load_trained_model(self, model_path):
        """Load a trained model from checkpoint."""
        try:
            # Load the saved model
            checkpoint = torch.load(model_path, map_location=DEVICE)
            print(f"Loaded checkpoint from {model_path}")
            
            # Handle different checkpoint formats
            model_state_dict, vocab_size, embedding_matrix, word2idx, mfcc_mean, mfcc_std = \
                self._parse_checkpoint(checkpoint)
            
            # Create default components if not available
            if embedding_matrix is None or word2idx is None:
                print("Creating default vocabulary and embeddings...")
                word2idx = text_processor.create_default_vocabulary(vocab_size)
                embedding_matrix = self._create_embedding_matrix(len(word2idx))
            
            # Set components
            text_processor.set_vocabulary(word2idx)
            audio_processor.set_mfcc_stats(mfcc_mean, mfcc_std)
            
            # Initialize and load model
            model = create_model(len(word2idx), embedding_matrix, len(EMOTION_CLASSES))
            model.to(DEVICE)
            model.load_state_dict(model_state_dict)
            emotion_predictor.set_model(model)
            
            print("Model loaded and initialized successfully")
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading trained model: {e}")
            print("ERROR: Failed to load the real trained model.")
            return False
    
    def _parse_checkpoint(self, checkpoint):
        """Parse different checkpoint formats."""
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # New format with complete checkpoint
                return (
                    checkpoint['model_state_dict'],
                    checkpoint.get('vocab_size', 1882),
                    checkpoint.get('embedding_matrix', None),
                    checkpoint.get('word2idx', None),
                    checkpoint.get('mfcc_mean', torch.zeros(N_MFCC)),
                    checkpoint.get('mfcc_std', torch.ones(N_MFCC))
                )
            else:
                # Legacy format - model state dict directly
                return (
                    checkpoint,
                    1882,  # Default vocab size
                    None, None,  # No embedding matrix or vocab
                    torch.zeros(N_MFCC),
                    torch.ones(N_MFCC)
                )
        else:
            # Very old format
            return (
                checkpoint,
                10000,
                None, None,
                torch.zeros(N_MFCC),
                torch.ones(N_MFCC)
            )
    
    def _create_embedding_matrix(self, vocab_size):
        """Create a random embedding matrix."""
        embedding_matrix = torch.randn(vocab_size, EMBEDDING_DIM)
        # Zero out PAD token embedding
        embedding_matrix[0] = torch.zeros(EMBEDDING_DIM)
        return embedding_matrix
    
    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            'model_loaded': self.model_loaded,
            'device': str(DEVICE),
            'emotion_classes': EMOTION_CLASSES,
            'vocab_info': text_processor.get_vocabulary_info() if self.model_loaded else None
        }
    
    def ensure_models_loaded(self):
        """Ensure all models are loaded, load if necessary."""
        if not self.model_loaded:
            success = self.load_model_and_preprocessing()
            if not success:
                raise RuntimeError("Failed to load models")
        
        # Load Whisper model
        audio_processor.load_whisper_model()


# Global model loader instance
model_loader = ModelLoader()