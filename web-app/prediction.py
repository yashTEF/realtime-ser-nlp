"""
Prediction utilities for emotion recognition.
Handles model inference, real-time predictions, and result processing.
"""

import time
import numpy as np
import torch
from config import EMOTION_CLASSES, DEVICE
from audio_processing import audio_processor
from text_processing import text_processor


class EmotionPredictor:
    """Handles emotion prediction using the trained multimodal model."""
    
    def __init__(self):
        self.model = None
    
    def set_model(self, model):
        """Set the trained model for predictions."""
        self.model = model
        if self.model:
            self.model.eval()
    
    def predict_emotion_with_model(self, text, mfcc_features):
        """
        Predict emotion using the trained multimodal model.
        
        Args:
            text: Input text string
            mfcc_features: MFCC audio features
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Preprocess inputs
            input_ids = text_processor.preprocess_text(text)
            mfcc_tensor, seq_lens = audio_processor.preprocess_mfcc(mfcc_features)
            
            # Move to device
            input_ids = input_ids.to(DEVICE)
            mfcc_tensor = mfcc_tensor.to(DEVICE)
            seq_lens = seq_lens.to(DEVICE)
            
            print("Running model prediction...")
            start_time = time.time()
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids, mfcc_tensor, seq_lens)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # Create prediction result
            predictions = {}
            for i, emotion in enumerate(EMOTION_CLASSES):
                predictions[emotion] = round(float(probabilities[i]), 3)
            
            predicted_emotion = EMOTION_CLASSES[np.argmax(probabilities)]
            confidence = round(float(np.max(probabilities)), 3)

            end_time = time.time()
            print(f"Prediction completed in {end_time - start_time:.2f} seconds")
            
            return {
                'predicted_emotion': predicted_emotion,
                'probabilities': predictions,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return self._fallback_prediction()
    
    def _fallback_prediction(self):
        """
        Generate a fallback prediction when model fails.
        Uses neutral emotion with low confidence as a safe default.
        """
        # Return neutral emotion as safe fallback
        predictions = {
            'angry': 0.1,
            'happy': 0.1, 
            'sad': 0.1,
            'neutral': 0.7
        }
        
        return {
            'predicted_emotion': 'neutral',
            'probabilities': predictions,
            'confidence': 0.7,
            'fallback_reason': 'model_unavailable'
        }
    
    def create_audio_features_summary(self, audio_features):
        """
        Create a summary of audio features for response.
        
        Args:
            audio_features: Raw audio features (numpy array or list)
            
        Returns:
            Dictionary with feature statistics
        """
        try:
            # Convert to numpy array if needed
            if not isinstance(audio_features, np.ndarray):
                audio_features = np.array(audio_features)
            
            # Handle empty arrays
            if audio_features.size == 0:
                return {
                    'mfcc_count': 0,
                    'mean': 0.0,
                    'std': 0.0,
                    'shape': 'empty'
                }
            
            # Flatten if multidimensional
            if audio_features.ndim > 1:
                flat_features = audio_features.flatten()
            else:
                flat_features = audio_features
            
            # Calculate statistics
            return {
                'mfcc_count': int(flat_features.size),
                'mean': round(float(np.mean(flat_features)), 3),
                'std': round(float(np.std(flat_features)), 3),
                'min': round(float(np.min(flat_features)), 3),
                'max': round(float(np.max(flat_features)), 3),
                'shape': str(audio_features.shape)
            }
            
        except Exception as e:
            print(f"Error creating audio features summary: {e}")
            return {
                'mfcc_count': 0,
                'mean': 0.0,
                'std': 1.0,
                'min': -1.0,
                'max': 1.0,
                'shape': 'error',
                'error': str(e)
            }


# Global emotion predictor instance
emotion_predictor = EmotionPredictor()