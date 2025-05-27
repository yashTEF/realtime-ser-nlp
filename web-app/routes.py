"""
Flask route handlers for the emotion recognition web application.
Handles all HTTP endpoints including prediction, testing, and real-time processing.
"""

import tempfile
import os
import numpy as np
from flask import Blueprint, render_template, request, jsonify
from audio_processing import audio_processor
from prediction import emotion_predictor
from model_loader import model_loader
from config import EMOTION_CLASSES


# Create blueprint for routes
main_routes = Blueprint('main', __name__)


@main_routes.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@main_routes.route('/predict', methods=['POST'])
def predict():
    """Handle audio file upload and prediction."""
    try:
        # Ensure models are loaded
        model_loader.ensure_models_loaded()
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Save the uploaded audio file temporarily
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, 'uploaded_audio.wav')
        audio_file.save(audio_path)
        
        try:
            # Step 1: Convert speech to text
            print("Converting speech to text...")
            transcribed_text = audio_processor.speech_to_text(audio_path)

            # Step 2: Extract audio features (MFCC)
            print("Extracting audio features...")
            audio_features = audio_processor.extract_mfcc_features(audio_path)

            # Step 3: Predict emotion using multimodal model
            print("Predicting emotion...")
            emotion_result = emotion_predictor.predict_emotion_with_model(transcribed_text, audio_features)
            
            print(f"Emotion prediction result: {emotion_result}")
            
            # Create audio features summary - Fixed: Handle numpy array properly
            try:
                if isinstance(audio_features, np.ndarray):
                    features_for_summary = audio_features.flatten()
                else:
                    features_for_summary = audio_features
                audio_features_summary = emotion_predictor.create_audio_features_summary(features_for_summary)
            except Exception as summary_error:
                print(f"Warning: Could not create audio features summary: {summary_error}")
                audio_features_summary = {
                    'mean': 0.0,
                    'std': 1.0,
                    'shape': str(audio_features.shape) if hasattr(audio_features, 'shape') else 'unknown'
                }
            
            # Prepare response
            response = {
                'success': True,
                'transcribed_text': transcribed_text,
                'audio_features_summary': audio_features_summary,
                'emotion_prediction': emotion_result,
                'model_info': {
                    'device': str(model_loader.get_model_info()['device'])
                }
            }
            
            return jsonify(response)
            
        finally:
            # Clean up temporary file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            os.rmdir(temp_dir)
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@main_routes.route('/test', methods=['GET'])
def test_prediction():
    """Test endpoint with predefined realistic test cases."""
    try:
        # Ensure models are loaded
        model_loader.ensure_models_loaded()
        
        # Use realistic test data that demonstrates the model capabilities
        test_cases = [
            {
                'text': "I am so excited about this opportunity!",
                'emotion': 'happy',
                'description': 'Positive excited statement'
            },
            {
                'text': "This is completely unacceptable and frustrating.",
                'emotion': 'angry', 
                'description': 'Negative angry statement'
            },
            {
                'text': "I feel so disappointed and let down by this news.",
                'emotion': 'sad',
                'description': 'Sad disappointed statement'
            },
            {
                'text': "Please let me know when the meeting is scheduled.",
                'emotion': 'neutral',
                'description': 'Neutral informational request'
            }
        ]
        
        # Select a test case (rotate through them based on current time)
        import time
        case_index = int(time.time()) % len(test_cases)
        selected_case = test_cases[case_index]
        
        # Create realistic MFCC features (based on typical emotional speech patterns)
        # Need to import numpy for array creation
        import numpy as np
        
        emotion_mfcc_patterns = {
            'happy': [1.2, 0.8, 0.5, -0.3, 0.4, 0.2, -0.1, 0.3, -0.2, 0.1, 0.0, -0.1, 0.2],
            'angry': [2.1, 1.5, 0.9, 0.7, 0.8, 0.6, 0.4, 0.5, 0.3, 0.2, 0.1, 0.0, 0.3],
            'sad': [-0.8, -0.5, -0.3, -0.6, -0.2, -0.4, -0.1, -0.3, -0.2, -0.1, 0.0, 0.1, -0.2],
            'neutral': [0.1, 0.0, -0.1, 0.2, 0.0, 0.1, -0.1, 0.0, 0.1, 0.0, -0.1, 0.0, 0.1]
        }
        
        test_features_list = emotion_mfcc_patterns.get(selected_case['emotion'], 
                                                       emotion_mfcc_patterns['neutral'])
        
        # Convert to proper 2D numpy array format (time_frames, features)
        # Create multiple time frames for realistic test
        test_features = np.array([test_features_list] * 50)  # 50 time frames
        
        # Get prediction
        emotion_result = emotion_predictor.predict_emotion_with_model(
            selected_case['text'], 
            test_features
        )
        
        # Create features summary
        audio_features_summary = emotion_predictor.create_audio_features_summary(test_features)
        
        response = {
            'success': True,
            'test_mode': True,
            'test_case': {
                'description': selected_case['description'],
                'expected_emotion': selected_case['emotion'],
                'case_number': case_index + 1,
                'total_cases': len(test_cases)
            },
            'transcribed_text': selected_case['text'],
            'audio_features_summary': audio_features_summary,
            'emotion_prediction': emotion_result,
            'model_info': model_loader.get_model_info()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Test failed: {str(e)}'}), 500


@main_routes.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        model_info = model_loader.get_model_info()
        
        return jsonify({
            'status': 'healthy',
            'emotion_classes': EMOTION_CLASSES,
            'model_info': model_info
        })
    
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


# Error handlers
@main_routes.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@main_routes.errorhandler(405)
def method_not_allowed_error(error):
    """Handle 405 errors."""
    return jsonify({'error': 'Method not allowed'}), 405


@main_routes.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500