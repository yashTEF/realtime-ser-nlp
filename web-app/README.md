# Speech Emotion Recognition Web App

A modular Flask-based web application for speech emotion recognition using multimodal deep learning trained models.

## 🚀 Quick Start

```bash
cd web-app
python app.py
```

## 📁 Project Structure

```
web-app/
├── app.py                 # Main application entry point
├── audio_processing.py    # Audio processing utilities (Whisper + MFCC)
├── config.py              # Configuration constants and settings
├── model_loader.py        # Model loading and initialization
├── models.py              # Neural network model definitions
├── prediction.py          # Emotion prediction logic
├── requirements.txt       # Python dependencies
├── routes.py              # Flask route handlers
├── session_manager.py     # Simple session tracking
├── text_processing.py     # Text preprocessing utilities
└── templates/             # HTML templates
```

## 🔧 Module Overview

### Core Components
- **`app.py`** - Main Flask application with startup logic
- **`config.py`** - Centralized configuration and constants
- **`models.py`** - Multimodal LSTM emotion classifier architecture

### Processing Pipeline
- **`audio_processing.py`** - Speech-to-text (Whisper) + MFCC feature extraction
- **`text_processing.py`** - Text tokenization and preprocessing
- **`prediction.py`** - Emotion inference and result formatting

### Infrastructure
- **`model_loader.py`** - Model checkpoint loading and initialization
- **`session_manager.py`** - Simple file upload session tracking
- **`routes.py`** - HTTP endpoints and API handlers

## 📊 Features

- **Audio Recording**: Web-based audio capture
- **File Upload**: Support for WAV, MP3, M4A formats
- **Speech-to-Text**: Whisper-powered transcription
- **Feature Extraction**: MFCC audio features
- **Emotion Classification**: 4 classes (angry, happy, sad, neutral)
- **Trained Model Integration**: Uses real models from nlp-ser-v4.ipynb

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/predict` | POST | Upload audio for emotion prediction |
| `/test` | GET | Test with realistic example cases |
| `/health` | GET | System health and status |
| `/model_info` | GET | Model configuration details |

## 🛠️ Development

### Adding New Features
1. **Configuration**: Update `config.py`
2. **Models**: Modify `models.py`
3. **Processing**: Edit relevant processing modules
4. **Routes**: Add endpoints in `routes.py`

### Testing Components
```python
# Test audio processing
from audio_processing import audio_processor
text = audio_processor.speech_to_text("audio.wav")

# Test prediction
from prediction import emotion_predictor
result = emotion_predictor.predict_emotion_with_model(text, features)
```

## 📋 Requirements

- Python 3.8+
- PyTorch
- Flask
- librosa
- whisper-openai
- nltk
- numpy

Install with:
```bash
pip install -r requirements.txt
```

## 🎯 Model Architecture

- **Text Branch**: Embedding layer + Bidirectional LSTM
- **Audio Branch**: Layer normalization + Bidirectional LSTM
- **Fusion**: Concatenated features + Fully connected layers
- **Output**: 4-class emotion classification

## 📈 Performance

- **File Processing**: Optimized for accuracy over speed
- **Audio Support**: WAV, MP3, M4A formats
- **Model Quality**: Depends on trained model from nlp-ser-v4.ipynb
- **Session Tracking**: Simple upload session management

## 🔧 Configuration

Key settings in `config.py`:
- Model hyperparameters
- Audio processing parameters
- Server configuration
- File paths and constants

## 🚀 Deployment

1. Install dependencies: `pip install -r requirements.txt`
2. Place trained model in `../models/multimodal_emotion_classifier.pt`
3. Run server: `python app.py`
4. Access at: `http://localhost:5000`

## 🔍 Monitoring

- Health check: `/health` endpoint
- Model info: `/model_info` endpoint
- Console logging for debugging

The modular architecture ensures maintainable, testable, and scalable code while providing powerful emotion recognition capabilities using real trained models.
