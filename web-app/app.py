"""
Modular Flask application for Speech Emotion Recognition.
Main application file that initializes and runs the web server.
"""

from flask import Flask
from routes import main_routes
from model_loader import model_loader
from config import FLASK_CONFIG, EMOTION_CLASSES
import os
import ssl

def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Configured Flask app instance
    """
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(main_routes)
    
    return app

def create_ssl_context():
    """Create SSL context for HTTPS."""
    try:
        # Try to create self-signed certificates if they don't exist
        cert_dir = 'certs'
        os.makedirs(cert_dir, exist_ok=True)
        cert_file = os.path.join(cert_dir,'cert.pem')
        key_file = os.path.join(cert_dir,'key.pem')
        
        if not os.path.exists(cert_file) or not os.path.exists(key_file):
            print("Creating self-signed SSL certificates...")
            os.system(f'openssl req -x509 -newkey rsa:4096 -nodes -out {cert_file} -keyout {key_file} -days 365 -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"')
        
        if os.path.exists(cert_file) and os.path.exists(key_file):
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert_file, key_file)
            return context
        else:
            print("Could not create SSL certificates. Using adhoc SSL.")
            return 'adhoc'
    except Exception as e:
        print(f"SSL setup error: {e}. Using adhoc SSL.")
        return 'adhoc'


def initialize_models():
    """Initialize all models and preprocessing components."""
    print("Initializing Emotion Recognition System...")
    print("=" * 50)
    
    try:
        success = model_loader.load_model_and_preprocessing()
        if success:
            model_info = model_loader.get_model_info()
            print(f"✓ Models loaded successfully")
            print(f"✓ Device: {model_info['device']}")
            print(f"✓ Emotion classes: {', '.join(EMOTION_CLASSES)}")
            
            if model_info['vocab_info']:
                vocab_info = model_info['vocab_info']
                if 'vocab_size' in vocab_info:
                    print(f"✓ Vocabulary size: {vocab_info['vocab_size']}")
        else:
            print("✗ Failed to load models - Real trained model required")
            return False
            
    except Exception as e:
        print(f"✗ Error initializing models: {e}")
        return False
    
    print("=" * 50)
    return True


def print_startup_info():
    """Print application startup information."""
    print("\n" + "=" * 60)
    print("SPEECH EMOTION RECOGNITION WEB APPLICATION")
    print("=" * 60)
    print(f"🌐 Server: http://{FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}")
    print(f"🎯 Emotion Classes: {', '.join(EMOTION_CLASSES)}")
    print("\n📋 Available Features:")
    print("  • Audio file upload and processing")
    print("  • Web-based audio recording")
    print("  • Speech-to-text conversion (Whisper)")
    print("  • MFCC audio feature extraction")
    print("  • Multimodal emotion classification")
    print("  • Simple session tracking")
    print("\n🔗 API Endpoints:")
    print("  • GET  /              - Main interface")
    print("  • POST /predict       - Upload audio for prediction")
    print("  • GET  /test          - Test with realistic examples")
    print("  • GET  /health        - Health check and system info")
    print("  • GET  /model_info    - Get model information")
    print("\n💡 Usage Tips:")
    print("  • Supported audio formats: WAV, MP3, M4A")
    print("  • For best results, use clear speech")
    print("  • Requires trained model from nlp-ser-v4.ipynb")
    print("  • File-based processing optimized for accuracy")
    print("=" * 60)


def main():
    """Main application entry point."""
    try:
        # Initialize models
        if not initialize_models():
            print("Failed to initialize models. Exiting.")
            return
        
        # Create Flask app
        app = create_app()
        
        # Print startup information
        print_startup_info()
        
        ssl_context = create_ssl_context()

        # Start the server
        print("\n🚀 Starting server...")
        print("Press Ctrl+C to stop the server")
        print("-" * 60)
        
        app.run(
            debug=FLASK_CONFIG['debug'],
            host=FLASK_CONFIG['host'],
            port=FLASK_CONFIG['port'],
            ssl_context=ssl_context,  # Use SSL context for HTTPS
            # use_reloader=False  # Disable reloader to prevent double initialization
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
    finally:
        print("👋 Goodbye!")


if __name__ == '__main__':
    main()