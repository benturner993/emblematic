"""
AI Emblematic - Main Flask Application
A modular Flask web application for intelligent document processing with AI-powered analysis and FHIR conversion.
"""

from flask import Flask
import logging

# Import our modular components
from config import config
from ai_services import AIService
from fhir_services import FHIRService
from routes import Routes

def create_app():
    """Application factory pattern for creating Flask app."""
    
    # Create Flask app
    app = Flask(__name__)
    
    # Configure Flask app with our config
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
    app.config['TEXT_CACHE_FOLDER'] = config.TEXT_CACHE_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting AI Emblematic application")
    
    # Initialize services
    ai_service = AIService(config)
    fhir_service = FHIRService(config)
    
    # Register routes
    Routes(app, config, ai_service, fhir_service)
    
    # Log configuration status
    if config.is_openai_configured():
        logging.info("✅ OpenAI configuration loaded successfully")
    else:
        logging.warning("⚠️  OpenAI not configured - AI features will not work")
    
    logging.info(f"📁 Upload folder: {config.UPLOAD_FOLDER}")
    logging.info(f"📁 Cache folder: {config.TEXT_CACHE_FOLDER}")
    
    return app

def main():
    """Main entry point for the application."""
    app = create_app()
    
    # Run the Flask development server
    # Note: In production, use a proper WSGI server like Gunicorn
    try:
        logging.info("🚀 Starting Flask development server...")
        app.run(
            debug=True, 
            use_reloader=False,  # Disable reloader to avoid watchdog issues
            host='0.0.0.0', 
            port=5001
        )
    except Exception as e:
        logging.error(f"❌ Failed to start server: {e}")
        raise

if __name__ == '__main__':
    main()
