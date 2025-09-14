"""
Configuration module for AI Emblematic application.
Handles all configuration settings, API keys, and environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class."""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    UPLOAD_FOLDER = 'uploads'
    TEXT_CACHE_FOLDER = 'text_cache'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY = None
    AZURE_OPENAI_ENDPOINT = None
    AZURE_OPENAI_API_VERSION = None
    
    def __init__(self):
        """Initialize configuration by loading API keys."""
        self._load_api_keys()
        self._validate_config()
        self._ensure_directories()
    
    def _load_api_keys(self):
        """Load API keys from keys.py or environment variables."""
        try:
            from keys import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
            self.AZURE_OPENAI_API_KEY = AZURE_OPENAI_API_KEY
            self.AZURE_OPENAI_ENDPOINT = AZURE_OPENAI_ENDPOINT
            self.AZURE_OPENAI_API_VERSION = AZURE_OPENAI_API_VERSION
        except ImportError:
            # Fallback to environment variables if keys.py doesn't exist
            self.AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
            self.AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
            self.AZURE_OPENAI_API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
    
    def _validate_config(self):
        """Validate configuration and show warnings if needed."""
        if not self.AZURE_OPENAI_API_KEY or not self.AZURE_OPENAI_ENDPOINT:
            print("Warning: Azure OpenAI credentials not found in keys.py or environment variables")
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        directories = [self.UPLOAD_FOLDER, self.TEXT_CACHE_FOLDER]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    def get_openai_config(self):
        """Get OpenAI configuration as a dictionary."""
        return {
            'api_key': self.AZURE_OPENAI_API_KEY,
            'azure_endpoint': self.AZURE_OPENAI_ENDPOINT,
            'api_version': self.AZURE_OPENAI_API_VERSION
        }
    
    def is_openai_configured(self):
        """Check if OpenAI is properly configured."""
        return bool(self.AZURE_OPENAI_API_KEY and self.AZURE_OPENAI_ENDPOINT)

# Global configuration instance
config = Config()
