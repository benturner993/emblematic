"""
Utility functions for AI Emblematic application.
Handles file operations, text extraction, hashing, and general utilities.
"""

import os
import hashlib
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import PyPDF2
import logging

def get_file_hash(file_path):
    """Generate SHA-256 hash of a file for caching purposes."""
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256()
            while chunk := f.read(8192):
                file_hash.update(chunk)
        return file_hash.hexdigest()
    except Exception as e:
        logging.error(f"Error generating file hash: {e}")
        return None

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    ALLOWED_EXTENSIONS = {'pdf'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file using PyPDF2."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfFileReader(file)
            for page_num in range(pdf_reader.numPages):
                page = pdf_reader.getPage(page_num)
                text += page.extractText() + "\n"
    except Exception as e:
        logging.warning(f"Error extracting text from PDF (trying alternative method): {e}")
        # Try alternative method for newer PyPDF2 versions
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e2:
            logging.error(f"Error with alternative extraction method: {e2}")
            raise Exception(f"Could not extract text from PDF: {e2}")
    
    return text.strip()

def save_session_data(file_hash, filename, text, cache_folder):
    """Save session data to JSON file."""
    session_data = {
        'filename': filename,
        'text': text,
        'timestamp': datetime.now().isoformat()
    }
    
    session_file = os.path.join(cache_folder, f"{file_hash}_session.json")
    try:
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Error saving session data: {e}")
        return False

def load_session_data(file_hash, cache_folder):
    """Load session data from JSON file."""
    session_file = os.path.join(cache_folder, f"{file_hash}_session.json")
    
    if not os.path.exists(session_file):
        return None
    
    try:
        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading session data: {e}")
        return None

def save_text_to_cache(file_hash, text, cache_folder):
    """Save extracted text to cache file."""
    cache_file = os.path.join(cache_folder, f"{file_hash}.txt")
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        logging.error(f"Error saving text to cache: {e}")
        return False

def load_text_from_cache(file_hash, cache_folder):
    """Load extracted text from cache file."""
    cache_file = os.path.join(cache_folder, f"{file_hash}.txt")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error loading text from cache: {e}")
        return None

def write_judge_output_to_file(file_hash, judge_output, attempt_number, cache_folder):
    """Write judge output directly to a text file."""
    try:
        judge_file = os.path.join(cache_folder, f"{file_hash}_judge_logs.txt")
        timestamp = datetime.now().isoformat()
        
        with open(judge_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"JUDGE OUTPUT - Attempt {attempt_number}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Raw Judge Response:\n{judge_output}\n")
            f.write(f"{'='*80}\n\n")
        
        logging.info(f"Judge output written to {judge_file}")
        return True
    except Exception as e:
        logging.error(f"Error writing judge output to file: {e}")
        return False

def load_judge_logs(file_hash, cache_folder):
    """Load judge logs from text file."""
    judge_logs_file = os.path.join(cache_folder, f"{file_hash}_judge_logs.txt")
    
    if not os.path.exists(judge_logs_file):
        return "No judge logs found. Please run FHIR conversion first."
    
    try:
        with open(judge_logs_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading judge logs file: {e}")
        return f"Error reading judge logs: {str(e)}"

def clean_filename(filename):
    """Clean and secure filename for storage."""
    return secure_filename(filename)

def truncate_text(text, max_length=10000):
    """Truncate text to maximum length for API calls."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def validate_file_size(file, max_size_mb=16):
    """Validate uploaded file size."""
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # Get file size
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning
    
    return size <= max_size_bytes, size
