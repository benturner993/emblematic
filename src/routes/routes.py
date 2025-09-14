"""
Routes module for AI Emblematic application.
Handles all Flask route definitions and HTTP request processing.
"""

import os
import json
import logging
from flask import render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from src.utils import (
    get_file_hash, allowed_file, extract_text_from_pdf, 
    save_session_data, load_session_data, save_text_to_cache, 
    load_text_from_cache, load_judge_logs, validate_file_size
)

class Routes:
    """Class to handle all Flask routes."""
    
    def __init__(self, app, config, ai_service, fhir_service, patient_email_service=None):
        """Initialize routes with Flask app and services."""
        self.app = app
        self.config = config
        self.ai_service = ai_service
        self.fhir_service = fhir_service
        self.patient_email_service = patient_email_service
        self._register_routes()
    
    def _register_routes(self):
        """Register all routes with the Flask app."""
        
        @self.app.route('/')
        def index():
            """Home page"""
            return render_template('index.html')
        
        @self.app.route('/about')
        def about():
            """About page"""
            return render_template('about.html')
        
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            """Handle file upload and text extraction"""
            try:
                if 'file' not in request.files:
                    flash('No file selected')
                    return redirect(url_for('index'))
                
                file = request.files['file']
                if file.filename == '':
                    flash('No file selected')
                    return redirect(url_for('index'))
                
                # Validate file size
                is_valid_size, file_size = validate_file_size(file, max_size_mb=16)
                if not is_valid_size:
                    flash(f'File size ({file_size / (1024*1024):.1f}MB) exceeds 16MB limit')
                    return redirect(url_for('index'))
                
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(self.config.UPLOAD_FOLDER, filename)
                    
                    # Save the file
                    file.save(file_path)
                    logging.info(f"File saved to: {file_path}")
                    
                    # Generate hash for caching
                    file_hash = get_file_hash(file_path)
                    if not file_hash:
                        flash('Error processing file')
                        return redirect(url_for('index'))
                    
                    logging.info(f"File hash: {file_hash}")
                    
                    # Check if we have cached text
                    cached_text = load_text_from_cache(file_hash, self.config.TEXT_CACHE_FOLDER)
                    
                    if cached_text:
                        text = cached_text
                        flash('Using cached text extraction')
                        logging.info("Using cached text")
                    else:
                        # Extract text from PDF
                        logging.info("Extracting text from PDF...")
                        try:
                            text = extract_text_from_pdf(file_path)
                            if not text.strip():
                                flash('Could not extract text from PDF. The PDF might be image-based or corrupted.')
                                return redirect(url_for('index'))
                            
                            # Save to cache
                            if save_text_to_cache(file_hash, text, self.config.TEXT_CACHE_FOLDER):
                                flash('Text extracted and cached')
                            else:
                                flash('Text extracted (caching failed)')
                            
                            logging.info(f"Text extracted, length: {len(text)} characters")
                        except Exception as e:
                            logging.error(f"Error extracting text: {e}")
                            flash(f'Error extracting text from PDF: {str(e)}')
                            return redirect(url_for('index'))
                    
                    # Save session data
                    if not save_session_data(file_hash, filename, text, self.config.TEXT_CACHE_FOLDER):
                        flash('Error saving session data')
                        return redirect(url_for('index'))
                    
                    return redirect(url_for('text_display', file_hash=file_hash))
                else:
                    flash('Invalid file type. Please upload a PDF file.')
                    return redirect(url_for('index'))
            
            except Exception as e:
                logging.error(f"Upload error: {str(e)}")
                flash(f'An error occurred during upload: {str(e)}')
                return redirect(url_for('index'))
        
        @self.app.route('/text/<file_hash>')
        def text_display(file_hash):
            """Display extracted text"""
            session_data = load_session_data(file_hash, self.config.TEXT_CACHE_FOLDER)
            
            if not session_data:
                flash('Text display not found')
                return redirect(url_for('index'))
            
            return render_template('text_display.html', data=session_data)
        
        @self.app.route('/summarise/<file_hash>', methods=['POST'])
        def summarise_document(file_hash):
            """Analyze and summarize document using AI"""
            try:
                session_data = load_session_data(file_hash, self.config.TEXT_CACHE_FOLDER)
                
                if not session_data:
                    return jsonify({"error": "Document not found"}), 404
                
                text = session_data.get('text', '')
                if not text:
                    return jsonify({"error": "No text content found"}), 400
                
                # Check if we already have analysis cached
                analysis_file = os.path.join(self.config.TEXT_CACHE_FOLDER, f"{file_hash}_analysis.json")
                
                if os.path.exists(analysis_file):
                    try:
                        with open(analysis_file, 'r', encoding='utf-8') as f:
                            cached_analysis = json.load(f)
                        logging.info("Using cached analysis")
                        return jsonify(cached_analysis)
                    except Exception as e:
                        logging.warning(f"Error loading cached analysis: {e}")
                
                # Perform AI analysis
                analysis_result = self.ai_service.analyze_document_with_ai(text)
                
                if "error" not in analysis_result:
                    # Cache the successful analysis
                    try:
                        with open(analysis_file, 'w', encoding='utf-8') as f:
                            json.dump(analysis_result, f, indent=2)
                    except Exception as e:
                        logging.warning(f"Error caching analysis: {e}")
                
                return jsonify(analysis_result)
                
            except Exception as e:
                logging.error(f"Error in summarise_document: {e}")
                return jsonify({"error": f"Internal server error: {str(e)}"}), 500
        
        @self.app.route('/convert-fhir/<file_hash>', methods=['POST'])
        def convert_to_fhir(file_hash):
            """Convert document to FHIR format"""
            try:
                session_data = load_session_data(file_hash, self.config.TEXT_CACHE_FOLDER)
                
                if not session_data:
                    return jsonify({"error": "Document not found"}), 404
                
                text = session_data.get('text', '')
                if not text:
                    return jsonify({"error": "No text content found"}), 400
                
                # Always run fresh FHIR conversion (no caching for now)
                fhir_result = self.fhir_service.convert_to_fhir_with_agent(text, max_retries=5, feedback_history=[])
                
                if "error" in fhir_result:
                    return jsonify({
                        "error": fhir_result["error"], 
                        "conversion_notes": fhir_result.get("conversion_notes", ""),
                        "validation": fhir_result.get("validation", {}),
                        "groundedness": fhir_result.get("groundedness", {}),
                        "confidence_score": fhir_result.get("confidence_score", "N/A")
                    }), 500
                
                # Return the full FHIR result data for successful conversions
                # Note: detailed_logs removed to avoid circular reference serialization issues
                return jsonify({
                    "success": True,
                    "fhir_bundle": fhir_result.get("fhir_bundle"),
                    "validation": fhir_result.get("validation", {}),
                    "groundedness": fhir_result.get("groundedness", {}),
                    "confidence_score": fhir_result.get("confidence_score", "N/A"),
                    "conversion_notes": fhir_result.get("conversion_notes", "Successfully converted to FHIR")
                })
                
            except Exception as e:
                logging.error(f"Error in convert_to_fhir: {e}")
                return jsonify({"error": f"Internal server error: {str(e)}"}), 500
        
        @self.app.route('/fhir-judge-logs/<file_hash>')
        def fhir_judge_logs(file_hash):
            """Display detailed FHIR conversion logs"""
            try:
                # Load the original text first
                session_data = load_session_data(file_hash, self.config.TEXT_CACHE_FOLDER)
                original_text = session_data.get('text', '') if session_data else ""
                
                # Load judge logs from text file
                judge_logs_text = load_judge_logs(file_hash, self.config.TEXT_CACHE_FOLDER)
                
                # Create a simple structure for the template
                fhir_data = {
                    "judge_logs_text": judge_logs_text,
                    "has_judge_logs": bool(judge_logs_text and judge_logs_text != "No judge logs found. Please run FHIR conversion first."),
                    "conversion_notes": "Judge logs loaded from text file"
                }
                
                return render_template('fhir_logs.html', 
                                     file_hash=file_hash,
                                     fhir_data=fhir_data,
                                     original_text=original_text)
                
            except Exception as e:
                logging.error(f"Error loading FHIR logs: {e}")
                return render_template('fhir_logs.html', 
                                     fhir_data={
                                         "judge_logs_text": f"Error loading FHIR logs: {str(e)}",
                                         "has_judge_logs": False,
                                         "conversion_notes": "Error occurred while loading logs"
                                     }, 
                                     original_text="",
                                     file_hash=file_hash)
        
        # Register error handlers
        # Patient Email Routes
        @self.app.route('/patient-email/<file_hash>')
        def patient_email_interface(file_hash):
            """Render the patient email chat interface."""
            try:
                # Load the document text for context
                text = load_text_from_cache(file_hash, self.config.TEXT_CACHE_FOLDER)
                if not text:
                    flash('Document not found. Please upload a document first.')
                    return redirect(url_for('index'))
                
                return render_template('patient_email.html', 
                                     file_hash=file_hash,
                                     text_preview=text[:500] + "..." if len(text) > 500 else text)
            except Exception as e:
                logging.error(f"Error loading patient email interface: {e}")
                flash('Error loading patient email interface.')
                return redirect(url_for('index'))
        
        @self.app.route('/api/patient-email/start-chat', methods=['POST'])
        def start_patient_chat():
            """Start a new patient email chat session."""
            try:
                if not self.patient_email_service:
                    return jsonify({"error": "Patient email service not available"}), 500
                
                session_data = self.patient_email_service.start_chat_session()
                return jsonify(session_data)
                
            except Exception as e:
                logging.error(f"Error starting patient chat: {e}")
                return jsonify({"error": "Failed to start chat session"}), 500
        
        @self.app.route('/api/patient-email/chat', methods=['POST'])
        def process_patient_chat():
            """Process patient chat responses."""
            try:
                if not self.patient_email_service:
                    return jsonify({"error": "Patient email service not available"}), 500
                
                data = request.get_json()
                session_data = data.get('session_data', {})
                user_response = data.get('user_response', '')
                
                if not user_response.strip():
                    return jsonify({"error": "Please provide a response"}), 400
                
                updated_session = self.patient_email_service.process_chat_response(
                    session_data, user_response
                )
                
                return jsonify(updated_session)
                
            except Exception as e:
                logging.error(f"Error processing patient chat: {e}")
                return jsonify({"error": "Failed to process chat response"}), 500
        
        @self.app.route('/api/patient-email/generate-letter', methods=['POST'])
        def generate_patient_letter():
            """Generate patient letter from collected information."""
            try:
                if not self.patient_email_service:
                    return jsonify({"error": "Patient email service not available"}), 500
                
                data = request.get_json()
                patient_info = data.get('patient_info', {})
                file_hash = data.get('file_hash')
                
                # Get document context if available
                document_context = None
                if file_hash:
                    try:
                        document_context = load_text_from_cache(file_hash, self.config.TEXT_CACHE_FOLDER)
                    except Exception as e:
                        logging.warning(f"Could not load document context: {e}")
                
                result = self.patient_email_service.generate_patient_letter(
                    patient_info, document_context
                )
                
                return jsonify(result)
                
            except Exception as e:
                logging.error(f"Error generating patient letter: {e}")
                return jsonify({"error": "Failed to generate patient letter"}), 500

        @self.app.errorhandler(404)
        def not_found_error(error):
            """Handle 404 errors"""
            return render_template('404.html'), 404
        
        @self.app.errorhandler(413)
        def too_large(error):
            """Handle file too large errors"""
            flash('File too large. Please upload a file smaller than 16MB.')
            return redirect(url_for('index'))
        
        @self.app.errorhandler(500)
        def internal_error(error):
            """Handle 500 errors"""
            logging.error(f"Internal server error: {error}")
            return render_template('500.html'), 500
