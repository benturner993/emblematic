from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import hashlib
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import PyPDF2
import openai
from dotenv import load_dotenv
from models import DocumentAnalysis
import logging

# Import API keys from separate file
try:
    from keys import OPENAI_API_KEY
except ImportError:
    # Fallback to environment variable if keys.py doesn't exist
    OPENAI_API_KEY = None

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEXT_CACHE_FOLDER'] = 'text_cache'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# OpenAI Configuration
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    # Fallback to environment variable
    openai.api_key = os.environ.get('OPENAI_API_KEY')

if not openai.api_key:
    print("Warning: OPENAI_API_KEY not found in keys.py or environment variables")

# Setup logging
logging.basicConfig(level=logging.INFO)

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEXT_CACHE_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

def get_file_hash(file_path):
    """Generate hash for file to use as cache key"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def extract_text_from_pdf(file_path):
    """Extract text from PDF using PyPDF2"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfFileReader(file)
            for page_num in range(pdf_reader.numPages):
                page = pdf_reader.getPage(page_num)
                text += page.extractText() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        # Try alternative method for newer PyPDF2 versions
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e2:
            print(f"Error with alternative extraction method: {e2}")
    return text

def is_medical_document(text):
    """Check if the document is medical using a simple LLM call"""
    if not openai.api_key:
        return False
    
    # Truncate text for the medical check
    check_text = text[:3000] if len(text) > 3000 else text
    
    prompt = f"""
    Analyze the following text and determine if it is a medical document.
    
    Text: {check_text}
    
    Respond with only "YES" if this is a medical document (contains medical terms, patient information, diagnoses, treatments, medications, etc.) or "NO" if it is not medical.
    """
    
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a medical document classifier. Respond with only 'YES' or 'NO' to indicate if the text is medical."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip().upper()
        return result == "YES"
        
    except Exception as e:
        logging.error(f"Error in medical document detection: {e}")
        return False

def analyze_document_with_ai(text, max_retries=3):
    """Analyze document using OpenAI with conditional processing based on document type"""
    if not openai.api_key:
        return {"error": "OpenAI API key not configured"}
    
    # First, check if this is a medical document
    is_medical = is_medical_document(text)
    logging.info(f"Document medical classification: {is_medical}")
    
    # Truncate text if too long (OpenAI has token limits)
    if len(text) > 12000:  # Leave room for prompt and response
        text = text[:12000] + "..."
    
    if is_medical:
        # Use structured Pydantic approach for medical documents
        prompt = f"""
        Analyze the following MEDICAL document text and provide a comprehensive classification and summary.
        
        Document Text:
        {text}
        
        Please respond with a valid JSON object that matches this exact structure:
        {{
            "classification": {{
                "document_type": "medical",
                "confidence": 0.95,
                "keywords": ["medical", "keyword1", "keyword2", "keyword3"],
                "language": "English",
                "is_structured": true
            }},
            "summary": {{
                "main_topic": "Brief description of main medical topic",
                "key_points": ["Medical Point 1", "Medical Point 2", "Medical Point 3"],
                "summary": "2-3 sentence summary of the medical document",
                "action_items": ["Medical Action 1", "Medical Action 2"],
                "important_dates": ["Medical Date 1", "Medical Date 2"],
                "entities": ["Medical Entity 1", "Medical Entity 2"]
            }},
            "processing_notes": "Medical document processed with structured analysis"
        }}
        
        Focus on medical terminology, patient information, diagnoses, treatments, and medical procedures. Ensure all fields are properly filled.
        """
    else:
        # Use freeform approach for non-medical documents
        prompt = f"""
        Analyze the following document text and provide a comprehensive summary.
        
        Document Text:
        {text}
        
        Please provide a detailed analysis in the following format:
        
        DOCUMENT TYPE: [Type of document]
        CONFIDENCE: [Confidence level 0-100%]
        LANGUAGE: [Primary language]
        STRUCTURED: [Yes/No]
        
        MAIN TOPIC: [Brief description of main topic]
        
        KEY POINTS:
        - [Key point 1]
        - [Key point 2]
        - [Key point 3]
        
        SUMMARY: [2-3 sentence summary of the document]
        
        KEYWORDS: [Comma-separated list of important keywords]
        
        ACTION ITEMS: [Any action items or next steps mentioned]
        
        IMPORTANT DATES: [Any important dates mentioned]
        
        ENTITIES: [Important people, organizations, or locations mentioned]
        
        Be thorough but concise. Focus on the most important information.
        """
    
    for attempt in range(max_retries):
        try:
            client = openai.OpenAI(api_key=openai.api_key)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert document analyst. Analyze documents and provide structured classifications and summaries. You MUST respond with valid JSON only - no markdown formatting, no explanations, just the raw JSON object. The JSON must match the exact structure provided in the user prompt."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            # Extract the response content
            response_text = response.choices[0].message.content
            
            if is_medical:
                # Try to parse as JSON and validate with Pydantic for medical documents
                try:
                    # Clean the response (remove markdown formatting if present)
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    
                    response_data = json.loads(response_text.strip())
                    analysis = DocumentAnalysis(**response_data)
                    
                    logging.info(f"Successfully analyzed medical document on attempt {attempt + 1}")
                    return analysis.dict()
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logging.warning(f"Attempt {attempt + 1}: Invalid JSON or Pydantic validation failed: {e}")
                    if attempt == max_retries - 1:
                        return {"error": f"Failed to get valid structured response after {max_retries} attempts"}
                    continue
            else:
                # For non-medical documents, return the freeform response
                logging.info(f"Successfully analyzed non-medical document on attempt {attempt + 1}")
                return {
                    "is_medical": False,
                    "freeform_analysis": response_text,
                    "processing_notes": "Non-medical document processed with freeform analysis"
                }
                
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}: OpenAI API error: {e}")
            if attempt == max_retries - 1:
                return {"error": f"OpenAI API error after {max_retries} attempts: {str(e)}"}
            continue
    
    return {"error": "Unexpected error in document analysis"}

def convert_to_fhir_with_agent(text, max_retries=3):
    """Convert document text to FHIR format using OpenAI agent with evaluator optimizer"""
    if not openai.api_key:
        return {"error": "OpenAI API key not configured"}
    
    # Truncate text if too long (OpenAI has token limits)
    if len(text) > 10000:  # Leave room for prompt and response
        text = text[:10000] + "..."
    
    fhir_prompt = f"""
    Convert the following document text into a valid FHIR (Fast Healthcare Interoperability Resources) Bundle.
    
    Document Text:
    {text}
    
    Create a comprehensive FHIR Bundle that includes relevant resources such as:
    - Patient (if patient information is present)
    - Practitioner (if healthcare providers are mentioned)
    - Observation (for measurements, lab results, vital signs)
    - Condition (for diagnoses, medical conditions)
    - Medication (for medications mentioned)
    - Procedure (for medical procedures)
    - Encounter (for healthcare encounters)
    - Organization (for healthcare organizations)
    
    IMPORTANT: Respond with ONLY a valid JSON object representing a FHIR Bundle. Do not include any explanations, markdown formatting, or additional text. The response must be valid JSON that can be parsed directly.
    
    The Bundle should have this exact structure:
    {{
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {{
                "resource": {{
                    "resourceType": "Patient",
                    "id": "patient-1",
                    "name": [{{"family": "Doe", "given": ["John"]}}]
                }}
            }}
        ]
    }}
    
    Each resource should be properly structured according to FHIR R4 specifications.
    """
    
    for attempt in range(max_retries):
        try:
            client = openai.OpenAI(api_key=openai.api_key)
            
            # FHIR Conversion Agent
            fhir_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a FHIR conversion specialist. Convert medical documents into valid FHIR R4 Bundle format. CRITICAL: You must respond with ONLY valid JSON. No explanations, no markdown, no additional text. The response must be parseable JSON that represents a proper FHIR Bundle."
                    },
                    {"role": "user", "content": fhir_prompt}
                ],
                max_tokens=3000,
                temperature=0.1
            )
            
            fhir_json = fhir_response.choices[0].message.content
            
            # Debug logging
            logging.info(f"Raw FHIR response: {fhir_json[:200]}...")
            
            # Clean the response (remove markdown formatting if present)
            if fhir_json.startswith("```json"):
                fhir_json = fhir_json[7:]
            if fhir_json.endswith("```"):
                fhir_json = fhir_json[:-3]
            if fhir_json.startswith("```"):
                fhir_json = fhir_json[3:]
            if fhir_json.endswith("```"):
                fhir_json = fhir_json[:-3]
            
            # Strip whitespace and check if empty
            fhir_json = fhir_json.strip()
            if not fhir_json:
                logging.error("Empty FHIR response from OpenAI")
                if attempt == max_retries - 1:
                    return {"error": "Empty response from FHIR conversion agent"}
                continue
            
            # Parse the FHIR JSON
            try:
                fhir_data = json.loads(fhir_json)
            except json.JSONDecodeError as json_err:
                logging.error(f"JSON decode error: {json_err}")
                logging.error(f"Problematic JSON: {fhir_json[:500]}")
                if attempt == max_retries - 1:
                    return {"error": f"Invalid JSON response from FHIR agent: {str(json_err)}"}
                continue
            
            # Evaluator Optimizer - Validate FHIR structure
            validation_result = validate_fhir_structure(fhir_data)
            
            if validation_result["is_valid"]:
                logging.info(f"Successfully converted to FHIR on attempt {attempt + 1}")
                return {
                    "fhir_bundle": fhir_data,
                    "validation": validation_result,
                    "conversion_notes": "Document successfully converted to FHIR format"
                }
            else:
                logging.warning(f"Attempt {attempt + 1}: FHIR validation failed: {validation_result['errors']}")
                if attempt == max_retries - 1:
                    return {
                        "error": f"Failed to generate valid FHIR after {max_retries} attempts",
                        "validation": validation_result,
                        "raw_fhir": fhir_data
                    }
                continue
                
        except json.JSONDecodeError as e:
            logging.warning(f"Attempt {attempt + 1}: Invalid JSON in FHIR response: {e}")
            if attempt == max_retries - 1:
                return {"error": f"Failed to parse FHIR JSON after {max_retries} attempts: {str(e)}"}
            continue
            
        except openai.APIError as e:
            logging.error(f"Attempt {attempt + 1}: OpenAI API error: {e}")
            if attempt == max_retries - 1:
                return {"error": f"OpenAI API error after {max_retries} attempts: {str(e)}"}
            continue
            
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}: FHIR conversion error: {e}")
            if attempt == max_retries - 1:
                return {"error": f"FHIR conversion error after {max_retries} attempts: {str(e)}"}
            continue
    
    return {"error": "Unexpected error in FHIR conversion"}

def validate_fhir_structure(fhir_data):
    """Evaluator Optimizer - Validate FHIR Bundle structure"""
    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "resource_count": 0,
        "resource_types": [],
        "validation_score": 0
    }
    
    try:
        # Check if it's a Bundle
        if not isinstance(fhir_data, dict):
            validation_result["errors"].append("FHIR data must be a JSON object")
            validation_result["is_valid"] = False
            return validation_result
        
        if fhir_data.get("resourceType") != "Bundle":
            validation_result["errors"].append("Root resource must be of type 'Bundle'")
            validation_result["is_valid"] = False
        
        if fhir_data.get("type") != "collection":
            validation_result["warnings"].append("Bundle type should be 'collection'")
        
        # Check for entries
        entries = fhir_data.get("entry", [])
        if not entries:
            validation_result["errors"].append("Bundle must contain at least one entry")
            validation_result["is_valid"] = False
        else:
            validation_result["resource_count"] = len(entries)
            
            # Validate each entry
            for i, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    validation_result["errors"].append(f"Entry {i} must be an object")
                    validation_result["is_valid"] = False
                    continue
                
                resource = entry.get("resource")
                if not resource:
                    validation_result["errors"].append(f"Entry {i} missing 'resource' field")
                    validation_result["is_valid"] = False
                    continue
                
                if not isinstance(resource, dict):
                    validation_result["errors"].append(f"Entry {i} resource must be an object")
                    validation_result["is_valid"] = False
                    continue
                
                resource_type = resource.get("resourceType")
                if not resource_type:
                    validation_result["errors"].append(f"Entry {i} resource missing 'resourceType'")
                    validation_result["is_valid"] = False
                else:
                    validation_result["resource_types"].append(resource_type)
                
                # Check for required fields based on resource type
                if resource_type == "Patient":
                    if not resource.get("name"):
                        validation_result["warnings"].append(f"Patient resource {i} missing 'name' field")
                elif resource_type == "Observation":
                    if not resource.get("status"):
                        validation_result["warnings"].append(f"Observation resource {i} missing 'status' field")
                elif resource_type == "Condition":
                    if not resource.get("code"):
                        validation_result["warnings"].append(f"Condition resource {i} missing 'code' field")
        
        # Calculate validation score
        total_checks = 5  # Basic structure checks
        passed_checks = 0
        
        if fhir_data.get("resourceType") == "Bundle":
            passed_checks += 1
        if fhir_data.get("type") == "collection":
            passed_checks += 1
        if entries:
            passed_checks += 1
        if validation_result["resource_count"] > 0:
            passed_checks += 1
        if not validation_result["errors"]:
            passed_checks += 1
        
        validation_result["validation_score"] = (passed_checks / total_checks) * 100
        
        return validation_result
        
    except Exception as e:
        validation_result["errors"].append(f"Validation error: {str(e)}")
        validation_result["is_valid"] = False
        return validation_result

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
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
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(file_path)
            print(f"File saved to: {file_path}")
            
            # Generate hash for caching
            file_hash = get_file_hash(file_path)
            cache_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}.txt")
            print(f"File hash: {file_hash}")
            
            # Check if we have cached text
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                flash('Using cached text extraction')
                print("Using cached text")
            else:
                # Extract text from PDF
                print("Extracting text from PDF...")
                text = extract_text_from_pdf(file_path)
                if not text.strip():
                    flash('Could not extract text from PDF. The PDF might be image-based or corrupted.')
                    return redirect(url_for('index'))
                
                # Save to cache
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                flash('Text extracted and cached')
                print(f"Text extracted, length: {len(text)} characters")
            
            # Store results in session for display
            session_data = {
                'filename': filename,
                'text': text,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save session data to file (in production, use proper session management)
            session_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}_session.json")
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)
            
            return redirect(url_for('text_display', file_hash=file_hash))
        else:
            flash('Invalid file type. Please upload a PDF file.')
            return redirect(url_for('index'))
    
    except Exception as e:
        print(f"Upload error: {str(e)}")
        flash(f'An error occurred during upload: {str(e)}')
        return redirect(url_for('index'))

@app.route('/text/<file_hash>')
def text_display(file_hash):
    """Display extracted text"""
    session_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}_session.json")
    
    if not os.path.exists(session_file):
        flash('Text display not found')
        return redirect(url_for('index'))
    
    with open(session_file, 'r', encoding='utf-8') as f:
        session_data = json.load(f)
    
    return render_template('text_display.html', data=session_data)

@app.route('/summarise/<file_hash>', methods=['POST'])
def summarise_document(file_hash):
    """Analyze and summarize document using AI"""
    try:
        session_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}_session.json")
        
        if not os.path.exists(session_file):
            return jsonify({"error": "Document not found"}), 404
        
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        text = session_data.get('text', '')
        if not text:
            return jsonify({"error": "No text content found"}), 400
        
        # Check if we already have analysis cached
        analysis_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}_analysis.json")
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            return jsonify(analysis_data)
        
        # Perform AI analysis
        analysis_result = analyze_document_with_ai(text)
        
        if "error" in analysis_result:
            return jsonify(analysis_result), 500
        
        # Cache the analysis result
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2)
        
        return jsonify(analysis_result)
        
    except Exception as e:
        logging.error(f"Error in summarise_document: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/convert-fhir/<file_hash>', methods=['POST'])
def convert_to_fhir(file_hash):
    """Convert document to FHIR format"""
    try:
        session_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}_session.json")
        
        if not os.path.exists(session_file):
            return jsonify({"error": "Document not found"}), 404
        
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        text = session_data.get('text', '')
        if not text:
            return jsonify({"error": "No text content found"}), 400
        
        # Check if we already have FHIR conversion cached
        fhir_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}_fhir.json")
        if os.path.exists(fhir_file):
            with open(fhir_file, 'r', encoding='utf-8') as f:
                fhir_data = json.load(f)
            return jsonify(fhir_data)
        
        # Perform FHIR conversion
        fhir_result = convert_to_fhir_with_agent(text)
        
        if "error" in fhir_result:
            return jsonify(fhir_result), 500
        
        # Cache the FHIR result
        with open(fhir_file, 'w', encoding='utf-8') as f:
            json.dump(fhir_result, f, indent=2)
        
        return jsonify(fhir_result)
        
    except Exception as e:
        logging.error(f"Error in convert_to_fhir: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/hello')
def api_hello():
    """Simple API endpoint"""
    return jsonify({
        'message': 'Hello from Flask!',
        'status': 'success'
    })

@app.route('/api/echo', methods=['POST'])
def api_echo():
    """Echo API endpoint that returns the sent data"""
    data = request.get_json()
    return jsonify({
        'received': data,
        'message': 'Data received successfully'
    })

@app.errorhandler(404)
def not_found(error):
    """Custom 404 error handler"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Custom 500 error handler"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Run the app in debug mode for development
    app.run(debug=True, host='0.0.0.0', port=5000)
