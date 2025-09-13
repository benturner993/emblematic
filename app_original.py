from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response
import os
import hashlib
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import PyPDF2
import openai
import httpx
from dotenv import load_dotenv
from models import DocumentAnalysis
import logging
import copy

# Import API keys from separate file
try:
    from keys import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
except ImportError:
    # Fallback to environment variables if keys.py doesn't exist
    AZURE_OPENAI_API_KEY = None
    AZURE_OPENAI_ENDPOINT = None
    AZURE_OPENAI_API_VERSION = None

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEXT_CACHE_FOLDER'] = 'text_cache'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Azure OpenAI Configuration
if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_VERSION:
    # Store credentials for use in client constructors
    openai.api_key = AZURE_OPENAI_API_KEY
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_version = AZURE_OPENAI_API_VERSION
else:
    # Fallback to environment variables
    openai.api_key = os.environ.get('AZURE_OPENAI_API_KEY')
    openai.api_base = os.environ.get('AZURE_OPENAI_ENDPOINT')
    openai.api_version = os.environ.get('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')

if not openai.api_key or not openai.api_base:
    print("Warning: Azure OpenAI credentials not found in keys.py or environment variables")

# Setup logging
logging.basicConfig(level=logging.INFO)

def extract_judge_logs_safely(fhir_result):
    """Extract judge logs safely without circular references"""
    try:
        judge_logs_data = []
        
        if isinstance(fhir_result, dict) and 'detailed_logs' in fhir_result:
            for log in fhir_result['detailed_logs']:
                if isinstance(log, dict):
                    # Extract basic attempt info
                    attempt_info = {
                        "attempt_number": log.get("attempt_number", 0),
                        "timestamp": str(log.get("timestamp", "")),
                        "groundedness_score": log.get("groundedness_evaluation", {}).get("groundedness_score", 0),
                        "is_acceptable": log.get("groundedness_evaluation", {}).get("is_acceptable", False),
                        "judge_communications": []
                    }
                    
                    # Extract judge communications
                    judge_logs = log.get("groundedness_evaluation", {}).get("judge_logs", [])
                    for judge_log in judge_logs:
                        if isinstance(judge_log, dict):
                            # Create completely clean judge communication
                            clean_judge = {
                                "judge_attempt": judge_log.get("attempt_number", 0),
                                "timestamp": str(judge_log.get("timestamp", "")),
                                "prompt_sent_to_judge": str(judge_log.get("prompt", ""))[:1000],
                                "raw_response_from_judge": str(judge_log.get("raw_response", "")),
                                "judge_errors": [str(e) for e in judge_log.get("errors", [])]
                            }
                            
                            # Extract parsed response safely
                            parsed_resp = judge_log.get("parsed_response", {})
                            if isinstance(parsed_resp, dict):
                                clean_judge["judge_assessment"] = {
                                    "groundedness_score": parsed_resp.get("groundedness_score", 0),
                                    "is_acceptable": parsed_resp.get("is_acceptable", False),
                                    "overall_assessment": str(parsed_resp.get("overall_assessment", "")),
                                    "hallucinations_detected": [str(h) for h in parsed_resp.get("hallucinations_detected", [])],
                                    "missing_information": [str(m) for m in parsed_resp.get("missing_information", [])],
                                    "improvement_feedback": str(parsed_resp.get("improvement_feedback", "")),
                                    "priority_fixes": [str(p) for p in parsed_resp.get("priority_fixes", [])]
                                }
                            else:
                                clean_judge["judge_assessment"] = {"note": "Could not parse judge response"}
                            
                            attempt_info["judge_communications"].append(clean_judge)
                    
                    # Only add if there are judge communications or other relevant data
                    if attempt_info["judge_communications"] or attempt_info["groundedness_score"] > 0:
                        judge_logs_data.append(attempt_info)
        
        logging.info(f"Extracted {len(judge_logs_data)} judge log entries")
        return judge_logs_data
        
    except Exception as e:
        logging.error(f"Error extracting judge logs: {e}")
        return []

def clean_for_json(obj, seen=None):
    """Clean object for JSON serialization by removing circular references"""
    if seen is None:
        seen = set()
    
    if id(obj) in seen:
        return "[Circular Reference]"
    
    if isinstance(obj, dict):
        seen.add(id(obj))
        result = {}
        for key, value in obj.items():
            try:
                result[key] = clean_for_json(value, seen)
            except Exception as e:
                result[key] = f"[Error cleaning {key}: {str(e)}]"
        seen.remove(id(obj))
        return result
    elif isinstance(obj, list):
        seen.add(id(obj))
        result = []
        for i, item in enumerate(obj):
            try:
                result.append(clean_for_json(item, seen))
            except Exception as e:
                result.append(f"[Error cleaning item {i}: {str(e)}]")
        seen.remove(id(obj))
        return result
    elif hasattr(obj, '__dict__'):
        seen.add(id(obj))
        try:
            result = clean_for_json(obj.__dict__, seen)
        except Exception as e:
            result = f"[Error cleaning object: {str(e)}]"
        seen.remove(id(obj))
        return result
    else:
        try:
            # Try to serialize the object directly
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

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
        client = openai.AzureOpenAI(
            api_key=openai.api_key,
            azure_endpoint=openai.api_base,
            api_version=openai.api_version,
            http_client=httpx.Client()
        )
        
        response = client.chat.completions.create(
            model="gpt-4",  # This should match your Azure OpenAI deployment name
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
            client = openai.AzureOpenAI(
            api_key=openai.api_key,
            azure_endpoint=openai.api_base,
            api_version=openai.api_version,
            http_client=httpx.Client()
        )
            
            response = client.chat.completions.create(
                model="gpt-4",  # This should match your Azure OpenAI deployment name
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

def convert_to_fhir_with_agent(text, max_retries=5, feedback_history=None):
    """Convert document text to FHIR format using OpenAI agent with evaluator optimizer and groundedness judge"""
    if not openai.api_key:
        return {"error": "OpenAI API key not configured"}
    
    # Store original text for groundedness evaluation
    original_text = text
    
    # Truncate text if too long (OpenAI has token limits)
    if len(text) > 10000:  # Leave room for prompt and response
        text = text[:10000] + "..."
    
    # Initialize feedback history if not provided
    if feedback_history is None:
        feedback_history = []
    
    # Initialize detailed logs immediately - this is critical for capturing all attempts
    detailed_logs = []
    
    # Track the best attempt across all tries
    best_attempt = None
    best_score = 0
    
    logging.info(f"Starting FHIR conversion with {max_retries} max retries")
    logging.info(f"Text length: {len(text)} characters")
    
    # Base FHIR prompt template (feedback will be added per attempt)
    base_fhir_template = """
    Convert the following document text into a valid FHIR (Fast Healthcare Interoperability Resources) Bundle.
    
    Document Text:
    {text}{feedback_section}
    
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
        # Create attempt log with safe, serializable data
        attempt_log = {
            "attempt_number": attempt + 1,
            "timestamp": datetime.now().isoformat(),
            "fhir_conversion": {
                "prompt": "",
                "prompt_length": 0,
                "raw_response": "",
                "response_length": 0,
                "parsed_successfully": False,
                "parsed_data": None,
                "json_error": None
            },
            "structure_validation": {
                "is_valid": False,
                "validation_score": 0,
                "resource_count": 0,
                "resource_types": [],
                "errors": [],
                "warnings": []
            },
            "groundedness_evaluation": {
                "groundedness_score": 0,
                "is_acceptable": False,
                "overall_assessment": "",
                "hallucinations_detected": [],
                "missing_information": [],
                "accuracy_issues": [],
                "improvement_feedback": "",
                "priority_fixes": [],
                "judge_logs": []
            },
            "errors": [],
            "retry_reason": None
        }
        
        # CRITICAL: Append attempt log immediately to ensure it's captured
        detailed_logs.append(attempt_log)
        logging.info(f"Attempt {attempt + 1} log added to detailed_logs. Total logs: {len(detailed_logs)}")
        
        try:
            # Build feedback learning prompt for this attempt
            feedback_section = ""
            if feedback_history:
                feedback_section = "\n\nIMPORTANT FEEDBACK FROM PREVIOUS ATTEMPTS:\n"
                for i, feedback in enumerate(feedback_history, 1):
                    feedback_section += f"\nAttempt {i} Feedback:\n"
                    feedback_section += f"- Score: {feedback.get('groundedness_score', 'N/A')}/5\n"
                    feedback_section += f"- Issues: {', '.join(feedback.get('priority_fixes', []))}\n"
                    feedback_section += f"- Improvement needed: {feedback.get('improvement_feedback', 'N/A')}\n"
                    if feedback.get('hallucinations_detected'):
                        feedback_section += f"- Hallucinations to avoid: {', '.join(feedback.get('hallucinations_detected', []))}\n"
                    if feedback.get('missing_information'):
                        feedback_section += f"- Missing info to include: {', '.join(feedback.get('missing_information', []))}\n"
                feedback_section += "\nPlease address these specific issues in your conversion:\n"
                logging.info(f"Attempt {attempt + 1}: Including feedback from {len(feedback_history)} previous attempts")
            
            # Build the complete prompt with current feedback
            fhir_prompt = base_fhir_template.format(text=text, feedback_section=feedback_section)
            
            # Initialize Azure OpenAI client
            logging.info(f"Initializing Azure OpenAI client for attempt {attempt + 1}")
            client = openai.AzureOpenAI(
                api_key=openai.api_key,
                azure_endpoint=openai.api_base,
                api_version=openai.api_version,
                http_client=httpx.Client()
            )
            
            # Log the prompt being sent to FHIR agent
            attempt_log["fhir_conversion"]["prompt"] = fhir_prompt
            attempt_log["fhir_conversion"]["prompt_length"] = len(fhir_prompt)
            
            # FHIR Conversion Agent
            logging.info(f"Calling OpenAI API for attempt {attempt + 1}")
            fhir_response = client.chat.completions.create(
                model="gpt-4",  # This should match your Azure OpenAI deployment name
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
            
            # Log the raw response
            fhir_json = fhir_response.choices[0].message.content
            logging.info(f"Received response of length {len(fhir_json)} for attempt {attempt + 1}")
            attempt_log["fhir_conversion"]["raw_response"] = fhir_json
            attempt_log["fhir_conversion"]["response_length"] = len(fhir_json)
            
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
                attempt_log["fhir_conversion"]["parsed_successfully"] = True
                attempt_log["fhir_conversion"]["parsed_data"] = fhir_data
            except json.JSONDecodeError as json_err:
                logging.error(f"JSON decode error: {json_err}")
                logging.error(f"Problematic JSON: {fhir_json[:500]}")
                attempt_log["errors"].append(f"JSON decode error: {str(json_err)}")
                attempt_log["fhir_conversion"]["parsed_successfully"] = False
                attempt_log["fhir_conversion"]["json_error"] = str(json_err)
                if attempt == max_retries - 1:
                    final_result = {"error": f"Invalid JSON response from FHIR agent: {str(json_err)}", "detailed_logs": detailed_logs}
                    logging.info(f"Returning JSON error result with {len(detailed_logs)} detailed logs")
                    return final_result
                continue
            
            # Evaluator Optimizer - Validate FHIR structure
            validation_result = validate_fhir_structure(fhir_data)
            attempt_log["structure_validation"] = validation_result
            
            if not validation_result["is_valid"]:
                logging.warning(f"Attempt {attempt + 1}: FHIR validation failed: {validation_result['errors']}")
                attempt_log["errors"].append(f"FHIR validation failed: {', '.join(validation_result['errors'])}")
                if attempt == max_retries - 1:
                    final_result = {
                        "error": f"Failed to generate valid FHIR after {max_retries} attempts",
                        "validation": validation_result,
                        "raw_fhir": fhir_data,
                        "detailed_logs": detailed_logs
                    }
                    logging.info(f"Returning validation error result with {len(detailed_logs)} detailed logs")
                    return final_result
                continue
            
            # LLM Judge - Evaluate groundedness against original text
            # Get file_hash from session to pass to judge
            file_hash = None
            try:
                # Try to find the file_hash from the current request context
                # This is a bit hacky but necessary for the judge logging
                import glob
                session_files = glob.glob(os.path.join(app.config['TEXT_CACHE_FOLDER'], "*_session.json"))
                if session_files:
                    # Get the most recent session file
                    latest_session = max(session_files, key=os.path.getmtime)
                    file_hash = os.path.basename(latest_session).replace("_session.json", "")
            except Exception as e:
                logging.warning(f"Could not determine file_hash for judge logging: {e}")
            
            groundedness_result = evaluate_fhir_groundedness(original_text, fhir_data, max_retries=3, file_hash=file_hash, attempt_number=attempt + 1, feedback_history=feedback_history)
            attempt_log["groundedness_evaluation"] = groundedness_result
            
            # Debug logging for judge response
            logging.info(f"Judge result keys: {list(groundedness_result.keys()) if isinstance(groundedness_result, dict) else 'Not a dict'}")
            logging.info(f"Judge result has judge_logs: {'judge_logs' in groundedness_result if isinstance(groundedness_result, dict) else False}")
            if isinstance(groundedness_result, dict) and 'judge_logs' in groundedness_result:
                logging.info(f"Judge logs count: {len(groundedness_result['judge_logs'])}")
                for i, judge_log in enumerate(groundedness_result['judge_logs']):
                    logging.info(f"Judge log {i+1} keys: {list(judge_log.keys()) if isinstance(judge_log, dict) else 'Not a dict'}")
                    if isinstance(judge_log, dict) and 'raw_response' in judge_log:
                        logging.info(f"Judge log {i+1} raw response length: {len(judge_log['raw_response'])}")
            
            if "error" in groundedness_result:
                logging.warning(f"Attempt {attempt + 1}: Groundedness evaluation failed: {groundedness_result['error']}")
                attempt_log["errors"].append(f"Groundedness evaluation failed: {groundedness_result['error']}")
                if attempt == max_retries - 1:
                    return {
                        "error": f"Groundedness evaluation failed after {max_retries} attempts",
                        "validation": validation_result,
                        "groundedness_error": groundedness_result["error"],
                        "fhir_bundle": fhir_data,
                        "detailed_logs": detailed_logs
                    }
                continue
            
            # Check if groundedness score is acceptable
            groundedness_score = groundedness_result.get("groundedness_score", 0)
            is_acceptable = groundedness_result.get("is_acceptable", False)
            
            # Track the best attempt so far - create clean, serializable data
            if groundedness_score > best_score:
                best_score = groundedness_score
                
                # Create clean groundedness data without circular references
                clean_groundedness = {
                    "groundedness_score": groundedness_result.get("groundedness_score", 0),
                    "is_acceptable": groundedness_result.get("is_acceptable", False),
                    "overall_assessment": groundedness_result.get("overall_assessment", ""),
                    "hallucinations_detected": groundedness_result.get("hallucinations_detected", []),
                    "missing_information": groundedness_result.get("missing_information", []),
                    "accuracy_issues": groundedness_result.get("accuracy_issues", []),
                    "improvement_feedback": groundedness_result.get("improvement_feedback", ""),
                    "priority_fixes": groundedness_result.get("priority_fixes", [])
                }
                
                # Create clean validation data without circular references  
                clean_validation = {
                    "validation_score": validation_result.get("validation_score", 0),
                    "is_valid": validation_result.get("is_valid", False),
                    "resource_count": validation_result.get("resource_count", 0),
                    "resource_types": validation_result.get("resource_types", []),
                    "errors": validation_result.get("errors", []),
                    "warnings": validation_result.get("warnings", [])
                }
                
                best_attempt = {
                    "fhir_bundle": fhir_data,
                    "validation": clean_validation,
                    "groundedness": clean_groundedness,
                    "confidence_score": f"{groundedness_score}/5",
                    "conversion_notes": f"Best FHIR conversion result with confidence score {groundedness_score}/5 (attempt {attempt + 1})",
                    "attempt_number": attempt + 1
                }
                logging.info(f"New best score: {groundedness_score}/5 on attempt {attempt + 1}")
            
            # Return early if we get a perfect or very good score
            if groundedness_score >= 4 and is_acceptable:
                logging.info(f"Successfully converted to FHIR on attempt {attempt + 1} with groundedness score {groundedness_score}/5")
                return best_attempt
            elif attempt == max_retries - 1:
                # Last attempt - return the best result we found
                if best_attempt:
                    logging.info(f"Completed FHIR conversion after {max_retries} attempts. Returning best result with score {best_score}/5 from attempt {best_attempt['attempt_number']}")
                    return best_attempt
                else:
                    # Fallback if no best attempt was recorded - create clean data
                    logging.info(f"Completed FHIR conversion after {max_retries} attempts with final groundedness score {groundedness_score}/5")
                    clean_groundedness = {
                        "groundedness_score": groundedness_result.get("groundedness_score", 0),
                        "is_acceptable": groundedness_result.get("is_acceptable", False),
                        "overall_assessment": groundedness_result.get("overall_assessment", ""),
                        "hallucinations_detected": groundedness_result.get("hallucinations_detected", []),
                        "missing_information": groundedness_result.get("missing_information", []),
                        "accuracy_issues": groundedness_result.get("accuracy_issues", []),
                        "improvement_feedback": groundedness_result.get("improvement_feedback", ""),
                        "priority_fixes": groundedness_result.get("priority_fixes", [])
                    }
                    clean_validation = {
                        "validation_score": validation_result.get("validation_score", 0),
                        "is_valid": validation_result.get("is_valid", False),
                        "resource_count": validation_result.get("resource_count", 0),
                        "resource_types": validation_result.get("resource_types", []),
                        "errors": validation_result.get("errors", []),
                        "warnings": validation_result.get("warnings", [])
                    }
                    return {
                        "fhir_bundle": fhir_data,
                        "validation": clean_validation,
                        "groundedness": clean_groundedness,
                        "confidence_score": f"{groundedness_score}/5",
                        "conversion_notes": f"FHIR conversion completed with confidence score {groundedness_score}/5 after {max_retries} attempts"
                    }
            else:
                # Try to improve with feedback
                logging.warning(f"Attempt {attempt + 1}: Groundedness score {groundedness_score}/5. Attempting to improve...")
                
                # Add feedback to history for learning
                feedback_history.append(groundedness_result)
                attempt_log["retry_reason"] = f"Attempting to improve groundedness score {groundedness_score}/5"
                continue
                
        except json.JSONDecodeError as e:
            logging.warning(f"Attempt {attempt + 1}: Invalid JSON in FHIR response: {e}")
            attempt_log["errors"].append(f"JSON decode error: {str(e)}")
            if attempt == max_retries - 1:
                # Return the best attempt if we have one, even with JSON errors
                if best_attempt:
                    logging.info(f"JSON parsing failed but returning best attempt with score {best_score}/5")
                    return best_attempt
                return {"error": f"Failed to parse FHIR JSON after {max_retries} attempts: {str(e)}"}
            continue
            
        except openai.APIError as e:
            logging.error(f"Attempt {attempt + 1}: OpenAI API error: {e}")
            attempt_log["errors"].append(f"OpenAI API error: {str(e)}")
            if attempt == max_retries - 1:
                # Return the best attempt if we have one, even with API errors
                if best_attempt:
                    logging.info(f"API error occurred but returning best attempt with score {best_score}/5")
                    return best_attempt
                return {"error": f"OpenAI API error after {max_retries} attempts: {str(e)}"}
            continue
            
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}: FHIR conversion error: {e}")
            attempt_log["errors"].append(f"General error: {str(e)}")
            if attempt == max_retries - 1:
                # Return the best attempt if we have one, even with general errors
                if best_attempt:
                    logging.info(f"General error occurred but returning best attempt with score {best_score}/5")
                    return best_attempt
                return {"error": f"FHIR conversion error after {max_retries} attempts: {str(e)}"}
            continue
    
    logging.error("Unexpected end of FHIR conversion function")
    # Return the best attempt if we have one, even in unexpected cases
    if best_attempt:
        logging.info(f"Unexpected end but returning best attempt with score {best_score}/5")
        return best_attempt
    return {"error": "Unexpected error in FHIR conversion"}

def write_judge_output_to_file(file_hash, judge_output, attempt_number=1):
    """Write judge output directly to a text file"""
    try:
        judge_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}_judge_logs.txt")
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

def evaluate_fhir_groundedness(original_text, fhir_bundle, max_retries=3, file_hash=None, attempt_number=1, feedback_history=None):
    """LLM Judge - Evaluate groundedness of FHIR conversion against original text"""
    if not openai.api_key:
        return {"error": "OpenAI API key not configured"}
    
    # Truncate text if too long
    if len(original_text) > 8000:  # Leave room for prompt and response
        original_text = original_text[:8000] + "..."
    
    fhir_json = json.dumps(fhir_bundle, indent=2)
    if len(fhir_json) > 4000:  # Truncate FHIR if too long
        fhir_json = fhir_json[:4000] + "..."
    
    # Initialize feedback history if not provided
    if feedback_history is None:
        feedback_history = []
    
    # Build feedback context for judge
    feedback_context = ""
    if feedback_history and attempt_number > 1:
        feedback_context = f"\n\nPREVIOUS FEEDBACK HISTORY (for context):\n"
        feedback_context += f"This is attempt #{attempt_number}. Here's what feedback was given in previous attempts:\n\n"
        
        for i, feedback in enumerate(feedback_history, 1):
            feedback_context += f"Previous Attempt {i} Feedback:\n"
            feedback_context += f"  - Score Given: {feedback.get('groundedness_score', 'N/A')}/5\n"
            feedback_context += f"  - Issues Identified: {', '.join(feedback.get('priority_fixes', []))}\n"
            feedback_context += f"  - Guidance Provided: {feedback.get('improvement_feedback', 'N/A')}\n"
            feedback_context += f"  - Missing Information: {', '.join(feedback.get('missing_information', []))}\n\n"
        
        feedback_context += "IMPORTANT: Assess whether this current FHIR Bundle addresses the previous feedback. "
        feedback_context += "If improvements were made based on previous guidance, acknowledge them in your scoring.\n"
    
    # Store original inputs for logging
    judge_inputs = {
        "original_text": original_text,
        "fhir_json": fhir_json,
        "original_text_length": len(original_text),
        "fhir_json_length": len(fhir_json),
        "feedback_history_provided": len(feedback_history),
        "attempt_number": attempt_number
    }
    
    groundedness_prompt = f"""
    You are a medical data quality judge evaluating FHIR Bundle conversions. Your role is to provide constructive, improvement-focused feedback.
    
    Original Document Text:
    {original_text}
    
    Generated FHIR Bundle:
    {fhir_json}{feedback_context}
    
    EVALUATION CRITERIA:
    1. **Factual Accuracy**: Does the FHIR data accurately reflect the original text?
    2. **Completeness**: Are key medical facts captured appropriately?
    3. **No Hallucination**: Are there any facts in the FHIR that are NOT in the original text?
    4. **Proper Mapping**: Are medical concepts correctly mapped to FHIR resources?
    5. **Progressive Improvement**: Focus on the most important issues first
    
    SCORING GUIDELINES (be fair but thorough):
    - 5: Excellent - captures all essential information accurately with comprehensive detail
    - 4: Very Good - captures most important information with good detail, minor omissions acceptable  
    - 3: Good - captures core medical facts but missing significant details or context
    - 2: Needs Work - missing important information, lacks medical detail, or contains inaccuracies
    - 1: Poor - significant problems with accuracy, completeness, or medical context
    
    IMPORTANT: If this appears to be an improved version (more complete than typical first attempts), 
    be more generous with scoring to acknowledge the improvement effort.
    
    Respond with ONLY a JSON object in this exact format:
    {{
        "groundedness_score": <number 1-5>,
        "is_acceptable": <true/false - be generous, accept 4+ if no major hallucinations>,
        "hallucinations_detected": <list of any false information found>,
        "missing_information": <list of only the MOST important facts missing from FHIR>,
        "accuracy_issues": <list of any significant accuracy problems>,
        "overall_assessment": "<brief explanation focusing on what was done well>",
        "improvement_feedback": "<specific, actionable instructions for the single most important improvement>",
        "priority_fixes": <list of max 2 most critical issues to address>
    }}
    """
    
    for attempt in range(max_retries):
        judge_attempt_log = {
            "attempt_number": attempt + 1,
            "timestamp": datetime.now().isoformat(),
            "inputs": judge_inputs,
            "prompt": groundedness_prompt,
            "raw_response": "",
            "parsed_response": {},
            "errors": []
        }
        
        try:
            client = openai.AzureOpenAI(
            api_key=openai.api_key,
            azure_endpoint=openai.api_base,
            api_version=openai.api_version,
            http_client=httpx.Client()
        )
            
            judge_response = client.chat.completions.create(
                model="gpt-4",  # This should match your Azure OpenAI deployment name
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a medical data quality judge. Evaluate FHIR conversions for groundedness against source documents. Always respond with valid JSON only."
                    },
                    {"role": "user", "content": groundedness_prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            judge_json = judge_response.choices[0].message.content
            judge_attempt_log["raw_response"] = judge_json
            
            # Write judge output to static text file immediately
            if file_hash:
                write_judge_output_to_file(file_hash, judge_json, attempt_number)
            
            # Clean the response
            if judge_json.startswith("```json"):
                judge_json = judge_json[7:]
            if judge_json.endswith("```"):
                judge_json = judge_json[:-3]
            if judge_json.startswith("```"):
                judge_json = judge_json[3:]
            if judge_json.endswith("```"):
                judge_json = judge_json[:-3]
            
            judge_json = judge_json.strip()
            if not judge_json:
                logging.error("Empty judge response from OpenAI")
                if attempt == max_retries - 1:
                    return {"error": "Empty response from groundedness judge", "judge_logs": [judge_attempt_log]}
                continue
            
            # Parse the judge response
            try:
                judge_data = json.loads(judge_json)
                judge_attempt_log["parsed_response"] = judge_data
            except json.JSONDecodeError as json_err:
                logging.error(f"Judge JSON decode error: {json_err}")
                logging.error(f"Problematic JSON: {judge_json[:500]}")
                judge_attempt_log["errors"].append(f"JSON decode error: {str(json_err)}")
                if attempt == max_retries - 1:
                    return {"error": f"Invalid JSON response from judge: {str(json_err)}", "judge_logs": [judge_attempt_log]}
                continue
            
            # Validate the judge response structure
            required_fields = ["groundedness_score", "is_acceptable", "overall_assessment"]
            if not all(field in judge_data for field in required_fields):
                logging.warning(f"Judge response missing required fields: {judge_data}")
                judge_attempt_log["errors"].append(f"Missing required fields: {required_fields}")
                if attempt == max_retries - 1:
                    return {"error": "Judge response missing required fields", "judge_logs": [judge_attempt_log]}
                continue
            
            # Validate score is between 1-5
            score = judge_data.get("groundedness_score")
            if not isinstance(score, (int, float)) or score < 1 or score > 5:
                logging.warning(f"Invalid groundedness score: {score}")
                judge_attempt_log["errors"].append(f"Invalid score: {score}")
                if attempt == max_retries - 1:
                    return {"error": f"Invalid groundedness score: {score}", "judge_logs": [judge_attempt_log]}
                continue
            
            logging.info(f"Groundedness evaluation completed: Score {score}/5")
            judge_data["judge_logs"] = [judge_attempt_log]
            return judge_data
            
        except openai.APIError as e:
            logging.error(f"Attempt {attempt + 1}: OpenAI API error in judge: {e}")
            if attempt == max_retries - 1:
                return {"error": f"OpenAI API error in judge after {max_retries} attempts: {str(e)}", "judge_logs": [judge_attempt_log]}
            continue
            
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}: Judge evaluation error: {e}")
            if attempt == max_retries - 1:
                return {"error": f"Judge evaluation error after {max_retries} attempts: {str(e)}", "judge_logs": [judge_attempt_log]}
            continue
    
    return {"error": "Unexpected error in groundedness evaluation", "judge_logs": []}

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
    """Convert document to FHIR format with real-time updates"""
    try:
        session_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}_session.json")
        
        if not os.path.exists(session_file):
            return jsonify({"error": "Document not found"}), 404
        
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        text = session_data.get('text', '')
        if not text:
            return jsonify({"error": "No text content found"}), 400
        
        # No caching - always run fresh FHIR conversion
        
        # Always use regular conversion (no streaming) with feedback learning
        fhir_result = convert_to_fhir_with_agent(text, max_retries=5, feedback_history=[])
        
        # No caching - just return the result
        # Judge logs are already written to text file by write_judge_output_to_file()
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

@app.route('/fhir-logs/<file_hash>')
def fhir_logs(file_hash):
    """Display detailed FHIR conversion logs"""
    try:
        
        # Load the original text first
        session_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}_session.json")
        original_text = ""
        if os.path.exists(session_file):
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                original_text = session_data.get('text', '')
        
        # Load judge logs from text file
        judge_logs_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}_judge_logs.txt")
        judge_logs_text = ""
        
        if os.path.exists(judge_logs_file):
            try:
                with open(judge_logs_file, 'r', encoding='utf-8') as f:
                    judge_logs_text = f.read()
            except Exception as e:
                logging.error(f"Error reading judge logs file: {e}")
                judge_logs_text = f"Error reading judge logs: {str(e)}"
        else:
            judge_logs_text = "No judge logs found. Please run FHIR conversion first."
        
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

def stream_fhir_conversion(text, file_hash):
    """Stream FHIR conversion progress with real-time updates"""
    def generate():
        try:
            # Send initial status
            yield f"data: {json.dumps({'status': 'starting', 'message': 'Initializing FHIR conversion...'})}\n\n"
            
            # Perform conversion with feedback learning
            feedback_history = []
            max_retries = 3
            
            for attempt in range(max_retries):
                attempt_num = attempt + 1
                
                # Send attempt status
                yield f"data: {json.dumps({'status': 'attempt', 'attempt': attempt_num, 'message': f'FHIR conversion attempt {attempt_num}/{max_retries}'})}\n\n"
                
                # Perform conversion with feedback
                fhir_result = convert_to_fhir_with_agent(text, max_retries=1, feedback_history=feedback_history)
                
                if "error" in fhir_result:
                    yield f"data: {json.dumps({'status': 'error', 'message': fhir_result['error']})}\n\n"
                    break
                
                # Send structure validation status
                yield f"data: {json.dumps({'status': 'validating', 'message': 'Validating FHIR structure...'})}\n\n"
                
                validation = fhir_result.get('validation', {})
                if not validation.get('is_valid', False):
                    yield f"data: {json.dumps({'status': 'structure_error', 'message': 'FHIR structure validation failed', 'errors': validation.get('errors', [])})}\n\n"
                    continue
                
                # Send groundedness evaluation status
                yield f"data: {json.dumps({'status': 'evaluating', 'message': 'Evaluating groundedness against original text...'})}\n\n"
                
                groundedness = fhir_result.get('groundedness', {})
                score = groundedness.get('groundedness_score', 0)
                
                # Send groundedness results with detailed information
                groundedness_data = {
                    'status': 'groundedness_result', 
                    'score': score, 
                    'assessment': groundedness.get('overall_assessment', ''),
                    'feedback': groundedness.get('improvement_feedback', ''),
                    'is_acceptable': groundedness.get('is_acceptable', False),
                    'hallucinations': groundedness.get('hallucinations_detected', []),
                    'missing_info': groundedness.get('missing_information', []),
                    'accuracy_issues': groundedness.get('accuracy_issues', []),
                    'priority_fixes': groundedness.get('priority_fixes', []),
                    'attempt': attempt_num,
                    'max_attempts': max_retries
                }
                yield f"data: {json.dumps(groundedness_data)}\n\n"
                
                if score >= 5:
                    # Success - cache and return
                    fhir_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}_fhir.json")
                    with open(fhir_file, 'w', encoding='utf-8') as f:
                        json.dump(fhir_result, f, indent=2)
                    
                    yield f"data: {json.dumps({'status': 'success', 'result': fhir_result})}\n\n"
                    break
                else:
                    # Add feedback for next attempt
                    feedback_history.append(groundedness)
                    retry_data = {
                        'status': 'retry',
                        'message': f'Quality score {score}/5 below threshold. Retrying with feedback...',
                        'feedback': groundedness.get('improvement_feedback', ''),
                        'attempt': attempt_num,
                        'max_attempts': max_retries,
                        'next_attempt': attempt_num + 1
                    }
                    yield f"data: {json.dumps(retry_data)}\n\n"
            
            else:
                # All attempts failed - cache the result anyway
                fhir_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}_fhir.json")
                try:
                    with open(fhir_file, 'w', encoding='utf-8') as f:
                        json.dump(fhir_result, f, indent=2, default=str)
                except TypeError as json_err:
                    logging.error(f"JSON serialization error in streaming: {json_err}")
                    cleaned_result = clean_for_json(fhir_result)
                    with open(fhir_file, 'w', encoding='utf-8') as f:
                        json.dump(cleaned_result, f, indent=2, default=str)
                
                yield f"data: {json.dumps({'status': 'failed', 'message': 'All conversion attempts failed', 'result': fhir_result})}\n\n"
                
        except Exception as e:
            logging.error(f"Error in stream_fhir_conversion: {e}")
            # Try to cache a basic error result
            try:
                fhir_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}_fhir.json")
                error_result = {
                    "error": f"Conversion error: {str(e)}",
                    "detailed_logs": [],
                    "conversion_notes": "Conversion failed due to streaming error"
                }
                with open(fhir_file, 'w', encoding='utf-8') as f:
                    json.dump(error_result, f, indent=2, default=str)
            except Exception as cache_err:
                logging.error(f"Failed to cache streaming error result: {cache_err}")
            
            yield f"data: {json.dumps({'status': 'error', 'message': f'Conversion error: {str(e)}'})}\n\n"
    
    return generate()

@app.route('/fhir-judge-logs/<file_hash>')
def fhir_judge_logs(file_hash):
    """Display detailed FHIR conversion logs from text file"""
    try:
        # Load the original text first
        session_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}_session.json")
        original_text = ""
        if os.path.exists(session_file):
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                original_text = session_data.get('text', '')
        
        # Load judge logs from text file
        judge_logs_file = os.path.join(app.config['TEXT_CACHE_FOLDER'], f"{file_hash}_judge_logs.txt")
        judge_logs_text = ""
        
        if os.path.exists(judge_logs_file):
            try:
                with open(judge_logs_file, 'r', encoding='utf-8') as f:
                    judge_logs_text = f.read()
            except Exception as e:
                logging.error(f"Error reading judge logs file: {e}")
                judge_logs_text = f"Error reading judge logs: {str(e)}"
        else:
            judge_logs_text = "No judge logs found. Please run FHIR conversion first."
        
        # Create a simple structure for the template
        fhir_data = {
            "judge_logs_text": judge_logs_text,
            "has_judge_logs": bool(judge_logs_text and judge_logs_text != "No judge logs found. Please run FHIR conversion first."),
            "conversion_notes": "Judge logs loaded from text file"
        }
        
        return render_template('fhir_logs.html', 
                             fhir_data=fhir_data, 
                             original_text=original_text,
                             file_hash=file_hash)
        
    except Exception as e:
        logging.error(f"Error in fhir_logs route: {e}")
        return render_template('fhir_logs.html', 
                             fhir_data={
                                 "judge_logs_text": f"Error loading FHIR logs: {str(e)}",
                                 "has_judge_logs": False,
                                 "conversion_notes": "Error occurred while loading logs"
                             }, 
                             original_text="",
                             file_hash=file_hash)

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
    # Run the app with debug but without reloader to avoid watchdog issues
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)
