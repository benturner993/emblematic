"""
FHIR Services module for AI Emblematic application.
Handles FHIR conversion, validation, and groundedness evaluation.
"""

import openai
import httpx
import json
import logging
from datetime import datetime
from utils import truncate_text, write_judge_output_to_file

class FHIRService:
    """Service class for FHIR-related operations."""
    
    def __init__(self, config):
        """Initialize FHIR service with configuration."""
        self.config = config
        self.openai_config = config.get_openai_config()
        
    def _get_openai_client(self):
        """Get configured OpenAI client."""
        if not self.config.is_openai_configured():
            raise Exception("OpenAI not properly configured")
            
        return openai.AzureOpenAI(
            api_key=self.openai_config['api_key'],
            azure_endpoint=self.openai_config['azure_endpoint'],
            api_version=self.openai_config['api_version'],
            http_client=httpx.Client()
        )
    
    def convert_to_fhir_with_agent(self, text, max_retries=5, feedback_history=None):
        """Convert text to FHIR format using LLM with agent optimization and feedback learning."""
        if not self.config.is_openai_configured():
            return {"error": "OpenAI API key not configured"}
        
        # Store original text for groundedness evaluation
        original_text = text
        
        # Truncate text if too long (OpenAI has token limits)
        text = truncate_text(text, 10000)
        
        # Initialize feedback history if not provided
        if feedback_history is None:
            feedback_history = []
        
        # Initialize detailed logs immediately
        detailed_logs = []
        
        # Track the best attempt across all tries
        best_attempt = None
        best_score = 0
        
        logging.info(f"Starting FHIR conversion with {max_retries} max retries")
        logging.info(f"Text length: {len(text)} characters")
        
        # Base FHIR prompt template
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
                        // ... other patient fields
                    }}
                }}
                // ... other resources
            ]
        }}
        
        Each resource should be properly structured according to FHIR R4 specifications.
        """
        
        for attempt in range(max_retries):
            # Create attempt log
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
            
            # Append attempt log immediately
            detailed_logs.append(attempt_log)
            logging.info(f"Attempt {attempt + 1} log added to detailed_logs. Total logs: {len(detailed_logs)}")
            
            try:
                # Build feedback learning prompt for this attempt
                feedback_section = ""
                if feedback_history:
                    feedback_section = "\\n\\nIMPORTANT FEEDBACK FROM PREVIOUS ATTEMPTS:\\n"
                    for i, feedback in enumerate(feedback_history, 1):
                        feedback_section += f"\\nAttempt {i} Feedback:\\n"
                        feedback_section += f"- Score: {feedback.get('groundedness_score', 'N/A')}/5\\n"
                        feedback_section += f"- Issues: {', '.join(feedback.get('priority_fixes', []))}\\n"
                        feedback_section += f"- Improvement needed: {feedback.get('improvement_feedback', 'N/A')}\\n"
                        if feedback.get('hallucinations_detected'):
                            feedback_section += f"- Hallucinations to avoid: {', '.join(feedback.get('hallucinations_detected', []))}\\n"
                        if feedback.get('missing_information'):
                            feedback_section += f"- Missing info to include: {', '.join(feedback.get('missing_information', []))}\\n"
                    feedback_section += "\\nPlease address these specific issues in your conversion:\\n"
                    logging.info(f"Attempt {attempt + 1}: Including feedback from {len(feedback_history)} previous attempts")
                
                # Build the complete prompt with current feedback
                fhir_prompt = base_fhir_template.format(text=text, feedback_section=feedback_section)
                
                # Initialize Azure OpenAI client
                logging.info(f"Initializing Azure OpenAI client for attempt {attempt + 1}")
                client = self._get_openai_client()
                
                # Log the prompt being sent to FHIR agent
                attempt_log["fhir_conversion"]["prompt"] = fhir_prompt
                attempt_log["fhir_conversion"]["prompt_length"] = len(fhir_prompt)
                
                # FHIR Conversion Agent
                logging.info(f"Calling OpenAI API for attempt {attempt + 1}")
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
                
                # Log the raw response
                fhir_json = fhir_response.choices[0].message.content
                logging.info(f"Received response of length {len(fhir_json)} for attempt {attempt + 1}")
                attempt_log["fhir_conversion"]["raw_response"] = fhir_json
                attempt_log["fhir_conversion"]["response_length"] = len(fhir_json)
                
                # Clean and parse the JSON response
                cleaned_json = self._clean_json_response(fhir_json)
                
                try:
                    fhir_data = json.loads(cleaned_json)
                    attempt_log["fhir_conversion"]["parsed_successfully"] = True
                    attempt_log["fhir_conversion"]["parsed_data"] = fhir_data
                    logging.info(f"Successfully parsed FHIR JSON for attempt {attempt + 1}")
                except json.JSONDecodeError as json_err:
                    attempt_log["fhir_conversion"]["json_error"] = str(json_err)
                    attempt_log["errors"].append(f"JSON decode error: {str(json_err)}")
                    logging.warning(f"Attempt {attempt + 1}: Invalid JSON in FHIR response: {json_err}")
                    if attempt == max_retries - 1:
                        if best_attempt:
                            logging.info(f"JSON parsing failed but returning best attempt with score {best_score}/5")
                            return best_attempt
                        return {"error": f"Failed to parse FHIR JSON after {max_retries} attempts: {str(json_err)}"}
                    continue
                
                # Validate FHIR structure
                validation_result = self.validate_fhir_structure(fhir_data)
                attempt_log["structure_validation"] = validation_result
                
                # Evaluate groundedness
                groundedness_result = self.evaluate_fhir_groundedness(
                    original_text, fhir_data, max_retries=3, 
                    file_hash=None, attempt_number=attempt + 1, 
                    feedback_history=feedback_history
                )
                
                if "error" in groundedness_result:
                    attempt_log["errors"].append(f"Groundedness evaluation error: {groundedness_result['error']}")
                    attempt_log["groundedness_evaluation"]["groundedness_error"] = groundedness_result["error"]
                    if attempt == max_retries - 1:
                        if best_attempt:
                            logging.info(f"Groundedness evaluation failed but returning best attempt with score {best_score}/5")
                            return best_attempt
                        return {
                            "error": f"Groundedness evaluation failed: {groundedness_result['error']}",
                            "fhir_bundle": fhir_data
                        }
                    continue
                
                # Update attempt log with groundedness results
                attempt_log["groundedness_evaluation"].update(groundedness_result)
                
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
    
    def _clean_json_response(self, json_response):
        """Clean JSON response from OpenAI."""
        # Remove markdown formatting if present
        cleaned = json_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        return cleaned.strip()
    
    def evaluate_fhir_groundedness(self, original_text, fhir_bundle, max_retries=3, file_hash=None, attempt_number=1, feedback_history=None):
        """Evaluate FHIR Bundle groundedness against original text using LLM judge."""
        if not self.config.is_openai_configured():
            return {"error": "OpenAI API key not configured"}
        
        # Initialize feedback history if not provided
        if feedback_history is None:
            feedback_history = []
        
        # Build feedback context for judge
        feedback_context = ""
        if feedback_history and attempt_number > 1:
            feedback_context = f"\\n\\nPREVIOUS FEEDBACK HISTORY (for context):\\n"
            feedback_context += f"This is attempt #{attempt_number}. Here's what feedback was given in previous attempts:\\n\\n"
            
            for i, feedback in enumerate(feedback_history, 1):
                feedback_context += f"Previous Attempt {i} Feedback:\\n"
                feedback_context += f"  - Score Given: {feedback.get('groundedness_score', 'N/A')}/5\\n"
                feedback_context += f"  - Issues Identified: {', '.join(feedback.get('priority_fixes', []))}\\n"
                feedback_context += f"  - Guidance Provided: {feedback.get('improvement_feedback', 'N/A')}\\n"
                feedback_context += f"  - Missing Information: {', '.join(feedback.get('missing_information', []))}\\n\\n"
            
            feedback_context += "IMPORTANT: Assess whether this current FHIR Bundle addresses the previous feedback. "
            feedback_context += "If improvements were made based on previous guidance, acknowledge them in your scoring.\\n"
        
        # Store original inputs for logging
        judge_inputs = {
            "original_text": original_text,
            "fhir_json": json.dumps(fhir_bundle, indent=2) if isinstance(fhir_bundle, dict) else str(fhir_bundle),
            "original_text_length": len(original_text),
            "fhir_json_length": len(json.dumps(fhir_bundle, indent=2)) if isinstance(fhir_bundle, dict) else len(str(fhir_bundle)),
            "feedback_history_provided": len(feedback_history),
            "attempt_number": attempt_number
        }
        
        fhir_json = json.dumps(fhir_bundle, indent=2) if isinstance(fhir_bundle, dict) else str(fhir_bundle)
        
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
            "priority_fixes": <list of max 2 most critical issues to address first>
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
                client = self._get_openai_client()
                
                judge_response = client.chat.completions.create(
                    model="gpt-4",
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
                    write_judge_output_to_file(file_hash, judge_json, attempt_number, self.config.TEXT_CACHE_FOLDER)
                
                # Clean and parse the response
                cleaned_json = self._clean_json_response(judge_json)
                
                try:
                    judge_data = json.loads(cleaned_json)
                    judge_attempt_log["parsed_response"] = judge_data
                except json.JSONDecodeError as json_err:
                    logging.warning(f"Judge attempt {attempt + 1}: Invalid JSON response: {json_err}")
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
    
    def validate_fhir_structure(self, fhir_data):
        """Validate FHIR Bundle structure."""
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
                validation_result["errors"].append("Root resource must be a Bundle")
                validation_result["is_valid"] = False
            
            if "type" not in fhir_data:
                validation_result["errors"].append("Bundle must have a type")
                validation_result["is_valid"] = False
            
            if "entry" not in fhir_data:
                validation_result["warnings"].append("Bundle has no entries")
            else:
                entries = fhir_data["entry"]
                if not isinstance(entries, list):
                    validation_result["errors"].append("Bundle entries must be a list")
                    validation_result["is_valid"] = False
                else:
                    validation_result["resource_count"] = len(entries)
                    
                    for i, entry in enumerate(entries):
                        if not isinstance(entry, dict):
                            validation_result["errors"].append(f"Entry {i} must be an object")
                            validation_result["is_valid"] = False
                            continue
                        
                        if "resource" not in entry:
                            validation_result["errors"].append(f"Entry {i} must have a resource")
                            validation_result["is_valid"] = False
                            continue
                        
                        resource = entry["resource"]
                        if not isinstance(resource, dict):
                            validation_result["errors"].append(f"Entry {i} resource must be an object")
                            validation_result["is_valid"] = False
                            continue
                        
                        resource_type = resource.get("resourceType")
                        if not resource_type:
                            validation_result["errors"].append(f"Entry {i} resource must have a resourceType")
                            validation_result["is_valid"] = False
                        else:
                            validation_result["resource_types"].append(resource_type)
            
            # Calculate validation score
            total_checks = 4  # Basic structure checks
            passed_checks = 0
            
            if fhir_data.get("resourceType") == "Bundle":
                passed_checks += 1
            if "type" in fhir_data:
                passed_checks += 1
            if "entry" in fhir_data:
                passed_checks += 1
            if validation_result["resource_count"] > 0:
                passed_checks += 1
            
            validation_result["validation_score"] = (passed_checks / total_checks) * 100
            
            return validation_result
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["is_valid"] = False
            return validation_result
