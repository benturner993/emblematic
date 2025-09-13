"""
AI Services module for AI Emblematic application.
Handles OpenAI API interactions, document analysis, and medical document classification.
"""

import openai
import httpx
import json
import logging
from datetime import datetime
from models import DocumentAnalysis, DocumentType
from utils import truncate_text

class AIService:
    """Service class for AI-related operations."""
    
    def __init__(self, config):
        """Initialize AI service with configuration."""
        self.config = config
        self.openai_config = config.get_openai_config()
    
    def _map_document_type(self, ai_document_type):
        """Map AI-generated document type to valid enum values."""
        ai_type_lower = ai_document_type.lower().strip()
        
        # Direct matches
        valid_types = {
            "medical": DocumentType.MEDICAL,
            "legal": DocumentType.LEGAL,
            "financial": DocumentType.FINANCIAL,
            "technical": DocumentType.TECHNICAL,
            "business": DocumentType.BUSINESS,
            "academic": DocumentType.ACADEMIC,
            "personal": DocumentType.PERSONAL,
            "government": DocumentType.GOVERNMENT,
            "insurance": DocumentType.INSURANCE,
            "other": DocumentType.OTHER
        }
        
        # Direct match
        if ai_type_lower in valid_types:
            return valid_types[ai_type_lower]
        
        # Fuzzy matching for common variations
        medical_keywords = ["medical", "health", "patient", "clinical", "hospital", "doctor", "prescription", "lab", "diagnosis", "treatment"]
        legal_keywords = ["legal", "contract", "agreement", "court", "law", "litigation", "attorney"]
        financial_keywords = ["financial", "bank", "invoice", "receipt", "payment", "accounting", "budget", "tax"]
        technical_keywords = ["technical", "manual", "specification", "documentation", "engineering", "software"]
        business_keywords = ["business", "corporate", "company", "meeting", "proposal", "report", "memo"]
        academic_keywords = ["academic", "research", "paper", "study", "journal", "thesis", "publication"]
        insurance_keywords = ["insurance", "policy", "claim", "coverage", "premium", "benefit"]
        government_keywords = ["government", "official", "public", "regulatory", "compliance", "permit"]
        
        # Check for keyword matches
        for keyword in medical_keywords:
            if keyword in ai_type_lower:
                return DocumentType.MEDICAL
        
        for keyword in legal_keywords:
            if keyword in ai_type_lower:
                return DocumentType.LEGAL
                
        for keyword in financial_keywords:
            if keyword in ai_type_lower:
                return DocumentType.FINANCIAL
                
        for keyword in technical_keywords:
            if keyword in ai_type_lower:
                return DocumentType.TECHNICAL
                
        for keyword in business_keywords:
            if keyword in ai_type_lower:
                return DocumentType.BUSINESS
                
        for keyword in academic_keywords:
            if keyword in ai_type_lower:
                return DocumentType.ACADEMIC
                
        for keyword in insurance_keywords:
            if keyword in ai_type_lower:
                return DocumentType.INSURANCE
                
        for keyword in government_keywords:
            if keyword in ai_type_lower:
                return DocumentType.GOVERNMENT
        
        # Default to "other" if no match found
        logging.warning(f"Unknown document type '{ai_document_type}', defaulting to 'other'")
        return DocumentType.OTHER
        
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
    
    def is_medical_document(self, text):
        """Check if the document is medical using LLM."""
        if not self.config.is_openai_configured():
            return False
        
        # Truncate text for the medical check
        check_text = truncate_text(text, 3000)
        
        prompt = f"""
        Analyze the following text and determine if it is a medical document.
        
        Text: {check_text}
        
        Respond with only "YES" if this is a medical document (contains medical terms, patient information, diagnoses, treatments, medications, etc.) or "NO" if it is not medical.
        """
        
        try:
            client = self._get_openai_client()
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical document classifier. Respond with only YES or NO."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip().upper()
            return result == "YES"
            
        except Exception as e:
            logging.error(f"Error in medical document classification: {e}")
            return False
    
    def analyze_document_with_ai(self, text, max_retries=3):
        """Analyze document using AI with structured or freeform analysis."""
        if not self.config.is_openai_configured():
            return {"error": "OpenAI API key not configured"}
        
        # Truncate text if too long
        text = truncate_text(text, 8000)
        
        # Check if document is medical
        is_medical = self.is_medical_document(text)
        
        if is_medical:
            return self._analyze_medical_document(text, max_retries)
        else:
            return self._analyze_freeform_document(text, max_retries)
    
    def _analyze_medical_document(self, text, max_retries):
        """Analyze medical document with structured Pydantic model."""
        prompt = f"""
        Analyze the following medical document and provide a structured analysis.
        
        Document text: {text}
        
        IMPORTANT: For document_type, you must use EXACTLY one of these values:
        - "medical" (for medical reports, prescriptions, lab results, patient records, etc.)
        - "legal" (for legal documents)
        - "financial" (for financial documents)
        - "technical" (for technical documentation)
        - "business" (for business documents)
        - "academic" (for academic papers)
        - "personal" (for personal documents)
        - "government" (for government documents)
        - "insurance" (for insurance documents)
        - "other" (for documents that don't fit other categories)
        
        Please provide the analysis in the following JSON format:
        {{
            "classification": {{
                "document_type": "medical",
                "confidence": 0.95,
                "language": "English",
                "is_structured": true,
                "keywords": ["patient", "diagnosis", "treatment", "medication"]
            }},
            "summary": {{
                "main_topic": "Patient medical consultation",
                "summary": "Brief summary of the medical document content",
                "key_points": ["Key point 1", "Key point 2", "Key point 3"],
                "action_items": ["Follow-up appointment", "Take prescribed medication"],
                "important_dates": ["2025-01-15", "2025-02-01"],
                "entities": ["Dr. Smith", "General Hospital", "Patient Name"]
            }}
        }}
        
        Respond with ONLY the JSON object, no additional text or formatting.
        """
        
        for attempt in range(max_retries):
            try:
                client = self._get_openai_client()
                
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
                
                response_text = response.choices[0].message.content
                
                # Clean the response
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                # Parse JSON and validate with Pydantic
                try:
                    parsed_data = json.loads(response_text.strip())
                    
                    # Fix document_type if it's not a valid enum value
                    if "classification" in parsed_data and "document_type" in parsed_data["classification"]:
                        original_type = parsed_data["classification"]["document_type"]
                        mapped_type = self._map_document_type(original_type)
                        parsed_data["classification"]["document_type"] = mapped_type.value
                        
                        if original_type.lower() != mapped_type.value:
                            logging.info(f"Mapped document type '{original_type}' to '{mapped_type.value}'")
                    
                    # Validate with Pydantic model
                    validated_analysis = DocumentAnalysis(**parsed_data)
                    
                    # Return the validated data as dict
                    result = validated_analysis.dict()
                    result["is_medical"] = True
                    return result
                    
                except json.JSONDecodeError as json_err:
                    logging.warning(f"Attempt {attempt + 1}: JSON parsing error: {json_err}")
                    if attempt == max_retries - 1:
                        return {"error": f"Failed to parse AI response as JSON: {json_err}"}
                    continue
                    
                except Exception as validation_err:
                    logging.warning(f"Attempt {attempt + 1}: Pydantic validation error: {validation_err}")
                    if attempt == max_retries - 1:
                        return {"error": f"AI response validation failed: {validation_err}"}
                    continue
                    
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}: OpenAI API error: {e}")
                if attempt == max_retries - 1:
                    return {"error": f"OpenAI API error after {max_retries} attempts: {str(e)}"}
                continue
        
        return {"error": "Unexpected error in document analysis"}
    
    def _analyze_freeform_document(self, text, max_retries):
        """Analyze non-medical document with freeform analysis."""
        prompt = f"""
        Analyze the following document and provide a comprehensive analysis.
        
        Document text: {text}
        
        Please provide:
        1. Document type and classification
        2. Main topics and themes
        3. Key information and insights
        4. Summary of content
        5. Any notable patterns or structure
        
        Provide a detailed but concise analysis.
        """
        
        for attempt in range(max_retries):
            try:
                client = self._get_openai_client()
                
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert document analyst. Provide comprehensive, structured analysis of documents. Be thorough but concise."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.2
                )
                
                analysis_text = response.choices[0].message.content
                
                return {
                    "is_medical": False,
                    "freeform_analysis": analysis_text
                }
                
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}: OpenAI API error: {e}")
                if attempt == max_retries - 1:
                    return {"error": f"OpenAI API error after {max_retries} attempts: {str(e)}"}
                continue
        
        return {"error": "Unexpected error in document analysis"}
