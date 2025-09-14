"""
Patient Email Module for AI Emblematic
=====================================

This module handles patient communication through a chat interface that:
1. Collects patient information through structured questions
2. Generates professional patient letters using AI
3. Provides doctors with copy-ready communication templates

Author: AI Emblematic System
Created: September 2025
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import openai
import httpx

class PatientInfo(BaseModel):
    """Model for storing patient information collected through chat."""
    patient_name: str = Field(..., description="Patient's full name")
    doctors_practice: str = Field(..., description="Doctor's practice name and location")
    reason_for_medication: str = Field(..., description="Medical reason for prescribed medication")
    medication_instructions: str = Field(..., description="Detailed usage instructions for medication")
    additional_notes: Optional[str] = Field(None, description="Any additional notes or concerns")
    consultation_date: Optional[str] = Field(None, description="Date of consultation")
    doctor_name: Optional[str] = Field(None, description="Prescribing doctor's name")

class ChatMessage(BaseModel):
    """Model for chat messages in the patient communication flow."""
    role: str = Field(..., description="Role: 'system', 'assistant', or 'user'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)

class PatientEmailService:
    """Service class for handling patient email generation and chat interface."""
    
    def __init__(self, config):
        """Initialize the patient email service with configuration."""
        self.config = config
        self.openai_config = config.get_openai_config()
        
        # Define the structured question flow
        self.question_flow = [
            {
                "id": "patient_name",
                "question": "Hello! I'm here to help you generate a patient letter. What is the patient's full name?",
                "field": "patient_name",
                "validation": lambda x: len(x.strip()) > 0,
                "error_msg": "Please provide a valid patient name."
            },
            {
                "id": "doctors_practice",
                "question": "Thank you! What is the name and location of your doctor's practice? (e.g., 'Dr. Smith's Family Medicine, London')",
                "field": "doctors_practice", 
                "validation": lambda x: len(x.strip()) > 0,
                "error_msg": "Please provide the doctor's practice information."
            },
            {
                "id": "reason_for_medication",
                "question": "What is the medical reason or condition for which the medication is being prescribed?",
                "field": "reason_for_medication",
                "validation": lambda x: len(x.strip()) > 0,
                "error_msg": "Please provide the medical reason for the medication."
            },
            {
                "id": "medication_instructions",
                "question": "Please provide the detailed medication instructions including dosage, frequency, and any special instructions for usage:",
                "field": "medication_instructions",
                "validation": lambda x: len(x.strip()) > 0,
                "error_msg": "Please provide detailed medication instructions."
            },
            {
                "id": "additional_info",
                "question": "Are there any additional notes, concerns, or information you'd like to include in the patient letter? (Optional - you can type 'none' if not applicable)",
                "field": "additional_notes",
                "validation": lambda x: True,  # Optional field
                "error_msg": None
            }
        ]
    
    def _get_openai_client(self):
        """Get configured OpenAI client."""
        if not self.config.is_openai_configured():
            raise Exception("OpenAI not properly configured")
        
        return openai.AzureOpenAI(
            api_key=self.openai_config['api_key'],
            api_version=self.openai_config['api_version'],
            azure_endpoint=self.openai_config['azure_endpoint'],
            http_client=httpx.Client()
        )
    
    def start_chat_session(self) -> Dict:
        """Start a new patient email chat session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        initial_message = {
            "role": "assistant",
            "content": "👋 Welcome to the Patient Letter Generator! I'll help you create a professional patient letter by asking a few questions about the patient and their medication. Let's get started!",
            "timestamp": datetime.now().isoformat()
        }
        
        # Get the first question
        first_question = self.question_flow[0]
        question_message = {
            "role": "assistant", 
            "content": first_question["question"],
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "session_id": session_id,
            "messages": [initial_message, question_message],
            "current_step": 0,
            "patient_info": {},
            "completed": False
        }
    
    def process_chat_response(self, session_data: Dict, user_response: str) -> Dict:
        """Process user response and return next question or completion status."""
        current_step = session_data.get("current_step", 0)
        patient_info = session_data.get("patient_info", {})
        messages = session_data.get("messages", [])
        
        # Add user response to messages
        user_message = {
            "role": "user",
            "content": user_response,
            "timestamp": datetime.now().isoformat()
        }
        messages.append(user_message)
        
        # Get current question configuration
        if current_step >= len(self.question_flow):
            return {**session_data, "completed": True, "messages": messages}
        
        current_question = self.question_flow[current_step]
        
        # Validate response
        if not current_question["validation"](user_response):
            if current_question["error_msg"]:
                error_message = {
                    "role": "assistant",
                    "content": f"❌ {current_question['error_msg']} Please try again.",
                    "timestamp": datetime.now().isoformat()
                }
                messages.append(error_message)
                return {
                    **session_data,
                    "messages": messages
                }
        
        # Store the validated response
        field_name = current_question["field"]
        if user_response.lower().strip() == "none" and field_name == "additional_notes":
            patient_info[field_name] = None
        else:
            patient_info[field_name] = user_response.strip()
        
        # Move to next question
        next_step = current_step + 1
        
        if next_step >= len(self.question_flow):
            # All questions completed - automatically generate the letter
            completion_message = {
                "role": "assistant",
                "content": "✅ Thank you! I have all the information needed. Generating your professional patient letter now...",
                "timestamp": datetime.now().isoformat()
            }
            messages.append(completion_message)
            
            return {
                **session_data,
                "current_step": next_step,
                "patient_info": patient_info,
                "messages": messages,
                "completed": True,
                "auto_generate": True  # Signal to frontend to automatically generate letter
            }
        else:
            # Ask next question
            next_question = self.question_flow[next_step]
            question_message = {
                "role": "assistant",
                "content": next_question["question"],
                "timestamp": datetime.now().isoformat()
            }
            messages.append(question_message)
            
            return {
                **session_data,
                "current_step": next_step,
                "patient_info": patient_info,
                "messages": messages,
                "completed": False
            }
    
    def generate_patient_letter(self, patient_info: Dict, document_context: Optional[str] = None) -> Dict:
        """Generate a professional patient letter using AI based on collected information."""
        try:
            # Prepare comprehensive context from the document
            context_section = ""
            if document_context:
                # Use more of the document context for better letter generation
                context_section = f"""
            
            IMPORTANT - Use this medical document context to inform the letter:
            ---
            {document_context[:4000]}
            ---
            
            Please use relevant information from the above document to enhance the letter content, including:
            - Specific medical findings or test results mentioned
            - Treatment history or previous interventions
            - Clinical observations that support the current treatment plan
            - Any relevant patient history or background information
            """
            
            prompt = f"""
            You are an experienced medical professional writing a comprehensive patient letter. Generate a professional patient communication letter based on the following information:
            
            PATIENT INFORMATION:
            - Patient Name: {patient_info.get('patient_name', 'N/A')}
            - Doctor's Practice: {patient_info.get('doctors_practice', 'N/A')}
            - Medical Reason/Condition: {patient_info.get('reason_for_medication', 'N/A')}
            - Medication Instructions: {patient_info.get('medication_instructions', 'N/A')}
            - Additional Notes: {patient_info.get('additional_notes', 'None provided')}
            {context_section}
            
            LETTER REQUIREMENTS:
            1. Professional medical letter format with proper date and letterhead structure
            2. Clear, compassionate explanation of the patient's medical condition
            3. Detailed medication instructions with dosage, frequency, and administration guidelines
            4. Important safety information and potential side effects
            5. Follow-up care instructions and when to contact the practice
            6. Professional yet patient-friendly language
            7. Proper medical terminology where appropriate
            8. Clear next steps and expectations
            
            The letter should serve multiple purposes:
            ✓ Patient education and understanding
            ✓ Medical record documentation
            ✓ Insurance and pharmacy reference
            ✓ Continuity of care communication
            
            Format the letter with:
            - Professional letterhead format
            - Proper date and addressing
            - Clear section headings where appropriate
            - Professional closing and signature line
            - Contact information for questions
            
            Generate a comprehensive, professional letter that demonstrates medical expertise while being accessible to the patient.
            """
            
            client = self._get_openai_client()
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an experienced medical professional specializing in patient communication. Generate clear, professional, and compassionate patient letters that are medically accurate and easy to understand."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            letter_content = response.choices[0].message.content
            
            # Generate a summary of the letter
            summary_prompt = f"""
            Based on the following patient letter, provide a brief summary (2-3 sentences) of the key points:
            
            {letter_content}
            """
            
            summary_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Provide a concise summary of the patient letter's key points."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=150,
                temperature=0.2
            )
            
            summary = summary_response.choices[0].message.content
            
            return {
                "success": True,
                "letter_content": letter_content,
                "summary": summary,
                "patient_name": patient_info.get('patient_name'),
                "generated_date": datetime.now().isoformat(),
                "patient_info": patient_info
            }
            
        except Exception as e:
            logging.error(f"Error generating patient letter: {e}")
            return {
                "success": False,
                "error": f"Failed to generate patient letter: {str(e)}",
                "patient_info": patient_info
            }
    
    def get_chat_history_summary(self, messages: List[Dict]) -> str:
        """Generate a summary of the chat conversation for reference."""
        try:
            conversation_text = "\n".join([
                f"{msg['role'].title()}: {msg['content']}" 
                for msg in messages[-10:]  # Last 10 messages
            ])
            
            client = self._get_openai_client()
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize this patient information gathering conversation in 2-3 sentences."
                    },
                    {"role": "user", "content": f"Conversation:\n{conversation_text}"}
                ],
                max_tokens=100,
                temperature=0.2
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error generating chat summary: {e}")
            return "Chat conversation completed successfully."

# Utility functions for integration with main application

def create_patient_email_service(config):
    """Factory function to create PatientEmailService instance."""
    return PatientEmailService(config)

def validate_patient_info(patient_info: Dict) -> bool:
    """Validate that required patient information is present."""
    required_fields = ['patient_name', 'doctors_practice', 'reason_for_medication', 'medication_instructions']
    return all(field in patient_info and patient_info[field] for field in required_fields)
