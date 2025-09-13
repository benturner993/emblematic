from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class DocumentType(str, Enum):
    """Enumeration of possible document types"""
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    BUSINESS = "business"
    ACADEMIC = "academic"
    PERSONAL = "personal"
    GOVERNMENT = "government"
    INSURANCE = "insurance"
    OTHER = "other"

class DocumentClassification(BaseModel):
    """Document classification model"""
    document_type: DocumentType = Field(..., description="The primary type/category of the document")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    keywords: List[str] = Field(..., description="Key terms and concepts found in the document")
    language: str = Field(..., description="Primary language of the document")
    is_structured: bool = Field(..., description="Whether the document has a clear structure/form")

class DocumentSummary(BaseModel):
    """Document summary model"""
    main_topic: str = Field(..., description="The main topic or subject of the document")
    key_points: List[str] = Field(..., description="3-5 key points or findings from the document")
    summary: str = Field(..., description="A concise 2-3 sentence summary of the document content")
    action_items: Optional[List[str]] = Field(None, description="Any action items or next steps mentioned")
    important_dates: Optional[List[str]] = Field(None, description="Important dates mentioned in the document")
    entities: Optional[List[str]] = Field(None, description="Important people, organizations, or locations mentioned")

class DocumentAnalysis(BaseModel):
    """Complete document analysis combining classification and summary"""
    classification: DocumentClassification
    summary: DocumentSummary
    processing_notes: Optional[str] = Field(None, description="Any notes about the processing or analysis")
