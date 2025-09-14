from pydantic import BaseModel, Field
from typing import List, Optional, Union
from enum import Enum
from datetime import date

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

# Insurance Underwriting Models

class RiskLevel(str, Enum):
    """Risk level enumeration for underwriting"""
    LOW = "low"
    MODERATE = "moderate" 
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNINSURABLE = "uninsurable"

class ConditionSeverity(str, Enum):
    """Severity levels for medical conditions"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

class TreatmentStatus(str, Enum):
    """Status of medical treatment"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ONGOING = "ongoing"
    STABLE = "stable"
    PROGRESSIVE = "progressive"

class PreExistingCondition(BaseModel):
    """Model for pre-existing medical conditions"""
    condition_name: str = Field(..., description="Name of the medical condition")
    icd_10_code: Optional[str] = Field(None, description="ICD-10 diagnostic code if available")
    severity: Optional[ConditionSeverity] = Field(None, description="Severity of the condition")
    diagnosis_date: Optional[str] = Field(None, description="Date when condition was diagnosed")
    treatment_status: Optional[TreatmentStatus] = Field(None, description="Current treatment status")
    medications: Optional[List[str]] = Field(None, description="Medications for this condition")
    complications: Optional[List[str]] = Field(None, description="Any complications from this condition")
    risk_factors: Optional[List[str]] = Field(None, description="Risk factors associated with condition")

class VitalSigns(BaseModel):
    """Model for vital signs and measurements"""
    blood_pressure_systolic: Optional[int] = Field(None, description="Systolic blood pressure")
    blood_pressure_diastolic: Optional[int] = Field(None, description="Diastolic blood pressure")
    heart_rate: Optional[int] = Field(None, description="Heart rate in BPM")
    weight: Optional[float] = Field(None, description="Weight in pounds or kg")
    height: Optional[str] = Field(None, description="Height (e.g., '5\'10\"' or '175cm')")
    bmi: Optional[float] = Field(None, description="Body Mass Index")
    temperature: Optional[float] = Field(None, description="Body temperature")
    oxygen_saturation: Optional[int] = Field(None, description="Oxygen saturation percentage")

class LabResult(BaseModel):
    """Model for laboratory test results"""
    test_name: str = Field(..., description="Name of the laboratory test")
    result_value: str = Field(..., description="Test result value")
    reference_range: Optional[str] = Field(None, description="Normal reference range")
    units: Optional[str] = Field(None, description="Units of measurement")
    abnormal_flag: Optional[bool] = Field(None, description="Whether result is abnormal")
    test_date: Optional[str] = Field(None, description="Date test was performed")

class Medication(BaseModel):
    """Model for medications"""
    medication_name: str = Field(..., description="Name of the medication")
    dosage: Optional[str] = Field(None, description="Dosage information")
    frequency: Optional[str] = Field(None, description="How often taken")
    route: Optional[str] = Field(None, description="Route of administration (oral, IV, etc.)")
    indication: Optional[str] = Field(None, description="What condition it treats")
    start_date: Optional[str] = Field(None, description="When medication was started")
    duration: Optional[str] = Field(None, description="How long to take medication")

class FamilyHistory(BaseModel):
    """Model for family medical history"""
    relation: str = Field(..., description="Family relation (mother, father, sibling, etc.)")
    conditions: List[str] = Field(..., description="Medical conditions in family member")
    age_at_diagnosis: Optional[List[int]] = Field(None, description="Ages when conditions were diagnosed")
    deceased: Optional[bool] = Field(None, description="Whether family member is deceased")
    cause_of_death: Optional[str] = Field(None, description="Cause of death if applicable")

class LifestyleFactors(BaseModel):
    """Model for lifestyle and behavioral factors"""
    smoking_status: Optional[str] = Field(None, description="Smoking status (never, former, current)")
    alcohol_consumption: Optional[str] = Field(None, description="Alcohol consumption pattern")
    exercise_frequency: Optional[str] = Field(None, description="Exercise frequency and type")
    diet_type: Optional[str] = Field(None, description="Dietary patterns or restrictions")
    occupation: Optional[str] = Field(None, description="Current occupation")
    occupational_hazards: Optional[List[str]] = Field(None, description="Workplace health hazards")
    travel_history: Optional[List[str]] = Field(None, description="Recent travel to high-risk areas")

class InsuranceUnderwritingData(BaseModel):
    """Comprehensive model for insurance underwriting health information"""
    
    # Patient Demographics
    patient_age: Optional[int] = Field(None, description="Patient's age")
    patient_gender: Optional[str] = Field(None, description="Patient's gender")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    
    # Pre-existing Conditions
    pre_existing_conditions: List[PreExistingCondition] = Field(default=[], description="List of pre-existing medical conditions")
    
    # Current Health Status
    current_symptoms: Optional[List[str]] = Field(None, description="Current symptoms or complaints")
    vital_signs: Optional[VitalSigns] = Field(None, description="Current vital signs and measurements")
    
    # Medical History
    surgical_history: Optional[List[str]] = Field(None, description="Previous surgeries and procedures")
    hospitalization_history: Optional[List[str]] = Field(None, description="Previous hospitalizations")
    emergency_room_visits: Optional[List[str]] = Field(None, description="Recent ER visits")
    
    # Medications and Treatments
    current_medications: List[Medication] = Field(default=[], description="Current medications")
    allergies: Optional[List[str]] = Field(None, description="Known allergies and adverse reactions")
    
    # Laboratory and Diagnostic Results
    lab_results: List[LabResult] = Field(default=[], description="Recent laboratory test results")
    imaging_results: Optional[List[str]] = Field(None, description="Imaging study results (X-ray, MRI, etc.)")
    diagnostic_tests: Optional[List[str]] = Field(None, description="Other diagnostic test results")
    
    # Family and Genetic History
    family_history: List[FamilyHistory] = Field(default=[], description="Family medical history")
    genetic_conditions: Optional[List[str]] = Field(None, description="Known genetic conditions or predispositions")
    
    # Lifestyle and Risk Factors
    lifestyle_factors: Optional[LifestyleFactors] = Field(None, description="Lifestyle and behavioral factors")
    
    # Risk Assessment
    overall_risk_level: Optional[RiskLevel] = Field(None, description="Overall assessed risk level")
    risk_factors: Optional[List[str]] = Field(None, description="Identified risk factors")
    protective_factors: Optional[List[str]] = Field(None, description="Factors that reduce risk")
    
    # Underwriting Notes
    underwriting_notes: Optional[str] = Field(None, description="Additional notes for underwriters")
    requires_medical_exam: Optional[bool] = Field(None, description="Whether medical exam is required")
    recommended_exclusions: Optional[List[str]] = Field(None, description="Recommended policy exclusions")
    premium_adjustments: Optional[str] = Field(None, description="Recommended premium adjustments")

class DocumentAnalysis(BaseModel):
    """Complete document analysis combining classification and summary"""
    classification: DocumentClassification
    summary: DocumentSummary
    processing_notes: Optional[str] = Field(None, description="Any notes about the processing or analysis")

class InsuranceUnderwritingAnalysis(BaseModel):
    """Complete insurance underwriting analysis"""
    classification: DocumentClassification
    summary: DocumentSummary
    underwriting_data: InsuranceUnderwritingData
    processing_notes: Optional[str] = Field(None, description="Any notes about the processing or analysis")
