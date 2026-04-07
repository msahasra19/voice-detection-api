from typing import Dict, Any, List
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field

class ClassificationResult(str, Enum):
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"

class ConfidenceLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class SupportedLanguage(str, Enum):
    ENGLISH = "English"
    TAMIL = "Tamil"
    HINDI = "Hindi"
    TELUGU = "Telugu"
    SPANISH = "Spanish"

class AudioQualityScore(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class AudioQuality(BaseModel):
    snr: float = Field(..., description="Signal-to-Noise Ratio (dB)")
    clipping_detected: bool = Field(..., description="Whether audio clipping is detected")
    quality_check: AudioQualityScore = Field(..., description="Overall quality assessment")

class SegmentAnalysis(BaseModel):
    start_time: float
    end_time: float
    label: ClassificationResult
    confidence: float

class VoiceRequest(BaseModel):
    audio_data: Optional[str] = Field(None, description="Base64 encoded MP3 audio data")
    audio_url: Optional[str] = Field(None, description="URL pointing to an MP3 voice sample")
    audio_base64: Optional[str] = Field(None, description="Alternative field for Base64 data")



class ExplainabilityData(BaseModel):
    shap_results: Dict[str, float] = Field(..., description="SHAP feature importances")
    lime_results: Dict[str, float] = Field(..., description="LIME local explanations")
    gradcam_regions: List[Dict[str, Any]] = Field(..., description="High saliency regions in spectrogram")
    attention_segments: List[Dict[str, Any]] = Field(..., description="Critical time chunks affecting prediction")
    counterfactual_explanation: str = Field(..., description="Minimal change needed to flip prediction")

class VoiceResponse(BaseModel):
    classification: ClassificationResult
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    human_confidence: float = Field(..., ge=0.0, le=1.0)
    ai_confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    deepfake_risk_score: float = Field(..., ge=0.0, le=1.0)
    detected_language: SupportedLanguage
    audio_quality: AudioQuality
    explainability: ExplainabilityData
    segments: List[SegmentAnalysis] = Field(default_factory=list)
