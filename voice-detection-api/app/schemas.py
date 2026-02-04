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
    TAMIL = "Tamil"
    ENGLISH = "English"
    HINDI = "Hindi"
    MALAYALAM = "Malayalam"
    TELUGU = "Telugu"

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
    audio_data: str = Field(..., description="Base64 encoded MP3 audio data")

class VoiceResponse(BaseModel):
    classification: ClassificationResult
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    deepfake_risk_score: float = Field(..., ge=0.0, le=1.0)
    detected_language: SupportedLanguage
    audio_quality: AudioQuality
    explainability: List[str] = Field(..., description="Human-readable reasons for classification")
    segments: List[SegmentAnalysis] = Field(default_factory=list)
