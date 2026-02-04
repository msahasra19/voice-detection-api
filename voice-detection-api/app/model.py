from typing import Any, Dict
import numpy as np
from .schemas import ClassificationResult, ConfidenceLevel, VoiceResponse, AudioQuality
from .analysis import (
    analyze_audio_quality,
    extract_features_and_explain,
    detect_language_heuristic,
    analyze_segments
)

def detect_voice(audio: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrate the detection process.
    """
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    # Ensure we have a NumPy array (librosa/soundfile usually return float64 or float32)
    y = np.asarray(waveform, dtype=float)
    if y.ndim > 1:
        # Mix down to mono
        y = np.mean(y, axis=1)

    # 1. Quality Analysis
    quality_data = analyze_audio_quality(y, sample_rate)

    # 2. Main Detection & Explainability
    reasons, ai_score = extract_features_and_explain(y, sample_rate)

    # 3. Decision Logic
    is_ai = ai_score > 0.5
    classification = ClassificationResult.AI_GENERATED if is_ai else ClassificationResult.HUMAN
    
    # Map score to High/Medium/Low confidence
    # If score is extreme (0.0 or 1.0), confidence is High. If near 0.5, Low.
    # We define confidence as distance from 0.5 * 2
    raw_confidence = abs(ai_score - 0.5) * 2.0
    # Provide a minimum floor
    raw_confidence = max(0.1, raw_confidence)
    
    if raw_confidence > 0.8:
        conf_level = ConfidenceLevel.HIGH
    elif raw_confidence > 0.4:
        conf_level = ConfidenceLevel.MEDIUM
    else:
        conf_level = ConfidenceLevel.LOW

    # 4. Language Detection
    language = detect_language_heuristic(y, sample_rate)

    # 5. Segment Analysis
    segments = analyze_segments(y, sample_rate, ai_score)

    return {
        "classification": classification,
        "confidence_score": float(raw_confidence),
        "confidence_level": conf_level,
        "deepfake_risk_score": float(ai_score), # Using AI probability as risk score
        "detected_language": language,
        "audio_quality": quality_data,
        "explainability": reasons,
        "segments": segments
    }

