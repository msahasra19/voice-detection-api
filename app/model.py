import time
from typing import Any, Dict
import numpy as np
from .schemas import (
    ClassificationResult,
    ConfidenceLevel,
    VoiceResponse
)
from .analysis import (
    preprocess_audio,
    extract_ml_features,
    analyze_acoustic_indicators,
    predict_random_forest,
    perform_fusion,
    detect_language_and_transcript,
    analyze_timeline_segments
)

def detect_voice(audio: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrate dual-branch voice analysis system:
    1. Preprocessing & Normalization
    2. ML Feature Extraction & Supervised Random Forest Classifier
    3. Interpretable Signal Acoustics (SNR, Pitch/F0, Silence dynamics)
    4. Calibrated Decision Fusion (combines ML and physical acoustic evidence)
    5. Pretrained Faster Whisper Multilingual Language Identification
    6. Second-by-second short window analysis for suspicious segment localization
    7. Spliced & mixed-speech partial manipulation aggregation
    """
    t_start = time.perf_counter()

    raw_waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    # 1. Preprocessing
    y, sr = preprocess_audio(raw_waveform, sample_rate, target_sr=16000)

    # 2. Interpretable Acoustic Indicators
    acoustic_indicators, acoustic_raw, acoustic_explanations = analyze_acoustic_indicators(y, sr)

    # 3. Supervised Random Forest ML Classifier
    features_ml = extract_ml_features(y, sr)
    ml_prob, ml_explanations = predict_random_forest(features_ml)

    # 4. Calibrated Decision Fusion
    fusion_analysis, fusion_explanations = perform_fusion(ml_prob, acoustic_raw)
    final_risk = fusion_analysis.fusion_risk_score

    # 5. Pretrained Faster Whisper Language Branch
    detected_lang, lang_confidence, sample_transcript = detect_language_and_transcript(y, sr)

    # 6. Second-by-Second Window Timeline Analysis
    segments = analyze_timeline_segments(y, sr, fusion_analysis)

    # 7. Spliced / Mixed Speech Analysis
    ai_segments = [s for s in segments if s.label == ClassificationResult.AI_GENERATED]
    human_segments = [s for s in segments if s.label == ClassificationResult.HUMAN]
    
    total_segments = max(1, len(segments))
    ai_segment_ratio = len(ai_segments) / total_segments
    is_spliced = (len(ai_segments) >= 2 and len(human_segments) >= 2)

    if is_spliced and ai_segment_ratio >= 0.30:
        # Spliced / partially manipulated audio with distinct AI sections
        is_ai = True
        segment_avg_risk = float(np.mean([s.risk_score for s in ai_segments]))
        final_risk = max(final_risk, segment_avg_risk)
        classification = ClassificationResult.AI_GENERATED
    elif final_risk >= 0.50 or ai_segment_ratio >= 0.40:
        is_ai = True
        classification = ClassificationResult.AI_GENERATED
    else:
        is_ai = False
        classification = ClassificationResult.HUMAN

    # Confidence in assigned verdict
    if is_ai:
        confidence_score = float(np.clip(final_risk, 0.51, 0.99))
    else:
        confidence_score = float(np.clip(1.0 - final_risk, 0.51, 0.99))

    if confidence_score >= 0.80:
        conf_level = ConfidenceLevel.HIGH
    elif confidence_score >= 0.60:
        conf_level = ConfidenceLevel.MEDIUM
    else:
        conf_level = ConfidenceLevel.LOW

    # 8. Compile Comprehensive Forensic Explainability Report
    all_explanations = []

    if is_spliced and is_ai:
        ai_time_spans = f"{ai_segments[0].start_time:.1f}s - {ai_segments[-1].end_time:.1f}s"
        all_explanations.append(f"Spliced / Mixed Audio Detected: Found AI synthetic speech in {len(ai_segments)} of {total_segments} seconds ({ai_time_spans}), combined with natural human speech.")
    elif is_ai:
        all_explanations.append(f"AI/Synthetic Speech Detected with {final_risk*100:.1f}% risk score.")
    else:
        all_explanations.append(f"Bona Fide Human Voice Verified with {(1.0-final_risk)*100:.1f}% authenticity score.")

    all_explanations.extend(fusion_explanations)
    all_explanations.extend(ml_explanations)
    all_explanations.extend(acoustic_explanations)

    elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    return {
        "classification": classification,
        "confidence_score": round(confidence_score, 3),
        "confidence_level": conf_level,
        "deepfake_risk_score": round(final_risk, 3),
        "detected_language": detected_lang,
        "language_confidence": round(lang_confidence, 3),
        "sample_transcript": sample_transcript,
        "acoustic_indicators": acoustic_indicators,
        "fusion_analysis": fusion_analysis,
        "explainability": all_explanations,
        "segments": segments,
        "processing_time_ms": elapsed_ms
    }
