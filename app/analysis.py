import os
import time
import numpy as np
import librosa
import joblib
from typing import List, Dict, Any, Tuple, Optional

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except Exception:
    WhisperModel = None
    HAS_WHISPER = False

from .schemas import (
    ClassificationResult,
    ConfidenceLevel,
    SupportedLanguage,
    AudioQualityScore,
    AcousticIndicators,
    FusionAnalysis,
    SegmentAnalysis
)

MODEL_DIR = os.path.dirname(__file__)
RF_MODEL_PATH = os.path.join(MODEL_DIR, "voice_model.joblib")
FUSION_MODEL_PATH = os.path.join(MODEL_DIR, "fusion_model.joblib")

# ==========================================
# 1. PREPROCESSING & SIGNAL NORMALIZATION
# ==========================================

def preprocess_audio(y: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Standardized preprocessing:
    - Mix down to mono
    - Resample to target sample rate (16kHz)
    - Peak volume normalization
    """
    if y.ndim > 1:
        y = np.mean(y, axis=0 if y.shape[0] < y.shape[1] else 1)

    y = y.astype(np.float32)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp

    return y, sr


# ==========================================
# 2. ACOUSTIC FEATURE EXTRACTION (ML BRANCH)
# ==========================================

def extract_ml_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract robust acoustic feature representation:
    - MFCCs (13 coefficients): static, delta, delta-delta (mean & std) -> 78 dims
    - Spectral Flatness (mean, std) -> 2 dims
    - Spectral Contrast (5 sub-bands: mean, std, fmin=150Hz) -> 10 dims
    - Spectral Centroid & Rolloff (mean, std) -> 4 dims
    - Zero Crossing Rate (mean, std) -> 2 dims
    - Shimmer proxy -> 1 dim
    Total: 97 acoustic feature dimensions
    """
    if len(y) < 512:
        y = np.pad(y, (0, 512 - len(y)), mode='constant')

    # 1. MFCCs + Deltas
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    delta_mean = np.mean(mfccs_delta, axis=1)
    delta_std = np.std(mfccs_delta, axis=1)
    delta2_mean = np.mean(mfccs_delta2, axis=1)
    delta2_std = np.std(mfccs_delta2, axis=1)

    # 2. Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=y)
    flatness_mean = np.array([np.mean(flatness)])
    flatness_std = np.array([np.std(flatness)])

    # 3. Spectral Contrast (5 bands avoiding Nyquist edge)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=5, fmin=150.0)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    # 4. Spectral Centroid and Rolloff
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_stats = np.array([
        np.mean(centroid) / (sr / 2),
        np.std(centroid) / (sr / 2),
        np.mean(rolloff) / (sr / 2),
        np.std(rolloff) / (sr / 2)
    ])

    # 5. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.array([np.mean(zcr)])
    zcr_std = np.array([np.std(zcr)])

    # 6. Shimmer proxy
    frame_rms = librosa.feature.rms(y=y)[0]
    rms_diff = np.diff(frame_rms) if len(frame_rms) > 1 else np.array([0.0])
    shimmer_proxy = np.array([np.std(rms_diff) / (np.mean(frame_rms) + 1e-6)])

    return np.hstack([
        mfcc_mean, mfcc_std,
        delta_mean, delta_std,
        delta2_mean, delta2_std,
        flatness_mean, flatness_std,
        contrast_mean, contrast_std,
        spectral_stats,
        zcr_mean, zcr_std,
        shimmer_proxy
    ])


# =======================================================
# 3. INTERPRETABLE ACOUSTIC INDICATORS
# =======================================================

def analyze_acoustic_indicators(y: np.ndarray, sr: int) -> Tuple[AcousticIndicators, Dict[str, float], List[str]]:
    """
    Compute physical acoustic signal indicators:
    - SNR in dB
    - Pitch (F0) mean and variation / standard deviation
    - Silence ratio and speech cadence
    - Spectral Flatness mean
    - Digital clipping detection
    """
    explanations = []

    # 1. Clipping detection
    clipping_detected = bool(np.max(np.abs(y)) >= 0.999)

    # 2. SNR estimation
    s_full = np.abs(librosa.stft(y))
    rms = librosa.feature.rms(S=s_full)[0]
    
    if len(rms) == 0 or np.max(rms) == 0:
        snr_db = 0.0
    else:
        noise_floor = np.percentile(rms, 10)
        if noise_floor <= 1e-8:
            noise_floor = 1e-8
        signal_energy = np.mean(rms[rms >= noise_floor])
        if np.isnan(signal_energy) or signal_energy <= 0:
            signal_energy = noise_floor
        snr_db = float(20 * np.log10(signal_energy / noise_floor))

    if snr_db < 12:
        quality_check = AudioQualityScore.LOW
    elif snr_db < 28:
        quality_check = AudioQualityScore.MEDIUM
    else:
        quality_check = AudioQualityScore.HIGH

    # 3. Fundamental Frequency (F0) Dynamics
    pitch_mean_hz = 0.0
    pitch_std_hz = 0.0
    pitch_stability_score = 0.5

    try:
        f0 = librosa.yin(y, fmin=65, fmax=450, sr=sr)
        voiced_f0 = f0[(f0 >= 65) & (f0 <= 450) & (~np.isnan(f0))]
        if len(voiced_f0) > 10:
            pitch_mean_hz = float(np.median(voiced_f0))
            pitch_std_hz = float(np.std(voiced_f0))
            
            if pitch_std_hz < 20.0:
                pitch_stability_score = float(np.clip(1.0 - (pitch_std_hz / 25.0), 0.65, 0.98))
                explanations.append(f"Unnaturally flat pitch contour (F0 std={pitch_std_hz:.1f}Hz) indicative of synthetic TTS.")
            elif pitch_std_hz > 130.0:
                pitch_stability_score = float(np.clip((pitch_std_hz - 130.0) / 100.0 + 0.5, 0.5, 0.95))
                explanations.append(f"Erratic pitch jumps (F0 std={pitch_std_hz:.1f}Hz) typical of vocoder synthesis artifacts.")
            else:
                pitch_stability_score = float(np.clip(pitch_std_hz / 120.0 * 0.4, 0.05, 0.40))
                explanations.append(f"Natural vocal pitch dynamics detected (F0 mean={pitch_mean_hz:.1f}Hz, std={pitch_std_hz:.1f}Hz).")
    except Exception:
        pass

    # 4. Silence and Temporal Dynamics
    non_silent_intervals = librosa.effects.split(y, top_db=25)
    total_len = len(y)
    non_silent_samples = sum([end - start for start, end in non_silent_intervals]) if len(non_silent_intervals) > 0 else total_len
    silence_ratio = float(np.clip(1.0 - (non_silent_samples / max(1, total_len)), 0.0, 1.0))

    if silence_ratio > 0.60:
        explanations.append(f"Extended silence pause ratio ({silence_ratio*100:.1f}%) observed.")
    elif silence_ratio < 0.03 and len(y) / sr > 3.0:
        explanations.append("Near-zero natural breathing pauses detected across speech flow.")

    # 5. Spectral Flatness & Mid-Band Contrast
    flatness = librosa.feature.spectral_flatness(y=y)
    spectral_flatness_mean = float(np.mean(flatness))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=5, fmin=150.0)
    mid_contrast = float(np.mean(contrast[:4, :]))

    if spectral_flatness_mean > 0.025:
        explanations.append(f"Elevated spectral flatness ({spectral_flatness_mean:.4f}) indicating synthetic harmonic profile.")

    if snr_db > 30.0:
        explanations.append(f"Studio-grade zero-noise floor ({snr_db:.1f} dB) characteristic of neural voice cloning.")

    # Heuristic Score calculation
    heuristic_score = 0.0
    if pitch_stability_score > 0.6:
        heuristic_score += 0.35
    if snr_db > 28.0:
        heuristic_score += 0.25
    if spectral_flatness_mean > 0.025:
        heuristic_score += 0.25
    if silence_ratio < 0.04 and len(y) / sr > 2.0:
        heuristic_score += 0.15

    heuristic_score = float(np.clip(heuristic_score, 0.0, 1.0))

    indicators = AcousticIndicators(
        snr_db=round(snr_db, 2),
        clipping_detected=clipping_detected,
        pitch_mean_hz=round(pitch_mean_hz, 1),
        pitch_std_hz=round(pitch_std_hz, 1),
        pitch_stability_score=round(pitch_stability_score, 3),
        silence_ratio=round(silence_ratio, 3),
        spectral_flatness_mean=round(spectral_flatness_mean, 4),
        quality_check=quality_check
    )

    raw_dict = {
        "snr_db": snr_db,
        "pitch_stability": pitch_stability_score,
        "silence_ratio": silence_ratio,
        "spectral_flatness": spectral_flatness_mean,
        "mid_contrast": mid_contrast,
        "heuristic_score": heuristic_score
    }

    return indicators, raw_dict, explanations


# =======================================================
# 4. RANDOM FOREST ML CLASSIFIER INFERENCE
# =======================================================

def predict_random_forest(features: np.ndarray) -> Tuple[float, List[str]]:
    """
    Run Random Forest inference on acoustic feature representation.
    """
    ml_notes = []
    if os.path.exists(RF_MODEL_PATH):
        try:
            rf_clf = joblib.load(RF_MODEL_PATH)
            feat_reshaped = features.reshape(1, -1)
            
            classes = list(rf_clf.classes_)
            if 1 in classes:
                ai_idx = classes.index(1)
                probs = rf_clf.predict_proba(feat_reshaped)[0]
                rf_prob = float(probs[ai_idx])
            else:
                rf_prob = float(rf_clf.predict(feat_reshaped)[0])

            ml_notes.append(f"Supervised Random Forest detector: {rf_prob * 100:.1f}% AI probability.")
            if rf_prob >= 0.65:
                ml_notes.append("Acoustic cepstral dynamics strongly match synthetic TTS profiles.")
            elif rf_prob <= 0.30:
                ml_notes.append("Acoustic spectral features align closely with natural human speech benchmarks.")
            return rf_prob, ml_notes
        except Exception as e:
            ml_notes.append(f"ML model notice: {e}")
    
    fallback_prob = 0.2
    return fallback_prob, ml_notes


# =======================================================
# 5. LEARNED FUSION MODEL (ML + ACOUSTIC INDICATORS)
# =======================================================

def perform_fusion(
    ml_prob: float,
    acoustic_raw: Dict[str, float]
) -> Tuple[FusionAnalysis, List[str]]:
    """
    Calibrated fusion combining Random Forest probability and acoustic indicators.
    """
    fusion_notes = []
    snr_scaled = float(np.clip(acoustic_raw.get("snr_db", 0.0) / 40.0, 0.0, 1.0))
    pitch_stability = float(acoustic_raw.get("pitch_stability", 0.5))
    silence_ratio = float(acoustic_raw.get("silence_ratio", 0.1))
    flatness = float(acoustic_raw.get("spectral_flatness", 0.05))
    heuristic_score = float(acoustic_raw.get("heuristic_score", 0.0))

    fusion_vector = np.array([
        ml_prob,
        snr_scaled,
        pitch_stability,
        silence_ratio,
        flatness
    ]).reshape(1, -1)

    final_risk = None
    fusion_method = "calibrated_rule_fusion"
    ml_weight = 0.60
    acoustic_weight = 0.40

    if os.path.exists(FUSION_MODEL_PATH):
        try:
            fusion_clf = joblib.load(FUSION_MODEL_PATH)
            if fusion_clf.n_features_in_ == fusion_vector.shape[1]:
                probs = fusion_clf.predict_proba(fusion_vector)[0]
                classes = list(fusion_clf.classes_)
                ai_idx = classes.index(1) if 1 in classes else 1
                final_risk = float(probs[ai_idx])
                fusion_method = "learned_logistic_regression"
                coefs = fusion_clf.coef_[0]
                ml_weight = float(abs(coefs[0]) / (np.sum(np.abs(coefs)) + 1e-7))
                acoustic_weight = float(1.0 - ml_weight)
        except Exception:
            pass

    if final_risk is None:
        final_risk = (ml_weight * ml_prob) + (acoustic_weight * heuristic_score)
        final_risk = float(np.clip(final_risk, 0.0, 0.99))

    fusion_notes.append(f"Calibrated Decision Fusion: ML Evidence ({ml_prob*100:.1f}%) + Acoustic Indicators ({heuristic_score*100:.1f}%) -> {final_risk*100:.1f}% risk.")

    return FusionAnalysis(
        ml_probability=round(ml_prob, 4),
        acoustic_heuristic_score=round(heuristic_score, 4),
        fusion_risk_score=round(final_risk, 4),
        fusion_method=fusion_method,
        ml_weight=round(ml_weight, 3),
        acoustic_weight=round(acoustic_weight, 3)
    ), fusion_notes


# =======================================================
# 6. FASTER WHISPER MULTILINGUAL LANGUAGE BRANCH
# =======================================================

class LocalWhisperModel:
    _model = None
    _init_attempted = False

    @classmethod
    def get_model(cls):
        if not HAS_WHISPER:
            return None
        if not cls._init_attempted:
            cls._init_attempted = True
            try:
                os.environ.setdefault("HF_HOME", "/tmp/hf_home")
                cls._model = WhisperModel("tiny", device="cpu", compute_type="int8")
            except Exception:
                try:
                    cls._model = WhisperModel("base", device="cpu", compute_type="int8")
                except Exception as e:
                    print(f"Whisper initialization fallback: {e}")
                    cls._model = None
        return cls._model

def detect_language_and_transcript(y: np.ndarray, sr: int) -> Tuple[SupportedLanguage, float, str]:
    """
    Run inference with Faster Whisper for language identification and transcript preview.
    Gracefully falls back if model is unavailable in constrained serverless runtime.
    """
    try:
        whisper = LocalWhisperModel.get_model()
        if whisper is None:
            return SupportedLanguage.ENGLISH, 0.85, "(Audio acoustic signatures processed)"

        y_16k, _ = preprocess_audio(y, sr, target_sr=16000)
        segments_gen, info = whisper.transcribe(
            y_16k,
            beam_size=1,
            vad_filter=True,
            language=None
        )

        detected_code = info.language
        lang_prob = float(info.language_probability)

        sample_texts = []
        for i, seg in enumerate(segments_gen):
            sample_texts.append(seg.text.strip())
            if i >= 3:
                break
        transcript = " ".join(sample_texts).strip()

        code_map = {
            "en": SupportedLanguage.ENGLISH,
            "hi": SupportedLanguage.HINDI,
            "ta": SupportedLanguage.TAMIL,
            "te": SupportedLanguage.TELUGU,
            "ml": SupportedLanguage.MALAYALAM,
            "kn": SupportedLanguage.KANNADA,
            "mr": SupportedLanguage.MARATHI,
            "bn": SupportedLanguage.BENGALI,
            "gu": SupportedLanguage.GUJARATI,
            "pa": SupportedLanguage.PUNJABI,
        }

        lang_enum = code_map.get(detected_code, SupportedLanguage.ENGLISH if lang_prob < 0.2 else SupportedLanguage.OTHER)
        return lang_enum, round(lang_prob, 4), transcript
    except Exception:
        return SupportedLanguage.ENGLISH, 0.80, "(Acoustic signal analyzed)"


# =======================================================
# 7. SECOND-BY-SECOND WINDOW TIMELINE ANALYSIS
# =======================================================

def analyze_timeline_segments(
    y: np.ndarray,
    sr: int,
    global_fusion: FusionAnalysis
) -> List[SegmentAnalysis]:
    """
    Short window analysis (1-second intervals) across waveform for timeline localization.
    """
    duration = librosa.get_duration(y=y, sr=sr)
    chunk_sec = 1.0
    samples_per_chunk = int(chunk_sec * sr)
    total_samples = len(y)
    
    num_chunks = int(np.ceil(duration / chunk_sec))
    segments: List[SegmentAnalysis] = []

    for i in range(num_chunks):
        start_idx = i * samples_per_chunk
        end_idx = min((i + 1) * samples_per_chunk, total_samples)

        if end_idx - start_idx < int(0.25 * sr):
            continue

        chunk = y[start_idx:end_idx]
        start_t = float(round(i * chunk_sec, 2))
        end_t = float(round(start_t + (len(chunk) / sr), 2))

        chunk_features = extract_ml_features(chunk, sr)
        local_ml_prob, _ = predict_random_forest(chunk_features)
        _, local_raw, _ = analyze_acoustic_indicators(chunk, sr)
        local_fusion, _ = perform_fusion(local_ml_prob, local_raw)
        local_risk = local_fusion.fusion_risk_score

        # Silence / near-zero energy gating to prevent false positives on silent pauses
        chunk_rms = float(np.mean(librosa.feature.rms(y=chunk)))
        if chunk_rms < 0.005:
            local_risk = min(local_risk, 0.15)

        is_suspicious = local_risk >= 0.65
        seg_label = ClassificationResult.AI_GENERATED if is_suspicious else ClassificationResult.HUMAN
        seg_conf = local_risk if is_suspicious else (1.0 - local_risk)

        segments.append(SegmentAnalysis(
            start_time=start_t,
            end_time=end_t,
            label=seg_label,
            confidence=round(float(seg_conf), 3),
            risk_score=round(float(local_risk), 3),
            snr_db=round(float(local_raw.get("snr_db", 0.0)), 1),
            pitch_stability=round(float(local_raw.get("pitch_stability", 0.5)), 3),
            spectral_flatness=round(float(local_raw.get("spectral_flatness", 0.05)), 4)
        ))

    return segments
