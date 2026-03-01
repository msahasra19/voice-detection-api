import numpy as np
import librosa
import joblib
import os
from typing import List, Dict, Any, Tuple
from .schemas import ClassificationResult, SupportedLanguage, AudioQualityScore, SegmentAnalysis

def analyze_audio_quality(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Analyze audio quality: SNR and Clipping.
    """
    # 1. Clipping detection: check if any sample is close to max (assuming float -1.0 to 1.0)
    # If loaded as int, scale would be different. Librosa loads as float32 by default.
    max_amp = np.max(np.abs(y))
    clipping_detected = max_amp > 0.99

    # 2. SNR (Signal-to-Noise Ratio) estimation
    # Simple heuristic: assume lowest 10% energy frames are noise
    s_full = np.abs(librosa.stft(y))
    rms = librosa.feature.rms(S=s_full)[0]
    
    if len(rms) == 0:
        return {"snr": 0.0, "clipping_detected": False, "quality_check": AudioQualityScore.LOW}
        
    noise_thresh = np.percentile(rms, 10)
    # Avoid division by zero
    if noise_thresh == 0:
        noise_thresh = 1e-9
    
    signal_rms = np.mean(rms[rms > noise_thresh])
    if str(signal_rms) == 'nan':
         signal_rms = noise_thresh

    snr = 20 * np.log10(signal_rms / noise_thresh)
    
    # Bucket quality
    if snr < 10:
        quality = AudioQualityScore.LOW
    elif snr < 30:
        quality = AudioQualityScore.MEDIUM
    else:
        quality = AudioQualityScore.HIGH

    return {
        "snr": float(snr),
        "clipping_detected": bool(clipping_detected),
        "quality_check": quality
    }

def extract_ml_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract features consistent with the training script.
    """
    # 1. MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)
    
    # 2. Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=y)
    flatness_mean = np.mean(flatness)
    
    # 3. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    
    # 4. Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)
    
    return np.hstack([mfccs_mean, mfccs_std, flatness_mean, zcr_mean, contrast_mean])

def extract_features_and_explain(y: np.ndarray, sr: int) -> Tuple[List[str], float]:
    """
    Extract features to determine if AI or Human, and return explanations.
    Uses a trained Random Forest model if available, otherwise falls back to heuristics.
    """
    reasons = []
    ai_score = 0.0
    
    # Check if a trained model exists
    model_path = os.path.join(os.path.dirname(__file__), "voice_model.joblib")
    
    if os.path.exists(model_path):
        try:
            clf = joblib.load(model_path)
            features = extract_ml_features(y, sr)
            # Reshape for single sample
            features = features.reshape(1, -1)
            
            # Get probability of class 1 (AI)
            probs = clf.predict_proba(features)[0]
            ai_score = float(probs[1])
            
            reasons.append(f"ML Model Analysis: {ai_score*100:.1f}% AI probability.")
            if ai_score > 0.8:
                reasons.append("Strong patterns of synthetic generation detected by neural classifier.")
            elif ai_score < 0.2:
                reasons.append("Vocal characteristics strongly match natural human speech patterns.")
            
            return reasons, ai_score
        except Exception as e:
            print(f"DEBUG: ML model inference failed: {e}. Falling back to heuristics.")
            reasons.append("ML inference failed, using fallback heuristics.")

    # --- HEURISTIC FALLBACK ---
    # Feature 1: Spectral Flatness
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
    # Feature 2: Pitch Stability
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[magnitudes > np.median(magnitudes)]
    pitch_std = np.std(pitch_vals) if len(pitch_vals) > 0 else 0
        
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    if flatness > 0.2:
        ai_score += 0.4
        reasons.append("Abnormally high spectral flatness (robotic characteristics).")
    
    if pitch_std < 20: 
        ai_score += 0.3
        reasons.append("Unnatural pitch stability (monotone synthesis detected).")
    elif pitch_std > 500:
        ai_score += 0.2
        reasons.append("Erratic pitch variance inconsistent with natural speech.")

    # Silence gaps
    non_silent_intervals = librosa.effects.split(y, top_db=20)
    silence_ratio = 1.0 - (np.sum([end - start for start, end in non_silent_intervals]) / len(y))
    if silence_ratio > 0.8:
        reasons.append("Excessive silence detected.")
        ai_score = max(0.0, ai_score - 0.2)
    
    # Studio Quality check
    s_full = np.abs(librosa.stft(y))
    rms = librosa.feature.rms(S=s_full)[0]
    if len(rms) > 0:
        noise_approx = np.percentile(rms, 10) or 1e-9
        signal_approx = np.mean(rms[rms > noise_approx]) or noise_approx
        local_snr = 20 * np.log10(signal_approx / noise_approx)
    else:
        local_snr = 0

    if ai_score > 0.2 and local_snr > 38: # Higher threshold (Studio Quality)
         ai_score += 0.5
         reasons.append("High-fidelity studio quality detected (common in AI).")

    if ai_score == 0.0:
        reasons.append("Natural prosody and spectral characteristics observed.")
    
    return reasons, min(ai_score, 1.0)

from faster_whisper import WhisperModel
import os

class LocalMLModels:
    _instance = None
    _whisper_model = None

    @classmethod
    def get_whisper(cls):
        if cls._whisper_model is None:
            # Using 'base' for faster loading on local machines
            print("DEBUG: Initializing faster-whisper 'base' model on CPU...")
            try:
                cls._whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
                print("DEBUG: faster-whisper base model loaded successfully.")
            except Exception as e:
                print(f"DEBUG: Error loading faster-whisper model: {e}")
                raise
        return cls._whisper_model

def detect_language_ml(y: np.ndarray, sr_rate: int) -> SupportedLanguage:
    """
    Detect language using a local Whisper ML model.
    Resamples to 16kHz and normalizes audio for better accuracy.
    """
    try:
        model = LocalMLModels.get_whisper()
        
        # 1. Ensure 16kHz
        if sr_rate != 16000:
            print(f"DEBUG: Resampling from {sr_rate}Hz to 16000Hz.")
            y = librosa.resample(y, orig_sr=sr_rate, target_sr=16000)
            sr_rate = 16000

        # 2. Normalize Volume (important for Whisper accuracy)
        max_val = np.abs(y).max()
        if max_val > 0:
            y = y / max_val
            print("DEBUG: Audio normalized.")

        # Whisper expects float32
        if y.dtype != np.float32:
            y = y.astype(np.float32)

        # 3. Detect Language
        # We provide an initial prompt to bias the model towards the target languages
        initial_p = "This audio contains speech in English, Hindi, or Telugu."
        
        segments_gen, info = model.transcribe(
            y, 
            beam_size=10, 
            vad_filter=True, 
            initial_prompt=initial_p,
            language=None
        )
        
        detected_code = info.language
        prob = info.language_probability
        
        # Consume first few segments to get a transcript for debug
        segments = []
        for i, s in enumerate(segments_gen):
            segments.append(s.text)
            if i > 5: break # Don't consume everything if it's long
        
        transcript = " ".join(segments).strip()
        print(f"DEBUG: ML Result -> Lang: {detected_code}, Prob: {prob:.4f}, Sample Transcript: '{transcript}'")

        # Map Whisper codes to our SupportedLanguage enum
        mapping = {
            "en": SupportedLanguage.ENGLISH,
            "ta": SupportedLanguage.TAMIL,
            "hi": SupportedLanguage.HINDI,
            "ml": SupportedLanguage.MALAYALAM,
            "te": SupportedLanguage.TELUGU,
            "kn": SupportedLanguage.TELUGU,
            "mr": SupportedLanguage.HINDI,
            "bn": SupportedLanguage.HINDI,
            "pa": SupportedLanguage.HINDI,
            "gu": SupportedLanguage.HINDI
        }
        
        # If confidence is extremely low (< 0.1), it's likely noise or unidentifiable
        if prob < 0.1:
            print("DEBUG: Language probability too low, defaulting to English.")
            return SupportedLanguage.ENGLISH

        if detected_code in mapping:
            return mapping[detected_code]
            
        return SupportedLanguage.ENGLISH

    except Exception as e:
        print(f"DEBUG: Local ML language detection failed: {e}")
        return SupportedLanguage.ENGLISH


def analyze_segments(y: np.ndarray, sr: int, main_score: float) -> List[SegmentAnalysis]:
    """
    Split audio into segments and analyze each.
    """
    duration = librosa.get_duration(y=y, sr=sr)
    # Analyze 1-second chunks
    segments = []
    chunk_len_sec = 1.0
    total_samples = len(y)
    samples_per_chunk = int(chunk_len_sec * sr)
    
    num_chunks = int(duration // chunk_len_sec)
    
    for i in range(num_chunks + 1):
        start = i * samples_per_chunk
        end = min((i + 1) * samples_per_chunk, total_samples)
        
        if end - start < 512:
            continue
            
        chunk = y[start:end]
        
        # Simple local analysis (mirroring main logic simplified)
        flatness = np.mean(librosa.feature.spectral_flatness(y=chunk))
        local_score = main_score 
        if flatness > 0.3:
            local_score += 0.2
            
        local_score = min(local_score, 1.0)
        
        label = ClassificationResult.AI_GENERATED if local_score > 0.5 else ClassificationResult.HUMAN
        
        segments.append(SegmentAnalysis(
            start_time=float(i * chunk_len_sec),
            end_time=float(i * chunk_len_sec + (len(chunk)/sr)),
            label=label,
            confidence=float(local_score if label == ClassificationResult.AI_GENERATED else 1.0 - local_score)
        ))
        
    return segments
