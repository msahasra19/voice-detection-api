import numpy as np
import librosa
import joblib
import os
from typing import List, Dict, Any, Tuple
from .schemas import ClassificationResult, SupportedLanguage, AudioQualityScore, SegmentAnalysis, ConfidenceLevel
from faster_whisper import WhisperModel

# --- 1. Audio Quality ---
def analyze_audio_quality(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze audio quality: SNR and Clipping."""
    max_amp = np.max(np.abs(y))
    clipping_detected = max_amp > 0.99

    s_full = np.abs(librosa.stft(y))
    rms = librosa.feature.rms(S=s_full)[0]
    
    if len(rms) == 0:
        return {"snr": 0.0, "clipping_detected": False, "quality_check": AudioQualityScore.LOW}
        
    noise_thresh = np.percentile(rms, 10)
    if noise_thresh == 0:
        noise_thresh = 1e-9
    
    signal_rms = np.mean(rms[rms > noise_thresh])
    if str(signal_rms) == 'nan':
         signal_rms = noise_thresh

    snr = 20 * np.log10(signal_rms / noise_thresh)
    
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


# --- 2. XAI / Detection ---
def extract_ml_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract features matching the training script.
    """
    # 1. MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)
    
    # 2. Spectral Features
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # 3. Spectral Contrast
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    
    # 4. Chroma STFT
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

    # Combine all features
    features = np.hstack([
        mfccs_mean, 
        mfccs_std, 
        flatness, 
        zcr, 
        centroid,
        rolloff,
        contrast,
        chroma
    ])
    return features

def extract_features_and_explain(y: np.ndarray, sr: int) -> Tuple[Dict[str, Any], float]:
    """
    Computes AI score using the restricted feature set for realism.
    Matches the 1.5s / 6-MFCC training logic.
    """
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'voice_model_bundle.joblib')
        if not os.path.exists(model_path):
             print(f"DEBUG: Model bundle not found at {model_path}")
             return {"error": "Model not found"}, 0.5
             
        bundle = joblib.load(model_path)
        clf = bundle['model']
        scaler = bundle['scaler']
        
        # 1. Load 3.0s to match new training calibration
        y_short = y[:int(3.0 * sr)]
        y_norm = librosa.util.normalize(y_short)
        
        # 2. Extract 13 mean + 13 std MFCCs + Flatness + ZCR (28 features total)
        mfccs = librosa.feature.mfcc(y=y_norm, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        flatness = np.array([np.mean(librosa.feature.spectral_flatness(y=y_norm))])
        zcr = np.array([np.mean(librosa.feature.zero_crossing_rate(y_norm))])
        
        features = np.hstack([mfccs_mean, mfccs_std, flatness, zcr])
        
        # Scale
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # 3. Get AI & Human probabilities
        probs = clf.predict_proba(features_scaled)[0]
        human_prob = float(probs[0])
        ai_prob = float(probs[1])
        
        # Stability tweak
        human_prob = max(0.0001, min(0.9999, human_prob))
        ai_prob = 1.0 - human_prob
        ai_score = ai_prob
        
        # 4. Feature Categories for 28 features
        feature_categories = {
            "Voice Resonance (MFCC)": slice(0, 13),
            "Vocal Stability (STD)": slice(13, 26),
            "Spectral Flatness": slice(26, 27),
            "Zero Crossing Rate": slice(27, 28)
        }
        
        importances = clf.feature_importances_
        contributions = importances * features_scaled[0]
        
        shap_results = {}
        for cat, slc in feature_categories.items():
            impact = float(np.sum(contributions[slc]))
            shap_results[cat] = round(impact, 3)

        lime_results = {}
        sorted_contribs = sorted(shap_results.items(), key=lambda x: abs(x[1]), reverse=True)
        for cat, impact in sorted_contribs[:3]:
             if abs(impact) > 0.001:
                 label = "High" if impact > 0 else "Low"
                 lime_results[f"{label} {cat}"] = impact

        # --- Simplified XAI Markers ---
        times = librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=sr)
        gradcam_regions = []
        # Highlight regions where MFCC variance is highest
        mfcc_var = np.var(mfccs, axis=0)
        if len(mfcc_var) > 0:
            top_frames = np.argsort(mfcc_var)[-2:]
            for f_idx in top_frames:
                if f_idx < len(times):
                    gradcam_regions.append({
                        "start": float(max(0, times[f_idx] - 0.1)),
                        "end": float(min(1.5, times[f_idx] + 0.1)),
                        "intensity": float(mfcc_var[f_idx] / np.max(mfcc_var)) if np.max(mfcc_var) > 0 else 0.5,
                        "reason": "Feature Variance"
                    })

        # --- Counterfactuals ---
        if ai_score > 0.65:
            counterfactual = "Classification driven by synthetic spectral regularity. Human-like variation is missing in the 3s window."
        elif ai_score < 0.35:
            counterfactual = "Audio displays natural vocal tract resonances and healthy spectral variance typical of human speech."
        else:
            counterfactual = "Model is uncertain. Audio has mixed characteristics. Human voices often show more random fluctuation."

        explain_data = {
            "shap_results": shap_results,
            "lime_results": lime_results,
            "gradcam_regions": gradcam_regions,
            "attention_segments": [], # Not used for this pruned model
            "counterfactual_explanation": counterfactual
        }

        return explain_data, ai_score
    except Exception as e:
        print(f"DEBUG: Error in realistic XAI estimation: {e}")
        return {"error": str(e)}, 0.5

# --- 3. Language & Speech ---
class LocalMLModels:
    _instance = None
    _whisper_model = None

    @classmethod
    def get_whisper(cls):
        if cls._whisper_model is None:
            print("DEBUG: Initializing faster-whisper 'base' model on CPU...")
            try:
                cls._whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
                print("DEBUG: faster-whisper base model loaded successfully.")
            except Exception as e:
                print(f"DEBUG: Error loading faster-whisper model: {e}")
                raise
        return cls._whisper_model

def detect_language_ml(y: np.ndarray, sr_rate: int) -> SupportedLanguage:
    """Detect language using local Whisper. Maps cleanly to the 10 specified languages."""
    try:
        model = LocalMLModels.get_whisper()
        
        # Audio is pre-resampled by utils.py, but safeguard checks:
        if sr_rate != 16000:
            y = librosa.resample(y, orig_sr=sr_rate, target_sr=16000)
            sr_rate = 16000

        max_val = np.abs(y).max()
        if max_val > 0:
            y = y / max_val

        if y.dtype != np.float32:
            y = y.astype(np.float32)

        # Prompt hint covering the new languages
        initial_p = "English, Tamil, Hindi, Telugu, Spanish"
        
        segments_gen, info = model.transcribe(
            y, beam_size=10, vad_filter=True, initial_prompt=initial_p, language=None
        )
        
        detected_code = info.language
        prob = info.language_probability
        
        # Pull debug transc_text lazily
        transc_text = " ".join([s.text for i, s in enumerate(segments_gen) if i < 3])
        print(f"DEBUG: Lang: {detected_code}, Prob: {prob:.2f}, Text: {transc_text}")

        mapping = {
            "en": SupportedLanguage.ENGLISH,
            "ta": SupportedLanguage.TAMIL,
            "hi": SupportedLanguage.HINDI,
            "te": SupportedLanguage.TELUGU,
            "es": SupportedLanguage.SPANISH,
        }
        
        if prob < 0.1:
            return SupportedLanguage.ENGLISH

        return mapping.get(detected_code, SupportedLanguage.ENGLISH)

    except Exception as e:
        print(f"DEBUG: Local ML language detection failed: {e}")
        return SupportedLanguage.ENGLISH


# --- 4. Segmentation ---
def analyze_segments(y: np.ndarray, sr: int) -> List[SegmentAnalysis]:
    """
    Split audio into exactly 1-second chunks and analyze segments objectively.
    Each segment is evaluated independently to detect local manipulations.
    """
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'voice_model_bundle.joblib')
        if not os.path.exists(model_path):
             return []
             
        bundle = joblib.load(model_path)
        clf = bundle['model']
        scaler = bundle['scaler']

        duration = librosa.get_duration(y=y, sr=sr)
        segments = []
        chunk_len_sec = 1.0
        total_samples = len(y)
        samples_per_chunk = int(chunk_len_sec * sr)
        num_chunks = int(duration // chunk_len_sec)
        
        for i in range(num_chunks + 1):
            start = i * samples_per_chunk
            end = min((i + 1) * samples_per_chunk, total_samples)
            if end - start < int(0.5 * sr): # Minimum 0.5s for analysis
                continue
                
            chunk = y[start:end]
            chunk_norm = librosa.util.normalize(chunk)
            
            # Extract local features (13 MFCCs + Flatness + ZCR)
            mfccs = librosa.feature.mfcc(y=chunk_norm, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            mfccs_std = np.std(mfccs.T, axis=0)
            flatness = np.array([np.mean(librosa.feature.spectral_flatness(y=chunk_norm))])
            zcr = np.array([np.mean(librosa.feature.zero_crossing_rate(chunk_norm))])
            
            local_features = np.hstack([mfccs_mean, mfccs_std, flatness, zcr])
            local_features_scaled = scaler.transform(local_features.reshape(1, -1))
            
            probs = clf.predict_proba(local_features_scaled)[0]
            local_ai_prob = float(probs[1])
            
            # Align threshold with main prediction (0.7)
            label = ClassificationResult.AI_GENERATED if local_ai_prob > 0.7 else ClassificationResult.HUMAN
            confidence = local_ai_prob if label == ClassificationResult.AI_GENERATED else (1.0 - local_ai_prob)
            
            segments.append(SegmentAnalysis(
                start_time=float(i * chunk_len_sec),
                end_time=float(i * chunk_len_sec + (len(chunk)/sr)),
                label=label,
                confidence=float(confidence)
            ))
            
        return segments
    except Exception as e:
        print(f"DEBUG: Error in segment analysis: {e}")
        return []

# --- 5. Main Orchestration ---
def detect_voice(audio: Dict[str, Any]) -> Dict[str, Any]:
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    y = np.asarray(waveform, dtype=float)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # 1. MATCH TRAINING: Add base noise injection for robustness
    # This prevents the model from overfitting to clean silences.
    noise = np.random.normal(0, 0.001, len(y))
    y = y + noise

    quality_data = analyze_audio_quality(y, sample_rate)
    reasons, ai_score = extract_features_and_explain(y, sample_rate)

    # Classification using AI probability threshold (0.7) for stricter AI detection
    classification = ClassificationResult.AI_GENERATED if ai_score > 0.7 else ClassificationResult.HUMAN
    
    # Extract Human and AI probabilities for the response
    # (ai_score is calculated in extract_features_and_explain)
    ai_confidence = ai_score
    human_confidence = 1.0 - ai_score
    
    # Confidence score is the probability of the PREDICTED class
    confidence_score = ai_confidence if classification == ClassificationResult.AI_GENERATED else human_confidence
    
    # Correct confidence level based on the predicted class probability
    if confidence_score > 0.8:
        conf_level = ConfidenceLevel.HIGH
    elif confidence_score > 0.4:
        conf_level = ConfidenceLevel.MEDIUM
    else:
        conf_level = ConfidenceLevel.LOW

    language = detect_language_ml(y, sample_rate)
    segments = analyze_segments(y, sample_rate)

    print(f"DEBUG: Inference Result - Status: {classification}, Human: {human_confidence:.4f}, AI: {ai_confidence:.4f}, Conf: {confidence_score:.4f}")

    return {
        "classification": classification,
        "confidence_score": float(confidence_score),
        "human_confidence": float(human_confidence),
        "ai_confidence": float(ai_confidence),
        "confidence_level": conf_level,
        "deepfake_risk_score": float(ai_score),
        "detected_language": language,
        "audio_quality": quality_data,
        "explainability": reasons,
        "segments": segments
    }
