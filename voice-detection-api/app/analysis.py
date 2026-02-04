import numpy as np
import librosa
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

def extract_features_and_explain(y: np.ndarray, sr: int) -> Tuple[List[str], float]:
    """
    Extract features to determine if AI or Human, and return explanations.
    Returns: (Reasons, AI_Probability_Score)
    """
    reasons = []
    
    # Feature 1: Spectral Flatness (Robotic/Synthetic voices tend to be flatter)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
    # Feature 2: Pitch Stability / Variance
    # AI models sometimes have "too perfect" or "too jittery" pitch depending on quality
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    # Select pitches with magnitude
    pitch_vals = pitches[magnitudes > np.median(magnitudes)]
    if len(pitch_vals) > 0:
        pitch_std = np.std(pitch_vals)
    else:
        pitch_std = 0
        
    # Feature 3: Zero Crossing Rate (detects noise/fricatives)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Heuristic Logic (Simulation of a Model)
    # IMPORTANT: Real production needs a trained Deep Learning model here.
    # We are using signal processing heuristics to "detect" artifacts common in synthesis.
    
    ai_score = 0.0
    
    # Check Flatness
    if flatness > 0.2: # Threshold for 'buzzy' or 'flat' sound
        ai_score += 0.4
        reasons.append("Abnormally high spectral flatness (robotic characteristics).")
    
    # Check Pitch
    if pitch_std < 20: # Monotone
        ai_score += 0.3
        reasons.append("Unnatural pitch stability (monotone synthesis detected).")
    elif pitch_std > 500: # Random noise masquerading as speech
        # Base score is low because this can happen in noisy human audio too
        ai_score += 0.2
        reasons.append("Erratic pitch variance inconsistent with natural speech.")

    # Audio Silence/Gaps
    non_silent_intervals = librosa.effects.split(y, top_db=20)
    silence_ratio = 1.0 - (np.sum([end - start for start, end in non_silent_intervals]) / len(y))
    
    if silence_ratio > 0.8:
        reasons.append("Excessive silence detected.")
        ai_score = max(0.0, ai_score - 0.2)

    # Base score adjustment
    if zcr < 0.02:
        reasons.append("Low frequency variation (muffled/filtered signal).")
    
    # NEW: High Quality / Studio Quality Penalty
    # AI generators (ElevenLabs) produce near-perfect studio audio (High SNR).
    # Real user uploads often have background noise.
    
    # Calculate simple SNR context locally
    s_full = np.abs(librosa.stft(y))
    rms = librosa.feature.rms(S=s_full)[0]
    if len(rms) > 0:
        noise_approx = np.percentile(rms, 10) or 1e-9
        signal_approx = np.mean(rms[rms > noise_approx]) or noise_approx
        local_snr = 20 * np.log10(signal_approx / noise_approx)
    else:
        local_snr = 0

    if "Erratic pitch variance" in "".join(reasons):
        # AI typically has High SNR (> 30dB) + Erratic Pitch.
        # Noisy Human audio (< 30dB) also has Erratic Pitch due to noise.
        
        if local_snr > 38: # Higher threshold (Studio Quality)
             # Major boost for clean AI
             ai_score += 0.5
             reasons.append("High-fidelity studio quality detected (common in AI).")
        elif local_snr > 30:
             # Moderate boost
             ai_score += 0.3
 

    # Default to Human if no strong AI signs
    if ai_score == 0.0:
        reasons.append("Natural prosody and spectral characteristics observed.")
    
    return reasons, min(ai_score, 1.0)

import speech_recognition as sr
import soundfile as sf
import io


def detect_language_heuristic(y: np.ndarray, sr_rate: int) -> SupportedLanguage:
    """
    Attempt to detect language using Google Speech Recognition.
    Probs through supported languages and returns the one with highest confidence & linguistic validity.
    """
    recognizer = sr.Recognizer()
    
    # Minimal Stop Words / Common Tokens for Validation
    STOP_WORDS = {
        SupportedLanguage.ENGLISH: {"the", "is", "a", "to", "of", "in", "and", "you", "that", "it", "he", "was", "for", "on", "are", "as", "with", "his", "they", "i", "at", "be", "this", "have", "from"},
        SupportedLanguage.HINDI: {"है", "हैं", "का", "की", "के", "और", "से", "में", "को", "पर", "इस", "कि", "जो", "नहीं", "तो", "ही", "एक", "मैं", "तुम", "हम", "वे", "था", "थी", "थे"},
        SupportedLanguage.TAMIL: {"நான்", "என்", "என்னை", "அவன்", "அவள்", "அது", "இது", "அந்த", "இந்த", "ஒரு", "இல்லை", "உள்ளது", "வேண்டும்", "என்று", "ஆகும்", "ஆனால்", "அல்லது"},
        SupportedLanguage.TELUGU: {"నేను", "నా", "నన్ను", "అతను", "ఆమె", "అది", "ఇది", "ఆ", "ఈ", "ఒక", "కాదు", "ఉంది", "కావాలి", "అని", "కానీ", "లేదా", "మరియు"},
        SupportedLanguage.MALAYALAM: {"ഞാൻ", "എന്റെ", "എന്നെ", "അവൻ", "അവൾ", "അത്", "ഇത്", "ഒരു", "അല്ല", "ഉണ്ട്", "வேண்டும்", "എന്ന്", "ആണ്", "പക്ഷേ", "അല്ലെങ്കിൽ"}
    }
    
    try:
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, y, sr_rate, format='WAV')
        wav_buffer.seek(0)
        
        with sr.AudioFile(wav_buffer) as source:
            audio_data = recognizer.record(source)
            
        candidates = [
            ("en-IN", SupportedLanguage.ENGLISH),
            ("ta-IN", SupportedLanguage.TAMIL),
            ("hi-IN", SupportedLanguage.HINDI),
            ("ml-IN", SupportedLanguage.MALAYALAM),
            ("te-IN", SupportedLanguage.TELUGU),
        ]
        
        best_lang = SupportedLanguage.ENGLISH
        max_score = 0.0 
        
        results = []

        for code, lang_enum in candidates:
            try:
                response = recognizer.recognize_google(audio_data, language=code, show_all=True)
                
                if isinstance(response, dict) and 'alternative' in response:
                    alt = response['alternative'][0]
                    confidence = alt.get('confidence', 0.0)
                    transcript = alt.get('transcript', "")
                    
                    if 'confidence' not in alt and transcript:
                         confidence = 0.85 

                    # Stop Word Validation
                    found_stop_words = 0
                    tokens = set(transcript.lower().split()) if lang_enum == SupportedLanguage.ENGLISH else set(transcript.split())
                    
                    if lang_enum in STOP_WORDS:
                        relevant_stops = STOP_WORDS[lang_enum]
                        match_count = sum(1 for t in tokens if t in relevant_stops)
                        
                        print(f"DEBUG: {code} -> Match: {match_count} words. Text: {transcript[:50]}...")
                        
                        # Dynamic Boost based on linguistic validity
                        # More matches = Higher confidence that this is the correct language
                        if match_count > 0:
                            confidence += (0.1 * match_count) 
                            
                    results.append((lang_enum, confidence, transcript))
                    
                    if confidence > max_score:
                        max_score = confidence
                        best_lang = lang_enum
                        
            except (sr.UnknownValueError, sr.RequestError):
                print(f"DEBUG: {code} -> Failed")
                continue
        
        if not results:
             return SupportedLanguage.ENGLISH
             
        return best_lang

    except Exception:
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
