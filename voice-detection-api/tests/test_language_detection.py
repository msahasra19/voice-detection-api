import os
import librosa
import numpy as np
from app.analysis import detect_language_and_transcript
from app.schemas import SupportedLanguage

def test_language_detection():
    print("Testing Multilingual Identification with local Faster Whisper...")
    samples_dir = os.path.join("dataset", "languages")
    
    if not os.path.exists(samples_dir):
        print(f"Directory {samples_dir} not found.")
        return

    languages_files = [f for f in os.listdir(samples_dir) if f.endswith(('.wav', '.mp3'))]
    if not languages_files:
        print(f"No audio samples in {samples_dir}.")
        return

    for filename in languages_files:
        file_path = os.path.join(samples_dir, filename)
        print(f"\nProcessing: {filename}...")
        try:
            y, sr = librosa.load(file_path, sr=16000)
            detected_lang, prob, transcript = detect_language_and_transcript(y, sr)
            print(f">> DETECTED: {detected_lang} (Confidence: {prob*100:.1f}%)")
            print(f">> TRANSCRIPT: '{transcript}'")
        except Exception as e:
            print(f"Error testing {filename}: {e}")

if __name__ == "__main__":
    test_language_detection()
