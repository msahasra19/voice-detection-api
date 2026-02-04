import sys
import os
import json
import logging

# Add project root to path so we can import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging to show up in console
logging.basicConfig(level=logging.INFO)

print(f"DEBUG: Python Executable: {sys.executable}")
try:
    import speech_recognition as sr_test
    print(f"DEBUG: speech_recognition imported successfully from {sr_test.__file__}")
except ImportError as e:
    print(f"DEBUG: Failed to import speech_recognition directly: {e}")

from app.utils import load_audio_file
from app.analysis import detect_language_heuristic

def debug_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"Loading '{file_path}'...")
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    
    try:
        audio = load_audio_file(file_bytes)
        y = audio["waveform"]
        sr = audio["sample_rate"]
        
        # Mix down if needed (model.py logic duplicate)
        import numpy as np
        y_np = np.asarray(y, dtype=float)
        if y_np.ndim > 1:
            y_np = np.mean(y_np, axis=1)

        print("\n--- Starting Language Detection ---")
        lang = detect_language_heuristic(y_np, sr)
        print(f"\nFinal Detected Language: {lang}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/debug_lang.py <path_to_audio_file>")
    else:
        debug_file(sys.argv[1])
