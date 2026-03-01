from faster_whisper import WhisperModel
import time

print("Testing Whisper 'base' model initialization...")
start = time.time()
try:
    # This will check if 'base' is ready
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print(f"SUCCESS: 'base' model loaded in {time.time() - start:.2f} seconds.")
except Exception as e:
    print(f"FAILURE: {e}")
