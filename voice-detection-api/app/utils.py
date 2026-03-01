import base64
from io import BytesIO
from typing import Any

import soundfile as sf


def decode_audio_base64(b64_string: str) -> bytes:
    """
    Decode a Base64 string into raw audio bytes.
    """
    try:
        # Validate input (basic check)
        if "," in b64_string:
            # Handle data header if present (e.g., "data:audio/mp3;base64,.....")
            b64_string = b64_string.split(",")[1]
        return base64.b64decode(b64_string)
    except Exception as exc:
        raise ValueError("Invalid Base64 string") from exc


import librosa
import io

def load_audio_file(raw_bytes: bytes) -> Any:
    """
    Decode an audio file from raw bytes into a waveform array using librosa.
    Librosa is more flexible than soundfile for various formats.
    """
    if not raw_bytes:
        raise ValueError("Empty audio payload")

    try:
        # Load raw bytes as a file-like object
        # librosa.load will try to decoce it. If it fails, it might need ffmpeg.
        # But for basics, it uses soundfile or audioread.
        with io.BytesIO(raw_bytes) as audio_file:
            # We load at original sample rate first
            y, sr = librosa.load(audio_file, sr=None)
            
        if y is None or len(y) == 0:
            raise ValueError("Decoded audio is empty")
            
        return {"waveform": y, "sample_rate": sr}
    except Exception as exc:
        raise ValueError(f"Failed to decode audio: {str(exc)}") from exc

