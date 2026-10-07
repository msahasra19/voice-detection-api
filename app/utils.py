import base64
import io
import os
import tempfile
from typing import Any
import soundfile as sf
import scipy.io.wavfile as wavfile
import numpy as np

def decode_audio_base64(b64_string: str) -> bytes:
    """
    Decode a Base64 string into raw audio bytes.
    """
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        return base64.b64decode(b64_string)
    except Exception as exc:
        raise ValueError("Invalid Base64 string") from exc

def load_audio_file(raw_bytes: bytes) -> Any:
    """
    Decode an audio file from raw bytes into a waveform array.
    Supports: WAV, MP3, FLAC, M4A, OGG, WebM, AAC.
    """
    if not raw_bytes or len(raw_bytes) < 16:
        raise ValueError("Empty or invalid audio payload.")

    # 1. Try in-memory soundfile (fastest for WAV/FLAC)
    try:
        with io.BytesIO(raw_bytes) as audio_file:
            y, sr = sf.read(audio_file)
            if len(y.shape) > 1:
                y = y.mean(axis=1)
            return {"waveform": y.astype(np.float32), "sample_rate": int(sr)}
    except Exception:
        pass

    # 2. Try scipy wavfile
    try:
        with io.BytesIO(raw_bytes) as audio_file:
            sr, y = wavfile.read(audio_file)
            if len(y.shape) > 1:
                y = y.mean(axis=1)
            if np.issubdtype(y.dtype, np.integer):
                max_val = np.iinfo(y.dtype).max
                y = y.astype(np.float32) / max_val
            else:
                y = y.astype(np.float32)
            return {"waveform": y, "sample_rate": int(sr)}
    except Exception:
        pass

    # 3. Write to temporary file on disk for librosa / audioread (supports MP3, M4A, OGG, WebM, FLAC)
    temp_path = None
    try:
        # Detect header hint
        suffix = ".wav"
        if raw_bytes[:3] == b'ID3' or raw_bytes[:2] == b'\xff\xfb' or raw_bytes[:2] == b'\xff\xf3':
            suffix = ".mp3"
        elif raw_bytes[:4] == b'fLaC':
            suffix = ".flac"
        elif raw_bytes[:4] == b'OggS':
            suffix = ".ogg"
        elif b'ftyp' in raw_bytes[:16]:
            suffix = ".m4a"
        elif raw_bytes[:4] == b'\x1a\x45\xdf\xa3':
            suffix = ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw_bytes)
            temp_path = tmp.name

        import librosa
        y, sr = librosa.load(temp_path, sr=None)
        if y is not None and len(y) > 0:
            if y.ndim > 1:
                y = np.mean(y, axis=0 if y.shape[0] < y.shape[1] else 1)
            return {"waveform": y.astype(np.float32), "sample_rate": int(sr)}
    except Exception as lib_err:
        raise ValueError(f"Failed to decode audio. Please ensure it is a valid MP3, WAV, or FLAC file. Error details: {str(lib_err)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    raise ValueError("Failed to decode audio payload.")
