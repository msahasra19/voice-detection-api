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


def load_audio_file(raw_bytes: bytes) -> Any:
    """
    Decode an audio file from raw bytes into a waveform array.

    This uses soundfile, which supports common formats such as WAV, FLAC, OGG, etc.
    Returns dictionary with waveform (numpy array) and sample_rate.
    """
    if not raw_bytes:
        raise ValueError("Empty audio payload")

    try:
        # soundfile.read returns (data, samplerate)
        # We assume the file format is detectable from headers in the bytes
        data, sample_rate = sf.read(BytesIO(raw_bytes))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Failed to decode audio data. Please ensure the file is a valid WAV or MP3. M4A/AAC is not supported without ffmpeg.") from exc

    if data is None:
        raise ValueError("Decoded audio is empty")

    return {"waveform": data, "sample_rate": sample_rate}

