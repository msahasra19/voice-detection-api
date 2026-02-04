from fastapi import FastAPI, Header, HTTPException, status, Body
from typing import Annotated

from .auth import validate_api_key
from .model import detect_voice
from .utils import load_audio_file, decode_audio_base64
from .schemas import VoiceRequest, VoiceResponse

app = FastAPI(title="Voice Detection API", version="1.0.0")


@app.get("/")
async def root():
    """
    Root endpoint - provides API information and links.
    """
    return {
        "message": "Voice Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "openapi": "/openapi.json"
        },
        "status": "running"
    }


@app.post("/predict", response_model=VoiceResponse)
async def predict_endpoint(
    request: VoiceRequest = Body(...),
    x_api_key: str | None = Header(default=None),
):
    """
    Detect whether a voice is AI-generated or Human.
    Input: Base64 encoded MP3.
    """
    if not validate_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    try:
        audio_bytes = decode_audio_base64(request.audio_data)
        audio = load_audio_file(audio_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    result = detect_voice(audio)
    return result


@app.get("/health")
async def health_check():
    return {"status": "ok"}

