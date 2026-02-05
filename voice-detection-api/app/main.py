from fastapi import FastAPI, Header, HTTPException, status, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Annotated
from pathlib import Path

from .auth import validate_api_key
from .model import detect_voice
from .utils import load_audio_file, decode_audio_base64
from .schemas import VoiceRequest, VoiceResponse

app = FastAPI(title="Voice Detection API", version="1.0.0")

# Mount static files
# Try to resolve the static directory relative to this file, but also
# support deployments where the working directory is the project root.
static_dir_candidates = [
    Path(__file__).resolve().parent.parent / "static",
    Path.cwd() / "static",
]

static_dir = next((p for p in static_dir_candidates if p.exists()), None)

if static_dir is not None and static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """
    Serve the web interface.
    """
    # Try the resolved static directory first
    if static_dir is not None:
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)

    # Fallback: try a relative path from the current working directory
    cwd_index = Path("static") / "index.html"
    if cwd_index.exists():
        return FileResponse(cwd_index)
    
    # Fallback to API info if static files don't exist
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

