from fastapi import FastAPI, Header, HTTPException, status, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Annotated
from pathlib import Path

from .auth import validate_api_key
from .model import detect_voice
from .utils import load_audio_file, decode_audio_base64
from .schemas import VoiceRequest, VoiceResponse

import os
from pyngrok import ngrok

app = FastAPI(title="Voice Detection API", version="1.0.0")

# --- Cloud Tunnel Initialization (for Render/Heroku) ---
@app.on_event("startup")
async def startup_event():
    # Only run ngrok if tokens are provided (Cloud environment)
    auth_token = os.environ.get("NGROK_AUTHTOKEN")
    domain = os.environ.get("NGROK_DOMAIN")
    
    if auth_token and domain:
        try:
            print(f"DEBUG: Starting cloud tunnel for domain: {domain}")
            ngrok.set_auth_token(auth_token)
            
            # We tunnel to the local port uvicorn is running on. 
            local_port = int(os.environ.get("PORT", 10000))
            ngrok.connect(local_port, pyngrok_config=None, name="render_tunnel", url=domain)
            print(f"DEBUG: Tunnel established at https://{domain}")
        except Exception as e:
            print(f"WARNING: Could not start ngrok tunnel: {e}")
            print("The app will still be available at the Render URL.")


# Mount static files from the project-root-relative "static" folder.
# This matches both local runs (uvicorn from repo root) and Render's
# working directory (cloned repo root).
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """
    Serve the web interface.
    """
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    # Fallback to API info if static files don't exist at all
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
    request: dict = Body(...),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    """
    Detect whether a voice is AI-generated or Human.
    Accepts either Base64 encoded MP3 or a URL pointing to an MP3.
    """
    # Prefer X-API-Key, then Authorization header
    token = x_api_key or authorization
    if token and token.startswith("Bearer "):
        token = token[7:]

    if not validate_api_key(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    # Log request for debugging (visible in server terminal)
    print(f"DEBUG: Received request: {request}")

    try:
        audio_bytes = None
        
        # Case-insensitive search for any key containing 'audio' or 'base64'
        url = None
        data = None
        
        for k, v in request.items():
            k_lower = k.lower()
            if "url" in k_lower:
                url = v
            if "data" in k_lower or "base64" in k_lower or "file" in k_lower:
                data = v
                
        # If still not found, check specific keys as fallback
        if not url:
            url = request.get("audio_url") or request.get("url")
        if not data:
            data = request.get("audio_data") or request.get("audio_base64") or request.get("audioBase64") or request.get("base64") or request.get("file")

        if url:

            # Download from URL
            import requests as req
            resp = req.get(url, timeout=10)
            resp.raise_for_status()
            audio_bytes = resp.content
        elif data:
            # Handle string or potential list/dict from some tools
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            audio_bytes = decode_audio_base64(str(data))
        else:
            # Create a summary of what was received to help debugging
            received_info = {k: (f"Length: {len(str(v))}" if v else "Empty/None") for k, v in request.items()}
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No audio source found in request. Received data summary: {received_info}",
            )


        audio = load_audio_file(audio_bytes)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    result = detect_voice(audio)
    return result




@app.get("/health")
async def health_check():
    return {"status": "ok"}

