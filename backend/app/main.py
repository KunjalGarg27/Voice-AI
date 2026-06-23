"""FastAPI application for the local Voice AI assistant."""

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import UPLOADS_DIR
from app.ollama_service import generate_response
from app.whisper_service import transcribe_audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice AI Assistant",
    description="Local voice assistant: audio → Whisper → Qwen → JSON response",
    version="0.1.0",
)

# Ensure the uploads directory exists at startup
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _validate_upload(audio: UploadFile) -> None:
    """
    Validate that the uploaded file looks like an audio upload.

    Raises HTTPException if the file is missing or has no filename.
    """
    if audio.filename is None or not audio.filename.strip():
        raise HTTPException(status_code=400, detail="No audio file provided.")


def _build_upload_path(filename: str) -> Path:
    """
    Build a unique path inside uploads/ for a saved audio file.

    A UUID prefix prevents filename collisions when multiple requests
    upload files with the same name.
    """
    safe_name = Path(filename).name
    unique_name = f"{uuid.uuid4()}_{safe_name}"
    return UPLOADS_DIR / unique_name


async def _save_upload(audio: UploadFile, destination: Path) -> None:
    """
    Write the uploaded audio bytes to disk at the given path.

    Raises HTTPException if reading or writing the file fails.
    """
    try:
        contents = await audio.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
        destination.write_bytes(contents)
        logger.info("Saved upload to %s (%d bytes)", destination, len(contents))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to save uploaded audio")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded audio: {exc}",
        ) from exc


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Return a simple health status for uptime checks."""
    return {"status": "ok"}


@app.post("/voice-query")
async def voice_query(audio: UploadFile = File(...)) -> JSONResponse:
    """
    Accept an audio file, transcribe it with Whisper, and reply via Qwen.

    Pipeline: upload → save to uploads/ → Whisper STT → Ollama Qwen → JSON.

    Returns:
        JSON with ``transcript`` (speech-to-text) and ``response`` (LLM reply).
    """
    _validate_upload(audio)
    upload_path = _build_upload_path(audio.filename)
    await _save_upload(audio, upload_path)

    # Run CPU-bound Whisper transcription off the async event loop
    try:
        transcript = await asyncio.to_thread(transcribe_audio, upload_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        response_text = await generate_response(transcript)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return JSONResponse(
        content={
            "transcript": transcript,
            "response": response_text,
        }
    )
