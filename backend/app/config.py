"""Application configuration constants."""

from pathlib import Path

# Base directory for the backend package (backend/)
BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Directory where uploaded audio files are stored
UPLOADS_DIR = BACKEND_ROOT / "uploads"

# Ollama local server settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_TIMEOUT_SECONDS = 120.0

# Whisper model settings (faster-whisper)
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
