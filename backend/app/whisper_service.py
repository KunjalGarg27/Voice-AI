"""Speech-to-text transcription using faster-whisper."""

import logging
from pathlib import Path

from faster_whisper import WhisperModel

from app.config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL_SIZE

logger = logging.getLogger(__name__)

# Lazy-loaded Whisper model instance (loaded once on first transcription)
_whisper_model: WhisperModel | None = None


def _get_whisper_model() -> WhisperModel:
    """
    Load and return the Whisper model, initializing it on first use.

    The model is cached in memory so subsequent transcriptions do not
    incur the cost of reloading weights from disk.
    """
    global _whisper_model
    if _whisper_model is None:
        logger.info(
            "Loading Whisper model '%s' (device=%s, compute_type=%s)",
            WHISPER_MODEL_SIZE,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
        )
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _whisper_model


def transcribe_audio(audio_path: Path) -> str:
    """
    Transcribe an audio file to text using faster-whisper.

    Args:
        audio_path: Path to the audio file on disk (wav, mp3, webm, etc.).

    Returns:
        The transcribed text as a single string.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If transcription fails or produces no speech.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _get_whisper_model()

    try:
        segments, _info = model.transcribe(str(audio_path), beam_size=5)
        transcript_parts = [segment.text.strip() for segment in segments]
        transcript = " ".join(part for part in transcript_parts if part)
    except Exception as exc:
        logger.exception("Whisper transcription failed for %s", audio_path)
        raise RuntimeError(f"Speech-to-text transcription failed: {exc}") from exc

    if not transcript:
        raise RuntimeError("No speech detected in the audio file.")

    logger.info("Transcription complete (%d characters)", len(transcript))
    return transcript
