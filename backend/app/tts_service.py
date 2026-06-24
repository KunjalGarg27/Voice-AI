"""Text-to-speech synthesis using the local Piper binary."""

import logging
import subprocess
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Absolute path to the Piper executable on this machine
PIPER_BINARY = Path("/home/kunjal-garg/piper/piper/piper")

# Absolute path to the English Piper ONNX voice model
PIPER_MODEL = Path("/home/kunjal-garg/piper/en_US-lessac-medium.onnx")

# Directory where generated WAV files are written (backend/temp/)
TEMP_DIR = Path(__file__).resolve().parent.parent / "temp"

# Maximum seconds to wait for Piper to finish synthesis
PIPER_TIMEOUT_SECONDS = 60.0


def _ensure_prerequisites() -> None:
    """
    Verify that Piper, the voice model, and the output directory are available.

    Raises:
        FileNotFoundError: If the Piper binary or model file is missing.
        OSError: If the temp directory cannot be created.
    """
    if not PIPER_BINARY.is_file():
        raise FileNotFoundError(f"Piper binary not found: {PIPER_BINARY}")

    if not PIPER_MODEL.is_file():
        raise FileNotFoundError(f"Piper model not found: {PIPER_MODEL}")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def _build_output_path() -> Path:
    """
    Build a unique WAV file path inside backend/temp/.

    A UUID prefix prevents collisions when multiple requests synthesize
    speech at the same time.
    """
    unique_name = f"{uuid.uuid4()}.wav"
    return TEMP_DIR / unique_name


def generate_speech(text: str) -> str:
    """
    Synthesize speech from text and write a WAV file with Piper.

    Piper reads the input text from stdin and writes audio to the output file
    specified by ``--output_file``.

    Args:
        text: The text to speak aloud.

    Returns:
        The absolute path to the generated WAV file as a string.

    Raises:
        ValueError: If the input text is empty or whitespace-only.
        FileNotFoundError: If Piper or the model file is missing.
        RuntimeError: If Piper fails, times out, or produces no output file.
    """
    if not text or not text.strip():
        raise ValueError("Text is empty; cannot generate speech.")

    _ensure_prerequisites()
    output_path = _build_output_path()

    # Build the Piper command as a list to avoid shell injection
    command = [
        str(PIPER_BINARY),
        "--model",
        str(PIPER_MODEL),
        "--output_file",
        str(output_path),
        "--quiet",
    ]

    try:
        result = subprocess.run(
            command,
            input=text,
            text=True,
            capture_output=True,
            check=False,
            timeout=PIPER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        logger.exception("Piper synthesis timed out after %s seconds", PIPER_TIMEOUT_SECONDS)
        raise RuntimeError("Piper speech synthesis timed out.") from exc
    except FileNotFoundError as exc:
        logger.exception("Piper binary could not be executed")
        raise FileNotFoundError(f"Piper binary not found: {PIPER_BINARY}") from exc
    except OSError as exc:
        logger.exception("Failed to run Piper subprocess")
        raise RuntimeError(f"Failed to execute Piper: {exc}") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        logger.error("Piper exited with code %d: %s", result.returncode, stderr)
        raise RuntimeError(
            f"Piper speech synthesis failed (exit code {result.returncode})."
            + (f" Details: {stderr}" if stderr else "")
        )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Piper did not produce a valid WAV file.")

    logger.info("Generated speech at %s (%d bytes)", output_path, output_path.stat().st_size)
    return str(output_path.resolve())
