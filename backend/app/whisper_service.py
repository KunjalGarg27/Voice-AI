"""Speech-to-text transcription using faster-whisper."""

import logging
from pathlib import Path
import subprocess
import tempfile

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

    # Convert any uploaded audio (webm/wav/mp3/etc.) into a clean
    # 16 kHz mono WAV file before sending it to Whisper.
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav_path = temp_wav.name

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                wav_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    except Exception as exc:
        logger.exception("FFmpeg audio conversion failed")
        raise RuntimeError(
            f"Failed to convert audio for transcription: {exc}"
        ) from exc
    audio_for_whisper = wav_path    
    model = _get_whisper_model()

    try:
        segments, info = model.transcribe(
        audio_for_whisper,
        beam_size=5,
        initial_prompt="""
        Clean energy marketplace vocabulary:

        solar panel
        solar panels
        rooftop solar
        photovoltaic
        solar inverter
        lithium battery
        lithium ion battery
        battery storage
        energy storage system
        EV charger
        electric vehicle charger
        charging station
        renewable energy
        clean energy
        green energy
        sustainability
        carbon footprint
        energy efficient
        energy conservation
        smart meter
        smart grid
        microgrid
        wind turbine
        hydrogen fuel cell
        biogas
        biomass
        geothermal energy
        net zero
        electric vehicle
        EV battery
        solar water heater
        LED lighting
        energy management system
        watt
        wattage
        watt hour
        watt hour meter
        watt hour meter reading
        watt hour meter reading
        units
        Luminous
        Exide
        Amaron
        Livguard
        Tata Power Solar
        Waaree
        Adani Solar
        Vikram Solar
        Microtek
        Schneider Electric
        Havells
        Anchor


        User may speak English, Hindi, Punjabi, or Hinglish.
        """
)

        segments = list(segments)

        print("\n================ WHISPER DEBUG ================")
        print("LANGUAGE:", info.language)
        print("PROBABILITY:", info.language_probability)
        print("NUMBER OF SEGMENTS:", len(segments))

        for seg in segments:
            print("SEGMENT:", repr(seg.text))

        transcript_parts = [
            seg.text.strip()
            for seg in segments
            if seg.text.strip()
        ]

        transcript = " ".join(transcript_parts)

        print("FINAL TRANSCRIPT:", repr(transcript))
        print("=============================================\n")
        COMMON_FIXES = {
            "solar penal": "solar panel",
            "solar pannel": "solar panel",
            "lithium bettery": "lithium battery",
            "ev charge": "EV charger",
            "charging point": "charging station",
        }
        transcript_lower = transcript.lower()

        for wrong, right in COMMON_FIXES.items():
            transcript_lower = transcript_lower.replace(wrong, right)

        transcript = transcript_lower

    except Exception as exc:
        logger.exception("Whisper transcription failed for %s", audio_path)
        raise RuntimeError(
            f"Speech-to-text transcription failed: {exc}"
        ) from exc

    if not transcript.strip():
        raise RuntimeError("No speech detected in the audio file.")

    logger.info("Transcription complete (%d characters)", len(transcript))
    return transcript