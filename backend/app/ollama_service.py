"""LLM response generation via a local Ollama server."""

import logging

import httpx

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"


async def generate_response(transcript: str) -> str:
    """
    Send a transcript to the local Ollama server and return the model reply.

    Uses the Ollama chat API with model qwen2.5:7b.

    Args:
        transcript: The user's speech transcribed to text.

    Returns:
        The assistant's text response from Qwen.

    Raises:
        ValueError: If the transcript is empty.
        RuntimeError: If Ollama is unreachable or returns an error response.
    """
    if not transcript.strip():
        raise ValueError("Transcript is empty; cannot generate a response.")

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": transcript,
            }
        ],
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            response = await client.post(OLLAMA_CHAT_URL, json=payload)
    except httpx.ConnectError as exc:
        logger.exception("Could not connect to Ollama at %s", OLLAMA_BASE_URL)
        raise RuntimeError(
            f"Ollama is not reachable at {OLLAMA_BASE_URL}. "
            "Ensure Ollama is running locally."
        ) from exc
    except httpx.TimeoutException as exc:
        logger.exception("Ollama request timed out")
        raise RuntimeError("Ollama request timed out while generating a response.") from exc
    except httpx.HTTPError as exc:
        logger.exception("HTTP error while calling Ollama")
        raise RuntimeError(f"Failed to communicate with Ollama: {exc}") from exc

    if response.status_code != 200:
        logger.error(
            "Ollama returned status %d: %s",
            response.status_code,
            response.text,
        )
        raise RuntimeError(
            f"Ollama returned an error (status {response.status_code})."
        )

    try:
        data = response.json()
        message = data.get("message", {})
        assistant_text = message.get("content", "").strip()
    except (ValueError, AttributeError) as exc:
        logger.exception("Unexpected Ollama response format")
        raise RuntimeError("Received an invalid response from Ollama.") from exc

    if not assistant_text:
        raise RuntimeError("Ollama returned an empty response.")

    logger.info("Ollama response received (%d characters)", len(assistant_text))
    return assistant_text
