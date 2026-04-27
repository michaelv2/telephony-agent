"""Async Kokoro TTS client with audio format conversion for Twilio."""

import logging

import httpx

from audio import wav_bytes_to_mulaw_payloads
from config import TTS_URL, TTS_VOICE

logger = logging.getLogger("tts")

# Reusable client — avoids connection setup per request
_client = httpx.AsyncClient(timeout=30.0)


async def synthesize(
    text: str,
    voice: str = TTS_VOICE,
    speed: float = 1.0,
) -> list[str]:
    """Synthesize text and return base64 mulaw 8kHz chunks for Twilio."""
    resp = await _client.post(
        f"{TTS_URL}/tts",
        json={"text": text, "voice": voice, "speed": speed, "format": "wav"},
    )
    resp.raise_for_status()

    wav_data = resp.content
    logger.info(f"TTS: {len(wav_data)} bytes WAV for {len(text)} chars")
    return wav_bytes_to_mulaw_payloads(wav_data)
