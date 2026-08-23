"""Async Kokoro TTS client with audio format conversion for Twilio."""

import base64
import logging
import struct
from collections.abc import AsyncGenerator

import numpy as np

import httpx

from audio import wav_bytes_to_mulaw_payloads, resample_pcm, mulaw_encode, TWILIO_RATE
from config import TTS_URL, TTS_VOICE

logger = logging.getLogger("tts")

# Reusable client — avoids connection setup per request
_client = httpx.AsyncClient(timeout=30.0)

MULAW_CHUNK_SAMPLES = TWILIO_RATE * 20 // 1000  # 20ms at 8kHz = 160 samples


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


async def synthesize_stream(
    text: str,
    voice: str = TTS_VOICE,
    speed: float = 1.0,
) -> AsyncGenerator[str, None]:
    """Stream TTS: yield base64 mulaw chunks as PCM arrives from the server."""
    async with _client.stream(
        "POST",
        f"{TTS_URL}/tts/stream",
        json={"text": text, "voice": voice, "speed": speed},
    ) as resp:
        resp.raise_for_status()

        raw = bytearray()
        header_parsed = False
        sample_rate = 0

        async for data in resp.aiter_bytes():
            raw.extend(data)

            if not header_parsed and len(raw) >= 8:
                sample_rate, _ch, _bd = struct.unpack("<IHH", bytes(raw[:8]))
                del raw[:8]
                header_parsed = True

            if not header_parsed:
                continue

            while len(raw) >= 4:
                chunk_len = struct.unpack("<I", bytes(raw[:4]))[0]
                if chunk_len == 0:
                    del raw[:4]
                    return
                if len(raw) < 4 + chunk_len:
                    break

                pcm_f32 = np.frombuffer(bytes(raw[4 : 4 + chunk_len]), dtype=np.float32)
                del raw[: 4 + chunk_len]

                pcm_i16 = (pcm_f32 * 32767).astype(np.int16).tobytes()
                pcm_8k = resample_pcm(pcm_i16, sample_rate, TWILIO_RATE)
                mulaw = mulaw_encode(pcm_8k)

                for i in range(0, len(mulaw), MULAW_CHUNK_SAMPLES):
                    chunk = mulaw[i : i + MULAW_CHUNK_SAMPLES]
                    yield base64.b64encode(chunk).decode("ascii")
