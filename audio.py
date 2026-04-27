"""Audio format conversion utilities for Twilio <-> pipeline."""

import audioop
import base64
import io

import numpy as np
import soundfile as sf

# Twilio Media Streams: mulaw 8kHz mono
TWILIO_RATE = 8000
# Whisper expects 16kHz
WHISPER_RATE = 16000
# Kokoro outputs 24kHz
KOKORO_RATE = 24000


def mulaw_decode(data: bytes) -> bytes:
    """Decode mulaw bytes to PCM 16-bit signed."""
    return audioop.ulaw2lin(data, 2)


def mulaw_encode(pcm: bytes) -> bytes:
    """Encode PCM 16-bit signed to mulaw."""
    return audioop.lin2ulaw(pcm, 2)


def resample_pcm(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample PCM 16-bit mono using linear interpolation."""
    if from_rate == to_rate:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    ratio = to_rate / from_rate
    n_out = int(len(samples) * ratio)
    indices = np.arange(n_out) / ratio
    indices = np.clip(indices, 0, len(samples) - 1)
    idx_floor = indices.astype(np.int64)
    idx_ceil = np.minimum(idx_floor + 1, len(samples) - 1)
    frac = indices - idx_floor
    resampled = samples[idx_floor] * (1 - frac) + samples[idx_ceil] * frac
    return resampled.astype(np.int16).tobytes()


def twilio_payload_to_pcm16k(payload: str) -> bytes:
    """Decode a Twilio media payload (base64 mulaw 8kHz) to PCM 16kHz."""
    raw = base64.b64decode(payload)
    pcm_8k = mulaw_decode(raw)
    return resample_pcm(pcm_8k, TWILIO_RATE, WHISPER_RATE)


def wav_bytes_to_mulaw_payloads(wav_data: bytes, chunk_ms: int = 20) -> list[str]:
    """Convert Kokoro WAV output to base64 mulaw 8kHz chunks for Twilio.

    Returns a list of base64-encoded mulaw chunks, each ~chunk_ms long.
    """
    buf = io.BytesIO(wav_data)
    audio, rate = sf.read(buf, dtype="int16")

    # Ensure mono
    if audio.ndim > 1:
        audio = audio[:, 0]

    pcm = audio.astype(np.int16).tobytes()

    # Resample to 8kHz
    pcm_8k = resample_pcm(pcm, rate, TWILIO_RATE)

    # Encode to mulaw
    mulaw = mulaw_encode(pcm_8k)

    # Split into chunks
    chunk_size = TWILIO_RATE * chunk_ms // 1000  # bytes per chunk (1 byte per mulaw sample)
    chunks = []
    for i in range(0, len(mulaw), chunk_size):
        chunk = mulaw[i : i + chunk_size]
        chunks.append(base64.b64encode(chunk).decode("ascii"))

    return chunks
