"""Tests for TTS client streaming mode."""

import io
import struct
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("TWILIO_ACCOUNT_SID", "test")
os.environ.setdefault("TWILIO_AUTH_TOKEN", "test")
os.environ.setdefault("TWILIO_API_KEY", "test")
os.environ.setdefault("TWILIO_API_SECRET", "test")
os.environ.setdefault("TWILIO_PHONE", "+10000000000")

# Mock heavy deps
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("faster_whisper", MagicMock())
sys.modules.setdefault("twilio", MagicMock())
sys.modules.setdefault("twilio.request_validator", MagicMock())
sys.modules.setdefault("twilio.rest", MagicMock())


def _make_sine_chunk(n_samples: int = 480, sample_rate: int = 24000) -> np.ndarray:
    t = np.linspace(0, n_samples / sample_rate, n_samples, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _build_stream_response(chunks: list[np.ndarray], sample_rate: int = 24000) -> bytes:
    """Build a binary PCM stream response from chunks."""
    buf = io.BytesIO()
    buf.write(struct.pack("<IHH", sample_rate, 1, 32))
    for chunk in chunks:
        pcm = chunk.tobytes()
        buf.write(struct.pack("<I", len(pcm)))
        buf.write(pcm)
    buf.write(struct.pack("<I", 0))
    return buf.getvalue()


class FakeStreamResponse:
    """Mock httpx response that yields bytes incrementally."""

    def __init__(self, data: bytes, chunk_size: int = 512):
        self._data = data
        self._chunk_size = chunk_size
        self.status_code = 200

    def raise_for_status(self):
        pass

    async def aiter_bytes(self, chunk_size=None):
        cs = chunk_size or self._chunk_size
        for i in range(0, len(self._data), cs):
            yield self._data[i : i + cs]

    async def aclose(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
async def test_synthesize_stream_yields_mulaw_chunks():
    """Streaming synthesize should yield base64 mulaw chunks as PCM arrives."""
    audio_chunks = [_make_sine_chunk(2400), _make_sine_chunk(2400)]
    stream_data = _build_stream_response(audio_chunks)
    fake_resp = FakeStreamResponse(stream_data, chunk_size=256)

    with patch("tts._client") as mock_client:
        mock_client.stream = MagicMock(return_value=fake_resp)

        from tts import synthesize_stream
        mulaw_chunks = []
        async for chunk in synthesize_stream("Hello there"):
            mulaw_chunks.append(chunk)

    assert len(mulaw_chunks) > 0
    for chunk in mulaw_chunks:
        assert isinstance(chunk, str)  # base64 encoded


@pytest.mark.asyncio
async def test_synthesize_stream_handles_single_chunk():
    """Stream with just one audio chunk still works."""
    audio_chunks = [_make_sine_chunk(4800)]
    stream_data = _build_stream_response(audio_chunks)
    fake_resp = FakeStreamResponse(stream_data, chunk_size=1024)

    with patch("tts._client") as mock_client:
        mock_client.stream = MagicMock(return_value=fake_resp)

        from tts import synthesize_stream
        mulaw_chunks = []
        async for chunk in synthesize_stream("Hi"):
            mulaw_chunks.append(chunk)

    assert len(mulaw_chunks) > 0


@pytest.mark.asyncio
async def test_original_synthesize_still_works():
    """The non-streaming synthesize function should still work."""
    import soundfile as sf

    audio = _make_sine_chunk(4800, sample_rate=24000)
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, 24000, format="WAV")
    wav_bytes = wav_buf.getvalue()

    mock_resp = AsyncMock()
    mock_resp.content = wav_bytes
    mock_resp.raise_for_status = MagicMock()

    with patch("tts._client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_resp)

        from tts import synthesize
        chunks = await synthesize("Hello")

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, str)
