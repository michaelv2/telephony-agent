"""Faster-Whisper STT with Silero VAD for speech boundary detection."""

import asyncio
import logging

import numpy as np
import torch
from faster_whisper import WhisperModel

from .base import BaseSTT

logger = logging.getLogger("stt.whisper")

# Silero VAD works on 16kHz 512-sample (32ms) frames
VAD_FRAME_SAMPLES = 512
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit


class WhisperSTT(BaseSTT):
    def __init__(
        self,
        model_size: str = "base.en",
        silence_threshold_ms: int = 800,
        device: str = "cuda",
    ):
        logger.info(f"Loading Whisper model '{model_size}' on {device}...")
        self.whisper = WhisperModel(model_size, device=device, compute_type="float16")

        logger.info("Loading Silero VAD...")
        self.vad_model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self.vad_model.eval()

        self._silence_frames_needed = int(
            silence_threshold_ms / 1000 * SAMPLE_RATE / VAD_FRAME_SAMPLES
        )

        self._audio_buffer = bytearray()
        self._speech_active = False
        self._silence_count = 0
        self._pending_transcript: str | None = None
        self._vad_leftover = bytearray()

        logger.info("WhisperSTT ready")

    async def feed_audio(self, pcm_16k: bytes) -> None:
        self._vad_leftover.extend(pcm_16k)

        frame_bytes = VAD_FRAME_SAMPLES * BYTES_PER_SAMPLE

        while len(self._vad_leftover) >= frame_bytes:
            frame = bytes(self._vad_leftover[:frame_bytes])
            del self._vad_leftover[:frame_bytes]

            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            tensor = torch.from_numpy(samples)

            with torch.no_grad():
                prob = self.vad_model(tensor, SAMPLE_RATE).item()

            is_speech = prob > 0.5

            if is_speech:
                self._speech_active = True
                self._silence_count = 0
                self._audio_buffer.extend(frame)
            elif self._speech_active:
                self._silence_count += 1
                self._audio_buffer.extend(frame)

                if self._silence_count >= self._silence_frames_needed:
                    await self._transcribe()

    async def _transcribe(self) -> None:
        if not self._audio_buffer:
            self.reset()
            return

        audio = (
            np.frombuffer(bytes(self._audio_buffer), dtype=np.int16).astype(np.float32)
            / 32768.0
        )

        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None,
            lambda: self.whisper.transcribe(audio, language="en", beam_size=1),
        )
        text = " ".join(s.text.strip() for s in segments).strip()

        if text:
            logger.info(f"Transcript: {text}")
            self._pending_transcript = text

        self.reset()

    async def get_transcript(self) -> str | None:
        result = self._pending_transcript
        self._pending_transcript = None
        return result

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    def reset(self) -> None:
        self._audio_buffer.clear()
        self._speech_active = False
        self._silence_count = 0
        self._vad_leftover.clear()
        self.vad_model.reset_states()
