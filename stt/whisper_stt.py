"""Faster-Whisper STT with Silero VAD for speech boundary detection."""

import asyncio
import logging
import time

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
        early_threshold_ms: int | None = None,
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
        # Transcribe after the early threshold; the turn only ends at the full one. See
        # CHANGELOG-latency.md: this hides Whisper's time inside the remaining silence.
        self._early_frames_needed = min(
            self._silence_frames_needed,
            int((early_threshold_ms or silence_threshold_ms) / 1000 * SAMPLE_RATE / VAD_FRAME_SAMPLES),
        )

        self._audio_buffer = bytearray()
        self._speech_active = False
        self._silence_count = 0
        self._pending_transcript: str | None = None
        self._vad_leftover = bytearray()
        # time.monotonic() of the last speech frame of the most recent utterance
        self.speech_ended_at: float | None = None
        self._eot_silence: int | None = None  # silence frames so far, while waiting for end of turn
        self._end_of_turn = False

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
                self.speech_ended_at = time.monotonic()
                self._silence_count = 0
                self._eot_silence = None  # caller kept talking: the turn hasn't ended
                self._audio_buffer.extend(frame)
            elif self._speech_active:
                self._silence_count += 1
                self._audio_buffer.extend(frame)

                if self._silence_count >= self._early_frames_needed:
                    silence = self._silence_count
                    await self._transcribe()
                    self._eot_silence = silence
                    self._check_end_of_turn()
            elif self._eot_silence is not None:
                self._eot_silence += 1
                self._check_end_of_turn()

    def _check_end_of_turn(self) -> None:
        if self._eot_silence >= self._silence_frames_needed:
            self._eot_silence = None
            self._end_of_turn = True

    async def _transcribe(self) -> None:
        if not self._audio_buffer:
            self.reset()
            return

        audio = (
            np.frombuffer(bytes(self._audio_buffer), dtype=np.int16).astype(np.float32)
            / 32768.0
        )

        loop = asyncio.get_event_loop()
        t0 = time.monotonic()
        segments, _ = await loop.run_in_executor(
            None,
            lambda: self.whisper.transcribe(audio, language="en", beam_size=1),
        )
        text = " ".join(s.text.strip() for s in segments).strip()

        if text:
            done = time.monotonic()
            logger.info(
                f"Transcript (+{(done - self.speech_ended_at) * 1000:.0f}ms after speech end, "
                f"whisper {(done - t0) * 1000:.0f}ms): {text}"
            )
            self._pending_transcript = text

        self.reset()

    async def get_transcript(self) -> str | None:
        result = self._pending_transcript
        self._pending_transcript = None
        return result

    def pop_end_of_turn(self) -> bool:
        result = self._end_of_turn
        self._end_of_turn = False
        return result

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    def reset(self) -> None:
        self._audio_buffer.clear()
        self._speech_active = False
        self._silence_count = 0
        self._vad_leftover.clear()
        self._eot_silence = None
        self._end_of_turn = False
        self.vad_model.reset_states()
