"""Tests for two-stage endpointing in WhisperSTT.

The transcript should arrive at the early silence threshold, and the end of turn only at the
full threshold, unless the caller starts speaking again first. VAD and Whisper are faked.
"""

import asyncio
import os
import sys
import time
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(__file__))
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("faster_whisper", MagicMock())

from stt.whisper_stt import BYTES_PER_SAMPLE, VAD_FRAME_SAMPLES, WhisperSTT

FRAME = b"\x00" * (VAD_FRAME_SAMPLES * BYTES_PER_SAMPLE)  # one 32 ms VAD frame


class FakeVAD:
    def __init__(self):
        self.next_prob = 0.0

    def __call__(self, tensor, sr):
        prob = self.next_prob
        return MagicMock(item=lambda: prob)

    def reset_states(self):
        pass


def make_stt(early_ms, silence_ms=800):
    stt = WhisperSTT.__new__(WhisperSTT)
    stt.vad_model = FakeVAD()
    stt.whisper = MagicMock()
    stt.whisper.transcribe.return_value = ([MagicMock(text=" hello there.")], None)
    frames = lambda ms: int(ms / 1000 * 16000 / VAD_FRAME_SAMPLES)
    stt._silence_frames_needed = frames(silence_ms)
    stt._early_frames_needed = min(stt._silence_frames_needed, frames(early_ms))
    stt._audio_buffer = bytearray()
    stt._speech_active = False
    stt._speech_frames = 0
    stt._silence_count = 0
    stt._pending_transcript = None
    stt._vad_leftover = bytearray()
    stt.speech_ended_at = None
    stt._eot_silence = None
    stt._end_of_turn = False
    return stt


def feed(stt, prob, n):
    """Feed n frames with the given speech probability; return per-frame (transcript, end_of_turn)."""
    events = []
    for _ in range(n):
        stt.vad_model.next_prob = prob
        asyncio.run(stt.feed_audio(FRAME))
        events.append((asyncio.run(stt.get_transcript()), stt.pop_end_of_turn()))
    return events


def first(events, which):
    return next((i for i, e in enumerate(events) if e[which]), None)


class TestTwoStage:
    def test_transcript_at_early_threshold_end_of_turn_at_full(self):
        stt = make_stt(early_ms=400)
        feed(stt, 0.9, 20)
        events = feed(stt, 0.0, 40)
        assert first(events, 0) == 11  # 12th silent frame = 384 ms
        assert first(events, 1) == 24  # 25th silent frame = 800 ms
        assert sum(bool(e[1]) for e in events) == 1

    def test_speech_resuming_cancels_end_of_turn(self):
        stt = make_stt(early_ms=400)
        feed(stt, 0.9, 20)
        pause = feed(stt, 0.0, 15)  # transcribed, but under 800 ms of silence
        assert first(pause, 0) == 11 and first(pause, 1) is None
        resumed = feed(stt, 0.9, 10)
        assert first(resumed, 1) is None
        after = feed(stt, 0.0, 40)
        assert first(after, 0) == 11 and first(after, 1) == 24

    def test_equal_thresholds_end_turn_with_transcript(self):
        stt = make_stt(early_ms=800)
        feed(stt, 0.9, 20)
        events = feed(stt, 0.0, 40)
        assert first(events, 0) == 24 and first(events, 1) == 24

    def test_silence_without_speech_never_ends_a_turn(self):
        stt = make_stt(early_ms=400)
        events = feed(stt, 0.0, 60)
        assert first(events, 0) is None and first(events, 1) is None


class TestSpeechMs:
    """speech_ms counts speech frames only, so a cough stays short while speech_active stays on."""

    def test_counts_speech_frames_not_the_silence_after_them(self):
        stt = make_stt(early_ms=400)
        feed(stt, 0.9, 3)  # a 96 ms cough
        feed(stt, 0.0, 5)
        assert stt.speech_active and stt.speech_ms == 96
        feed(stt, 0.9, 5)
        assert stt.speech_ms == 256

    def test_resets_when_the_utterance_is_transcribed(self):
        stt = make_stt(early_ms=400)
        feed(stt, 0.9, 10)
        feed(stt, 0.0, 12)  # transcribed at the early threshold
        assert not stt.speech_active and stt.speech_ms == 0


class TestDecoding:
    def test_decoding_runs_off_the_event_loop(self):
        """faster-whisper decodes while its segment generator is consumed; that must not block."""
        stt = make_stt(early_ms=400)

        def slow_segments():
            time.sleep(0.2)
            yield MagicMock(text=" hello there.")

        stt.whisper.transcribe.return_value = (slow_segments(), None)

        async def main():
            ticks = 0

            async def ticker():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.01)
                    ticks += 1

            task = asyncio.create_task(ticker())
            stt.vad_model.next_prob = 0.9
            for _ in range(10):
                await stt.feed_audio(FRAME)
            stt.vad_model.next_prob = 0.0
            for _ in range(12):  # transcribed at the early threshold
                await stt.feed_audio(FRAME)
            task.cancel()
            return ticks, await stt.get_transcript()

        ticks, text = asyncio.run(main())
        assert text == "hello there."
        assert ticks >= 10  # the loop kept running during the 200 ms decode

    def test_no_temperature_retries(self):
        stt = make_stt(early_ms=400)
        feed(stt, 0.9, 10)
        feed(stt, 0.0, 12)
        assert stt.whisper.transcribe.call_args.kwargs["temperature"] == 0.0
