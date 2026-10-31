"""Scenario tests for turn-taking: when the agent replies, holds, or gets cut off.

Each test drives the real /ws/media handler with a scripted call. STT and the reply pipeline are
faked, so no audio, GPU, Claude or TTS is involved. A media frame's payload is a command for the
fake STT:
  "speech"       VAD hears speech (the caller is talking)
  "silence"      VAD hears silence
  "text:<...>"   Whisper's transcript arrives (in production: after VAD_EARLY_MS of silence)
  "eot"          end of turn (in production: after VAD_SILENCE_MS of silence)

Each test checks the log of what the agent did: ("reply", text) when a reply starts, ("done", text)
when it finishes sending, ("cancelled", text) on barge-in. ("note", ...) entries mark points in
the script, so a test can tell what happened before and after them.
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("faster_whisper", MagicMock())
sys.modules.setdefault("twilio", MagicMock())
sys.modules.setdefault("twilio.request_validator", MagicMock())
sys.modules.setdefault("twilio.rest", MagicMock())
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("TWILIO_ACCOUNT_SID", "test")
os.environ.setdefault("TWILIO_AUTH_TOKEN", "test")
os.environ.setdefault("TWILIO_API_KEY", "test")
os.environ.setdefault("TWILIO_API_SECRET", "test")
os.environ.setdefault("TWILIO_PHONE", "+10000000000")

from fastapi import WebSocketDisconnect  # noqa: E402

import server  # noqa: E402

HOLD_S = 0.1  # hold timeout used in these tests (production: 1s)
GREETING_S = 0.02  # how long the fake greeting takes to send
REPLY_S = 0.3  # how long a fake reply takes to send


@dataclass
class Wait:
    seconds: float


@dataclass
class Note:
    text: str


def media(cmd: str) -> dict:
    return {"event": "media", "media": {"payload": cmd}}


START = {"event": "start", "start": {"streamSid": "MZtest", "callSid": "CAtest"}}
STOP = {"event": "stop"}
SPEECH, SILENCE, EOT = media("speech"), media("silence"), media("eot")


def text(t: str) -> dict:
    return media(f"text:{t}")


class FakeWS:
    """Delivers the script one message at a time, then disconnects."""

    def __init__(self, script: list, log: list):
        self.script = list(script)
        self.log = log

    async def accept(self):
        pass

    async def send_text(self, data: str):
        pass

    async def receive_text(self) -> str:
        while self.script:
            item = self.script.pop(0)
            if isinstance(item, Wait):
                await asyncio.sleep(item.seconds)
            elif isinstance(item, Note):
                self.log.append(("note", item.text))
            else:
                await asyncio.sleep(0.001)  # let tasks run, like real network I/O
                return json.dumps(item)
        raise WebSocketDisconnect()


class FakeSTT:
    """Plays back the signals WhisperSTT would produce."""

    def __init__(self):
        self.speech_active = False
        self.speech_ended_at = 0.0
        self._transcript = None
        self._end_of_turn = False

    async def feed_audio(self, cmd: str) -> None:
        self.speech_active = cmd == "speech"
        if cmd.startswith("text:"):
            self._transcript = cmd[len("text:"):]
        elif cmd == "eot":
            self._end_of_turn = True

    async def get_transcript(self):
        t, self._transcript = self._transcript, None
        return t

    def pop_end_of_turn(self) -> bool:
        e, self._end_of_turn = self._end_of_turn, False
        return e


class FakeTranscriptLog:
    def __init__(self, *a, **k):
        pass

    def user(self, *a):
        pass

    def agent(self, *a):
        pass

    def save(self):
        pass


def run_call(monkeypatch, script: list, greeting_s: float = GREETING_S) -> list:
    """Run one scripted call through server.media_stream and return the event log."""
    log: list = []

    async def fake_reply(ws, agent, stream_sid, call_sid, connected, speaking,
                         transcript_log=None, transfer_number=None, user_text=None,
                         greeting=False, on_done=None, turn_t0=None):
        label = "greeting" if greeting else user_text
        log.append(("reply", label))
        speaking.set()
        try:
            await asyncio.sleep(greeting_s if greeting else REPLY_S)
            log.append(("done", label))
        except asyncio.CancelledError:
            log.append(("cancelled", label))
        finally:
            speaking.clear()
            if on_done:
                on_done.set()

    monkeypatch.setattr(server, "stt_engine", FakeSTT())
    monkeypatch.setattr(server, "_stream_agent_response", fake_reply)
    monkeypatch.setattr(server, "twilio_payload_to_pcm16k", lambda payload: payload)
    monkeypatch.setattr(server, "CallAgent", lambda context=None: object())
    monkeypatch.setattr(server, "TranscriptLog", FakeTranscriptLog)
    monkeypatch.setattr(server, "_HOLD_TIMEOUT_SECONDS", HOLD_S)

    async def main():
        await server.media_stream(FakeWS(script, log))
        await asyncio.sleep(0.02)  # let a reply started by "stop" begin
        snapshot = list(log)
        rest = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in rest:
            t.cancel()
        await asyncio.gather(*rest, return_exceptions=True)
        return snapshot

    return asyncio.run(main())


GREETED = [START, Wait(0.05)]  # call starts and the greeting finishes sending
GREETING_LOG = [("reply", "greeting"), ("done", "greeting")]


# --- Replying ---


def test_question_replies_at_the_early_mark(monkeypatch):
    log = run_call(monkeypatch, GREETED + [
        SPEECH, SPEECH, text("Do you have Tuesday?"), Wait(0.03), Note("before end of turn"), EOT,
        Wait(0.03),
    ])
    assert log == GREETING_LOG + [
        ("reply", "Do you have Tuesday?"),
        ("note", "before end of turn"),  # end of turn afterwards adds nothing
    ]


def test_statement_waits_for_end_of_turn(monkeypatch):
    log = run_call(monkeypatch, GREETED + [
        SPEECH, text("I'd like to book a cleaning."), SILENCE, Wait(0.03), Note("before end of turn"),
        EOT, Wait(0.03),
    ])
    assert log == GREETING_LOG + [
        ("note", "before end of turn"),
        ("reply", "I'd like to book a cleaning."),
    ]


def test_pieces_before_end_of_turn_are_joined(monkeypatch):
    log = run_call(monkeypatch, GREETED + [
        SPEECH, text("I want to"), SPEECH, text("book a cleaning."), EOT, Wait(0.03),
    ])
    assert log == GREETING_LOG + [("reply", "I want to book a cleaning.")]


# --- Holding unfinished turns ---


def test_unfinished_turn_is_held_then_flushed(monkeypatch):
    log = run_call(monkeypatch, GREETED + [
        SPEECH, text("I want to book a cleaning and"), EOT, Wait(0.03), Note("holding"),
        Wait(HOLD_S + 0.05),
    ])
    assert log == GREETING_LOG + [
        ("note", "holding"),
        ("reply", "I want to book a cleaning and"),
    ]


def test_speech_during_hold_cancels_flush_and_joins_continuation(monkeypatch):
    log = run_call(monkeypatch, GREETED + [
        SPEECH, text("I want to book a cleaning and"), EOT, Wait(0.03),
        SPEECH, Wait(HOLD_S + 0.05), Note("past the hold timeout"),
        text("for next Tuesday."), EOT, Wait(0.03),
    ])
    assert log == GREETING_LOG + [
        ("note", "past the hold timeout"),
        ("reply", "I want to book a cleaning and for next Tuesday."),
    ]


def test_noise_during_hold_restarts_the_hold(monkeypatch):
    # Speech with no transcript (a cough Whisper drops) cancels the hold, then the next end of
    # turn re-checks the held text and starts a fresh hold.
    log = run_call(monkeypatch, GREETED + [
        SPEECH, text("I want to book a cleaning and"), EOT, Wait(HOLD_S * 0.6),
        SPEECH, SILENCE, EOT, Wait(HOLD_S * 0.6), Note("first hold would have fired"),
        Wait(HOLD_S),
    ])
    assert log == GREETING_LOG + [
        ("note", "first hold would have fired"),
        ("reply", "I want to book a cleaning and"),
    ]


def test_question_after_held_text_replies_with_both(monkeypatch):
    log = run_call(monkeypatch, GREETED + [
        SPEECH, text("so I was thinking"), EOT, Wait(0.03),
        SPEECH, text("can you do Tuesday?"), Wait(0.03),
    ])
    assert log == GREETING_LOG + [("reply", "so I was thinking can you do Tuesday?")]


def test_stop_flushes_held_text(monkeypatch):
    log = run_call(monkeypatch, GREETED + [
        SPEECH, text("I want to book a cleaning and"), STOP,
    ])
    assert log == GREETING_LOG + [("reply", "I want to book a cleaning and")]


# --- Barge-in ---


def test_speech_during_reply_cancels_it(monkeypatch):
    log = run_call(monkeypatch, GREETED + [
        SPEECH, text("Do you have Tuesday?"), Wait(0.03), SPEECH, Wait(0.03),
    ])
    assert log == GREETING_LOG + [
        ("reply", "Do you have Tuesday?"),
        ("cancelled", "Do you have Tuesday?"),
    ]


def test_speech_after_reply_finished_sending_does_not_cancel(monkeypatch):
    # The reply has been sent but would still be playing on the phone. Barge-in isn't armed
    # (TODO.md: track playback with Twilio marks).
    log = run_call(monkeypatch, GREETED + [
        SPEECH, text("Do you have Tuesday?"), Wait(REPLY_S + 0.05), SPEECH, Wait(0.03),
    ])
    assert log == GREETING_LOG + [
        ("reply", "Do you have Tuesday?"),
        ("done", "Do you have Tuesday?"),
    ]


def test_no_barge_in_during_greeting(monkeypatch):
    log = run_call(monkeypatch, [START, Wait(0.02), SPEECH, Wait(0.03)], greeting_s=0.3)
    assert log == [("reply", "greeting")]


# --- Current quirks (pinned so a change to them is deliberate) ---


def test_quirk_question_during_greeting_starts_second_reply(monkeypatch):
    # Both replies then send audio at the same time.
    log = run_call(monkeypatch, [
        START, Wait(0.02), SPEECH, text("Hello?"), Wait(0.03),
    ], greeting_s=0.3)
    assert log == [("reply", "greeting"), ("reply", "Hello?")]


def test_quirk_greeting_finishing_disarms_barge_in_for_overlapping_reply(monkeypatch):
    # Both replies share one `speaking` flag, so the greeting clears it while "Hello?" is still
    # being sent, and speech can no longer cut "Hello?" off.
    log = run_call(monkeypatch, [
        START, Wait(0.02), SPEECH, text("Hello?"), Wait(0.15), SPEECH, Wait(0.03),
    ], greeting_s=0.1)
    assert log == [("reply", "greeting"), ("reply", "Hello?"), ("done", "greeting")]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
