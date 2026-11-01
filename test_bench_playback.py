"""Tests for the bench's stand-in for Twilio's playback buffer (bench_turn_latency.Call).

The barge-in numbers depend on this model, so it's checked against Twilio's documented behaviour:
media plays in order in real time, a mark comes back once the audio before it has played (or at
once if nothing is buffered), and `clear` drops unplayed audio and sends back every pending mark.
"""

import base64
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from audio import mulaw_decode, mulaw_encode  # noqa: E402
from bench_turn_latency import FRAME_S, SILENCE, Call, speech_blip  # noqa: E402

MEDIA = {"event": "media", "media": {"payload": ""}}


def mark(name):
    return {"event": "mark", "mark": {"name": name}}


def feed(call, t0, n, step=0.0):
    """n agent frames arriving from t0, `step` seconds apart."""
    for i in range(n):
        call.on_message(MEDIA, t0 + i * step)


def test_frames_queue_and_play_in_real_time():
    call = Call(None)
    feed(call, 0.0, 3)  # a burst, faster than real time
    assert [s for _, s in call.frames] == pytest.approx([0.0, 0.02, 0.04])
    assert call.play_end == pytest.approx(0.06)


def test_mark_comes_back_after_its_audio_has_played():
    call = Call(None)
    feed(call, 0.0, 50)  # 1 s of audio
    call.on_message(mark("end of sentence"), 0.001)
    assert call.due_marks(0.99) == []
    assert call.due_marks(1.001) == ["end of sentence"]
    assert call.due_marks(1.1) == []  # sent back once


def test_mark_with_nothing_buffered_comes_back_at_once():
    call = Call(None)
    call.on_message(mark("m"), 5.0)
    assert call.due_marks(5.0) == ["m"]


def test_clear_drops_unplayed_audio_and_returns_pending_marks():
    call = Call(None)
    feed(call, 0.0, 50)
    call.on_message(mark("m"), 0.0)
    call.on_message({"event": "clear"}, 0.3)
    assert len(call.frames) == 15  # the frames that started before 0.3 s
    assert call.due_marks(0.3) == ["m"]
    assert call.heard_after(0.1) == pytest.approx(0.2)
    feed(call, 0.5, 1)  # new audio after a clear plays straight away
    assert call.frames[-1][1] == pytest.approx(0.5)


def test_heard_after_stops_at_a_gap_in_playback():
    call = Call(None)
    feed(call, 0.0, 25)  # 0-0.5 s
    feed(call, 1.0, 25)  # 1.0-1.5 s, after a 0.5 s gap
    assert call.heard_after(0.2) == pytest.approx(0.3)
    assert call.heard_after(0.7) is None  # nothing playing


def test_heard_after_bridges_a_short_gap():
    call = Call(None)
    feed(call, 0.0, 25)  # 0-0.5 s
    feed(call, 0.6, 25)  # 0.6-1.1 s
    assert call.heard_after(0.2) == pytest.approx(0.9)


def test_heard_after_ignores_the_next_reply():
    call = Call(None)
    feed(call, 0.0, 100)  # a 2 s reply, sent at once
    feed(call, 1.5, 50)  # the next reply arrives while it plays, and queues behind it (2-3 s)
    assert call.heard_after(0.5) == pytest.approx(2.5)
    assert call.heard_after(0.5, arrived_before=1.5) == pytest.approx(1.5)


def test_reply_starts_split_on_gaps():
    call = Call(None)
    feed(call, 0.0, 10, step=0.05)  # reply 1, arriving in pieces
    feed(call, 3.0, 10, step=0.05)  # reply 2
    assert call.reply_starts(-1) == pytest.approx([0.0, 3.0])
    assert call.reply_starts(0.2) == pytest.approx([3.0])


def test_blip_is_160ms_cut_from_the_start_of_speech():
    tone = (8000 * np.sin(2 * np.pi * 200 * np.arange(8000) / 8000)).astype(np.int16)  # 1 s, all loud
    mu = mulaw_encode(tone.tobytes())
    speech = ([SILENCE] * 5 + [base64.b64encode(mu[i:i + 160]).decode() for i in range(0, len(mu), 160)], 5, 54)
    frames, first, last = speech_blip(speech)
    assert len(frames) * FRAME_S == pytest.approx(0.16)
    assert (first, last) == (0, 7)
    edge = np.frombuffer(mulaw_decode(base64.b64decode(frames[0])), np.int16)
    assert abs(int(edge[0])) < 100  # faded in
