"""Measure turn latency and barge-in end to end by replaying synthetic caller speech into /ws/media.

The harness stands in for Twilio. Caller speech is synthesized by the running tts-server and
streamed in real time as 20 ms mulaw frames, surrounded by silence. Agent audio plays from a buffer
in real time, as on a phone. As Twilio documents, a `mark` from the agent is sent back once the
audio queued before it has played, and a `clear` drops the audio not yet played and sends back
every pending mark.

By default, the latency is the time from the last voiced frame sent to the first agent audio
frame received. The harness then waits for the reply to finish playing before the next utterance.

Each run makes real Claude calls (a greeting plus one reply per utterance). Don't run it during a
live call: the server's STT instance is shared. Transcripts land in transcripts/ under BENCH_*.

With --pauses, each turn is instead a line, a pause of the given length, then a continuation, to
find where the agent starts cutting in. Outcomes, from the caller's side:
  clean    no agent audio until the caller finished the continuation
  cut in   agent audio arrived during the pause or the continuation. Barge-in stops further
           audio, but what was already sent still plays from Twilio's buffer
Whether a clean turn had a reply started and dropped ("before any audio") is in the server log.

With --barge-in, the caller asks for a detailed answer, then makes a sound while the reply plays:
  interrupt  "Sorry, wait, can I ask about something else?", 1 s or 3 s into the reply
  blip       160 ms of the caller's voice, 1.5 s in: a stand-in for a cough or throat-clear.
             Silero scores it as 2-5 speech frames (64-160 ms), so it sets off today's one-frame
             trigger but should not pass a 250 ms gate. Synthetic coughs and plain noise are missed
             by Silero at most alignments to its 32 ms frames, so they would test nothing.
  mm-hmm     a synthesized "Mm-hmm.", 1.5 s in
Measured for each, from the caller's side:
  heard after  agent audio heard from the caller's first voiced frame until the reply goes quiet.
               For an interruption, less is better. After a blip or "mm-hmm" the reply should
               carry on, so this should be the rest of the reply.
  clear        the agent sent Twilio `clear`
  new reply    audio arrived after the sound: an answer (expected for an interruption) or, after
               an acknowledgement, the cut-off reply resuming
With --server-log PATH, it also counts the server's "Barge-in detected" and "Resuming after" lines
for each case.

usage: .venv/bin/python bench_turn_latency.py [runs=3] [--voice NAME] [--url ws://localhost:9002/ws/media]
       .venv/bin/python bench_turn_latency.py [runs=2] --pauses 0.5,1,1.5,2,2.5,3
       .venv/bin/python bench_turn_latency.py [runs=3] --barge-in [--server-log PATH]
"""

import argparse
import asyncio
import base64
import json
import random
import re
import statistics
import time
from pathlib import Path

import httpx
import numpy as np
import websockets

from audio import mulaw_decode, mulaw_encode, wav_bytes_to_mulaw_payloads

UTTERANCES = [
    ("statement", "Hi, I'm calling to book a dental cleaning for next week."),
    ("question", "Do you have anything on Tuesday morning?"),
    ("statement", "Tuesday at ten works for me."),
    ("question", "Can you send me a confirmation text afterwards?"),
]
PAUSE_LINES = [
    ("question", "Are you open on Saturday?", "Or maybe Sunday instead?"),
    ("statement", "I'd like to book a cleaning.", "Sometime next week, if possible."),
]
BARGE_QUESTION = "Could you walk me through what happens during a dental cleaning, step by step?"
BARGE_LINES = {"interrupt": "Sorry, wait, can I ask about something else?", "mm-hmm": "Mm-hmm."}
BARGE_CASES = [("interrupt", 1.0), ("interrupt", 3.0), ("blip", 1.5), ("mm-hmm", 1.5)]
FRAME_S = 0.02
SILENCE = base64.b64encode(b"\xff" * 160).decode("ascii")  # 20 ms of mulaw silence
VOICED_RMS = 300  # int16 RMS above which a frame counts as speech
QUIET_GAP_S = 1.5  # no agent audio for this long = reply finished
PLAY_GAP_S = 0.3  # playback quiet for this long = the agent stopped talking
REPLY_GAP_S = 0.8  # agent audio arriving after a gap this long starts a new reply (Claude takes >1 s)
SETTLE_S = 8.0  # after a barge-in case, wait until the agent has been quiet this long


def voiced_span(frames: list[str]) -> tuple[int, int]:
    """Indexes of the first and last frames loud enough to count as speech."""
    rms = [np.sqrt(np.mean(np.frombuffer(mulaw_decode(base64.b64decode(f)), np.int16).astype(np.float64) ** 2))
           for f in frames]
    voiced = [i for i, r in enumerate(rms) if r > VOICED_RMS] or [0, len(frames) - 1]
    return voiced[0], voiced[-1]


def synthesize(tts_url: str, text: str, voice: str) -> tuple[list[str], int, int]:
    """Return 20 ms mulaw frames for text, and the indexes of the first and last voiced frames."""
    wav = httpx.post(f"{tts_url}/tts", json={"text": text, "voice": voice}, timeout=60).content
    frames = wav_bytes_to_mulaw_payloads(wav)
    return (frames, *voiced_span(frames))


def speech_blip(speech: tuple[list[str], int, int], n: int = 8) -> tuple[list[str], int, int]:
    """The first n frames (160 ms) of real speech with 10 ms fades: a stand-in for a cough or a
    throat-clear. Silero scores it as 2-5 speech frames (64-160 ms) wherever it lands against its
    32 ms frames. Synthetic coughs are a poor stand-in: Silero misses them at most alignments."""
    frames, first, _ = speech
    pcm = np.frombuffer(b"".join(mulaw_decode(base64.b64decode(f)) for f in frames[first:first + n]), np.int16)
    i = np.arange(len(pcm))
    faded = (pcm * np.minimum(1, np.minimum(i, i[::-1]) / 80)).astype(np.int16)
    mu = mulaw_encode(faded.tobytes())
    out = [base64.b64encode(mu[k:k + 160]).decode("ascii") for k in range(0, len(mu), 160)]
    return (out, *voiced_span(out))


class Call:
    """One simulated call: a paced caller, and a stand-in for Twilio's playback buffer."""

    def __init__(self, ws):
        self.ws = ws
        self.queue: list[str] = []  # frames to send ahead of silence
        self.marks: dict[int, str] = {}  # frame sequence number -> event name
        self.times: dict[str, float] = {}  # event name -> when that frame was sent
        self.media_times: list[float] = []  # arrival time of every agent frame
        self.frames: list[tuple[float, float]] = []  # (arrival, play start) of agent frames not cleared
        self.play_end = 0.0  # when the agent audio queued so far finishes playing
        self.pending_marks: list[tuple[float, str]] = []  # (due, name) of agent marks not yet sent back
        self.clears: list[float] = []  # when the agent sent `clear`
        self.seq = 0

    async def sender(self):
        start = time.monotonic()
        while True:
            payload = self.queue.pop(0) if self.queue else SILENCE
            await self.ws.send(json.dumps({"event": "media", "streamSid": "MZbench",
                                           "media": {"track": "inbound", "chunk": str(self.seq),
                                                     "timestamp": str(self.seq * 20), "payload": payload}}))
            if self.seq in self.marks:
                self.times[self.marks.pop(self.seq)] = time.monotonic()
            for name in self.due_marks(time.monotonic()):
                await self.ws.send(json.dumps({"event": "mark", "streamSid": "MZbench", "mark": {"name": name}}))
            self.seq += 1
            await asyncio.sleep(max(0.0, start + self.seq * FRAME_S - time.monotonic()))

    async def receiver(self):
        async for raw in self.ws:
            self.on_message(json.loads(raw), time.monotonic())

    def on_message(self, msg: dict, now: float) -> None:
        """Handle one message from the agent the way Twilio's playback buffer does."""
        event = msg.get("event")
        if event == "media":
            self.media_times.append(now)
            start = max(now, self.play_end)
            self.frames.append((now, start))
            self.play_end = start + FRAME_S
        elif event == "mark":
            self.pending_marks.append((max(now, self.play_end), msg["mark"]["name"]))
        elif event == "clear":
            self.clears.append(now)
            self.frames = [(a, s) for a, s in self.frames if s < now]
            self.play_end = min(self.play_end, now)
            self.pending_marks = [(now, name) for _, name in self.pending_marks]

    def due_marks(self, now: float) -> list[str]:
        """Remove and return the agent marks whose audio has finished playing."""
        due = [name for t, name in self.pending_marks if t <= now]
        self.pending_marks = [(t, name) for t, name in self.pending_marks if t > now]
        return due

    def heard_after(self, t: float, arrived_before: float = float("inf")) -> float | None:
        """Seconds of agent audio heard from t until playback goes quiet, counting only frames that
        arrived before `arrived_before`. None if nothing was playing at t."""
        end = t
        for a, s in self.frames:
            if a >= arrived_before or s + FRAME_S <= t:
                continue
            if s - end > PLAY_GAP_S:
                break
            end = max(end, s + FRAME_S)
        return None if end == t else end - t

    def reply_starts(self, since: float) -> list[float]:
        """Arrival times of the first frame of each agent reply after `since`."""
        starts, last = [], None
        for a in self.media_times:
            if a > since and (last is None or a - last > REPLY_GAP_S):
                starts.append(a)
            last = a
        return starts

    async def wait_reply(self, since: float, timeout: float = 20.0) -> float | None:
        """Wait for agent audio after `since`, then until it has finished playing. Return first-frame time."""
        deadline = since + timeout
        while time.monotonic() < deadline:
            frames = [t for t in self.media_times if t > since]
            if frames and time.monotonic() - frames[-1] > QUIET_GAP_S and time.monotonic() > self.play_end:
                return frames[0]
            await asyncio.sleep(0.05)
        frames = [t for t in self.media_times if t > since]
        return frames[0] if frames else None

    async def settle(self, quiet_s: float = SETTLE_S, timeout: float = 45.0) -> None:
        """Wait until all agent audio has played and none has arrived for quiet_s."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            now = time.monotonic()
            last = self.media_times[-1] if self.media_times else 0.0
            if now > self.play_end and now - last > quiet_s:
                return
            await asyncio.sleep(0.05)

    async def barge_case(self, question, sound, offset_s: float) -> dict:
        """Ask `question`, then `offset_s` into the reply's playback, make `sound`. Return what the caller heard."""
        q_frames, _, q_last = question
        self.enqueue(q_frames, {q_last: "q_end"})
        q_end = await self.sent("q_end")
        while not (starts := self.reply_starts(q_end)):
            if time.monotonic() > q_end + 20:
                return {"heard": None}
            await asyncio.sleep(0.01)
        await asyncio.sleep(max(0.0, starts[0] + offset_s - time.monotonic()))  # buffer was empty: arrival = playback
        s_frames, s_first, _ = sound
        self.enqueue(s_frames, {s_first: "s_start"})
        t_s = await self.sent("s_start")
        wall = time.time() - (time.monotonic() - t_s)
        await self.settle()
        later = [a for a in self.reply_starts(q_end)[1:] if a > t_s]
        return {
            "heard": self.heard_after(t_s, arrived_before=later[0] if later else float("inf")),
            "clear": any(c > t_s for c in self.clears),
            "new_reply": bool(later),
            "window": (wall, time.time()),
        }

    def enqueue(self, frames: list[str], marks: dict[int, str]) -> None:
        base = self.seq + len(self.queue)
        self.marks.update({base + i: name for i, name in marks.items()})
        self.queue.extend(frames)

    async def sent(self, name: str) -> float:
        while name not in self.times:
            await asyncio.sleep(0.005)
        return self.times.pop(name)

    async def say(self, frames: list[str], first_voiced: int, last_voiced: int) -> float | None:
        self.enqueue(frames, {last_voiced: "end"})
        end = await self.sent("end")
        first = await self.wait_reply(end)
        return None if first is None else first - end

    async def say_with_pause(self, a, b, pause_s: float) -> tuple[str, float | None]:
        """Say a, pause (voiced-to-voiced) for pause_s, say b. Return (outcome, reply latency after b)."""
        (fa, _, la), (fb, b0, lb) = a, b
        gap = max(0, round(pause_s / FRAME_S) - (len(fa) - 1 - la) - b0)
        self.enqueue(fa, {la: "a_end"})
        self.enqueue([SILENCE] * gap, {})
        self.enqueue(fb, {lb: "b_end"})
        a_end, b_end = await self.sent("a_end"), await self.sent("b_end")
        early = [t for t in self.media_times if a_end < t < b_end]
        first = await self.wait_reply(b_end)
        return ("cut in" if early else "clean"), (None if first is None else first - b_end)


async def run_once(url: str, speech, label: str, pauses: list[float] | None = None, seed: int = 0,
                   barge: tuple | None = None):
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({"event": "start", "streamSid": "MZbench",
                                  "start": {"streamSid": "MZbench", "callSid": label, "tracks": ["inbound"]}}))
        call = Call(ws)
        tasks = [asyncio.create_task(call.sender()), asyncio.create_task(call.receiver())]
        try:
            await call.wait_reply(time.monotonic() - 0.1)  # greeting
            results = []
            if barge:
                question, sounds = barge
                plan = list(BARGE_CASES)
                random.Random(seed).shuffle(plan)
                for kind, offset in plan:
                    results.append((kind, offset, await call.barge_case(question, sounds[kind], offset)))
            elif pauses:
                plan = [(kind, a, b, p) for kind, a, b in speech for p in pauses]
                random.Random(seed).shuffle(plan)
                for kind, a, b, p in plan:
                    outcome, latency = await call.say_with_pause(a, b, p)
                    results.append((kind, p, outcome, latency))
            else:
                for kind, *synth in speech:
                    results.append((kind, await call.say(*synth)))
            await ws.send(json.dumps({"event": "stop", "streamSid": "MZbench"}))
            return results
        finally:
            for t in tasks:
                t.cancel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", type=int, default=3)
    ap.add_argument("--voice", default="default", help="tts-server voice for the simulated caller")
    ap.add_argument("--url", default="ws://localhost:9002/ws/media")
    ap.add_argument("--tts-url", default="http://localhost:9100")
    ap.add_argument("--pauses", help="comma-separated pause lengths in seconds, e.g. 0.5,1,1.5,2")
    ap.add_argument("--barge-in", action="store_true", help="interrupt, blip and mm-hmm while the agent talks")
    ap.add_argument("--server-log", help="with --barge-in: the server's log, to count barge-ins per case")
    args = ap.parse_args()

    if args.barge_in:
        return barge_in_sweep(args)
    if args.pauses:
        return pause_sweep(args, [float(p) for p in args.pauses.split(",")])
    speech = [(kind, *synthesize(args.tts_url, text, args.voice)) for kind, text in UTTERANCES]
    results = []
    for r in range(args.runs):
        label = f"BENCH_{time.strftime('%H%M%S')}_{r}"
        run = asyncio.run(run_once(args.url, speech, label))
        print(f"run {r + 1}: " + ", ".join(f"{k} {'-' if v is None else f'{v * 1000:.0f}ms'}" for k, v in run))
        results += run

    print("\n| turn | n | median | min | max |\n|---|---|---|---|---|")
    for kind in ("statement", "question", "all"):
        vals = [v * 1000 for k, v in results if v is not None and (kind == "all" or k == kind)]
        if vals:
            print(f"| {kind} | {len(vals)} | {statistics.median(vals):.0f} ms | {min(vals):.0f} ms | {max(vals):.0f} ms |")
    missing = sum(v is None for _, v in results)
    if missing:
        print(f"\n{missing} turn(s) got no reply within 20 s")


def pause_sweep(args, pauses: list[float]) -> None:
    speech = [(kind, synthesize(args.tts_url, a, args.voice), synthesize(args.tts_url, b, args.voice))
              for kind, a, b in PAUSE_LINES]
    results = []
    for r in range(args.runs):
        label = f"BENCH_{time.strftime('%H%M%S')}_pauses_{r}"
        results += asyncio.run(run_once(args.url, speech, label, pauses, seed=r))
        print(f"run {r + 1} done")

    kinds = [k for k, *_ in PAUSE_LINES]
    print("\n| pause | " + " | ".join(kinds) + " |\n|---|" + "---|" * len(kinds))
    for p in pauses:
        cells = []
        for kind in kinds:
            rs = [(o, l) for k, pp, o, l in results if k == kind and pp == p]
            cut = sum(o == "cut in" for o, _ in rs)
            lat = [l * 1000 for o, l in rs if l is not None]
            cells.append(f"{cut}/{len(rs)} cut in" + (f", reply +{statistics.median(lat):.0f}ms" if lat else ""))
        print(f"| {p:g} s | " + " | ".join(cells) + " |")


def logged(log: str, what: str, start: float, end: float) -> int:
    """Count server log lines starting with `what` between two wall-clock times (1 s resolution)."""
    lo, hi = (time.strftime("%H:%M:%S", time.localtime(t)) for t in (start - 1, end + 1))
    pattern = rf"^(\d\d:\d\d:\d\d) \[\w+\] {re.escape(what)}"
    return sum(lo <= m[1] <= hi for m in re.finditer(pattern, log, re.M))


def barge_in_sweep(args) -> None:
    question = synthesize(args.tts_url, BARGE_QUESTION, args.voice)
    sounds = {kind: synthesize(args.tts_url, text, args.voice) for kind, text in BARGE_LINES.items()}
    sounds["blip"] = speech_blip(sounds["interrupt"])
    for kind, (_, first, last) in sounds.items():
        print(f"{kind}: {(last - first + 1) * FRAME_S * 1000:.0f} ms loud enough to count as speech")
    results = []
    for r in range(args.runs):
        label = f"BENCH_{time.strftime('%H%M%S')}_barge_{r}"
        run = asyncio.run(run_once(args.url, None, label, seed=r, barge=(question, sounds)))
        print(f"run {r + 1}: " + ", ".join(
            f"{k} @{o:g}s " + ("skipped" if res["heard"] is None else f"heard {res['heard'] * 1000:.0f}ms")
            for k, o, res in run))
        results += run

    log = Path(args.server_log).read_text() if args.server_log else None
    print("\n| case | n | heard after (median, range) | clear sent | barge-in fired | resumed | new reply |")
    print("|---|---|---|---|---|---|---|")
    for kind, offset in BARGE_CASES:
        ok = [res for k, o, res in results if (k, o) == (kind, offset) and res["heard"] is not None]
        heard = [res["heard"] * 1000 for res in ok]
        cell = f"{statistics.median(heard):.0f} ms ({min(heard):.0f}-{max(heard):.0f})" if heard else "-"

        def count(what: str) -> str:
            return f"{sum(logged(log, what, *res['window']) > 0 for res in ok)}/{len(ok)}" if log else "-"

        print(f"| {kind} @{offset:g}s | {len(ok)} | {cell} | {sum(res['clear'] for res in ok)}/{len(ok)} "
              f"| {count('Barge-in detected')} | {count('Resuming after')} | {sum(res['new_reply'] for res in ok)}/{len(ok)} |")
    skipped = sum(res["heard"] is None for _, _, res in results)
    if skipped:
        print(f"\n{skipped} case(s) skipped: no reply came, or it had finished playing before the sound")


if __name__ == "__main__":
    main()
