"""Measure turn latency end to end by replaying synthetic caller speech into /ws/media like Twilio.

Caller speech is synthesized by the running tts-server. Each utterance is streamed in real time
as 20 ms mulaw frames, surrounded by silence, and the latency is the time from the last voiced
frame sent to the first agent audio frame received. The harness then waits for the reply to
finish "playing" before the next utterance.

Each run makes real Claude calls (a greeting plus one reply per utterance). Don't run it during a
live call: the server's STT instance is shared. Transcripts land in transcripts/ under BENCH_*.

With --pauses, each turn is instead a line, a pause of the given length, then a continuation, to
find where the agent starts cutting in. Outcomes, from the caller's side:
  clean    no agent audio until the caller finished the continuation
  cut in   agent audio arrived during the pause or the continuation. Barge-in stops further
           audio, but what was already sent still plays from Twilio's buffer
Whether a clean turn had a reply started and dropped ("before any audio") is in the server log.

usage: .venv/bin/python bench_turn_latency.py [runs=3] [--voice NAME] [--url ws://localhost:9002/ws/media]
       .venv/bin/python bench_turn_latency.py [runs=2] --pauses 0.5,1,1.5,2,2.5,3
"""

import argparse
import asyncio
import base64
import json
import random
import statistics
import time

import httpx
import numpy as np
import websockets

from audio import mulaw_decode, wav_bytes_to_mulaw_payloads

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
FRAME_S = 0.02
SILENCE = base64.b64encode(b"\xff" * 160).decode("ascii")  # 20 ms of mulaw silence
VOICED_RMS = 300  # int16 RMS above which a frame counts as speech
QUIET_GAP_S = 1.5  # no agent audio for this long = reply finished


def synthesize(tts_url: str, text: str, voice: str) -> tuple[list[str], int, int]:
    """Return 20 ms mulaw frames for text, and the indexes of the first and last voiced frames."""
    wav = httpx.post(f"{tts_url}/tts", json={"text": text, "voice": voice}, timeout=60).content
    frames = wav_bytes_to_mulaw_payloads(wav)
    rms = [np.sqrt(np.mean(np.frombuffer(mulaw_decode(base64.b64decode(f)), np.int16).astype(np.float64) ** 2))
           for f in frames]
    voiced = [i for i, r in enumerate(rms) if r > VOICED_RMS]
    return frames, voiced[0], voiced[-1]


class Call:
    """One simulated call: a paced sender and a receiver that timestamps agent audio."""

    def __init__(self, ws):
        self.ws = ws
        self.queue: list[str] = []  # frames to send ahead of silence
        self.marks: dict[int, str] = {}  # frame sequence number -> event name
        self.times: dict[str, float] = {}  # event name -> when that frame was sent
        self.media_times: list[float] = []
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
            self.seq += 1
            await asyncio.sleep(max(0.0, start + self.seq * FRAME_S - time.monotonic()))

    async def receiver(self):
        async for raw in self.ws:
            if json.loads(raw).get("event") == "media":
                self.media_times.append(time.monotonic())

    async def wait_reply(self, since: float, timeout: float = 20.0) -> float | None:
        """Wait for agent audio after `since`, then until it has finished playing. Return first-frame time."""
        deadline = since + timeout
        while time.monotonic() < deadline:
            frames = [t for t in self.media_times if t > since]
            if frames and time.monotonic() - frames[-1] > QUIET_GAP_S and \
                    time.monotonic() > frames[0] + len(frames) * FRAME_S:
                return frames[0]
            await asyncio.sleep(0.05)
        frames = [t for t in self.media_times if t > since]
        return frames[0] if frames else None

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


async def run_once(url: str, speech, label: str, pauses: list[float] | None = None, seed: int = 0):
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({"event": "start", "streamSid": "MZbench",
                                  "start": {"streamSid": "MZbench", "callSid": label, "tracks": ["inbound"]}}))
        call = Call(ws)
        tasks = [asyncio.create_task(call.sender()), asyncio.create_task(call.receiver())]
        try:
            await call.wait_reply(time.monotonic() - 0.1)  # greeting
            results = []
            if pauses:
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
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
