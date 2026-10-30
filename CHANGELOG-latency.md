# Latency Optimization Changelog

This log tracks changes to the turn-to-turn latency pipeline. Its purpose is to
prevent regression loops — changing something, forgetting why, hitting an issue,
reverting, and ending up back where we started.

Every entry records **what** changed, **why**, **what it replaced**, and
**what to watch for** if the change causes problems. If you're about to revert
something, read this first.

---

## Baseline: 2026-08-23

Measured from two live call transcripts. The latency waterfall per turn was:

| Stage                    | Duration   | Notes                                    |
|--------------------------|------------|------------------------------------------|
| VAD silence threshold    | 800ms      | Fixed, before Whisper even starts         |
| Whisper transcription    | 100-500ms  | Varies with utterance length              |
| Debounce wait            | 300ms      | Fixed, after every transcript             |
| LLM turn detection       | 200-500ms  | Haiku API call, ~60% of turns             |
| Claude TTFT + 1st sent   | 500-1200ms | Sonnet streaming, time to first sentence  |
| TTS first sentence       | 200-500ms  | Full WAV buffered before any audio sent   |
| **Total**                | **2.1-3.8s** |                                         |

Observed in transcripts: short turns ~1-2s, longer conversational turns 3-4s.

### Components at baseline

- `_DEBOUNCE_SECONDS = 0.3` — 300ms pause after each Whisper transcript before
  acting on it. Introduced to prevent responding during mid-thought pauses where
  VAD detected 800ms of silence between clauses.

- `_is_complete_turn()` — Haiku API call to classify whether a transcript is a
  complete conversational turn. Fast-path catches questions (trailing `?`) and a
  set of keyword patterns. Everything else hits the network.

- `_HOLD_TIMEOUT_SECONDS = 2.0` — safety flush for held incomplete turns.

- `agent.py` uses sync `anthropic.Anthropic` client with `messages.stream()`.
  Blocks the event loop between sentence yields, preventing audio processing and
  barge-in detection during Claude generation.

- TTS pipeline: Kokoro generates all audio chunks, concatenates into one array,
  server writes full WAV, client downloads full WAV, converts to mulaw. No
  streaming at any stage.

---

## Applied changes

### 1. Eliminate debounce + replace LLM turn detection (2026-08-23)

**Status:** Applied. Tests in `test_turn_detection.py` (37 cases).

**What changed:**
- Removed `_DEBOUNCE_SECONDS = 0.3` — no delay after transcripts.
- Replaced `_is_complete_turn` async Haiku API call with synchronous local
  heuristic using `_INCOMPLETE_TRAIL` trailing-word detection.
- Reduced `_HOLD_TIMEOUT_SECONDS` from 2s to 1s.
- Word threshold for sentence-final punctuation: 3+ words (not 4 — "That sounds
  amazing!" is clearly complete).

**Why:** The debounce added 300ms to every turn. The Haiku call added 200-500ms
to ~60% of turns (any sentence ending with a period that doesn't match the
fast-path patterns). Together they accounted for 500-800ms of pure serial
latency on the critical path.

**What it replaced:** The debounce was originally added because VAD silence
detection alone (800ms) was too aggressive — the system would respond during
mid-thought pauses. The Haiku turn detection was added as a more intelligent
guard against this. However, both were redundant with each other, and the VAD
threshold already provides an adequate pause boundary for the vast majority of
conversational turns.

**New heuristic logic:** Any transcript with sentence-final punctuation (`.`,
`!`, `?`) after 3+ words is treated as complete. Transcripts trailing with
conjunctions, prepositions, articles, or copulas (see `_INCOMPLETE_TRAIL`) are
held. Everything else (short fragments, no punctuation) is held. The hold
timeout at 1s provides a safety net.

**What to watch for if this causes problems:**
- If users report being talked over mid-thought: the root cause is that the user
  paused for 800ms+ between clauses and the new heuristics treated the fragment
  as complete. The fix is to add the specific trailing pattern to the
  `_INCOMPLETE_TRAIL` list, NOT to re-add the debounce or Haiku call. Those
  were too expensive for the problem they solved.
- If fragmentary transcripts are being responded to: tighten the heuristic
  (require more words, check for sentence-final verbs, etc.).
- Do NOT re-add the Haiku turn detection without measuring its latency impact.
  It was 200-500ms per call. If a smarter guard is needed, consider a local
  classifier or a rule-based approach first.

**Expected savings:** 500-800ms per turn.

---

### 2. Switch to async Anthropic client (2026-08-23)

**Status:** Applied. Tests in `test_agent.py` (7 cases).

**What changed:**
- `agent.py` line 13: `anthropic.Anthropic` → `anthropic.AsyncAnthropic`.
- `respond_stream`: `with` → `async with`, `for` → `async for`.

**Why:** The sync streaming client blocked the event loop while waiting for each
token from Claude. Between sentence-boundary yields, no other async work ran.
This meant:
- Incoming Twilio audio packets queued up during generation.
- Silero VAD couldn't process audio, so barge-in detection stalled mid-sentence.
- TTS for the next sentence couldn't start until the current sentence's audio
  was fully sent, because the event loop was blocked reading tokens.

**What it replaced:** The sync client was likely chosen for simplicity — the
Anthropic SDK's sync streaming works fine in a standalone script. But in an
async server with concurrent WebSocket I/O, it created pipeline bubbles.

**What to watch for if this causes problems:**
- The async streaming API has the same interface shape but different import
  paths. If the SDK version doesn't support async streaming, pin a version that
  does (the async API has been stable since anthropic SDK 0.25+).
- If Claude responses seem to arrive more slowly: this is a perception issue —
  the async client doesn't change Claude's speed, but it does interleave other
  work between tokens, which could change timing characteristics. Measure
  time-to-first-audio, not token arrival rate.
- The sync client was running in the calling coroutine's thread. The async
  client runs cooperatively. If any code in the sentence-processing loop does
  blocking I/O (it shouldn't), it will now stall the entire event loop instead
  of just the streaming coroutine.

**Expected savings:** 200-400ms direct (faster barge-in), significant indirect
improvement from enabling TTS pipelining.

---

### 3. Stream TTS audio end-to-end (2026-08-23)

**Status:** Applied. Tests in `tts-server/test_streaming.py` (5 cases),
`telephony-agent/test_tts_streaming.py` (3 cases).

**What changed:** Three-part change across both repos:
1. `tts-server/backends/__init__.py`: Added `generate_stream()` to `TTSBackend`
   base class. Default yields full `generate()` result as one chunk.
2. `tts-server/backends/kokoro.py`: Overrides `generate_stream()` to yield
   chunks as Kokoro's pipeline produces them, instead of concatenating.
3. `tts-server/server.py`: New `/tts/stream` endpoint that streams raw PCM via
   chunked transfer with a binary protocol (8-byte header + length-prefixed
   chunks + zero terminator). Original `/tts` WAV endpoint preserved.
4. `telephony-agent/tts.py`: Added `synthesize_stream()` that reads the binary
   stream, converts each PCM chunk to mulaw, and yields base64 chunks for Twilio.
5. `telephony-agent/server.py`: `_stream_agent_response` now uses
   `tts.synthesize_stream()` instead of `tts.synthesize()`.

**Binary protocol:**
```
Header (8 bytes): sample_rate (u32 LE) | channels (u16 LE) | bit_depth (u16 LE)
Per chunk:        length (u32 LE) | PCM float32 data (length bytes)
Terminator:       length = 0 (u32 LE)
```

**Why:** Kokoro's pipeline already yielded audio in chunks internally, but the
backend concatenated them all before returning. The server wrote a full WAV, and
the client downloaded the full WAV before converting to mulaw. 100% of TTS audio
was buffered before any of it reached the caller.

With streaming, the first chunk reaches Twilio while subsequent chunks are still
being generated.

**What it replaced:** The full-buffer approach was simpler — WAV is
self-describing, so the client didn't need to know sample rate in advance.
The binary protocol now communicates that in the header.

**What to watch for if this causes problems:**
- Audio glitches at chunk boundaries: each PCM chunk is resampled independently.
  If chunk boundaries don't align on clean sample periods, there may be subtle
  clicks. Fix: add a small crossfade or ensure Kokoro chunk sizes are multiples
  of the resampling ratio.
- Backend fallback: backends without `generate_stream` override (e.g.,
  Chatterbox) fall back to the base class's single-chunk yield. This still works
  but won't benefit from streaming — the entire audio buffers in one chunk.
- The `/tts` WAV endpoint is preserved for non-telephony clients.
- If `synthesize_stream` fails mid-stream (network drop, server crash), the
  caller gets a partial audio response. The `connected.is_set()` check in
  `server.py` handles the Twilio side, but the httpx stream may raise.

**Expected savings:** 200-400ms on time-to-first-audio per sentence.

---

### 4. Two-stage end-of-turn detection (2026-10-27)

**Status:** Applied on branch `turn-latency`. Tests in `test_endpointing.py` (4 cases).
Measured with `bench_turn_latency.py`.

**What changed:**
- `stt/whisper_stt.py`: Whisper now runs after `VAD_EARLY_MS` (default 400ms) of silence instead
  of 800ms. The end of turn is signalled separately, through `pop_end_of_turn()`, once silence
  reaches `VAD_SILENCE_MS` (default 800ms). Speech before that cancels it, and the next
  transcript is appended to the held text.
- `server.py`: a transcript that `_is_obviously_complete()` accepts responds immediately, at the
  early mark. That means a `?` near the end, or a turn-yielding phrase. Everything else waits
  for the end of turn, then goes through `_is_complete_turn()` and the hold logic as before.
- `server.py`: the hold-timeout flush is cancelled when the caller starts speaking again, so
  held text is no longer flushed in the middle of their next utterance.
- `config.py`: new `VAD_EARLY_MS` and `VAD_SILENCE_MS` variables.
- Logging per turn:
  - `Transcript (+N ms after speech end, whisper M ms)`
  - `Latency: respond +A, first sentence +B, first audio +C` (all measured from the end of speech)

**Why:** Whisper only started after the full 800ms of silence, so its ~200ms (base.en) was on
the critical path of every turn. Questions are the clearest end-of-turn signal, and they don't
need the full wait.

**What it replaced:** a single 800ms threshold, where transcription and the end of turn were the
same event. Simply lowering that threshold was rejected: entry 1 records that 800ms already
caught people mid-thought.

**Measured** (2026-10-27): the same code with `VAD_EARLY_MS=800` and `=400`. Each ran 12
synthetic turns, with Phonon TTS.

| Stage | 800 (old) | 400/800 (new) |
|---|---|---|
| End of turn, statements | 1022ms | 800ms |
| End of turn, questions | 1042ms | 622ms |
| Claude first sentence | 1871ms | 1988ms (noise: 1.1-4.2s per turn) |
| TTS first audio | 22ms | 20ms |
| End to end, harness | 3084ms | 2879ms |

**Pause tolerance** (`bench_turn_latency.py 2 --pauses ...`): the caller says a line, pauses for
the given time (voiced to voiced), then keeps going. The cell counts runs where agent audio
arrived before the caller finished, out of 2:

| Pause | Question, old | Question, new | Statement, old | Statement, new |
|---|---|---|---|---|
| 0.5-1.5s | 0/2 | 0/2 | 0/2 | 0/2 |
| 2s | 0/2 | 0/2 | 1/2 | 1/2 |
| 2.5s | 0/2 | **2/2** | 1/2 | 0/2 |
| 3s | 2/2 | 2/2 | 2/2 | 2/2 |

Replies started and then dropped before any audio: 14 (old) and 15 (new).
- **Where the boundary sits.** A pause is safe while it is shorter than end-of-turn time plus
  Claude's time to its first sentence plus TTS. Step 1 moved the boundary for questions about
  0.5s earlier, from 3s to 2.5s. Statements didn't change measurably.
- **Anything that makes Claude faster moves the boundary earlier by the same amount.** The two
  effects interact.
- **Barge-in doesn't flush Twilio's buffer.** No `clear` message is sent, so once audio is out
  the caller hears that whole sentence, even if they talk over it.

**What to watch for if this causes problems:**
- **The agent jumps in after a question when the caller keeps going.** For example "Do you have
  Tuesday? ... or maybe Wednesday." The reply starts about 620ms after the question. Barge-in
  cancels it, but the caller hears its first syllables. The fix is to raise `VAD_EARLY_MS` (to
  500-600). Don't collapse back to a single threshold.
- **Damage where a mid-sentence pause splits the transcript.** A pause of 400-800ms gets the two
  halves transcribed separately and joined. Watch for words dropped or garbled at the join.
- **Rollback:** `VAD_EARLY_MS=800` restores the old timing exactly. The hold-timeout cancel stays.

**Expected savings:** about 220ms per statement and 420ms per question. Both measured.

---

## Decision log

| Date       | Decision                            | Rationale                                                     |
|------------|-------------------------------------|---------------------------------------------------------------|
| 2026-08-23 | Baseline measured at 2.1-3.8s       | Two live call transcripts analyzed                            |
| 2026-08-23 | Prioritized debounce removal first  | Biggest savings (500-800ms), lowest complexity, pure deletion  |
| 2026-08-23 | Chose local heuristics over Haiku   | Network round-trip too expensive for latency-critical path     |
| 2026-08-23 | Async client before TTS streaming   | Unblocks event loop, prerequisite for proper TTS pipelining    |
| 2026-10-27 | Two-stage endpointing, not a shorter threshold | Hides Whisper in the silence; statements still wait 800ms |
| 2026-10-27 | Per-stage latency logs + bench harness  | Claude's 1.1-4.2s variance hides 200-400ms wins end to end    |
