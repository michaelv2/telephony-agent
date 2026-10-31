# TODO

## Latency plan (from 2026-10-27)

Step 1, two-stage end-of-turn detection, is applied on branch `turn-latency` (CHANGELOG-latency.md, entry 4). Planned order: barge-in fix, then step 2, then adaptive waits. Measure each one with `bench_turn_latency.py`, using both the standard turns and `--pauses`, and with a live call before starting the next. Steps 1 and 2 interact: a pause is safe only while it's shorter than end-of-turn time plus Claude's first sentence plus TTS.

### Barge-in (next; doesn't change reply timing)

Ship marks, `clear` and the 250ms gate as one change: `clear` does little without marks, and without the gate a cough cuts the agent off mid-sentence. Do the harness work under **Measure** first. `bench_turn_latency.py` ignores `mark` and `clear` today, so it can't show the fix.

- [ ] **Track playback with Twilio `mark` messages**: Keep `speaking` set until the last mark comes back. Today it clears when sending ends, so barge-in stops working while most of the reply is still playing. Phonon sends 5 s of audio in under 1 s.
- [ ] **Send Twilio `clear` on barge-in**: Discard the queued audio. Today the caller hears every sentence already sent, even after talking over it.
- [ ] **Require ~250ms of continuous speech before barge-in**: Today a single 32ms VAD frame triggers it. Once `clear` is sent, a cough or "mm-hmm" would cut the agent off.
- [ ] **Overlapping replies during the greeting**: Barge-in is off during the greeting, so a caller's "Hello?" starts a second reply that sends audio alongside the greeting. Both share one `speaking` flag, so once the greeting finishes, barge-in is also off for the rest of the second reply. Pinned by the `test_quirk_*` tests in `test_turn_taking.py`.
- [ ] **Record what the caller actually heard in Claude's history**: Today an interrupted reply leaves nothing in the history, although the caller may have heard a whole sentence.
- [ ] **Measure**:
  - Add a harness metric for agent audio played after the caller starts talking. Today that's about a sentence; the target is a few hundred ms.
  - Add a false-trigger test: a cough and a "mm-hmm" while the agent is speaking.
  - Re-run the pause sweep to confirm the cut-in boundaries don't move.

### Claude's first sentence (step 2)

- [ ] **First-chunk clause splitting first** (see Latency optimizations below). Measure it, then decide whether to try a faster reply model such as Haiku. That's a quality trade-off, best judged on a call. Every ms saved also shortens the safe pause, so re-run the pause sweep afterwards.

### Streaming transcription (step 3)

- [ ] **Transcribe while the caller is still speaking**: Step 1 already hides most of Whisper's ~200ms inside the silence wait. The remaining gain is mostly on long utterances, and this is the most complex change. Re-evaluate once steps 1 and 2 are measured.

### Dynamic patience (turn detection)

- [ ] **Adapt the silence wait per caller**: Lengthen it after evidence of mid-thought pauses, for example +300ms after a reply dropped before any audio and +500ms after a heard cut-in, capped at ~2s. Joined transcripts are also evidence. Let it drift back down on clean turns. Log the wait on every turn so changes can still be told apart.
- [ ] **Expectation-aware waits**: Claude tags its own question, like `[TRANSFER]`, e.g. `[EXPECT: yes_no|phone|date|open]`. Use a short wait after yes/no, wait for enough digits after asking for a phone number, and be patient after open-ended or emotional questions.
- [ ] **`[PACE: patient]` tag**: For conversations that turn reflective. Complements adaptive waits, which react to how the caller actually pauses.
- [ ] **Hold on trailing digits**: Phone numbers, dates and addresses get read out with pauses ("555… 867… 5309"), and `_is_complete_turn` doesn't treat trailing digits as unfinished.
- [ ] **If retrying a turn-detection model, only let it extend the wait, never shorten it**: A wrong guess then costs a slower reply, not an interruption. Judge it against the pause sweep and live-call logs.

### Infrastructure

- [ ] **Per-call STT state**: One `WhisperSTT` instance holds the VAD state and buffers for every call, so simultaneous calls collide. Per-call waits need it split; keep the Whisper model shared.
- [ ] **Whisper decodes on the event loop**: `WhisperModel.transcribe()` returns a lazy generator, so `run_in_executor` only sets it up, and the decoding runs in the `" ".join(...)` on the event loop. For the Whisper time logged each turn (100-200ms on base.en), no audio frames are read and no reply audio is sent. Fix: consume the segments inside the executor. It changes timing, so it needs a changelog entry and a bench run.
- [ ] **Let preset env vars override `.env` in `start.sh`**: It sources `.env` after the environment, so `TTS_VOICE=freya ./start.sh phonon` can't change the voice. On 2026-10-27 that took a hand-written launcher.

## Turn detection improvements

- [ ] **Transcript accumulation**: Buffer for a short window (~500ms) after transcription. If another segment arrives in that window, concatenate before responding. Catches mid-sentence pauses without relying solely on linguistic heuristics. (Partly covered by two-stage endpointing: speech between 400 and 800ms of silence is joined.)
- [x] **LLM-based turn detection** (added 2026-04-27, removed 2026-08-23): The Haiku call added 200-500ms to most turns, so local rules (`_INCOMPLETE_TRAIL`) and a 1s hold timeout replaced it. CHANGELOG-latency.md, entry 1.

## Latency optimizations

- [ ] **Speculative Claude call during Haiku classification**: For ambiguous turns that go to Haiku, start the Claude API call in parallel. If Haiku says COMPLETE, Claude is already streaming (~1s saved). If INCOMPLETE, cancel it. Low waste since the fast-path handles obvious cases.
- [ ] **TTS prefetching**: Start TTS for sentence N+1 while sending audio chunks for sentence N. Reduces gaps between sentences on multi-sentence responses (~0.5-1s savings per sentence).
- [ ] **First-chunk clause splitting**: For the first chunk only, split on clause boundaries (commas) instead of sentence boundaries to get something to TTS faster. Revert to sentence boundaries after the first chunk is playing (~0.3-0.5s off time-to-first-audio).

## Call transfer improvements

- [ ] **Whisper announcement on transfer**: Use Twilio's `<Dial url="...">` to play a short context message to the user before bridging (e.g. "Incoming transfer: dental appointment scheduling"). Gives the user context about who's on the line and why.
