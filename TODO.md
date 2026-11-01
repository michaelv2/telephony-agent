# TODO

## Latency plan (from 2026-10-27)

Step 1, two-stage end-of-turn detection, is applied on branch `turn-latency` (CHANGELOG-latency.md, entry 4). Planned order: barge-in fix, then step 2, then adaptive waits. Measure each one with `bench_turn_latency.py`, using both the standard turns and `--pauses`, and with a live call before starting the next. Steps 1 and 2 interact: a pause is safe only while it's shorter than end-of-turn time plus Claude's first sentence plus TTS.

### Barge-in (doesn't change reply timing)

Marks, `clear` and the 250ms gate shipped together on 2026-11-01 (CHANGELOG-latency.md, entry 7). Measure with `bench_turn_latency.py --barge-in`, which models Twilio's `mark` and `clear`. Twilio's marks were confirmed on a live call the same day. Acknowledgements resume the reply (entry 8).

- [x] **Track playback with Twilio `mark` messages**: The agent counts as talking until the last mark comes back. It used to stop when sending ended, so barge-in stops working while most of the reply is still playing. Phonon sends 5 s of audio in under 1 s.
  - **Replies stack up behind unplayed audio**: The 2026-11-01 live call showed what this does over a conversation. Barge-in fired once; after that, each thing the caller said started a new reply that queued behind the audio still playing, and the agent fell up to ~37s behind (asked for oatmeal at 10:42:04, heard the oatmeal recipe from ~10:42:44). Marks plus `clear` fix it: barge-in stays armed during playback and empties the queue before the next reply.
- [x] **Send Twilio `clear` on barge-in**: Discard the queued audio. The caller used to hear every sentence already sent, even after talking over it. Now they hear about 300ms.
- [x] **Require ~250ms of speech before barge-in** (`BARGE_IN_MS`): A single 32ms VAD frame used to trigger it. Notes from building it:
  - **Count speech frames, not `speech_active`** (done: `speech_ms`): `speech_active` stays true for up to `VAD_EARLY_MS` (400ms) of silence after any speech, until Whisper runs, so a 32ms cough would pass a "250ms of `speech_active`" check. `WhisperSTT` needs to report how much actual speech it has heard in the current utterance.
  - **250ms doesn't filter out "mm-hmm"**: Confirmed after the fix: "mm-hmm" stops the agent in 3 of 3 runs, and the agent then replies to "Mm.". See the open item below. From the 2026-11-01 baseline: the synthesized "Mm-hmm." is 740ms of speech, Whisper hears it as "Mm.", which counts as unfinished (under 3 words), and the agent replied to it after the 1s hold in 3 of 3 runs. A 160ms blip was once transcribed as "Sorry." and got a reply too, so ignoring short sounds needs a transcript check as well as the gate.
  - **The gate moves the pause boundary earlier**: A cut-off reply gets up to 250ms more to produce audio before it's cancelled. If the pause sweep gets worse, apply the gate only to replies the caller can already hear, and keep cancelling silent ones at the first speech frame. Not needed so far: the 2026-11-01 sweep after the fix showed no measurable move (entry 7).
- [x] **Don't let "mm-hmm" or "okay" stop the agent** (option (a), entry 8: the reply resumes from the sentence the caller was hearing): These acknowledgements run longer than 250ms, so they stop the agent, which then answers them after the 1s hold. Options: (a) when the speech that stopped the agent turns out to be a short acknowledgement, resume the cut-off reply from the sentence the caller was hearing (the marks say which; needs the history item below), not answer it; (b) raise `BARGE_IN_MS` to ~800ms, which costs every real interruption ~0.8s of overlap; (c) both, with a smaller increase.
- [x] **Don't reply to speech that didn't barge in while the agent was talking**: If the gate lets a short sound pass without cutting the agent off, a reply to its transcript would queue behind the playing audio, or cut it off. Ignore the transcript instead. On the 2026-11-01 call, "Mm hmm." and "Okay." each got a reply, and "Okay." replayed an 18s recipe. Ship with the gate.
- [ ] **Replies started during the greeting**: Barge-in is off until the greeting has finished playing, so a caller's "Hello?" starts a second reply whose audio queues behind the greeting. Pinned by `test_quirk_question_during_greeting_starts_second_reply`. (Fixed on 2026-11-01: that reply no longer loses barge-in when the greeting finishes.) Possible fix: hold the caller's text until the greeting has finished playing (`TurnTaking.greeting_done`).
- [x] **Resume when what cut the agent off has no words** (entry 9): On the 2026-11-01 13:28 call, a breath or noise stopped the agent twice, Whisper found no words, and the agent stayed silent for 8-9s.
- [x] **Record what the caller actually heard in Claude's history** (entry 8): A reply cut off while being written used to leave nothing, and one cut off after it was sent (a few seconds in) stayed in full. Now the history keeps the sentences the caller heard.
- [ ] **Measure** with `bench_turn_latency.py 3 --barge-in --server-log PATH`:
  - [x] Harness metric: agent audio heard after the caller starts talking ("heard after"). The target is a few hundred ms.
  - [x] False-trigger cases: a 160ms blip of speech (a stand-in for a cough; Silero misses synthetic coughs at most alignments) and a "mm-hmm", 1.5s into the reply.
  - Baseline (2026-11-01, Phonon, 3 runs per case):

    | Case | Heard after (median, range) | `clear` sent | Barge-in fired | Agent replied to it |
    |---|---|---|---|---|
    | Interrupt 1s in | 5.8s (3.8-9.4s) | 0/3 | 3/3 | 3/3 (expected) |
    | Interrupt 3s in | 16.6s (11.7-17.0s) | 0/3 | 0/3 (sending had ended) | 3/3 (expected) |
    | Blip 1.5s in | 10.8s (6.7-16.1s) | 0/3 | 3/3 | 1/3 |
    | "Mm-hmm" 1.5s in | 11.8s (8.9-13.7s) | 0/3 | 3/3 | 3/3 |

    When barge-in fires today, it drops the sentences not yet sent, but the caller still hears everything already queued at Twilio. So the blip and "mm-hmm" both silently cut their replies short, and an interruption still leaves 4-9s of speech.
  - [x] After the fix (2026-11-01): heard after an interruption fell to 0.28-0.30s from 5.8s and 16.6s; a blip no longer stops the agent (0/3); "mm-hmm" still does (3/3). The pause sweep didn't move. Full tables in CHANGELOG-latency.md, entry 7.
- [ ] **Transfer waits for sending, not playing**: `_execute_transfer` runs as soon as the farewell has been sent, so the caller may not hear all of it before the call is redirected. Wait for its mark instead.

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
