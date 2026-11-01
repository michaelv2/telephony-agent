# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A real-time telephony agent that handles phone calls via Twilio Media Streams. Incoming audio goes through Whisper STT, the transcript is sent to Claude for a response, and the response is streamed sentence-by-sentence through TTS (Phonon by default) back to the caller.

## Running

```bash
# Full stack (TTS server + ngrok + server) — requires ngrok installed
./start.sh            # Phonon TTS (voice Freya)
./start.sh kokoro     # or another tts-server model

# Server only (if you manage tunneling separately or have PUBLIC_URL set)
.venv/bin/python server.py
```

`start.sh` launches the TTS server from `../tts-server` (override with `TTS_SERVER_DIR`). Running `server.py` alone needs one already listening at `TTS_URL` (default `http://localhost:9100`).

Set Twilio voice webhook to `{PUBLIC_URL}/call/inbound` (POST).

## Required Environment Variables (.env)

- `ANTHROPIC_API_KEY`
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_API_KEY`, `TWILIO_API_SECRET`, `TWILIO_PHONE`

Optional: `TTS_URL`, `TTS_VOICE` (`Freya`; must be a voice of the running TTS model, and `start.sh` picks one per model when unset), `WHISPER_MODEL` (`base.en`), `PORT` (`9002`), `PUBLIC_URL` (auto-detected from ngrok), `AGENT_PROMPT`, `ALLOWED_NUMBERS`, `VAD_EARLY_MS` (`400`), `VAD_SILENCE_MS` (`800`), `BARGE_IN_MS` (`250`).

## Architecture

The real-time pipeline is: **Twilio mulaw 8kHz -> PCM 16kHz -> Silero VAD -> Whisper -> Claude (streaming) -> TTS (Phonon) -> mulaw 8kHz -> Twilio**.

**`server.py`** — FastAPI app. Four HTTP endpoints (`/call/inbound`, `/call/outbound` for TwiML, `/sms/inbound` to start a call by text, `/api/call` to initiate outbound calls) and one WebSocket endpoint (`/ws/media`) that handles the Twilio Media Stream protocol. Each media event is decoded and fed to STT, and the STT signals go to `TurnTaking`, which starts and cancels `_stream_agent_response` tasks (Claude -> TTS -> Twilio).

**`turn_taking.py`** — When the agent replies and when caller speech cuts it off. `TurnTaking` holds one call's state: caller phases `LISTENING` / `WAITING` (for end of turn) / `HOLDING` (unfinished text, 1s timeout), plus the current reply, `Playback` (Twilio marks not yet sent back, so it knows when the caller is still hearing audio) and barge-in. It also has the completeness rules (`_is_obviously_complete`, `_is_complete_turn`). No I/O; `test_turn_taking.py` drives it with scripted STT signals.

**`agent.py`** — `CallAgent` uses the async Anthropic client (`AsyncAnthropic`, `client.messages.stream`). Maintains conversation history per call. Splits Claude's streamed text on sentence boundaries (`[.!?]\s+`) and yields each sentence to TTS immediately for low latency.

**`audio.py`** — Bidirectional format conversion. Twilio sends base64 mulaw 8kHz; Whisper needs PCM 16kHz; Kokoro outputs WAV (24kHz). Key functions: `twilio_payload_to_pcm16k` (inbound) and `wav_bytes_to_mulaw_payloads` (outbound, chunks WAV into 20ms mulaw frames).

**`tts.py`** — Async HTTP client to the external TTS server. `synthesize_stream` reads PCM from `/tts/stream` and converts each chunk to 20ms mulaw frames as it arrives (CHANGELOG-latency.md, entry 3).

**`stt/`** — Pluggable STT behind `BaseSTT` ABC. `WhisperSTT` uses Silero VAD (512-sample frames at 16kHz) to detect speech boundaries, accumulates audio during speech, transcribes with faster-whisper after 400ms silence (`VAD_EARLY_MS`), and signals end of turn after 800ms (`VAD_SILENCE_MS`); questions respond at the early mark. VAD runs on the event loop. Whisper is handed to an executor, but faster-whisper decodes lazily, so the decoding actually happens on the event loop (TODO.md).

**`config.py`** — All config from env vars via `python-dotenv`. `get_public_url()` auto-detects ngrok tunnel URL by querying `localhost:4040/api/tunnels`.

## Latency Changelog

`CHANGELOG-latency.md` tracks all changes to the turn-to-turn latency pipeline — VAD thresholds, debounce/turn-detection logic, TTS streaming behavior, and anything else that affects how fast the agent responds after the user stops speaking. Before modifying any of these, read the changelog to understand why the current values exist. After making a change, add an entry recording what changed, why, what it replaced, and what to watch for if it regresses. The goal is to prevent revert loops where we fix a latency issue, forget why, hit a side effect, revert, and end up back where we started.

## Turn Timing Map

`docs/turn-timing-map.html` diagrams one caller turn: when the reply starts, the caller phases in `turn_taking.py`, and when barge-in is armed. It is published at https://claude.ai/artifact/XTV1eBEfKrf1rqEkFGCpmP. Any change that alters turn timing, the phases, a threshold or barge-in updates this page in the same commit, alongside its `CHANGELOG-latency.md` entry. Then republish it with the Artifact tool, passing that URL as `url`. The file is in artifact format: no `<html>` or `<head>`, which the host adds when it publishes.

## Key Design Decisions

- **Sentence-level streaming**: Claude's response is split on sentence boundaries and each sentence is sent to TTS independently, rather than waiting for the full response. This is the core latency optimization.
- **VAD-gated transcription**: Audio is only sent to Whisper after Silero VAD detects a pause (400ms silence), avoiding constant transcription of silence/noise. The turn ends at 800ms unless the transcript is a question (CHANGELOG-latency.md, entry 4).
- **Async Anthropic client**: `agent.py` uses `anthropic.AsyncAnthropic`. The sync client blocked the event loop between tokens, stalling audio processing and barge-in (CHANGELOG-latency.md, entry 2).
- **Barge-in tracks playback**: the agent counts as talking until Twilio sends back the mark after its last sentence. `BARGE_IN_MS` (250) of speech while it talks cancels the reply and sends Twilio `clear`. Shorter sounds are ignored, and so is their transcript (CHANGELOG-latency.md, entry 7). If the speech that cut the agent off is an acknowledgement ("mm-hmm", "okay"), the reply resumes from the sentence the caller was hearing; otherwise Claude's history keeps only the sentences the caller heard (`SpokenReply`, entry 8).
- **One STT instance**: `WhisperSTT` is loaded once at startup as a global. Models are GPU-resident (`cuda`, `float16`).
