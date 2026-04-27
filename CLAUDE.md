# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A real-time telephony agent that handles phone calls via Twilio Media Streams. Incoming audio goes through Whisper STT, the transcript is sent to Claude for a response, and the response is streamed sentence-by-sentence through Kokoro TTS back to the caller.

## Running

```bash
# Full stack (ngrok + server) — requires ngrok installed
./start.sh

# Server only (if you manage tunneling separately or have PUBLIC_URL set)
.venv/bin/python server.py
```

Requires a Kokoro TTS server running at `TTS_URL` (default `http://localhost:9100`).

Set Twilio voice webhook to `{PUBLIC_URL}/call/inbound` (POST).

## Required Environment Variables (.env)

- `ANTHROPIC_API_KEY`
- `TWILIO_ACCOUNT_SID`, `TWILIO_API_KEY`, `TWILIO_API_SECRET`, `TWILIO_PHONE`

Optional: `TTS_URL`, `TTS_VOICE` (`af_heart`), `WHISPER_MODEL` (`base.en`), `PORT` (`9002`), `PUBLIC_URL` (auto-detected from ngrok), `AGENT_PROMPT`.

## Architecture

The real-time pipeline is: **Twilio mulaw 8kHz -> PCM 16kHz -> Silero VAD -> Whisper -> Claude (streaming) -> Kokoro TTS -> mulaw 8kHz -> Twilio**.

**`server.py`** — FastAPI app. Three HTTP endpoints (`/call/inbound`, `/call/outbound` for TwiML, `/api/call` to initiate outbound calls) and one WebSocket endpoint (`/ws/media`) that handles the Twilio Media Stream protocol. Each media event is decoded, fed to STT, and when a transcript is ready, `_stream_agent_response` is fired as an async task.

**`agent.py`** — `CallAgent` wraps the Anthropic SDK's synchronous streaming API (`client.messages.stream`). Maintains conversation history per call. Splits Claude's streamed text on sentence boundaries (`[.!?]\s+`) and yields each sentence to TTS immediately for low latency.

**`audio.py`** — Bidirectional format conversion. Twilio sends base64 mulaw 8kHz; Whisper needs PCM 16kHz; Kokoro outputs WAV (24kHz). Key functions: `twilio_payload_to_pcm16k` (inbound) and `wav_bytes_to_mulaw_payloads` (outbound, chunks WAV into 20ms mulaw frames).

**`tts.py`** — Async HTTP client to external Kokoro TTS server. Sends text, receives WAV, converts to Twilio-ready mulaw chunks.

**`stt/`** — Pluggable STT behind `BaseSTT` ABC. `WhisperSTT` uses Silero VAD (512-sample frames at 16kHz) to detect speech boundaries, accumulates audio during speech, transcribes with faster-whisper after 800ms silence. VAD + transcription run in the event loop / executor respectively.

**`config.py`** — All config from env vars via `python-dotenv`. `get_public_url()` auto-detects ngrok tunnel URL by querying `localhost:4040/api/tunnels`.

## Key Design Decisions

- **Sentence-level streaming**: Claude's response is split on sentence boundaries and each sentence is sent to TTS independently, rather than waiting for the full response. This is the core latency optimization.
- **VAD-gated transcription**: Audio is only sent to Whisper after Silero VAD detects end-of-speech (800ms silence), avoiding constant transcription of silence/noise.
- **Synchronous Anthropic client**: `agent.py` uses `anthropic.Anthropic` (sync) with `.messages.stream()`, wrapped in async generators. The sync streaming runs in the calling coroutine's thread.
- **One STT instance**: `WhisperSTT` is loaded once at startup as a global. Models are GPU-resident (`cuda`, `float16`).
