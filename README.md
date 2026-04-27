# Telephony Agent

AI-powered phone agent using Twilio Media Streams, Whisper STT, Claude, and streaming TTS.

Handles both inbound and outbound calls. Streams Claude's response sentence-by-sentence through TTS for low-latency conversational audio.

## Pipeline

```
Caller → Twilio (mulaw 8kHz) → Silero VAD → Whisper STT → Claude (streaming) → TTS → Twilio → Caller
```

## Prerequisites

- Python 3.11+
- NVIDIA GPU (for Whisper + Silero VAD)
- [ngrok](https://ngrok.com/) (for local development)
- A Twilio account with a phone number and API keys
- An Anthropic API key
- The [kokoro-tts-server](../kokoro-tts-server) running (supports Kokoro, Chatterbox, Qwen3, and Orpheus backends)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:

```
ANTHROPIC_API_KEY=
TWILIO_ACCOUNT_SID=
TWILIO_API_KEY=
TWILIO_API_SECRET=
TWILIO_PHONE=+1...
```

### Optional environment variables

| Variable | Default | Description |
|---|---|---|
| `TTS_URL` | `http://localhost:9100` | TTS server endpoint |
| `TTS_VOICE` | `af_heart` | Voice name for TTS (see table below) |
| `WHISPER_MODEL` | `base.en` | Faster-whisper model size |
| `PORT` | `9002` | Server port |
| `PUBLIC_URL` | auto-detected from ngrok | Public URL for Twilio webhooks |
| `AGENT_PROMPT` | *(built-in)* | System prompt for the agent |

### Available TTS backends

The TTS server supports multiple backends. Set `TTS_VOICE` accordingly:

| Backend | `TTS_VOICE` | Notes |
|---|---|---|
| Kokoro | `af_heart` |
| Chatterbox | `default` |
| Qwen3 (clone) | `heart` (folder name in `voices/qwen3/`) | Too slow for real-time |
| Qwen3 (design) | A natural language description, e.g. `a warm female voice with a slight British accent` |
| Orpheus | `tara` |

For Qwen3 voice design mode, `TTS_VOICE` is passed directly as the voice design instruction — describe the voice you want in plain English.

## Running

### Full stack (ngrok + server)

```bash
./start.sh
```

This starts ngrok, waits for the tunnel, then launches the server. Set your Twilio voice webhook to the printed URL + `/call/inbound`.

### Server only

```bash
.venv/bin/python server.py
```

Use this if you manage tunneling separately or have `PUBLIC_URL` set.

## API

### `POST /call/inbound`

Twilio webhook for incoming calls. Returns TwiML to connect a Media Stream.

### `POST /api/call`

Initiate an outbound call.

```json
{
  "to": "+15551234567",
  "context": "Call Dr. Smith's office to schedule a routine checkup for Jane Doe, DOB 04/22/1990."
}
```

The `context` field is optional. When provided, it's appended to the agent's system prompt for that call.

### `WS /ws/media`

Twilio Media Stream WebSocket. Handles the real-time audio pipeline.

## CLI testing

Test the agent's conversational behavior without the voice pipeline:

```bash
# Default prompt
.venv/bin/python cli.py

# With call context
.venv/bin/python cli.py "Call a doctor's office to schedule an appointment"

# Override prompt via env var
AGENT_PROMPT="You are a dental clinic receptionist." .venv/bin/python cli.py
```
