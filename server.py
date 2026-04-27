"""Telephony agent — FastAPI server with Twilio Media Streams."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from twilio.rest import Client as TwilioClient

import config
import tts
from agent import CallAgent
from audio import twilio_payload_to_pcm16k
from stt import WhisperSTT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

app = FastAPI(title="Telephony Agent")

# Shared STT instance (model loaded once)
stt_engine: WhisperSTT | None = None

# Call context stored by call SID — set via /api/call, consumed when WebSocket connects
_call_contexts: dict[str, str] = {}

# Words/patterns that suggest the speaker hasn't finished their thought
_INCOMPLETE_SUFFIXES = (
    " and", " but", " or", " so", " because", " since", " although",
    " though", " while", " if", " when", " that", " which", " who",
    " like", " um", " uh", " well", " I mean", " you know",
    " actually", " basically", " the", " a", " an", " my", " your",
    " their", " to", " for", " with", " about", " of",
)


def _looks_incomplete(text: str) -> bool:
    """Check if a transcript looks like an unfinished thought."""
    # Strip trailing punctuation that Whisper adds (periods, ellipses, etc.)
    lower = text.lower().rstrip().rstrip(".!?…").rstrip()
    return any(lower.endswith(suffix) for suffix in _INCOMPLETE_SUFFIXES)


# Debounce window: wait this long after a transcript before responding,
# in case more speech is coming.
_DEBOUNCE_SECONDS = 0.3

# Transcript archive directory
_TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"
_TRANSCRIPTS_DIR.mkdir(exist_ok=True)


class TranscriptLog:
    """Records user/agent turns for a single call with timestamps."""

    def __init__(self, call_sid: str | None = None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = call_sid or "unknown"
        self._path = _TRANSCRIPTS_DIR / f"{ts}_{label}.txt"
        self._lines: list[str] = []

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def user(self, text: str) -> None:
        self._lines.append(f"[{self._ts()}] User: {text}")

    def agent(self, text: str) -> None:
        self._lines.append(f"[{self._ts()}] Agent: {text}")

    def save(self) -> None:
        if self._lines:
            self._path.write_text("\n".join(self._lines) + "\n")
            logger.info(f"Transcript saved: {self._path}")


@app.on_event("startup")
async def startup():
    global stt_engine
    stt_engine = WhisperSTT(model_size=config.WHISPER_MODEL)


# --- TwiML Webhooks ---


@app.post("/call/inbound")
async def call_inbound(request: Request):
    """Twilio webhook for incoming calls. Returns TwiML to connect Media Stream."""
    ws_url = config.get_public_url().replace("http", "ws") + "/ws/media"
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@app.post("/call/outbound")
async def call_outbound(request: Request):
    """TwiML for outbound calls — same Stream connection."""
    ws_url = config.get_public_url().replace("http", "ws") + "/ws/media"
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


# --- Outbound Call API ---


@app.post("/api/call")
async def initiate_call(request: Request):
    """Initiate an outbound call."""
    body = await request.json()
    to_number = body["to"]
    context = body.get("context")

    twilio = TwilioClient(config.TWILIO_API_KEY, config.TWILIO_API_SECRET, config.TWILIO_ACCOUNT_SID)
    twiml_url = config.get_public_url() + "/call/outbound"

    call = twilio.calls.create(
        to=to_number,
        from_=config.TWILIO_PHONE,
        url=twiml_url,
    )

    if context:
        _call_contexts[call.sid] = context

    logger.info(f"Outbound call initiated: {call.sid} -> {to_number}")
    return {"call_sid": call.sid, "status": call.status}


# --- WebSocket Media Stream ---


@app.websocket("/ws/media")
async def media_stream(ws: WebSocket):
    """Handle Twilio Media Stream WebSocket."""
    await ws.accept()
    logger.info("WebSocket connected")

    agent = None
    stream_sid = None
    call_sid = None
    connected = asyncio.Event()
    connected.set()
    speaking = asyncio.Event()  # Set while agent TTS is being sent
    response_task: asyncio.Task | None = None
    debounce_task: asyncio.Task | None = None
    pending_text = ""  # Accumulates incomplete utterances
    transcript_log: TranscriptLog | None = None

    async def _debounced_respond(text: str) -> None:
        """Wait briefly for more speech, then fire the response."""
        await asyncio.sleep(_DEBOUNCE_SECONDS)
        nonlocal response_task
        response_task = asyncio.create_task(
            _stream_agent_response(
                ws, agent, stream_sid, connected, speaking, transcript_log, user_text=text
            )
        )

    try:
        greeting_sent = False

        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                call_sid = msg["start"].get("callSid")
                context = _call_contexts.pop(call_sid, None) if call_sid else None
                agent = CallAgent(context=context)
                transcript_log = TranscriptLog(call_sid)
                logger.info(f"Stream started: {stream_sid} (call={call_sid}, context={'yes' if context else 'none'})")

                if not greeting_sent:
                    greeting_sent = True
                    response_task = asyncio.create_task(
                        _stream_agent_response(ws, agent, stream_sid, connected, speaking, transcript_log, greeting=True)
                    )

            elif event == "media":
                payload = msg["media"]["payload"]
                pcm = twilio_payload_to_pcm16k(payload)

                await stt_engine.feed_audio(pcm)

                # VAD-level barge-in: cancel as soon as speech is detected,
                # don't wait for a full transcript
                if stt_engine.speech_active and speaking.is_set() and response_task and not response_task.done():
                    response_task.cancel()
                    speaking.clear()
                    logger.info("Barge-in detected (VAD)")

                transcript = await stt_engine.get_transcript()

                if transcript:
                    # Cancel any pending debounce — new speech arrived
                    if debounce_task and not debounce_task.done():
                        debounce_task.cancel()

                    # Accumulate if the utterance looks incomplete
                    combined = (pending_text + " " + transcript).strip() if pending_text else transcript

                    if _looks_incomplete(combined):
                        pending_text = combined
                        logger.info(f"Holding incomplete utterance: {combined}")
                    else:
                        pending_text = ""
                        debounce_task = asyncio.create_task(_debounced_respond(combined))

            elif event == "stop":
                # Cancel debounce and flush any pending text
                if debounce_task and not debounce_task.done():
                    debounce_task.cancel()
                if pending_text and agent:
                    response_task = asyncio.create_task(
                        _stream_agent_response(
                            ws, agent, stream_sid, connected, speaking, transcript_log, user_text=pending_text
                        )
                    )
                    pending_text = ""
                logger.info("Stream stopped")
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        connected.clear()
        if transcript_log:
            transcript_log.save()


async def _stream_agent_response(
    ws: WebSocket,
    agent: CallAgent,
    stream_sid: str,
    connected: asyncio.Event,
    speaking: asyncio.Event,
    transcript_log: TranscriptLog | None = None,
    user_text: str | None = None,
    greeting: bool = False,
) -> None:
    """Stream Claude's response sentence-by-sentence through TTS to Twilio."""
    try:
        if greeting:
            sentence_stream = agent.greeting_stream()
        else:
            sentence_stream = agent.respond_stream(user_text)
            if transcript_log:
                transcript_log.user(user_text)

        total_chunks = 0
        full_response = ""
        speaking.set()

        async for sentence in sentence_stream:
            if not connected.is_set():
                logger.info("Connection closed, aborting response")
                return

            full_response += (" " if full_response else "") + sentence

            # Send each sentence to TTS immediately — don't wait for full response
            chunks = await tts.synthesize(sentence)

            for chunk in chunks:
                if not connected.is_set():
                    logger.info("Connection closed, aborting response")
                    return
                media_msg = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": chunk},
                }
                await ws.send_text(json.dumps(media_msg))

            total_chunks += len(chunks)

        if transcript_log and full_response:
            transcript_log.agent(full_response)

        logger.info(f"Sent {total_chunks} audio chunks (streamed)")

    except asyncio.CancelledError:
        logger.info("Response cancelled (barge-in)")
    except Exception:
        logger.exception("Error in agent response pipeline")
    finally:
        speaking.clear()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
