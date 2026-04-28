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
# Transfer-to number stored by call SID
_transfer_numbers: dict[str, str] = {}

# Debounce window: wait this long after a transcript before responding,
# in case more speech is coming.
_DEBOUNCE_SECONDS = 0.3

# If held text hasn't been released after this many seconds, flush it anyway.
_HOLD_TIMEOUT_SECONDS = 2.0

# Patterns that indicate a clearly complete turn — skip LLM classification
_COMPLETE_PATTERNS = (
    "what do you think",
    "what's your view",
    "what's your take",
    "your thoughts",
    "does that make sense",
    "if that makes sense",
    "what do you say",
    "do you agree",
    "would you say",
    "can you elaborate",
    "i can elaborate",
    "go ahead",
    "over to you",
    "right?",
    "yeah?",
)


def _is_obviously_complete(text: str) -> bool:
    """Fast check for clearly complete turns — avoids the LLM round-trip."""
    lower = text.lower().rstrip()
    # Any question in the last ~60 chars is likely turn-final
    tail = lower[-60:]
    if "?" in tail:
        return True
    return any(p in lower for p in _COMPLETE_PATTERNS)


async def _is_complete_turn(text: str) -> bool:
    """Determine if a transcript is a complete conversational turn.

    Fast-paths obvious cases, falls back to Haiku for ambiguous ones.
    """
    if _is_obviously_complete(text):
        return True

    import anthropic as _anthropic

    client = _anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    try:
        # Only send the last ~200 chars for speed
        context = text[-200:] if len(text) > 200 else text
        resp = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            system=(
                "You are a turn-detection classifier for a voice conversation. "
                "Given a transcript segment, determine if the speaker has finished "
                "their turn (complete thought) or is mid-sentence/pausing briefly. "
                "Respond with exactly one word: COMPLETE or INCOMPLETE."
            ),
            messages=[{"role": "user", "content": context}],
        )
        result = resp.content[0].text.strip().upper()
        return result == "COMPLETE"
    except Exception:
        # On error, assume complete to avoid holding indefinitely
        logger.warning("Turn detection failed, assuming complete")
        return True

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


# --- SMS Webhook ---


@app.post("/sms/inbound")
async def sms_inbound(request: Request):
    """Twilio webhook for incoming SMS. Parses the message and initiates an outbound call."""
    form = await request.form()
    from_number = form.get("From", "")
    body = form.get("Body", "").strip()

    logger.info(f"SMS from {from_number}: {body}")

    # Only allow authorized numbers
    if config.ALLOWED_NUMBERS and from_number not in config.ALLOWED_NUMBERS:
        logger.warning(f"Unauthorized SMS from {from_number}")
        twiml = '<?xml version="1.0" encoding="UTF-8"?><Response><Message>Not authorized.</Message></Response>'
        return Response(content=twiml, media_type="application/xml")

    if not body:
        twiml = '<?xml version="1.0" encoding="UTF-8"?><Response><Message>Send a message like: Call 555-123-4567 schedule a dental cleaning</Message></Response>'
        return Response(content=twiml, media_type="application/xml")

    # Use Claude to parse the SMS into a phone number and call context
    parsed = await _parse_sms(body)

    if not parsed:
        twiml = '<?xml version="1.0" encoding="UTF-8"?><Response><Message>I couldn\'t find a phone number in your message. Try: Call 555-123-4567 schedule a dental cleaning</Message></Response>'
        return Response(content=twiml, media_type="application/xml")

    to_number, context = parsed

    # Initiate the call
    try:
        twilio = TwilioClient(config.TWILIO_API_KEY, config.TWILIO_API_SECRET, config.TWILIO_ACCOUNT_SID)
        twiml_url = config.get_public_url() + "/call/outbound"

        call = twilio.calls.create(
            to=to_number,
            from_=config.TWILIO_PHONE,
            url=twiml_url,
        )

        if context:
            _call_contexts[call.sid] = context
        _transfer_numbers[call.sid] = from_number

        logger.info(f"SMS-triggered call: {call.sid} -> {to_number}")

        reply = f"Got it! Calling {to_number} now. I'll text you when it's done."
        twiml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{reply}</Message></Response>'
        return Response(content=twiml, media_type="application/xml")

    except Exception:
        logger.exception("Failed to initiate SMS-triggered call")
        twiml = '<?xml version="1.0" encoding="UTF-8"?><Response><Message>Something went wrong initiating the call. Please try again.</Message></Response>'
        return Response(content=twiml, media_type="application/xml")


async def _parse_sms(body: str) -> tuple[str, str] | None:
    """Use Claude to extract a phone number and call context from an SMS."""
    import anthropic as _anthropic

    client = _anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=(
            "Extract a phone number and call purpose from the user's message. "
            "Respond in exactly this format:\n"
            "PHONE: <number in E.164 format, e.g. +15551234567>\n"
            "CONTEXT: <what the agent should do on the call>\n\n"
            "If there is no phone number, respond with just: NO_PHONE"
        ),
        messages=[{"role": "user", "content": body}],
    )

    text = resp.content[0].text.strip()

    if "NO_PHONE" in text:
        return None

    phone = None
    context = ""
    for line in text.split("\n"):
        if line.startswith("PHONE:"):
            phone = line.split(":", 1)[1].strip()
        elif line.startswith("CONTEXT:"):
            context = line.split(":", 1)[1].strip()

    if not phone:
        return None

    return phone, context


# --- Outbound Call API ---


@app.post("/api/call")
async def initiate_call(request: Request):
    """Initiate an outbound call."""
    body = await request.json()
    to_number = body["to"]
    context = body.get("context")
    transfer_to = body.get("transfer_to")

    twilio = TwilioClient(config.TWILIO_API_KEY, config.TWILIO_API_SECRET, config.TWILIO_ACCOUNT_SID)
    twiml_url = config.get_public_url() + "/call/outbound"

    call = twilio.calls.create(
        to=to_number,
        from_=config.TWILIO_PHONE,
        url=twiml_url,
    )

    if context:
        _call_contexts[call.sid] = context
    if transfer_to:
        _transfer_numbers[call.sid] = transfer_to

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
    greeting_done = asyncio.Event()  # Set once greeting finishes playing
    response_task: asyncio.Task | None = None
    debounce_task: asyncio.Task | None = None
    hold_timeout_task: asyncio.Task | None = None
    pending_text = ""  # Accumulates incomplete utterances
    transcript_log: TranscriptLog | None = None

    transfer_number: str | None = None

    async def _debounced_respond(text: str) -> None:
        """Wait for debounce, check turn completeness, then fire the response."""
        await asyncio.sleep(_DEBOUNCE_SECONDS)
        nonlocal pending_text, response_task, hold_timeout_task

        # Ask Haiku if this is a complete turn
        is_complete = await _is_complete_turn(text)

        if not is_complete:
            pending_text = text
            logger.info(f"LLM: holding incomplete turn: {text[-80:]}")
            # Start a hold timeout — flush after N seconds if nothing new arrives
            if hold_timeout_task and not hold_timeout_task.done():
                hold_timeout_task.cancel()
            hold_timeout_task = asyncio.create_task(_flush_after_timeout())
        else:
            pending_text = ""
            if hold_timeout_task and not hold_timeout_task.done():
                hold_timeout_task.cancel()
            response_task = asyncio.create_task(
                _stream_agent_response(
                    ws, agent, stream_sid, call_sid, connected, speaking, transcript_log,
                    transfer_number, user_text=text
                )
            )

    async def _flush_after_timeout() -> None:
        """Flush held text after a timeout in case the speaker is done."""
        await asyncio.sleep(_HOLD_TIMEOUT_SECONDS)
        nonlocal pending_text, response_task
        if pending_text:
            text = pending_text
            pending_text = ""
            logger.info(f"Hold timeout — flushing: {text[-80:]}")
            response_task = asyncio.create_task(
                _stream_agent_response(
                    ws, agent, stream_sid, call_sid, connected, speaking, transcript_log,
                    transfer_number, user_text=text
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
                transfer_number = _transfer_numbers.pop(call_sid, None) if call_sid else None
                agent = CallAgent(context=context)
                transcript_log = TranscriptLog(call_sid)
                logger.info(f"Stream started: {stream_sid} (call={call_sid}, context={'yes' if context else 'none'})")

                if not greeting_sent:
                    greeting_sent = True
                    response_task = asyncio.create_task(
                        _stream_agent_response(
                            ws, agent, stream_sid, call_sid, connected, speaking,
                            transcript_log, transfer_number, greeting=True,
                            on_done=greeting_done,
                        )
                    )

            elif event == "media":
                payload = msg["media"]["payload"]
                pcm = twilio_payload_to_pcm16k(payload)

                await stt_engine.feed_audio(pcm)

                # VAD-level barge-in: cancel as soon as speech is detected,
                # but NOT during the greeting (let it play first)
                if (
                    stt_engine.speech_active
                    and speaking.is_set()
                    and greeting_done.is_set()
                    and response_task
                    and not response_task.done()
                ):
                    response_task.cancel()
                    speaking.clear()
                    logger.info("Barge-in detected (VAD)")

                transcript = await stt_engine.get_transcript()

                if transcript:
                    # Cancel any pending debounce — new speech arrived
                    if debounce_task and not debounce_task.done():
                        debounce_task.cancel()
                    # Cancel hold timeout — new speech resets it
                    if hold_timeout_task and not hold_timeout_task.done():
                        hold_timeout_task.cancel()

                    # Accumulate with any held text
                    combined = (pending_text + " " + transcript).strip() if pending_text else transcript
                    pending_text = ""

                    debounce_task = asyncio.create_task(_debounced_respond(combined))

            elif event == "stop":
                # Cancel debounce/timeout and flush any pending text
                if debounce_task and not debounce_task.done():
                    debounce_task.cancel()
                if hold_timeout_task and not hold_timeout_task.done():
                    hold_timeout_task.cancel()
                if pending_text and agent:
                    response_task = asyncio.create_task(
                        _stream_agent_response(
                            ws, agent, stream_sid, call_sid, connected, speaking,
                            transcript_log, transfer_number, user_text=pending_text
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


_TRANSFER_TAG = "[TRANSFER]"
_SUMMARY_RE = __import__("re").compile(r"\[SUMMARY:\s*(.+?)\]")


async def _stream_agent_response(
    ws: WebSocket,
    agent: CallAgent,
    stream_sid: str,
    call_sid: str | None,
    connected: asyncio.Event,
    speaking: asyncio.Event,
    transcript_log: TranscriptLog | None = None,
    transfer_number: str | None = None,
    user_text: str | None = None,
    greeting: bool = False,
    on_done: asyncio.Event | None = None,
) -> None:
    """Stream Claude's response sentence-by-sentence through TTS to Twilio."""
    should_transfer = False

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

            # Check for transfer signal
            if _TRANSFER_TAG in sentence:
                sentence = sentence.replace(_TRANSFER_TAG, "").strip()
                should_transfer = True

            # Strip summary tag from spoken text (will be sent as SMS)
            summary_match = _SUMMARY_RE.search(sentence)
            if summary_match:
                sentence = _SUMMARY_RE.sub("", sentence).strip()

            full_response += (" " if full_response else "") + sentence

            # Send each sentence to TTS immediately — don't wait for full response
            if sentence:
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

            if should_transfer:
                break

        if transcript_log and full_response:
            transcript_log.agent(full_response)

        logger.info(f"Sent {total_chunks} audio chunks (streamed)")

        # Execute transfer after speaking the farewell
        if should_transfer and transfer_number and call_sid:
            await _execute_transfer(call_sid, transfer_number)

        # Send SMS summary to user if the call achieved its objective
        summary_match = _SUMMARY_RE.search(full_response)
        if summary_match and transfer_number:
            await _send_sms(transfer_number, summary_match.group(1))

    except asyncio.CancelledError:
        logger.info("Response cancelled (barge-in)")
    except Exception:
        logger.exception("Error in agent response pipeline")
    finally:
        speaking.clear()
        if on_done:
            on_done.set()


async def _execute_transfer(call_sid: str, transfer_to: str) -> None:
    """Transfer an active call to another number via Twilio API."""
    try:
        twilio = TwilioClient(config.TWILIO_API_KEY, config.TWILIO_API_SECRET, config.TWILIO_ACCOUNT_SID)
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>{transfer_to}</Dial>
</Response>"""
        twilio.calls(call_sid).update(twiml=twiml)
        logger.info(f"Call {call_sid} transferred to {transfer_to}")
    except Exception:
        logger.exception(f"Failed to transfer call {call_sid}")


async def _send_sms(to: str, message: str) -> None:
    """Send an SMS summary to the user via Twilio."""
    try:
        twilio = TwilioClient(config.TWILIO_API_KEY, config.TWILIO_API_SECRET, config.TWILIO_ACCOUNT_SID)
        twilio.messages.create(
            to=to,
            from_=config.TWILIO_PHONE,
            body=f"Call complete: {message}",
        )
        logger.info(f"SMS sent to {to}: {message}")
    except Exception:
        logger.exception(f"Failed to send SMS to {to}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
