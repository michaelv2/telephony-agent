"""Telephony agent — FastAPI server with Twilio Media Streams."""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.request_validator import RequestValidator
from twilio.rest import Client as TwilioClient

import config
import tts
from agent import CallAgent
from audio import twilio_payload_to_pcm16k
from stt import WhisperSTT
from turn_taking import TurnTaking

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

app = FastAPI(title="Telephony Agent")

_twilio_validator = RequestValidator(config.TWILIO_AUTH_TOKEN)


async def _require_twilio_signature(request: Request) -> None:
    """Validate that the request was signed by Twilio."""
    signature = request.headers.get("X-Twilio-Signature", "")
    # Twilio signs against the public URL, not the local one behind ngrok
    url = config.get_public_url() + request.url.path
    form = await request.form()
    params = dict(form)
    if not _twilio_validator.validate(url, params, signature):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

# Shared STT instance (model loaded once)
stt_engine: WhisperSTT | None = None

# Call context stored by call SID — set via /api/call, consumed when WebSocket connects
_call_contexts: dict[str, str] = {}
# Transfer-to number stored by call SID
_transfer_numbers: dict[str, str] = {}

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
    stt_engine = WhisperSTT(
        model_size=config.WHISPER_MODEL,
        silence_threshold_ms=config.VAD_SILENCE_MS,
        early_threshold_ms=config.VAD_EARLY_MS,
    )


# --- TwiML Webhooks ---


@app.post("/call/inbound", dependencies=[Depends(_require_twilio_signature)])
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


@app.post("/call/outbound", dependencies=[Depends(_require_twilio_signature)])
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


@app.post("/sms/inbound", dependencies=[Depends(_require_twilio_signature)])
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
    """Handle one call's Twilio Media Stream. Turn-taking decisions are in turn_taking.py."""
    await ws.accept()
    logger.info("WebSocket connected")

    agent = None
    stream_sid = None
    call_sid = None
    transfer_number: str | None = None
    transcript_log: TranscriptLog | None = None
    connected = asyncio.Event()
    connected.set()

    def start_reply(text: str | None) -> asyncio.Task:
        """Start a reply to text, or the greeting when text is None."""
        if text is None:
            reply = _stream_agent_response(
                ws, agent, stream_sid, call_sid, connected, turns.speaking,
                transcript_log, transfer_number, greeting=True, on_done=turns.greeting_done,
            )
        else:
            reply = _stream_agent_response(
                ws, agent, stream_sid, call_sid, connected, turns.speaking,
                transcript_log, transfer_number, user_text=text, turn_t0=stt_engine.speech_ended_at,
            )
        return asyncio.create_task(reply)

    turns = TurnTaking(start_reply)

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
                    turns.greet()

            elif event == "media":
                pcm = twilio_payload_to_pcm16k(msg["media"]["payload"])
                await stt_engine.feed_audio(pcm)

                turns.on_audio(stt_engine.speech_active)
                transcript = await stt_engine.get_transcript()
                if transcript:
                    turns.on_transcript(transcript)
                if stt_engine.pop_end_of_turn():
                    turns.on_end_of_turn()

            elif event == "stop":
                turns.on_stop()
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
    turn_t0: float | None = None,
) -> None:
    """Stream Claude's response sentence-by-sentence through TTS to Twilio.

    turn_t0 is the time.monotonic() at which the caller stopped speaking, for latency logging.
    """
    should_transfer = False
    started_at = time.monotonic()
    first_sentence_at: float | None = None
    total_chunks = 0

    try:
        if greeting:
            sentence_stream = agent.greeting_stream()
        else:
            sentence_stream = agent.respond_stream(user_text)
            if transcript_log:
                transcript_log.user(user_text)

        full_response = ""
        speaking.set()

        async for sentence in sentence_stream:
            if not connected.is_set():
                logger.info("Connection closed, aborting response")
                return
            first_sentence_at = first_sentence_at or time.monotonic()

            # Check for transfer signal
            if _TRANSFER_TAG in sentence:
                sentence = sentence.replace(_TRANSFER_TAG, "").strip()
                should_transfer = True

            # Strip summary tag from spoken text (will be sent as SMS)
            summary_match = _SUMMARY_RE.search(sentence)
            if summary_match:
                sentence = _SUMMARY_RE.sub("", sentence).strip()

            full_response += (" " if full_response else "") + sentence

            # Stream TTS audio to Twilio as chunks arrive — don't wait for full sentence audio
            if sentence:
                async for chunk in tts.synthesize_stream(sentence):
                    if not connected.is_set():
                        logger.info("Connection closed, aborting response")
                        return
                    media_msg = {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": chunk},
                    }
                    await ws.send_text(json.dumps(media_msg))
                    total_chunks += 1
                    if total_chunks == 1 and turn_t0 is not None:
                        now = time.monotonic()
                        logger.info(
                            f"Latency: respond +{(started_at - turn_t0) * 1000:.0f}ms, "
                            f"first sentence +{(first_sentence_at - turn_t0) * 1000:.0f}ms, "
                            f"first audio +{(now - turn_t0) * 1000:.0f}ms after end of speech"
                        )

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
        # Audio already sent keeps playing from Twilio's buffer, so "after" means the caller heard it.
        logger.info(
            f"Response cancelled (barge-in) after {total_chunks} audio chunks"
            if total_chunks else "Response cancelled (barge-in) before any audio"
        )
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
