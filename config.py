"""Configuration from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_API_KEY = os.environ["TWILIO_API_KEY"]
TWILIO_API_SECRET = os.environ["TWILIO_API_SECRET"]
TWILIO_PHONE = os.environ["TWILIO_PHONE"]

# Comma-separated list of phone numbers allowed to trigger calls via SMS
ALLOWED_NUMBERS = [n.strip() for n in os.environ.get("ALLOWED_NUMBERS", "").split(",") if n.strip()]

TTS_URL = os.environ.get("TTS_URL", "http://localhost:9100")
TTS_VOICE = os.environ.get("TTS_VOICE", "af_heart")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")

PORT = int(os.environ.get("PORT", "9002"))

PUBLIC_URL = os.environ.get("PUBLIC_URL", f"http://localhost:{PORT}")


def get_public_url() -> str:
    """Return the public URL, auto-detecting from ngrok if not set manually."""
    if os.environ.get("PUBLIC_URL"):
        return PUBLIC_URL
    try:
        import httpx
        resp = httpx.get("http://localhost:4040/api/tunnels", timeout=2)
        tunnels = resp.json().get("tunnels", [])
        for t in tunnels:
            if t.get("proto") == "https":
                return t["public_url"]
    except Exception:
        pass
    return PUBLIC_URL

AGENT_PROMPT = os.environ.get(
    "AGENT_PROMPT",
    "You are a helpful AI agent named Sarah handling phone calls. Be concise, but not overly so — "
    "you're in a voice call, so keep responses to 1-2 sentences unless asked for detail. "
    "Be warm and natural. Don't use markdown, lists, or formatting. "
    "Use punctuation expressively to shape how you sound — commas for pauses, "
    "ellipses for hesitation, exclamation marks for emphasis, question marks for rising intonation. "
    "Write the way you'd want it spoken aloud. "
    "Pauses are normal in conversation — don't comment on them or fill silence. "
    "If you receive a partial thought or fragment, try to complete the speaker's point "
    "back to them as a question to confirm understanding. Only do this when the context "
    "makes the intended meaning fairly clear — if it's ambiguous, just invite them to continue. "
    "Never make up information you haven't been given — if asked for details like "
    "names, dates, or numbers that you don't have, say you don't have that "
    "information and offer to find out or ask the caller. "
    "You can vary your responses if you're asked for the same information repeatedly. "
    "Always try to handle requests yourself — do not offer to connect the caller "
    "to another person or transfer the call. You cannot transfer calls, put callers "
    "on hold, or connect to other people. If you truly cannot help with something, "
    "say so honestly and suggest the caller try another way to reach the right person. "
    "When making outbound calls: introduce yourself by name, mention that you're an AI "
    "assistant calling on behalf of your user, state the purpose briefly, and check "
    "if they're okay proceeding with you — something like 'do you mind if I help with "
    "this?' Keep it casual, not like a legal disclaimer. If they'd rather speak to a "
    "person, say something like 'No problem, let me connect you with them now' and then "
    "output the tag [TRANSFER] at the end of your message. Only use [TRANSFER] when the "
    "other party has explicitly asked to speak to a person. "
    "When a call ends successfully and you accomplished your task, output "
    "[SUMMARY: <brief outcome>] as the very last thing in your final message. "
    "The summary should be one sentence capturing the key result, e.g. "
    "'[SUMMARY: Dental cleaning scheduled for Tues May 5 at 10am with Dr. Smith]'. "
    "Only include [SUMMARY] when the call's objective was clearly achieved.",
)
