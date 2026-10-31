"""Turn-taking for one call: when the agent replies, and when the caller's speech cuts it off.

server.py passes in the STT signals for every 20ms media frame. TurnTaking makes the decisions and
starts or cancels replies through a callback. It does no audio or network I/O, so
test_turn_taking.py can drive it with scripted signals.

Caller side, one of three phases (TurnTaking.phase):

  LISTENING  Nothing pending.
  WAITING    A transcript is pending. Whisper ran after VAD_EARLY_MS (400ms) of silence, and the
             text wasn't a question or a hand-off phrase, so wait for the end of turn at
             VAD_SILENCE_MS (800ms). More speech before then is appended.
  HOLDING    The end of turn came but the text looks unfinished ("...and"). Reply anyway after
             _HOLD_TIMEOUT_SECONDS, unless the caller speaks first (back to WAITING).

Agent side: a reply is running or it isn't. Caller speech while a reply is being generated or
sent cancels it (barge-in), except during the greeting. Audio already sent keeps playing on the
phone, and once a reply has been sent, barge-in is off even though it may still be playing.

CHANGELOG-latency.md records why each threshold has its current value.
"""

import asyncio
import logging
from collections.abc import Callable
from enum import Enum

logger = logging.getLogger("turn")

# Unfinished text waits this long after the end of turn, then gets a reply anyway.
_HOLD_TIMEOUT_SECONDS = 1.0

# Phrases that hand the turn back to the agent. Like a question, they reply at the early mark
# (VAD_EARLY_MS) instead of waiting for the end of turn.
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


_INCOMPLETE_TRAIL = (
    " and", " but", " or", " so", " because", " since", " although",
    " however", " yet", " while", " when", " if", " that", " which",
    " who", " where", " then", " also", " to", " for", " with",
    " about", " like", " just", " the", " a", " an", " my", " your",
    " their", " its", " of", " in", " on", " at", " is", " are",
    " was", " were", " i'm", " we're", " they're", " it's",
)


def _is_obviously_complete(text: str) -> bool:
    """Fast check for clearly complete turns — questions and turn-yielding phrases."""
    lower = text.lower().rstrip()
    tail = lower[-60:]
    if "?" in tail:
        return True
    return any(p in lower for p in _COMPLETE_PATTERNS)


def _is_complete_turn(text: str) -> bool:
    """Determine if a transcript is a complete conversational turn.

    Uses local heuristics only — no network calls.
    See CHANGELOG-latency.md for why the Haiku LLM fallback was removed.
    """
    if _is_obviously_complete(text):
        return True

    stripped = text.strip()
    if not stripped:
        return False

    if stripped[-1] in ".!?":
        words = stripped.split()
        if len(words) >= 3:
            return True

    lower = stripped.lower()
    for trail in _INCOMPLETE_TRAIL:
        if lower.endswith(trail):
            return False

    return False


class Phase(Enum):
    LISTENING = "listening"
    WAITING = "waiting for end of turn"
    HOLDING = "holding"


class TurnTaking:
    """One call's turn-taking state. Feed it STT signals in this order for each media frame:
    on_audio, then on_transcript if there is one, then on_end_of_turn if it fired.
    """

    def __init__(self, start_reply: Callable[[str | None], asyncio.Task]):
        """start_reply(text) starts a reply to text, or the greeting when text is None, and
        returns its task. The reply must set and clear `speaking`, and the greeting must set
        `greeting_done` when it ends.
        """
        self._start_reply = start_reply
        self._hold_timeout_s = _HOLD_TIMEOUT_SECONDS
        self.speaking = asyncio.Event()  # Set while a reply is being generated or sent; not while it plays
        self.greeting_done = asyncio.Event()  # Set once the greeting has been sent (it may still be playing)
        self.pending = ""  # Caller text not replied to yet
        self._reply: asyncio.Task | None = None  # The most recent reply
        self._hold: asyncio.Task | None = None  # The hold timer, while HOLDING

    @property
    def phase(self) -> Phase:
        if self._hold is not None:
            return Phase.HOLDING
        return Phase.WAITING if self.pending else Phase.LISTENING

    def greet(self) -> None:
        """Start the greeting. Caller speech can't cut it off."""
        self._reply = self._start_reply(None)

    def on_audio(self, speech_active: bool) -> None:
        """Every media frame, once STT has processed it."""
        if not speech_active:
            return
        # Barge-in: the caller talks while a reply is being generated or sent.
        if (
            self.speaking.is_set()
            and self.greeting_done.is_set()
            and self._reply
            and not self._reply.done()
        ):
            self._reply.cancel()
            self.speaking.clear()
            logger.info("Barge-in detected (VAD)")
        # HOLDING -> WAITING: don't flush held text in the middle of the caller's next utterance.
        self._cancel_hold()

    def on_transcript(self, text: str) -> None:
        """Whisper's text for the latest stretch of speech, VAD_EARLY_MS after it ended."""
        self._cancel_hold()
        combined = (self.pending + " " + text).strip() if self.pending else text
        self.pending = ""
        if _is_obviously_complete(combined):
            self._reply_to(combined)  # A question or hand-off phrase: reply now
        else:
            self.pending = combined  # WAITING for the end of turn

    def on_end_of_turn(self) -> None:
        """VAD_SILENCE_MS of silence after the last transcript."""
        if not self.pending:
            return
        if _is_complete_turn(self.pending):
            self._reply_to(self._take_pending())
        else:
            logger.info(f"Holding unfinished turn: {self.pending[-80:]}")
            self._cancel_hold()
            self._hold = asyncio.create_task(self._flush_after_hold())  # HOLDING

    def on_stop(self) -> None:
        """The call ended. Reply to any held text (the reply aborts once the socket closes)."""
        self._cancel_hold()
        if self.pending:
            self._reply_to(self._take_pending())

    def _reply_to(self, text: str) -> None:
        self._cancel_hold()
        self._reply = self._start_reply(text)

    def _take_pending(self) -> str:
        text, self.pending = self.pending, ""
        return text

    def _cancel_hold(self) -> None:
        if self._hold is not None:
            self._hold.cancel()
            self._hold = None

    async def _flush_after_hold(self) -> None:
        await asyncio.sleep(self._hold_timeout_s)
        self._hold = None
        if self.pending:
            text = self._take_pending()
            logger.info(f"Hold timeout — flushing: {text[-80:]}")
            self._reply_to(text)
