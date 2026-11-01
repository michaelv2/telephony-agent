"""Turn-taking for one call: when the agent replies, and when the caller's speech cuts it off.

server.py passes in the STT signals for every 20ms media frame, and the Twilio marks that come
back as audio finishes playing. TurnTaking makes the decisions, and starts replies and stops the
agent through callbacks. It does no audio or network I/O, so test_turn_taking.py can drive it
with scripted signals.

Caller side, one of three phases (TurnTaking.phase):

  LISTENING  Nothing pending.
  WAITING    A transcript is pending. Whisper ran after VAD_EARLY_MS (400ms) of silence, and the
             text wasn't a question or a hand-off phrase, so wait for the end of turn at
             VAD_SILENCE_MS (800ms). More speech before then is appended.
  HOLDING    The end of turn came but the text looks unfinished ("...and"). Reply anyway after
             _HOLD_TIMEOUT_SECONDS, unless the caller speaks first (back to WAITING).

Agent side: the agent is talking while a reply is being written or sent, and until Twilio says
its audio has finished playing (Playback). Once the caller has spoken for _BARGE_IN_MS while it
talks, the agent stops: the reply is cancelled and the audio queued at Twilio is cleared
(barge-in). Shorter sounds, like a cough, don't stop it, and their transcript gets no reply. If
the speech that stopped it turns out to be an acknowledgement ("mm-hmm", "okay"), the reply picks
up again from the sentence the caller was hearing. Otherwise Claude's history keeps only the
sentences the caller heard. The greeting can't be cut off until it has finished playing.

CHANGELOG-latency.md records why each threshold has its current value.
"""

import asyncio
import logging
import re
from collections.abc import Callable
from enum import Enum

logger = logging.getLogger("turn")

# Unfinished text waits this long after the end of turn, then gets a reply anyway.
_HOLD_TIMEOUT_SECONDS = 1.0

# Speech needed while the agent talks before it stops (config.BARGE_IN_MS overrides it).
_BARGE_IN_MS = 250

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


# Words a caller says to show they're listening rather than to take the turn. A transcript made
# only of these, after it cut the agent off mid-reply, resumes the reply instead of getting one.
_ACKNOWLEDGEMENTS = frozenset(
    "mm mmm mmhmm mhm hmm hm uh uhhuh huh okay ok yeah yep yup yes right sure alright gotcha "
    "got it i see go on ahead".split()
)


def _is_acknowledgement(text: str) -> bool:
    words = re.sub(r"[^a-z ]", " ", text.lower().replace("-", "")).split()
    return 0 < len(words) <= 4 and all(w in _ACKNOWLEDGEMENTS for w in words)


class Phase(Enum):
    LISTENING = "listening"
    WAITING = "waiting for end of turn"
    HOLDING = "holding"


class SpokenReply:
    """One reply's sentences, and how many of them the caller has heard in full."""

    def __init__(self):
        self.sentences: list[str] = []  # As spoken (tags stripped), in order
        self.heard = 0  # The first `heard` sentences have finished playing
        self.written = False  # The reply ran to the end: no more sentences are coming
        self.cut_off = False  # Barge-in stopped it

    @property
    def unheard(self) -> list[str]:
        return self.sentences[self.heard:]

    @property
    def heard_text(self) -> str:
        return " ".join(self.sentences[: self.heard])

    @property
    def cut_short(self) -> bool:
        """It was cut off before the caller heard all of it, so Claude's history needs fixing."""
        return self.cut_off and (self.heard < len(self.sentences) or not self.written)


class Playback:
    """Audio queued at Twilio that the caller hasn't heard yet, tracked with Twilio marks.

    A reply sends a mark after each sentence's audio. Twilio sends it back once that sentence has
    played, or straight away after a `clear` (by then the mark is forgotten, so it doesn't count).
    """

    def __init__(self):
        self._pending: dict[str, tuple[SpokenReply, int]] = {}
        self._count = 0

    def mark(self, spoken: SpokenReply, index: int) -> str:
        """Name a new mark for sentence `index` of `spoken`, to send after its audio."""
        name = f"m{self._count}"
        self._count += 1
        self._pending[name] = (spoken, index)
        return name

    def on_mark(self, name: str) -> None:
        entry = self._pending.pop(name, None)
        if entry is not None:
            spoken, index = entry
            spoken.heard = max(spoken.heard, index + 1)

    def clear(self) -> None:
        self._pending.clear()

    @property
    def playing(self) -> bool:
        return bool(self._pending)


class TurnTaking:
    """One call's turn-taking state. Feed it STT signals in this order for each media frame:
    on_audio, then on_transcript if there is one, then on_end_of_turn if it fired. Pass the marks
    Twilio sends back to on_mark.
    """

    def __init__(
        self,
        start_reply: Callable[[str | None, SpokenReply, SpokenReply | None], asyncio.Task],
        stop_agent: Callable[[asyncio.Task | None], None],
        resume_reply: Callable[[SpokenReply], asyncio.Task] | None = None,
        barge_in_ms: int | None = None,
    ):
        """start_reply(text, spoken, cut_short) starts a reply to text, or the greeting when text
        is None, and returns its task. The reply adds each sentence to `spoken` as it says it and
        marks its audio with `playback`. The greeting sets `greeting_sent` when it has sent all of
        its audio. cut_short, when set, is the previous reply: the caller cut it off, so Claude's
        history should keep only its heard_text.

        stop_agent(reply) is called on barge-in, after the reply (if still running) has been
        cancelled. It must clear the audio queued at Twilio once the reply has stopped sending.

        resume_reply(spoken) speaks spoken.unheard after an acknowledgement, and returns its task.
        """
        self._start_reply = start_reply
        self._stop_agent = stop_agent
        self._resume_reply = resume_reply
        self._hold_timeout_s = _HOLD_TIMEOUT_SECONDS
        self._barge_in_ms = _BARGE_IN_MS if barge_in_ms is None else barge_in_ms
        self.playback = Playback()
        self.greeting_sent = asyncio.Event()  # The greeting's audio has all been sent
        self._greeting_played = False  # ...and has finished playing
        self.pending = ""  # Caller text not replied to yet
        self.last_spoken: SpokenReply | None = None  # The most recent reply's sentences
        self._reply: asyncio.Task | None = None  # The most recent reply, or a resume of it
        self._hold: asyncio.Task | None = None  # The hold timer, while HOLDING
        self._in_speech = False  # Between an utterance's first speech frame and its transcript
        self._ignore_utterance = False  # It began while the agent talked, and hasn't barged in
        self._cut_by_utterance: SpokenReply | None = None  # The reply this utterance cut off

    @property
    def phase(self) -> Phase:
        if self._hold is not None:
            return Phase.HOLDING
        return Phase.WAITING if self.pending else Phase.LISTENING

    @property
    def _reply_running(self) -> bool:
        return self._reply is not None and not self._reply.done()

    @property
    def agent_talking(self) -> bool:
        """A reply is being written or sent, or the caller is still hearing its audio."""
        return self._reply_running or self.playback.playing

    @property
    def greeting_done(self) -> bool:
        """The greeting has been sent and has finished playing. Barge-in is off until then."""
        if not self._greeting_played and self.greeting_sent.is_set() and not self.playback.playing:
            self._greeting_played = True
        return self._greeting_played

    def greet(self) -> None:
        """Start the greeting. Caller speech can't cut it off."""
        self.last_spoken = SpokenReply()
        self._reply = self._start_reply(None, self.last_spoken, None)

    def on_mark(self, name: str) -> None:
        """Twilio has played the audio sent before this mark."""
        was_playing = self.playback.playing
        self.playback.on_mark(name)
        if was_playing and not self.playback.playing and not self._reply_running:
            logger.info("Playback finished: the caller has heard all the audio sent")

    def on_audio(self, speech_active: bool, speech_ms: int = 0) -> None:
        """Every media frame, once STT has processed it. speech_ms is how much actual speech the
        current utterance has had."""
        if not speech_active:
            self._in_speech = False
            return
        if not self._in_speech:  # The utterance's first speech frame
            self._in_speech = True
            self._ignore_utterance = self.greeting_done and self.agent_talking
            self._cut_by_utterance = None
        if self.greeting_done and self.agent_talking and speech_ms >= self._barge_in_ms:
            self._barge_in(speech_ms)
        # HOLDING -> WAITING: don't flush held text in the middle of the caller's next utterance.
        self._cancel_hold()

    def _barge_in(self, speech_ms: int) -> None:
        reply, self._reply = self._reply, None
        if reply is not None and not reply.done():
            reply.cancel()
        self.playback.clear()
        self._ignore_utterance = False
        if self.last_spoken is not None:
            self.last_spoken.cut_off = True
        self._cut_by_utterance = self.last_spoken
        logger.info(f"Barge-in detected (VAD, {speech_ms}ms of speech)")
        self._stop_agent(reply)

    def on_transcript(self, text: str) -> None:
        """Whisper's text for the latest stretch of speech, VAD_EARLY_MS after it ended."""
        if self._ignore_utterance:
            # Too short to stop the agent, so don't answer it either: the reply would queue
            # behind the audio still playing.
            self._ignore_utterance = False
            logger.info(f"Ignoring {text[-80:]!r}: said while the agent talked, too short to stop it")
            return
        cut, self._cut_by_utterance = self._cut_by_utterance, None
        if cut is not None and cut.unheard and self._resume_reply and _is_acknowledgement(text):
            # The caller was only showing they're listening: carry on from the sentence they
            # were hearing instead of answering it.
            logger.info(
                f"Resuming after {text!r}: from sentence {cut.heard + 1} of {len(cut.sentences)}"
            )
            self._reply = self._resume_reply(cut)
            return
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
        last = self.last_spoken
        cut_short = last if last is not None and last.cut_short else None
        self.last_spoken = SpokenReply()
        self._reply = self._start_reply(text, self.last_spoken, cut_short)

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
