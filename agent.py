"""Claude conversation manager for phone calls — streaming."""

import logging
import re
from collections.abc import AsyncGenerator

import anthropic

from config import AGENT_PROMPT, ANTHROPIC_API_KEY

logger = logging.getLogger("agent")

client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = AGENT_PROMPT

# Split on sentence boundaries, keeping the delimiter attached
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# A sentence longer than this is spoken in clauses, each at least _MIN_CLAUSE_CHARS long. Each
# piece is sent to TTS and marked on its own, so a long list starts playing sooner, and a resume
# after "mm-hmm" repeats one clause rather than the whole sentence. CHANGELOG-latency.md, entry 11.
_LONG_CHARS = 100
_MIN_CLAUSE_CHARS = 40
_CLAUSE_END_RE = re.compile(r"(?:[,;:]| [—–])\s+")


def _clause_cut(text: str) -> int | None:
    """Where to cut a long sentence (finished or still streaming): after its last clause boundary
    within the first _LONG_CHARS, or else its first boundary after that. None if it's short, has no
    boundary yet, or the only boundaries are inside a [TAG: ...]."""
    if len(text) <= _LONG_CHARS:
        return None
    cuts = [
        m.end() for m in _CLAUSE_END_RE.finditer(text)
        if m.start() + 1 >= _MIN_CLAUSE_CHARS and text.count("[", 0, m.start()) == text.count("]", 0, m.start())
    ]
    if not cuts:
        return None
    within = [c for c in cuts if c <= _LONG_CHARS]
    return within[-1] if within else cuts[0]


def _clauses(sentence: str) -> list[str]:
    """A finished sentence, split into clauses if it's long."""
    pieces = []
    while (cut := _clause_cut(sentence)) is not None:
        pieces.append(sentence[:cut].strip())
        sentence = sentence[cut:]
    pieces.append(sentence.strip())
    return [p for p in pieces if p]


class CallAgent:
    """Manages a single phone call conversation with Claude."""

    def __init__(self, context: str | None = None):
        self.messages: list[dict] = []
        self.system = SYSTEM_PROMPT
        self._has_context = bool(context)
        if context:
            self.system += f"\n\nAdditional context for this call: {context}"

    async def respond_stream(self, user_text: str) -> AsyncGenerator[str, None]:
        """Stream Claude's response, yielding complete sentences as they form."""
        # A retried reply (cut off before it said anything) finds its question still unanswered
        if self.messages[-1:] != [{"role": "user", "content": user_text}]:
            self.messages.append({"role": "user", "content": user_text})
        logger.info(f"User: {user_text}")

        buffer = ""
        full_response = ""

        async with client.messages.stream(
            model="claude-sonnet-5",
            max_tokens=500,
            system=self.system,
            messages=self.messages,
        ) as stream:
            async for text in stream.text_stream:
                buffer += text
                full_response += text

                # Yield complete sentences as they accumulate, long ones in clauses
                parts = _SENTENCE_RE.split(buffer)
                buffer = parts.pop()  # The sentence still being written
                for sentence in parts:
                    for piece in _clauses(sentence):
                        yield piece
                # Don't wait for the end of a long sentence: yield its clauses as they arrive
                while (cut := _clause_cut(buffer)) is not None:
                    yield buffer[:cut].strip()
                    buffer = buffer[cut:]

        # Yield whatever remains
        for piece in _clauses(buffer):
            yield piece

        self.messages.append({"role": "assistant", "content": full_response})
        logger.info(f"Agent: {full_response}")

    def revise_last_reply(self, heard: str) -> None:
        """The caller cut the last reply off: keep only the part they heard in the history.

        If the reply was still being written when it was cut, its assistant message was never
        added, so `heard` is added instead (when there is any). Consecutive user messages are fine:
        the API combines them.
        """
        if self.messages and self.messages[-1]["role"] == "assistant":
            if heard:
                self.messages[-1]["content"] = heard
            else:
                self.messages.pop()
        elif heard:
            self.messages.append({"role": "assistant", "content": heard})
        logger.info(f"Agent (as heard): {heard or '(nothing)'}")

    async def greeting_stream(self) -> AsyncGenerator[str, None]:
        """Stream an initial greeting."""
        if self._has_context:
            prompt = (
                "[The call just connected. Introduce yourself, disclose that you're an AI "
                "assistant calling on behalf of your user, briefly state why you're calling, "
                "and check if they're okay to proceed with you.]"
            )
        else:
            prompt = "[The call just connected. Greet the caller briefly.]"
        async for sentence in self.respond_stream(prompt):
            yield sentence
