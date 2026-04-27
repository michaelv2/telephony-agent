"""Claude conversation manager for phone calls — streaming."""

import logging
import re
from collections.abc import AsyncGenerator

import anthropic

from config import AGENT_PROMPT, ANTHROPIC_API_KEY

logger = logging.getLogger("agent")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = AGENT_PROMPT

# Split on sentence boundaries, keeping the delimiter attached
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class CallAgent:
    """Manages a single phone call conversation with Claude."""

    def __init__(self, context: str | None = None):
        self.messages: list[dict] = []
        self.system = SYSTEM_PROMPT
        if context:
            self.system += f"\n\nAdditional context for this call: {context}"

    async def respond_stream(self, user_text: str) -> AsyncGenerator[str, None]:
        """Stream Claude's response, yielding complete sentences as they form."""
        self.messages.append({"role": "user", "content": user_text})
        logger.info(f"User: {user_text}")

        buffer = ""
        full_response = ""

        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=self.system,
            messages=self.messages,
        ) as stream:
            for text in stream.text_stream:
                buffer += text
                full_response += text

                # Yield complete sentences as they accumulate
                parts = _SENTENCE_RE.split(buffer)
                if len(parts) > 1:
                    # All but the last part are complete sentences
                    for sentence in parts[:-1]:
                        sentence = sentence.strip()
                        if sentence:
                            yield sentence
                    buffer = parts[-1]

        # Yield whatever remains
        if buffer.strip():
            yield buffer.strip()

        self.messages.append({"role": "assistant", "content": full_response})
        logger.info(f"Agent: {full_response}")

    async def greeting_stream(self) -> AsyncGenerator[str, None]:
        """Stream an initial greeting."""
        async for sentence in self.respond_stream(
            "[The call just connected. Greet the caller briefly.]"
        ):
            yield sentence
