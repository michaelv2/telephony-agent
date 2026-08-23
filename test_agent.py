"""Tests for CallAgent — verifies async streaming and sentence splitting."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("TWILIO_ACCOUNT_SID", "test")
os.environ.setdefault("TWILIO_AUTH_TOKEN", "test")
os.environ.setdefault("TWILIO_API_KEY", "test")
os.environ.setdefault("TWILIO_API_SECRET", "test")
os.environ.setdefault("TWILIO_PHONE", "+10000000000")


class MockTextStream:
    """Async iterator that yields token strings one at a time."""

    def __init__(self, tokens: list[str]):
        self._tokens = tokens
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._tokens):
            raise StopAsyncIteration
        token = self._tokens[self._index]
        self._index += 1
        return token


class MockStreamContext:
    """Mocks the async context manager returned by client.messages.stream()."""

    def __init__(self, tokens: list[str]):
        self.text_stream = MockTextStream(tokens)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def get_final_message(self):
        return MagicMock()


@pytest.mark.asyncio
async def test_respond_stream_yields_sentences():
    """Verify sentence splitting: yields on .!? boundaries."""
    tokens = [
        "Hello", " there", "!", " How", " can", " I", " help", " you", " today", "?"
    ]
    mock_stream = MockStreamContext(tokens)

    with patch("agent.client") as mock_client:
        mock_client.messages.stream.return_value = mock_stream

        from agent import CallAgent
        agent = CallAgent()
        sentences = []
        async for sentence in agent.respond_stream("Hi"):
            sentences.append(sentence)

    assert sentences == ["Hello there!", "How can I help you today?"]


@pytest.mark.asyncio
async def test_respond_stream_single_sentence():
    """A response with no mid-sentence boundary yields once."""
    tokens = ["Sure", " thing"]
    mock_stream = MockStreamContext(tokens)

    with patch("agent.client") as mock_client:
        mock_client.messages.stream.return_value = mock_stream

        from agent import CallAgent
        agent = CallAgent()
        sentences = []
        async for sentence in agent.respond_stream("Thanks"):
            sentences.append(sentence)

    assert sentences == ["Sure thing"]


@pytest.mark.asyncio
async def test_respond_stream_multiple_sentences():
    """Three sentences separated by different punctuation."""
    tokens = [
        "That", " sounds", " great", ".", " I", "'ll", " check", " on",
        " that", "!", " Any", "thing", " else", "?"
    ]
    mock_stream = MockStreamContext(tokens)

    with patch("agent.client") as mock_client:
        mock_client.messages.stream.return_value = mock_stream

        from agent import CallAgent
        agent = CallAgent()
        sentences = []
        async for sentence in agent.respond_stream("Check my order"):
            sentences.append(sentence)

    assert sentences == ["That sounds great.", "I'll check on that!", "Anything else?"]


@pytest.mark.asyncio
async def test_respond_stream_maintains_history():
    """Verify conversation history is updated after streaming."""
    tokens = ["Hi", "!"]
    mock_stream = MockStreamContext(tokens)

    with patch("agent.client") as mock_client:
        mock_client.messages.stream.return_value = mock_stream

        from agent import CallAgent
        agent = CallAgent()
        async for _ in agent.respond_stream("Hello"):
            pass

    assert len(agent.messages) == 2
    assert agent.messages[0] == {"role": "user", "content": "Hello"}
    assert agent.messages[1] == {"role": "assistant", "content": "Hi!"}


@pytest.mark.asyncio
async def test_greeting_stream_inbound():
    """Inbound call (no context) greeting."""
    tokens = ["Hey", " there", "!"]
    mock_stream = MockStreamContext(tokens)

    with patch("agent.client") as mock_client:
        mock_client.messages.stream.return_value = mock_stream

        from agent import CallAgent
        agent = CallAgent(context=None)
        sentences = []
        async for sentence in agent.greeting_stream():
            sentences.append(sentence)

    assert sentences == ["Hey there!"]
    assert len(agent.messages) == 2


@pytest.mark.asyncio
async def test_greeting_stream_outbound():
    """Outbound call (with context) greeting includes AI disclosure prompt."""
    tokens = ["Hi", ",", " I", "'m", " Sarah", "."]
    mock_stream = MockStreamContext(tokens)

    with patch("agent.client") as mock_client:
        mock_client.messages.stream.return_value = mock_stream

        from agent import CallAgent
        agent = CallAgent(context="Schedule a dental cleaning")
        sentences = []
        async for sentence in agent.greeting_stream():
            sentences.append(sentence)

    assert agent._has_context is True
    assert len(agent.messages) == 2
    assert "AI" in agent.messages[0]["content"] or "disclose" in agent.messages[0]["content"]


@pytest.mark.asyncio
async def test_client_is_async():
    """Verify the module-level client is AsyncAnthropic, not sync."""
    from agent import client
    import anthropic
    assert isinstance(client, anthropic.AsyncAnthropic), (
        f"Expected AsyncAnthropic, got {type(client).__name__}. "
        "The sync client blocks the event loop during streaming."
    )
