"""Abstract STT interface — swap implementations without changing the pipeline."""

from abc import ABC, abstractmethod


class BaseSTT(ABC):
    @abstractmethod
    async def feed_audio(self, pcm_16k: bytes) -> None:
        """Feed a chunk of PCM 16-bit 16kHz audio. Non-blocking."""

    @abstractmethod
    async def get_transcript(self) -> str | None:
        """Return transcript if speech just ended, else None."""

    @abstractmethod
    def reset(self) -> None:
        """Clear buffers for a new utterance."""

    @property
    @abstractmethod
    def speech_active(self) -> bool:
        """True if VAD currently detects speech."""
