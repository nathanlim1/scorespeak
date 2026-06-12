"""
Structured types for voice input preprocessing.

The voice package intentionally emits score-agnostic speech artifacts. These
objects can later be consumed by an agent or score-editing layer, but they do
not assume measures, beats, parts, keys, or MusicXML state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union


@dataclass
class VoiceWarning:
    """Non-fatal warning produced during voice preprocessing."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioInput:
    """Audio payload supplied to a voice backend.

    ``data`` may be a filesystem path or in-memory bytes. Byte inputs are
    supported for STT uploads and receive a best-effort filename.
    """

    data: Union[str, Path, bytes]
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    duration_seconds: Optional[float] = None

    @classmethod
    def from_value(cls, value: Union["AudioInput", str, Path, bytes]) -> "AudioInput":
        """Normalize a caller-supplied audio value into ``AudioInput``."""

        if isinstance(value, AudioInput):
            return value
        if isinstance(value, (str, Path)):
            path = Path(value)
            return cls(data=path, filename=path.name)
        if isinstance(value, bytes):
            return cls(data=value, filename="audio.wav")
        raise TypeError(
            f"Cannot interpret {type(value).__name__} as audio input. "
            "Expected AudioInput, path string, pathlib.Path, or bytes."
        )

    @property
    def is_path(self) -> bool:
        """Return true when this input points to a local file."""

        return isinstance(self.data, (str, Path))

    @property
    def path(self) -> Optional[Path]:
        """Return the filesystem path for path-backed input, if any."""

        if not self.is_path:
            return None
        return Path(self.data)


@dataclass
class SpeechTranscript:
    """Speech-to-text output from the ASR backend."""

    text: str
    raw_text: str
    model: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration_seconds: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[VoiceWarning] = field(default_factory=list)


@dataclass
class VoiceRequest:
    """Speech transcription request envelope for later agent integration."""

    audio: AudioInput
    speech_prompt: Optional[str] = None
    language: Optional[str] = "en"


@dataclass
class VoiceProcessingResult:
    """Top-level result from the voice preprocessing pipeline."""

    success: bool
    request: VoiceRequest
    speech: Optional[SpeechTranscript] = None
    warnings: list[VoiceWarning] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
