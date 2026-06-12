"""
Standalone voice-input preprocessing for ScoreSpeak.

This package converts speech audio into structured transcript artifacts. It
does not modify MusicXML and does not depend on ScoreSpeak's agent/tool layer.
"""

from .pipeline import VoiceInputProcessor
from .stt import DEFAULT_STT_MODEL, OpenAITranscriber, VoiceInputError
from .types import (
    AudioInput,
    SpeechTranscript,
    VoiceProcessingResult,
    VoiceRequest,
    VoiceWarning,
)

__all__ = [
    "AudioInput",
    "DEFAULT_STT_MODEL",
    "OpenAITranscriber",
    "SpeechTranscript",
    "VoiceInputError",
    "VoiceInputProcessor",
    "VoiceProcessingResult",
    "VoiceRequest",
    "VoiceWarning",
]
