"""
Speech-only voice preprocessing pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .stt import OpenAITranscriber
from .types import (
    AudioInput,
    VoiceProcessingResult,
    VoiceRequest,
    VoiceWarning,
)


class VoiceInputProcessor:
    """Coordinate speech transcription for voice input."""

    def __init__(self, speech_transcriber: Optional[OpenAITranscriber] = None) -> None:
        """Initialize the processor with an optional STT backend."""
        self.speech_transcriber = speech_transcriber or OpenAITranscriber()

    def process(
        self,
        audio: AudioInput | str | Path | bytes,
        *,
        speech_prompt: Optional[str] = None,
        language: Optional[str] = "en",
    ) -> VoiceProcessingResult:
        """Process audio into a spoken transcript."""

        audio_input = AudioInput.from_value(audio)
        request = VoiceRequest(
            audio=audio_input,
            speech_prompt=speech_prompt,
            language=language,
        )

        try:
            speech = self.speech_transcriber.transcribe(
                request.audio,
                prompt=request.speech_prompt,
                language=request.language,
            )
        except Exception as exc:
            return self._error_result(request, exc)

        warnings = list(speech.warnings)
        success = bool(speech.text.strip())
        error = None if success else "No usable speech transcript was detected."

        return VoiceProcessingResult(
            success=success,
            request=request,
            speech=speech,
            warnings=warnings,
            error=error,
        )

    def _error_result(
        self,
        request: VoiceRequest,
        exc: Exception,
    ) -> VoiceProcessingResult:
        """Return a structured failure result for backend errors."""

        warning = VoiceWarning(
            code="voice_backend_error",
            message=str(exc),
            details={"error_type": type(exc).__name__},
        )
        return VoiceProcessingResult(
            success=False,
            request=request,
            warnings=[warning],
            error=str(exc),
        )
