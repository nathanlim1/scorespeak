"""
OpenAI speech-to-text adapter for voice preprocessing.
"""

from __future__ import annotations

import io
import math
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from .types import AudioInput, SpeechTranscript, VoiceWarning


DEFAULT_STT_MODEL = "gpt-4o-mini-transcribe"
SUPPORTED_STT_MODELS = {
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
}


class VoiceInputError(RuntimeError):
    """Raised when a voice backend cannot process the supplied audio."""


def _clean_transcript(text: str) -> str:
    """Normalize STT text without trying to interpret the command."""

    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned


def _read_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from SDK response objects or dict-like test doubles."""

    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _extract_logprobs(value: Any) -> list[float]:
    """Extract token logprobs from a best-effort set of response shapes."""

    logprobs = _read_attr_or_key(value, "logprobs")
    if logprobs is None:
        return []

    if isinstance(logprobs, list):
        raw_items = logprobs
    else:
        raw_items = _read_attr_or_key(logprobs, "content", [])

    extracted: list[float] = []
    for item in raw_items or []:
        if isinstance(item, (int, float)):
            extracted.append(float(item))
            continue
        token_logprob = _read_attr_or_key(item, "logprob")
        if token_logprob is not None:
            extracted.append(float(token_logprob))
    return extracted


def _confidence_from_logprobs(logprobs: list[float]) -> Optional[float]:
    """Convert average token logprob to a rough 0-1 confidence score."""

    if not logprobs:
        return None
    avg = sum(logprobs) / len(logprobs)
    return max(0.0, min(1.0, math.exp(avg)))


@contextmanager
def _audio_file_for_openai(audio: AudioInput) -> Iterator[Any]:
    """Yield a file-like object suitable for OpenAI audio upload."""

    if audio.is_path:
        path = audio.path
        if path is None:
            raise VoiceInputError("Path-backed audio input is missing a path.")
        try:
            with path.open("rb") as handle:
                yield handle
        except OSError as exc:
            raise VoiceInputError(f"Cannot open audio file '{path}': {exc}") from exc
        return

    if not isinstance(audio.data, bytes):
        raise VoiceInputError("In-memory audio input must be bytes.")

    buffer = io.BytesIO(audio.data)
    buffer.name = audio.filename or "audio.wav"
    yield buffer


class OpenAITranscriber:
    """Batch OpenAI speech-to-text backend.

    The OpenAI SDK is imported lazily so the voice package remains importable
    without installing the optional ``voice`` dependencies.
    """

    def __init__(self, client: Any = None, default_model: str = DEFAULT_STT_MODEL):
        self.client = client
        self.default_model = default_model

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise VoiceInputError(
                "OpenAI STT requires the openai package. Install the project "
                "dependencies with 'pip install -e .'."
            ) from exc
        self.client = OpenAI()
        return self.client

    def transcribe(
        self,
        audio: AudioInput | str | Path | bytes,
        *,
        prompt: Optional[str] = None,
        language: Optional[str] = "en",
        model: str = DEFAULT_STT_MODEL,
    ) -> SpeechTranscript:
        """Transcribe speech audio into normalized command text."""

        audio_input = AudioInput.from_value(audio)
        selected_model = model or self.default_model
        if selected_model not in SUPPORTED_STT_MODELS:
            raise VoiceInputError(
                f"Unsupported STT model '{selected_model}'. "
                f"Supported models: {', '.join(sorted(SUPPORTED_STT_MODELS))}."
            )

        kwargs: dict[str, Any] = {
            "model": selected_model,
            "language": language,
        }
        if prompt:
            kwargs["prompt"] = prompt

        try:
            with _audio_file_for_openai(audio_input) as file_obj:
                response = self._client().audio.transcriptions.create(
                    file=file_obj,
                    **kwargs,
                )
        except VoiceInputError:
            raise
        except Exception as exc:
            raise VoiceInputError(f"OpenAI transcription failed: {exc}") from exc

        raw_text = str(_read_attr_or_key(response, "text", "") or "")
        text = _clean_transcript(raw_text)
        logprobs = _extract_logprobs(response)
        confidence = _confidence_from_logprobs(logprobs)

        warnings: list[VoiceWarning] = []
        if not text:
            warnings.append(
                VoiceWarning(
                    code="empty_transcript",
                    message="OpenAI returned an empty speech transcript.",
                )
            )

        return SpeechTranscript(
            text=text,
            raw_text=raw_text,
            model=selected_model,
            language=language,
            confidence=confidence,
            duration_seconds=audio_input.duration_seconds,
            metadata={
                "filename": audio_input.filename
                or (audio_input.path.name if audio_input.path else None),
                "logprob_count": len(logprobs),
            },
            warnings=warnings,
        )
