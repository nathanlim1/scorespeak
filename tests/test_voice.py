"""Tests for standalone speech-only voice input preprocessing."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from scorespeak.voice import (
    AudioInput,
    OpenAITranscriber,
    SpeechTranscript,
    VoiceInputProcessor,
    VoiceProcessingResult,
    VoiceRequest,
    VoiceWarning,
)
from scorespeak.voice import cli


def test_voice_types_are_serialization_friendly() -> None:
    """Voice dataclasses produce plain nested structures via asdict."""

    audio = AudioInput(data=b"abc", filename="clip.wav", duration_seconds=1.25)
    speech = SpeechTranscript(
        text="Add a C4 quarter note",
        raw_text=" Add a C4 quarter note ",
        model="gpt-4o-mini-transcribe",
        language="en",
        confidence=0.9,
    )
    request = VoiceRequest(
        audio=audio,
        speech_prompt="Music editing commands",
        language="en",
    )
    result = VoiceProcessingResult(
        success=True,
        request=request,
        speech=speech,
    )

    payload = asdict(result)

    assert audio.filename == "clip.wav"
    assert payload["speech"]["text"] == "Add a C4 quarter note"
    assert payload["speech"]["confidence"] == 0.9
    assert payload["request"]["speech_prompt"] == "Music editing commands"


class _FakeTranscriptions:
    """Fake OpenAI transcriptions resource for adapter tests."""

    def __init__(self, response: Any) -> None:
        """Initialize with a deterministic SDK-like response."""
        self.response = response
        self.calls = []

    def create(self, **kwargs: Any) -> Any:
        """Record request kwargs and return the configured response."""
        self.calls.append(kwargs)
        return self.response


class _FakeAudio:
    """Fake OpenAI audio resource."""

    def __init__(self, response: Any) -> None:
        """Initialize nested fake transcriptions API."""
        self.transcriptions = _FakeTranscriptions(response)


class _FakeOpenAIClient:
    """Fake OpenAI client exposing audio transcription calls."""

    def __init__(self, response: Any) -> None:
        """Initialize the fake client with one response."""
        self.audio = _FakeAudio(response)


def test_openai_transcriber_passes_model_prompt_language_and_cleans_text(
    tmp_path: Path,
) -> None:
    """OpenAI adapter keeps STT details structured and normalizes whitespace."""

    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"fake wav")
    response = {
        "text": "  Add   a C4 quarter note.  ",
        "logprobs": [{"logprob": -0.1}, {"logprob": -0.2}],
    }
    client = _FakeOpenAIClient(response)

    transcript = OpenAITranscriber(client=client).transcribe(
        audio_path,
        prompt="Music editing commands",
        language="en",
        model="gpt-4o-transcribe",
    )

    call = client.audio.transcriptions.calls[0]
    assert call["model"] == "gpt-4o-transcribe"
    assert call["prompt"] == "Music editing commands"
    assert call["language"] == "en"
    assert call["file"].name.endswith("speech.wav")
    assert transcript.text == "Add a C4 quarter note."
    assert transcript.raw_text == "  Add   a C4 quarter note.  "
    assert transcript.confidence is not None
    assert transcript.confidence > 0.0


def test_openai_transcriber_empty_transcript_adds_warning(tmp_path: Path) -> None:
    """Empty STT output is reported as a warning instead of hidden."""

    audio_path = tmp_path / "empty.wav"
    audio_path.write_bytes(b"fake wav")
    client = _FakeOpenAIClient({"text": "   "})

    transcript = OpenAITranscriber(client=client).transcribe(audio_path)

    assert transcript.text == ""
    assert transcript.warnings[0].code == "empty_transcript"


class _FakeSpeechBackend:
    """Fake speech backend for processor tests."""

    def __init__(
        self,
        transcript: SpeechTranscript | None = None,
        exc: Exception | None = None,
    ) -> None:
        """Initialize with either a transcript or an exception."""
        self.transcript = transcript
        self.exc = exc
        self.calls = []

    def transcribe(self, *args: Any, **kwargs: Any) -> SpeechTranscript:
        """Record the transcription call and return or raise."""
        self.calls.append((args, kwargs))
        if self.exc is not None:
            raise self.exc
        if self.transcript is None:
            raise AssertionError("Fake speech backend requires a transcript")
        return self.transcript


def _speech(text: str) -> SpeechTranscript:
    """Build a normalized fake speech transcript."""
    return SpeechTranscript(
        text=text,
        raw_text=text,
        model="gpt-4o-mini-transcribe",
        language="en",
    )


def test_pipeline_transcribes_speech() -> None:
    """The processor returns successful speech results."""

    backend = _FakeSpeechBackend(_speech("Set tempo to 120"))
    processor = VoiceInputProcessor(speech_transcriber=backend)

    result = processor.process(b"audio", speech_prompt="commands")

    assert result.success is True
    assert result.speech is not None
    assert result.speech.text == "Set tempo to 120"
    assert result.error is None


def test_pipeline_empty_transcript_fails_with_warning() -> None:
    """The processor treats an empty transcript as unsuccessful."""

    transcript = _speech("")
    transcript.warnings.append(
        VoiceWarning(
            code="empty_transcript",
            message="OpenAI returned an empty speech transcript.",
        )
    )
    processor = VoiceInputProcessor(speech_transcriber=_FakeSpeechBackend(transcript))

    result = processor.process("speech.wav")

    assert result.success is False
    assert result.error == "No usable speech transcript was detected."
    assert result.warnings[0].code == "empty_transcript"


def test_pipeline_forwards_prompt_and_language() -> None:
    """The processor forwards prompt and language to the speech backend."""

    backend = _FakeSpeechBackend(_speech("Add a note"))
    processor = VoiceInputProcessor(speech_transcriber=backend)

    processor.process(
        "speech.wav",
        speech_prompt="Music editing commands",
        language="fr",
    )

    args, kwargs = backend.calls[0]
    assert AudioInput.from_value(args[0]).filename == "speech.wav"
    assert kwargs["prompt"] == "Music editing commands"
    assert kwargs["language"] == "fr"


def test_pipeline_backend_error_is_wrapped() -> None:
    """Backend failures become structured unsuccessful results."""

    processor = VoiceInputProcessor(
        speech_transcriber=_FakeSpeechBackend(exc=RuntimeError("network down")),
    )

    result = processor.process("speech.wav")

    assert result.success is False
    assert result.error == "network down"
    assert result.warnings[0].code == "voice_backend_error"
    assert result.warnings[0].details["error_type"] == "RuntimeError"


def test_cli_outputs_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI parses a file path and emits structured JSON."""

    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"fake wav")

    class FakeProcessor:
        """Fake processor for CLI testing."""

        def process(
            self,
            audio: AudioInput | str | Path | bytes,
            *,
            speech_prompt: str | None = None,
            language: str | None = "en",
        ) -> VoiceProcessingResult:
            """Return deterministic CLI output."""
            assert audio == audio_path
            assert speech_prompt == "commands"
            assert language == "en"
            request_audio = AudioInput.from_value(audio)
            return VoiceProcessingResult(
                success=True,
                request=VoiceRequest(audio=request_audio),
                speech=_speech("Add a note"),
            )

    monkeypatch.setattr(cli, "VoiceInputProcessor", FakeProcessor)

    exit_code = cli.main([str(audio_path), "--prompt", "commands"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["speech"]["text"] == "Add a note"
    assert "detected_mode" not in output
