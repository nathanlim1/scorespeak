"""
Flask backend for the MusicXML web renderer with ScoreSpeak agent integration.

Provides API endpoints for:
- Loading/creating scores
- Chat interface to the ScoreSpeak agent
- Retrieving the current score as MusicXML (JSON or file download)
"""

from __future__ import annotations

import logging
import json
import os
import sys
import tempfile
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from flask_cors import CORS
from werkzeug.datastructures import FileStorage

from scorespeak import ScoreSpeak
from scorespeak.agent import (
    AgentMemoryStore,
    DEFAULT_PROMPT_SPLIT_CONFIG,
    PromptSplitConfig,
    run_prompt,
    run_prompt_stream,
    should_use_prompt_split,
)
from scorespeak.agent.defaults import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_AGENT_REASONING_EFFORT,
    DEFAULT_RECURSION_LIMIT,
    DEFAULT_RETRIEVAL_THRESHOLD,
    agent_model_options_payload,
    agent_reasoning_effort_options_payload,
    chat_openai_reasoning_kwargs,
    normalize_agent_reasoning_effort,
    normalize_agent_model,
)
from scorespeak.retrieval import LexicalContextRetriever
from scorespeak.types import OperationResult
from scorespeak.voice import (
    AudioInput,
    VoiceInputError,
    VoiceInputProcessor,
    VoiceProcessingResult,
    VoiceWarning,
)
from web.musicxml_window import (
    export_scorespeak_window_musicxml,
    extract_musicxml_window,
    normalize_measure_window,
    show_rests_for_empty_space as _show_rests_for_empty_space,
)

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../web', static_url_path='')
CORS(app)

WEB_ROOT = Path(__file__).parent
_DEFAULT_MODEL = DEFAULT_AGENT_MODEL
_DEFAULT_THRESHOLD = DEFAULT_RETRIEVAL_THRESHOLD
_DEFAULT_PROMPT_SPLIT_ENABLED = True
_DEFAULT_PROMPT_SPLIT_MIN_SENTENCES = (
    DEFAULT_PROMPT_SPLIT_CONFIG.sentence_threshold + 1
)
_SUPPORTED_SCORE_SUFFIXES = {".musicxml", ".xml", ".mxl"}
_SUPPORTED_AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".wav",
    ".webm",
}
_AUDIO_SUFFIX_BY_MIME_TYPE = {
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "audio/x-m4a": ".m4a",
}
_VOICE_SPEECH_PROMPT = (
    "The user is controlling a MusicXML score editor. Transcribe music editing "
    "commands, measure numbers, beats, part names, note names, intervals, "
    "durations, dynamics, articulations, and spoken correction context as "
    "literally as possible."
)
_TARGET_MEASURE_PARTS_PER_RENDER = 200
_LARGE_SCORE_COMPLEXITY_LIMIT = _TARGET_MEASURE_PARTS_PER_RENDER
_MIN_MEASURE_WINDOW = 8
_MAX_MUSICXML_WINDOW_CACHE = 8


_GLOBAL_EDIT_TOOL_NAMES = {
    "add_part",
    "remove_part",
    "transpose",
    "transpose_to_concert_pitch",
    "transpose_to_written_pitch",
    "set_title",
    "set_subtitle",
    "set_composer",
}


def _uploaded_score_suffix(filename: str | None) -> str:
    """Return the temporary-file suffix to use for an uploaded score."""
    suffix = Path(filename or "").suffix.lower()
    if suffix in _SUPPORTED_SCORE_SUFFIXES:
        return suffix
    return ".musicxml"


def _uploaded_audio_suffix(
    filename: str | None,
    mime_type: str | None = None,
) -> str:
    """Return the temporary-file suffix to use for an uploaded audio clip."""
    suffix = Path(filename or "").suffix.lower()
    if suffix in _SUPPORTED_AUDIO_SUFFIXES:
        return suffix
    if mime_type:
        base_mime_type = mime_type.split(";", 1)[0].strip().lower()
        inferred_suffix = _AUDIO_SUFFIX_BY_MIME_TYPE.get(base_mime_type)
        if inferred_suffix:
            return inferred_suffix
    return ".webm"


def _load_score_from_upload(file: FileStorage) -> ScoreSpeak:
    """Load a ScoreSpeak from an uploaded MusicXML or compressed MXL file."""
    score_state, _ = _load_score_and_display_xml_from_upload(file)
    return score_state


def _decode_xml_bytes(xml_bytes: bytes) -> str:
    """Decode MusicXML bytes for browser rendering."""
    return xml_bytes.decode("utf-8-sig")


def _mxl_rootfile_path(archive: zipfile.ZipFile) -> str:
    """Return the main MusicXML path from a compressed MXL archive."""
    try:
        container_xml = archive.read("META-INF/container.xml")
        root = etree.fromstring(container_xml)
        rootfile_paths = root.xpath(".//*[local-name()='rootfile']/@full-path")
        if rootfile_paths:
            return str(rootfile_paths[0])
    except (KeyError, etree.XMLSyntaxError):
        logger.warning("Could not read MXL container metadata; using XML fallback")

    for name in archive.namelist():
        lower_name = name.lower()
        if lower_name.endswith((".musicxml", ".xml")) and not lower_name.startswith(
            "meta-inf/"
        ):
            return name

    raise ValueError("Compressed MXL archive does not contain a MusicXML score.")


def _read_display_musicxml(path: Path) -> str:
    """Read the MusicXML text to return to the browser for initial rendering."""
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path) as archive:
            rootfile_path = _mxl_rootfile_path(archive)
            return _decode_xml_bytes(archive.read(rootfile_path))

    return path.read_text(encoding="utf-8-sig")


def _load_score_and_display_xml_from_upload(
    file: FileStorage,
) -> tuple[ScoreSpeak, str]:
    """Load a ScoreSpeak and browser-renderable MusicXML from an uploaded score."""
    temp_path: Path | None = None
    suffix = _uploaded_score_suffix(file.filename)

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            file.save(temp_file)

        score_state = ScoreSpeak.from_musicxml(temp_path)
        display_musicxml = _read_display_musicxml(temp_path)
        return score_state, display_musicxml
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _save_voice_audio_upload(file: FileStorage) -> Path:
    """Save an uploaded audio clip to a temporary path and return it."""
    suffix = _uploaded_audio_suffix(file.filename, file.mimetype)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        file.save(temp_file)
    return temp_path


def _optional_float_form(name: str) -> float | None:
    """Read an optional float form field, returning None when absent/blank."""
    raw_value = request.form.get(name)
    if raw_value is None or not raw_value.strip():
        return None
    return float(raw_value)


def _voice_render_range_from_form() -> dict[str, int] | None:
    """Return the current browser render range from voice form fields."""
    raw_start = request.form.get("render_start")
    raw_end = request.form.get("render_end")
    if not raw_start or not raw_end:
        return None

    start = int(raw_start)
    end = int(raw_end)
    start, end = normalize_measure_window(start, end)
    return {"start": start, "end": end}


def _warning_payload(warnings: list[VoiceWarning]) -> list[dict[str, Any]]:
    """Serialize voice warnings for JSON responses."""
    return [
        {
            "code": warning.code,
            "message": warning.message,
            "details": warning.details,
        }
        for warning in warnings
    ]


def _speech_text(result: VoiceProcessingResult) -> str:
    """Return the normalized speech text from a voice result."""
    if result.speech is None:
        return ""
    return result.speech.text.strip()


def _voice_agent_message(
    result: VoiceProcessingResult,
    render_range: dict[str, int] | None,
) -> str:
    """Build the natural-language request sent from voice mode to the agent."""
    speech_text = _speech_text(result)
    range_text = "(unknown)"
    if render_range is not None:
        range_text = f"{render_range['start']}-{render_range['end']}"

    lines = [
        "Voice input was captured from the web app.",
        f"Current rendered measure range: {range_text}",
        "",
        "Interpretation rules:",
        "- Use the spoken transcript as the user's editing intent.",
        (
            "- If the target measure, part, or starting pitch is missing and "
            "required, first use available score context, memory, search, and "
            "inspection to resolve it. Ask one short clarification before "
            "editing only if the target remains genuinely ambiguous."
        ),
    ]

    if speech_text:
        lines.extend(["", f'Spoken transcript: "{speech_text}"'])

    if speech_text:
        lines.extend(["", f"User request: {speech_text}"])
    else:
        lines.extend(["", "User request: Interpret the voice input if possible."])

    return "\n".join(lines)


def _voice_response_payload(
    result: VoiceProcessingResult,
    render_range: dict[str, int] | None,
) -> dict[str, Any]:
    """Return the normalized API payload for a processed voice request."""
    return {
        "success": result.success,
        "speech_text": _speech_text(result),
        "agent_message": _voice_agent_message(result, render_range),
        "warnings": _warning_payload(result.warnings),
        "error": result.error,
    }


def _score_metadata(score_state: ScoreSpeak) -> dict:
    """Return size metadata used by the web renderer."""
    return {
        "part_count": score_state.part_count,
        "measure_count": score_state.measure_count,
    }


def _measure_window_size(part_count: int) -> int:
    """Return the number of measures to render for a part count."""
    dynamic_window = _TARGET_MEASURE_PARTS_PER_RENDER // max(1, part_count)
    return max(_MIN_MEASURE_WINDOW, dynamic_window)


def _default_render_range(score_state: ScoreSpeak, start: int = 1) -> dict[str, int]:
    """Return the default render range for the current score size."""
    measure_count = max(0, score_state.measure_count)
    if measure_count == 0:
        return {"start": 1, "end": 1}

    safe_start = max(1, min(int(start), measure_count))
    if not _uses_measure_paging(score_state):
        return {"start": 1, "end": measure_count}

    window_size = _measure_window_size(score_state.part_count)
    safe_end = min(measure_count, safe_start + window_size - 1)
    return {"start": safe_start, "end": safe_end}


def _render_range_from_payload(
    data: dict[str, Any] | None,
    score_state: ScoreSpeak,
) -> dict[str, int] | None:
    """Return the browser's current render range from a JSON request payload."""
    raw_range = data.get("render_range") if isinstance(data, dict) else None
    if not isinstance(raw_range, dict):
        return None

    try:
        start = int(raw_range["start"])
        end = int(raw_range["end"])
    except (KeyError, TypeError, ValueError):
        return None

    start, end = normalize_measure_window(start, end)
    measure_count = score_state.measure_count
    if measure_count > 0:
        start = min(start, measure_count)
        end = min(end, measure_count)

    return {"start": start, "end": end}


def _current_render_range(
    data: dict[str, Any] | None,
    score_state: ScoreSpeak,
) -> dict[str, int]:
    """Return the client render range, falling back to the default first window."""
    render_range = _render_range_from_payload(data, score_state)
    if render_range is not None:
        return render_range
    return _default_render_range(score_state)


def _uses_measure_paging(score_state: ScoreSpeak) -> bool:
    """Return whether the score is large enough to page by measure window."""
    window_size = _measure_window_size(score_state.part_count)
    complexity = score_state.measure_count * max(1, score_state.part_count)
    return (
        score_state.measure_count > window_size
        and complexity > _LARGE_SCORE_COMPLEXITY_LIMIT
    )


def _initial_score_payload(score_state: ScoreSpeak, musicxml: str) -> dict:
    """Build an initial render payload with a windowed MusicXML document."""
    render_range = _default_render_range(score_state)
    windowed_musicxml = extract_musicxml_window(
        musicxml,
        render_range["start"],
        render_range["end"],
    )
    return {
        "musicxml": windowed_musicxml,
        "render_range": render_range,
        **_score_metadata(score_state),
    }


def _extract_changed_measure_range(
    tool_results: list[tuple[str, dict, OperationResult]],
    current_range: dict[str, int],
) -> dict[str, int] | None:
    """Infer the smallest changed measure range from completed tool results."""
    measures: list[int] = []
    has_global_edit = False

    for tool_name, kwargs, result in tool_results:
        if tool_name in _GLOBAL_EDIT_TOOL_NAMES:
            has_global_edit = True
        measures.extend(_collect_measure_numbers(kwargs))
        measures.extend(_collect_measure_numbers(result.details))

    if measures:
        return {"start": min(measures), "end": max(measures)}
    if has_global_edit:
        return current_range
    return None


def _collect_measure_numbers(value: object) -> list[int]:
    """Return measure-like integers recursively found in operation metadata."""
    measures: list[int] = []

    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if _is_measure_key(key_text):
                measures.extend(_measure_values(item))
            else:
                measures.extend(_collect_measure_numbers(item))
        return measures

    if isinstance(value, (list, tuple)):
        for item in value:
            measures.extend(_collect_measure_numbers(item))

    return measures


def _is_measure_key(key: str) -> bool:
    """Return whether a metadata key describes a measure number or range."""
    return (
        key == "measure"
        or key == "measure_number"
        or key == "measures_renumbered_from"
        or key.startswith("start_measure")
        or key.startswith("end_measure")
        or key.endswith("_measure")
        or key.endswith("_measure_number")
    )


def _measure_values(value: object) -> list[int]:
    """Normalize a value from a measure-like key into positive integers."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value] if value > 0 else []
    if isinstance(value, float) and value.is_integer():
        measure = int(value)
        return [measure] if measure > 0 else []
    if isinstance(value, str) and value.strip().isdigit():
        measure = int(value.strip())
        return [measure] if measure > 0 else []
    if isinstance(value, (list, tuple)):
        measures: list[int] = []
        for item in value:
            measures.extend(_measure_values(item))
        return measures
    if isinstance(value, dict):
        return _collect_measure_numbers(value)
    return []


def _normalize_prompt_split_enabled(
    value: object | None,
    default: bool = _DEFAULT_PROMPT_SPLIT_ENABLED,
) -> bool:
    """Return a normalized prompt split enabled flag."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError("prompt_split_enabled must be a boolean.")


def _normalize_prompt_split_min_sentences(value: object | None) -> int:
    """Return a positive minimum sentence count for prompt split activation."""
    if value is None:
        return _DEFAULT_PROMPT_SPLIT_MIN_SENTENCES
    if isinstance(value, bool):
        raise ValueError("prompt_split_min_sentences must be a positive integer.")
    try:
        sentence_count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "prompt_split_min_sentences must be a positive integer."
        ) from exc
    if sentence_count < 1:
        raise ValueError("prompt_split_min_sentences must be at least 1.")
    return sentence_count


class AgentSession:
    """Maintains state for a single editing session."""

    def __init__(
        self,
        score_state: ScoreSpeak,
        display_musicxml: str | None = None,
        model: str = _DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_AGENT_REASONING_EFFORT,
        prompt_split_enabled: bool = _DEFAULT_PROMPT_SPLIT_ENABLED,
        prompt_split_min_sentences: int = _DEFAULT_PROMPT_SPLIT_MIN_SENTENCES,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self.score_state = score_state
        self.retriever = LexicalContextRetriever(score_state, threshold=threshold)
        self.memory_store = AgentMemoryStore()
        self.model = normalize_agent_model(model)
        self.reasoning_effort = normalize_agent_reasoning_effort(reasoning_effort)
        self.prompt_split_enabled = _normalize_prompt_split_enabled(
            prompt_split_enabled,
        )
        self.prompt_split_min_sentences = _normalize_prompt_split_min_sentences(
            prompt_split_min_sentences,
        )
        self.llm = None
        self.splitter_llm = None
        self._api_key = os.environ.get("OPENAI_API_KEY")
        self.score_version = 0
        self._tool_results: list[tuple[str, dict, OperationResult]] = []
        self._musicxml_window_cache: OrderedDict[tuple[int, int, int], str] = OrderedDict()
        self._musicxml_cache = (
            _show_rests_for_empty_space(display_musicxml)
            if display_musicxml is not None
            else None
        )

    def _build_llm(self) -> Any:
        """Build a chat model for the current session options."""
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        from langchain_openai import ChatOpenAI
        kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": self._api_key,
        }
        kwargs.update(chat_openai_reasoning_kwargs(self.reasoning_effort))
        return ChatOpenAI(**kwargs)

    def get_llm(self) -> Any:
        """Lazy-load the main LLM to avoid import errors if OpenAI isn't configured."""
        if self.llm is None:
            self.llm = self._build_llm()
        return self.llm

    def get_splitter_llm(self) -> Any:
        """Lazy-load a separate LLM instance for background prompt splitting."""
        if self.splitter_llm is None:
            self.splitter_llm = self._build_llm()
        return self.splitter_llm

    def set_model(self, model: str | None) -> str:
        """Switch the chat model for future turns and return the selected id."""
        selected_model = normalize_agent_model(model)
        if selected_model != self.model:
            self.model = selected_model
            self.llm = None
            self.splitter_llm = None
        return self.model

    def set_reasoning_effort(self, reasoning_effort: object | None) -> str:
        """Switch the OpenAI reasoning effort for future turns."""
        selected_effort = normalize_agent_reasoning_effort(reasoning_effort)
        if selected_effort != self.reasoning_effort:
            self.reasoning_effort = selected_effort
            self.llm = None
            self.splitter_llm = None
        return self.reasoning_effort

    def set_prompt_split_enabled(self, enabled: object | None) -> bool:
        """Switch prompt split mode for future turns."""
        selected_enabled = _normalize_prompt_split_enabled(
            enabled,
            default=self.prompt_split_enabled,
        )
        self.prompt_split_enabled = selected_enabled
        return self.prompt_split_enabled

    def set_prompt_split_min_sentences(self, sentence_count: object | None) -> int:
        """Switch the minimum sentence count for prompt split activation."""
        selected_sentence_count = _normalize_prompt_split_min_sentences(
            sentence_count,
        )
        self.prompt_split_min_sentences = selected_sentence_count
        return self.prompt_split_min_sentences

    def prompt_split_config(self) -> PromptSplitConfig:
        """Return the effective prompt split configuration for this session."""
        if not self.prompt_split_enabled:
            return PromptSplitConfig(
                sentence_threshold=sys.maxsize,
                window_sentences=DEFAULT_PROMPT_SPLIT_CONFIG.window_sentences,
                fallback_split_sentences=(
                    DEFAULT_PROMPT_SPLIT_CONFIG.fallback_split_sentences
                ),
            )
        return PromptSplitConfig(
            sentence_threshold=max(0, self.prompt_split_min_sentences - 1),
            window_sentences=DEFAULT_PROMPT_SPLIT_CONFIG.window_sentences,
            fallback_split_sentences=(
                DEFAULT_PROMPT_SPLIT_CONFIG.fallback_split_sentences
            ),
        )

    def run_turn(self, user_text: str) -> str:
        """Execute one agent turn and return the response."""
        try:
            self._tool_results = []
            llm = self.get_llm()
            prompt_split_config = self.prompt_split_config()
            splitter_llm = (
                self.get_splitter_llm()
                if should_use_prompt_split(user_text, prompt_split_config)
                else None
            )
            response = run_prompt(
                self.score_state,
                self.retriever,
                llm,
                user_text,
                self.memory_store,
                recursion_limit=DEFAULT_RECURSION_LIMIT,
                mutation_recorder=self.record_tool_result,
                splitter_llm=splitter_llm,
                prompt_split_config=prompt_split_config,
            )
            if self._tool_results:
                self.score_version += 1
                self._clear_render_caches()
            return response
        except Exception as exc:
            logger.exception("Agent turn failed")
            return f"ERROR: {type(exc).__name__}: {exc}"

    def run_turn_stream(self, user_text: str) -> Iterator[dict[str, Any]]:
        """Execute one agent turn and yield progress events."""
        try:
            self._tool_results = []
            llm = self.get_llm()
            prompt_split_config = self.prompt_split_config()
            splitter_llm = (
                self.get_splitter_llm()
                if should_use_prompt_split(user_text, prompt_split_config)
                else None
            )
            for event in run_prompt_stream(
                self.score_state,
                self.retriever,
                llm,
                user_text,
                self.memory_store,
                recursion_limit=DEFAULT_RECURSION_LIMIT,
                mutation_recorder=self.record_tool_result,
                splitter_llm=splitter_llm,
                prompt_split_config=prompt_split_config,
            ):
                if event.get("type") == "final" and self._tool_results:
                    self.score_version += 1
                    self._clear_render_caches()
                yield event
        except Exception as exc:
            logger.exception("Streaming agent turn failed")
            yield {
                "type": "error",
                "error": f"ERROR: {type(exc).__name__}: {exc}",
            }

    def record_tool_result(
        self,
        tool_name: str,
        kwargs: dict,
        result: OperationResult,
    ) -> None:
        """Record successful mutation results for render-window selection."""
        if not result.success:
            return
        if result.details.get("changed") is False:
            return
        self._tool_results.append((tool_name, kwargs, result))

    def _clear_render_caches(self) -> None:
        """Clear exported MusicXML caches after a successful score mutation."""
        self._musicxml_cache = None
        self._musicxml_window_cache.clear()

    def get_musicxml(self) -> str:
        """Export the current score as MusicXML string."""
        if self._musicxml_cache is None:
            self._musicxml_cache = _show_rests_for_empty_space(
                self.score_state.to_musicxml_string()
            )
        return self._musicxml_cache

    def get_musicxml_window(self, start_measure: int, end_measure: int) -> str:
        """Export a render-sized MusicXML window for the current score."""
        cache_key = (self.score_version, start_measure, end_measure)
        cached = self._musicxml_window_cache.get(cache_key)
        if cached is not None:
            self._musicxml_window_cache.move_to_end(cache_key)
            return cached

        if self._musicxml_cache is not None:
            musicxml = extract_musicxml_window(
                self._musicxml_cache,
                start_measure,
                end_measure,
            )
        else:
            musicxml = _show_rests_for_empty_space(
                export_scorespeak_window_musicxml(
                    self.score_state,
                    start_measure,
                    end_measure,
                )
            )

        self._musicxml_window_cache[cache_key] = musicxml
        if len(self._musicxml_window_cache) > _MAX_MUSICXML_WINDOW_CACHE:
            self._musicxml_window_cache.popitem(last=False)

        return musicxml

    def changed_range(self, current_range: dict[str, int]) -> dict[str, int] | None:
        """Return the changed range from the most recent agent turn."""
        return _extract_changed_measure_range(self._tool_results, current_range)


session: Optional[AgentSession] = None


def _current_session_settings_kwargs() -> dict[str, object]:
    """Return settings to carry across score-session replacement."""
    if session is None:
        return {}
    return {
        "model": session.model,
        "reasoning_effort": session.reasoning_effort,
        "prompt_split_enabled": session.prompt_split_enabled,
        "prompt_split_min_sentences": session.prompt_split_min_sentences,
    }


def _ensure_session() -> None:
    """Initialize session with empty score if it doesn't exist."""
    global session
    if session is None:
        score_state = ScoreSpeak.create(measures=8)
        session = AgentSession(score_state)


def _agent_settings_response_payload() -> dict[str, Any]:
    """Return agent settings and selectable options for API responses."""
    _ensure_session()
    assert session is not None
    return {
        "default_model": DEFAULT_AGENT_MODEL,
        "model": session.model,
        "current_model": session.model,
        "models": agent_model_options_payload(),
        "default_reasoning_effort": DEFAULT_AGENT_REASONING_EFFORT,
        "reasoning_effort": session.reasoning_effort,
        "current_reasoning_effort": session.reasoning_effort,
        "reasoning_efforts": agent_reasoning_effort_options_payload(),
        "default_prompt_split_enabled": _DEFAULT_PROMPT_SPLIT_ENABLED,
        "prompt_split_enabled": session.prompt_split_enabled,
        "default_prompt_split_min_sentences": _DEFAULT_PROMPT_SPLIT_MIN_SENTENCES,
        "prompt_split_min_sentences": session.prompt_split_min_sentences,
    }


def _agent_models_response_payload() -> dict[str, Any]:
    """Return model options and the current session model for API responses."""
    return _agent_settings_response_payload()


def _apply_session_options_from_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Apply request-scoped agent option overrides to the current session."""
    _ensure_session()
    assert session is not None
    selected_model = session.model
    selected_reasoning_effort = session.reasoning_effort
    selected_prompt_split_enabled = session.prompt_split_enabled
    selected_prompt_split_min_sentences = session.prompt_split_min_sentences
    if "model" in data:
        selected_model = normalize_agent_model(data.get("model"))
    if "reasoning_effort" in data:
        selected_reasoning_effort = normalize_agent_reasoning_effort(
            data.get("reasoning_effort"),
        )
    if "prompt_split_enabled" in data:
        selected_prompt_split_enabled = _normalize_prompt_split_enabled(
            data.get("prompt_split_enabled"),
            default=session.prompt_split_enabled,
        )
    if "prompt_split_min_sentences" in data:
        selected_prompt_split_min_sentences = _normalize_prompt_split_min_sentences(
            data.get("prompt_split_min_sentences"),
        )

    session.set_model(selected_model)
    session.set_reasoning_effort(selected_reasoning_effort)
    session.set_prompt_split_enabled(selected_prompt_split_enabled)
    session.set_prompt_split_min_sentences(selected_prompt_split_min_sentences)
    return {
        "model": session.model,
        "reasoning_effort": session.reasoning_effort,
        "prompt_split_enabled": session.prompt_split_enabled,
        "prompt_split_min_sentences": session.prompt_split_min_sentences,
    }


@app.route('/')
def index():
    """Serve the main web app."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/vendor/osmd_musicxml_fixes.js')
def osmd_musicxml_fixes() -> Response:
    """Serve browser OSMD MusicXML fix helpers."""
    return send_from_directory(WEB_ROOT / 'vendor', 'osmd_musicxml_fixes.js')


@app.route('/api/agent/models', methods=['GET'])
def agent_models():
    """Return supported agent models and the current session selection."""
    return jsonify({
        'success': True,
        **_agent_models_response_payload(),
    })


@app.route('/api/agent/settings', methods=['GET'])
def agent_settings():
    """Return all configurable agent settings for the current session."""
    return jsonify({
        'success': True,
        **_agent_settings_response_payload(),
    })


@app.route('/api/agent/settings', methods=['PATCH'])
def update_agent_settings():
    """Update configurable agent settings for future turns."""
    try:
        data = request.get_json(silent=True) or {}
        _apply_session_options_from_payload(data)
    except ValueError as exc:
        return jsonify({
            'success': False,
            'error': str(exc),
            **_agent_settings_response_payload(),
        }), 400

    return jsonify({
        'success': True,
        **_agent_settings_response_payload(),
    })


@app.route('/api/agent/model', methods=['PATCH'])
def update_agent_model():
    """Update the current session's agent model for future turns."""
    _ensure_session()
    assert session is not None
    try:
        data = request.get_json(silent=True) or {}
        _apply_session_options_from_payload(data)
    except ValueError as exc:
        return jsonify({
            'success': False,
            'error': str(exc),
            **_agent_models_response_payload(),
        }), 400

    return jsonify({
        'success': True,
        **_agent_models_response_payload(),
    })


@app.route('/api/new', methods=['POST'])
def new_score():
    """Create a new empty score."""
    global session
    try:
        data = request.get_json() or {}
        session_settings = _current_session_settings_kwargs()
        measures = data.get('measures', 8)
        score_state = ScoreSpeak.create(
            measures=measures,
            parts=[{
                "instrument": "piano",
                "name": "Piano",
                "grand_staff": True,
            }],
        )
        session = AgentSession(score_state, **session_settings)
        musicxml = session.get_musicxml()
        return jsonify({
            'success': True,
            'message': f'Created new score with {measures} measures',
            'score_version': session.score_version,
            **_initial_score_payload(score_state, musicxml),
        })
    except Exception as exc:
        logger.exception("Failed to create new score")
        return jsonify({
            'success': False,
            'error': f"{type(exc).__name__}: {exc}"
        }), 500


@app.route('/api/load', methods=['POST'])
def load_score():
    """Load a score from uploaded MusicXML or compressed MXL."""
    global session
    try:
        if 'musicxml' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No musicxml file provided'
            }), 400

        file = request.files['musicxml']
        session_settings = _current_session_settings_kwargs()
        score_state, display_musicxml = _load_score_and_display_xml_from_upload(file)
        session = AgentSession(score_state, display_musicxml, **session_settings)

        return jsonify({
            'success': True,
            'message': 'Score loaded successfully',
            'score_version': session.score_version,
            **_initial_score_payload(score_state, display_musicxml),
        })
    except Exception as exc:
        logger.exception("Failed to load score")
        return jsonify({
            'success': False,
            'error': f"{type(exc).__name__}: {exc}"
        }), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a chat message through the agent."""
    _ensure_session()

    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400

        user_message = data['message']
        turn_options = _apply_session_options_from_payload(data)
        assert session is not None
        agent_response = session.run_turn(user_message)
        render_range = _current_render_range(data, session.score_state)
        changed_range = session.changed_range(render_range)

        return jsonify({
            'success': True,
            'response': agent_response,
            'model': turn_options["model"],
            'reasoning_effort': turn_options["reasoning_effort"],
            'prompt_split_enabled': turn_options["prompt_split_enabled"],
            'prompt_split_min_sentences': turn_options["prompt_split_min_sentences"],
            'score_version': session.score_version,
            'changed_range': changed_range,
            **_score_metadata(session.score_state),
        })
    except ValueError as exc:
        return jsonify({
            'success': False,
            'error': str(exc),
            **_agent_models_response_payload(),
        }), 400
    except Exception as exc:
        logger.exception("Chat request failed")
        return jsonify({
            'success': False,
            'error': f"{type(exc).__name__}: {exc}"
        }), 500


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Process a chat message through the agent and stream progress events."""
    _ensure_session()

    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400

        user_message = data['message']
        turn_options = _apply_session_options_from_payload(data)

        @stream_with_context
        def generate():
            assert session is not None
            for event in session.run_turn_stream(user_message):
                if event.get("type") == "final":
                    render_range = _current_render_range(data, session.score_state)
                    event = {
                        **event,
                        "model": turn_options["model"],
                        "reasoning_effort": turn_options["reasoning_effort"],
                        "prompt_split_enabled": (
                            turn_options["prompt_split_enabled"]
                        ),
                        "prompt_split_min_sentences": (
                            turn_options["prompt_split_min_sentences"]
                        ),
                        "score_version": session.score_version,
                        "changed_range": session.changed_range(render_range),
                        **_score_metadata(session.score_state),
                    }
                yield json.dumps(event) + "\n"

        return Response(
            generate(),
            mimetype='application/x-ndjson',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
            },
        )
    except ValueError as exc:
        return jsonify({
            'success': False,
            'error': str(exc),
            **_agent_models_response_payload(),
        }), 400
    except Exception as exc:
        logger.exception("Streaming chat request failed")
        return jsonify({
            'success': False,
            'error': f"{type(exc).__name__}: {exc}"
        }), 500


@app.route('/api/voice', methods=['POST'])
def voice():
    """Process uploaded voice audio into a structured agent message."""
    _ensure_session()

    if 'audio' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No audio file provided'
        }), 400

    temp_path: Path | None = None
    try:
        file = request.files['audio']
        mode = request.form.get("mode")
        if mode is not None and mode != "speech":
            return jsonify({
                'success': False,
                'error': "Unsupported voice mode. Only 'speech' is accepted.",
            }), 400
        language = request.form.get("language", "en") or None
        render_range = _voice_render_range_from_form()
        duration_seconds = _optional_float_form("duration_seconds")

        temp_path = _save_voice_audio_upload(file)
        audio_input = AudioInput(
            data=temp_path,
            filename=file.filename or temp_path.name,
            mime_type=file.mimetype,
            duration_seconds=duration_seconds,
        )
        result = VoiceInputProcessor().process(
            audio_input,
            speech_prompt=_VOICE_SPEECH_PROMPT,
            language=language,
        )
        payload = _voice_response_payload(result, render_range)
        status_code = 200 if result.success else 422
        return jsonify(payload), status_code
    except (TypeError, ValueError, VoiceInputError) as exc:
        logger.info("Voice request rejected: %s", exc)
        return jsonify({
            'success': False,
            'error': str(exc),
        }), 400
    except Exception as exc:
        logger.exception("Voice request failed")
        return jsonify({
            'success': False,
            'error': f"{type(exc).__name__}: {exc}"
        }), 500
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.route('/api/musicxml', methods=['GET'])
def get_musicxml():
    """Get the current score as MusicXML."""
    _ensure_session()
    try:
        return jsonify({
            'success': True,
            'musicxml': session.get_musicxml(),
            'score_version': session.score_version,
            **_score_metadata(session.score_state),
        })
    except Exception as exc:
        logger.exception("Failed to get MusicXML")
        return jsonify({
            'success': False,
            'error': f"{type(exc).__name__}: {exc}"
        }), 500


@app.route('/api/musicxml/download', methods=['GET'])
def download_musicxml():
    """Download the current score as a MusicXML file."""
    _ensure_session()
    try:
        xml = session.get_musicxml()
        return Response(
            xml,
            mimetype='application/vnd.recordare.musicxml+xml',
            headers={
                'Content-Disposition': 'attachment; filename="score.musicxml"',
                'Cache-Control': 'no-store',
            },
        )
    except Exception as exc:
        logger.exception("Failed to export MusicXML download")
        return jsonify({
            'success': False,
            'error': f"{type(exc).__name__}: {exc}"
        }), 500


@app.route('/api/musicxml/window', methods=['GET'])
def get_musicxml_window():
    """Get a render-sized MusicXML measure window for the current score."""
    _ensure_session()
    assert session is not None
    try:
        raw_start = request.args.get("start", 1)
        raw_end = request.args.get("end")
        start = int(raw_start)
        if raw_end is None:
            render_range = _default_render_range(session.score_state, start)
            end = render_range["end"]
        else:
            end = int(raw_end)
            start, end = normalize_measure_window(start, end)
            measure_count = session.score_state.measure_count
            if measure_count > 0:
                start = min(start, measure_count)
                end = min(end, measure_count)
        musicxml = session.get_musicxml_window(start, end)
        return jsonify({
            'success': True,
            'musicxml': musicxml,
            'score_version': session.score_version,
            'render_range': {'start': start, 'end': end},
            **_score_metadata(session.score_state),
        })
    except Exception as exc:
        logger.exception("Failed to get MusicXML window")
        return jsonify({
            'success': False,
            'error': f"{type(exc).__name__}: {exc}"
        }), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Get current session status."""
    _ensure_session()
    try:
        from scorespeak.agent.overview import build_score_overview, format_overview_for_prompt
        overview = build_score_overview(session.score_state)
        return jsonify({
            'success': True,
            'has_score': session is not None,
            'model': session.model if session else None,
            'overview': format_overview_for_prompt(overview)
        })
    except Exception as exc:
        logger.exception("Failed to get status")
        return jsonify({
            'success': False,
            'error': f"{type(exc).__name__}: {exc}"
        }), 500


def main():
    """Run the Flask development server."""
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(levelname)s %(name)s: %(message)s'
    )

    print(f"Starting ScoreSpeak web server on http://localhost:{port}")
    print("Make sure OPENAI_API_KEY is set in your environment")

    app.run(host='0.0.0.0', port=port, debug=debug)


if __name__ == '__main__':
    main()
