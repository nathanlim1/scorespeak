"""
Wrap ``ScoreSpeak`` public methods as LangChain ``StructuredTool`` objects.

For each :class:`~scorespeak.retrieval.MethodRecord` returned by the lexical
retriever, :func:`make_tool_from_record` builds a pydantic schema from the
bound method's signature and exposes a thin wrapper that calls the method
on a shared live ``ScoreSpeak``.  Tool calls that raise are captured and
returned as ``"ERROR: ..."`` strings so the agent can recover mid-loop
without crashing the graph.
"""

from __future__ import annotations

import inspect
import json
import logging
import typing
from dataclasses import dataclass, field, is_dataclass
from inspect import Parameter, Signature
from typing import Any, Callable, Iterable, Optional, Union, get_args, get_origin

from langchain_core.tools import StructuredTool
from pydantic import ConfigDict, Field, create_model

from ..core import ScoreSpeak
from ..music.pitch_space import (
    part_stores_sounding_pitch,
    part_transposition_interval,
    stored_key_signature_for_concert_key,
)
from ..retrieval import MethodRecord
from ..types import OperationResult
from ..music.validation import validate_voice_number
from .context_renderers import render_exact_context, render_summary_context
from .defaults import DEFAULT_TOOL_SEARCH_LIMIT, MAX_TOOL_SEARCH_LIMIT
from .tool_catalog import ToolCatalog

logger = logging.getLogger(__name__)


_MAX_RESULT_CHARS = 1500
_MAX_INSPECTION_RESULT_CHARS = 8000
_SIMPLE_TYPES: tuple[type, ...] = (str, int, float, bool)
AGENT_EXCLUDED_TOOL_NAMES = frozenset(
    {
        "create",
        "from_musicxml",
        "search_score",
        "to_musicxml",
        "to_musicxml_string",
    }
)

_COMMON_ARG_DESCRIPTIONS = {
    "measure": "1-based measure number.",
    "measure_number": "1-based measure number.",
    "start_measure": "1-based measure number where the span or range starts.",
    "end_measure": "1-based measure number where the span or range ends.",
    "source_start": "1-based first source measure number to copy from.",
    "target_start": "1-based first target measure number to replace.",
    "beat": "1-based beat position in the measure.",
    "start_beat": "1-based beat position where the span starts.",
    "end_beat": "1-based beat position where the span ends.",
    "total_duration": (
        "Total duration of a rest-spelling range, such as 'whole', 'half', "
        "or a quarter-length number."
    ),
    "part": (
        "Part identifier: 0-based part index, part name, or None for the "
        "tool's documented default scope. Prefer an explicit part when the "
        "user names one."
    ),
    "grand_staff": (
        "For add_part: set True when adding a piano/keyboard grand staff "
        "with RH and LH staves; leave False for a single staff."
    ),
    "source_part": (
        "Optional source part identifier. Omit with target_part to copy "
        "corresponding measures in all parts."
    ),
    "target_part": (
        "Optional target part identifier. Provide only when source_part is "
        "also provided for an explicit cross-part copy."
    ),
    "voice": (
        "1-based rhythmic voice number. Use voice 1 unless exact inspection "
        "shows the target belongs to another simultaneous rhythmic line."
    ),
    "duration": (
        "Duration value such as 'quarter', '8', '16th', or a "
        "quarter-length number."
    ),
    "slur_to_principal": (
        "For add_grace_note: set True only when the grace note should be "
        "slurred to the same-beat principal note or chord."
    ),
    "dots": "Number of augmentation dots.",
    "pitch": "Pitch such as 'C4', 'c#4', 'C♯4', or MIDI integer 60.",
    "pitches": "List of pitches such as ['C4', 'E4', 'G4'].",
    "ottava_type": "Ottava type such as '8va', '8vb', '15ma', or '15mb'.",
    "line_type": "Line style such as 'wavy' or 'solid'.",
    "label": "Optional displayed text label.",
    "placement": "Vertical placement: 'above' or 'below'.",
    "rewrite_pitches": (
        "For ottava: set True when the prompt asks to move written notes "
        "down/up to reduce ledger lines while adding or removing the ottava. "
        "False only adds or removes the octave-shift mark."
    ),
    "ornament_type": (
        "Ornament name such as 'trill', 'mordent', 'turn', or 'tremolo'."
    ),
    "tremolo_marks": "Number of tremolo slash marks.",
    "finger_number": "Fingering number or label, such as 1, 2, 3, 4, 5, or 'T'.",
    "lyric_number": "Lyric verse or line number, starting at 1.",
    "number": "Ending or volta number, such as 1 or '1, 2'.",
    "mark_type": "Navigation mark type to remove.",
}

_ADD_NOTES_NOTE_ITEM_SCHEMA = create_model(
    "AddNotesNoteItem",
    __config__=ConfigDict(extra="forbid"),
    pitch=(
        Any,
        Field(
            ...,
            description=(
                "Required note pitch, for example 'C4', 'c#4', 'C♯4', "
                "or MIDI integer 60."
            ),
        ),
    ),
    beat=(
        float,
        Field(..., description="Required 1-based beat position in the measure."),
    ),
    duration=(
        Any,
        Field(
            ...,
            description=(
                "Required note duration, for example 'quarter', '8', "
                "'16th', or a float quarter-length."
            ),
        ),
    ),
    dots=(
        int,
        Field(..., description="Required number of augmentation dots, usually 0."),
    ),
)
_REMOVE_NOTES_NOTE_ITEM_SCHEMA = create_model(
    "RemoveNotesNoteItem",
    __config__=ConfigDict(extra="forbid"),
    beat=(
        float,
        Field(..., description="Required 1-based beat position in the measure."),
    ),
    pitch=(
        Any,
        Field(
            default=None,
            description=(
                "Optional pitch guard, for example 'C4'. Omit to remove the "
                "whole note/chord/rest event at this beat."
            ),
        ),
    ),
)
_RESHAPE_RESTS_REST_ITEM_SCHEMA = create_model(
    "ReshapeRestsRestItem",
    __config__=ConfigDict(extra="forbid"),
    duration=(
        Any,
        Field(
            ...,
            description=(
                "Required visible rest duration, for example 'quarter', "
                "'half', or a float quarter-length."
            ),
        ),
    ),
    dots=(
        int,
        Field(default=0, description="Number of augmentation dots, usually 0."),
    ),
)
_SET_SCORE_PARTS_PART_ITEM_SCHEMA = create_model(
    "SetScorePartsPartItem",
    __config__=ConfigDict(extra="forbid"),
    instrument=(
        str,
        Field(
            ...,
            description=(
                "Required playable instrument, including transposition or "
                "variant when relevant. Use values like 'B♭ clarinet', "
                "'E♭ horn', or 'C trumpet' instead of generic 'clarinet', "
                "'horn', or 'trumpet' when the displayed part name includes "
                "that key. The optional name field is only the displayed "
                "staff label."
            ),
        ),
    ),
    name=(
        Optional[str],
        Field(
            default=None,
            description=(
                "Optional displayed part name. Omit to use the instrument "
                "default."
            ),
        ),
    ),
)


def filter_agent_method_records(records: Iterable[MethodRecord]) -> list[MethodRecord]:
    """Return method records that should be exposed as agent tools."""
    return [
        record
        for record in records
        if record.name not in AGENT_EXCLUDED_TOOL_NAMES
    ]


@dataclass
class ToolExpansionRequests:
    """Mutable per-run record of tool names loaded through ``tool_search``."""

    loaded_tool_names: list[str] = field(default_factory=list)

    def add(self, names: list[str]) -> None:
        """Record newly loaded tool names while preserving request order."""
        seen = set(self.loaded_tool_names)
        for name in names:
            if name in seen:
                continue
            self.loaded_tool_names.append(name)
            seen.add(name)


def _resolve_annotation(annotation: Any) -> Any:
    """Return a pydantic-friendly annotation, falling back to ``Any``.

    music21 signatures include forward references (e.g. ``"music21.pitch.Pitch"``)
    and permissive unions like ``Union[str, int, "music21.pitch.Pitch"]``.  To
    keep the tool schema generation robust we:

    * Preserve simple builtins (``str``, ``int``, ``float``, ``bool``).
    * Preserve ``Optional[...]`` and ``Union[...]`` when every member is a
      simple builtin or ``None``.
    * Preserve bare ``list`` / ``dict`` / ``tuple``.
    * Fall back to :class:`typing.Any` for everything else (forward refs,
      music21 classes, complex generics).
    """
    if annotation is inspect._empty:
        return Any
    if annotation is None or annotation is type(None):  # noqa: E721
        return type(None)
    if isinstance(annotation, str):
        return Any
    if annotation in _SIMPLE_TYPES:
        return annotation
    if annotation is list or annotation is dict or annotation is tuple:
        return annotation

    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        resolved_args = []
        for arg in args:
            if arg is type(None):
                resolved_args.append(type(None))
                continue
            if arg in _SIMPLE_TYPES:
                resolved_args.append(arg)
                continue
            return Any
        return Union[tuple(resolved_args)]  # type: ignore[return-value]
    if origin in (list, tuple, set, frozenset):
        return origin
    if origin is dict:
        return dict

    return Any


def _build_args_schema(
    method_name: str,
    signature: Signature,
    type_hints: dict[str, Any] | None = None,
    descriptions: dict[str, str] | None = None,
):
    """Build a pydantic model describing ``signature`` for use as ``args_schema``.

    The first parameter (``self``) is skipped because the tool is always
    bound to a live ``ScoreSpeak`` before the function is exposed.
    """
    fields: dict[str, tuple[Any, Any]] = {}
    type_hints = type_hints or {}
    descriptions = descriptions or {}
    for param_name, param in signature.parameters.items():
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue
        if method_name == "add_notes" and param_name == "notes":
            annotation = list[_ADD_NOTES_NOTE_ITEM_SCHEMA]
        elif method_name == "remove_notes" and param_name == "notes":
            annotation = list[_REMOVE_NOTES_NOTE_ITEM_SCHEMA]
        elif (
            method_name == "add_tuplet"
            and param_name == "pitches_and_durations"
        ):
            annotation = list[tuple[Any, Any]]
        elif method_name == "reshape_rests" and param_name == "rests":
            annotation = list[_RESHAPE_RESTS_REST_ITEM_SCHEMA]
        elif method_name == "set_score_parts" and param_name == "parts":
            annotation = list[_SET_SCORE_PARTS_PART_ITEM_SCHEMA]
        else:
            annotation = _resolve_annotation(
                type_hints.get(param_name, param.annotation)
            )
        description = descriptions.get(
            param_name,
            _COMMON_ARG_DESCRIPTIONS.get(param_name, f"parameter {param_name}"),
        )
        if param.default is inspect._empty:
            default = Field(..., description=description)
        else:
            default = Field(
                default=param.default,
                description=description,
            )
        fields[param_name] = (annotation, default)

    model_name = f"{method_name.title().replace('_', '')}Args"
    if not fields:
        return create_model(model_name, __config__=ConfigDict(extra="forbid"))
    return create_model(model_name, __config__=ConfigDict(extra="forbid"), **fields)


def _format_signature_for_description(signature: Signature) -> str:
    """Render a call signature string suitable for tool descriptions."""
    parts = []
    for param_name, param in signature.parameters.items():
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect._empty:
            parts.append(param_name)
        else:
            parts.append(f"{param_name}={param.default!r}")
    return "(" + ", ".join(parts) + ")"


def _short_docstring(docstring: str, limit: int = 400) -> str:
    """Return the first paragraph of ``docstring``, capped to ``limit`` chars."""
    if not docstring:
        return ""
    first_paragraph = docstring.strip().split("\n\n", 1)[0].strip()
    if len(first_paragraph) > limit:
        return first_paragraph[: limit - 1].rstrip() + "…"
    return first_paragraph


def _tool_description_body(tool: StructuredTool) -> str:
    """Return a core tool description without its leading signature line."""
    description = tool.description or ""
    paragraphs = description.strip().split("\n\n", 1)
    if len(paragraphs) == 2 and paragraphs[0].startswith(f"{tool.name}("):
        return paragraphs[1]
    return description


def _extract_arg_descriptions(docstring: str) -> dict[str, str]:
    """Extract simple Google-style Args descriptions from a docstring."""
    descriptions: dict[str, str] = {}
    if not docstring:
        return descriptions

    lines = inspect.cleandoc(docstring).splitlines()
    in_args = False
    current_name: str | None = None
    current_parts: list[str] = []

    def flush_current() -> None:
        """Store the currently buffered argument description."""
        nonlocal current_name, current_parts
        if current_name is not None and current_parts:
            descriptions[current_name] = " ".join(current_parts).strip()
        current_name = None
        current_parts = []

    for line in lines:
        stripped = line.strip()
        if stripped == "Args:":
            in_args = True
            continue
        if not in_args:
            continue
        if stripped in {"Returns:", "Raises:", "Examples:", "Yields:"}:
            flush_current()
            break
        if not stripped:
            continue
        if ":" in stripped:
            candidate, remainder = stripped.split(":", 1)
            candidate = candidate.strip()
            if candidate and " " not in candidate:
                flush_current()
                current_name = candidate
                current_parts = [remainder.strip()]
                continue
        if current_name is not None:
            current_parts.append(stripped)

    flush_current()
    return descriptions


def _serialize_result(value: Any) -> str:
    """Convert a ScoreSpeak return value into a short string for the agent."""
    if isinstance(value, OperationResult):
        prefix = "OK" if value.success else "FAIL"
        text = f"{prefix}: {value.description}"
        if value.details:
            text += f" | details={value.details}"
        return text[:_MAX_RESULT_CHARS]

    if is_dataclass(value):
        return f"{type(value).__name__}: {value}"[:_MAX_RESULT_CHARS]

    if isinstance(value, (list, tuple)):
        rendered = repr(list(value))
        return rendered[:_MAX_RESULT_CHARS]

    if isinstance(value, dict):
        return repr(value)[:_MAX_RESULT_CHARS]

    if value is None:
        return "OK"

    return str(value)[:_MAX_RESULT_CHARS]


def _serialize_tool_result(
    method_name: str,
    value: Any,
    *,
    kwargs: dict[str, Any],
    method: Callable[..., Any],
) -> str:
    """Serialize a tool result with method-specific agent annotations."""
    if method_name == "get_active_key_signature" and isinstance(value, str):
        annotated = _annotate_active_key_signature_tool_result(
            value,
            kwargs=kwargs,
            method=method,
        )
        if annotated is not None:
            return annotated[:_MAX_RESULT_CHARS]
    return _serialize_result(value)


def _annotate_active_key_signature_tool_result(
    key_signature: str,
    *,
    kwargs: dict[str, Any],
    method: Callable[..., Any],
) -> str | None:
    """Return an agent-facing key-signature label, if context is available."""
    score_state = getattr(method, "__self__", None)
    if not isinstance(score_state, ScoreSpeak):
        return None

    measure_number = kwargs.get("measure_number")
    part = kwargs.get("part")
    if part is None:
        return f"concert key: {key_signature}"
    if measure_number is None:
        return None

    try:
        numeric_measure = int(measure_number)
        concert_key = score_state.get_active_key_signature(
            numeric_measure,
            part=None,
        )
        part_obj, _ = score_state._resolve_part(part)
        active_part_key = score_state._get_active_key_signature_obj(
            part_obj,
            numeric_measure,
        )
        active_concert_key = score_state._get_active_concert_key_signature_obj(
            numeric_measure,
        )
        expected_part_key = stored_key_signature_for_concert_key(
            part_obj,
            active_concert_key,
        )
        expected_part_key_is_active = score_state._key_signatures_equal(
            active_part_key,
            expected_part_key,
        )
    except Exception:
        return None

    has_transposition = part_transposition_interval(part_obj) is not None
    if not expected_part_key_is_active:
        return f"local key: {key_signature} (concert key: {concert_key})"
    if not has_transposition:
        return (
            f"part key: {key_signature} "
            f"(non-transposing; concert key: {concert_key})"
        )
    if part_stores_sounding_pitch(part_obj):
        return (
            f"sounding key: {key_signature} "
            f"(concert key: {concert_key})"
        )
    return f"written key: {key_signature} (concert key: {concert_key})"


def _serialize_json_result(value: dict[str, Any], *, limit: int) -> str:
    """Serialize a dict for tool output and truncate with a clear warning."""
    text = json.dumps(value, default=str, separators=(",", ":"))
    if len(text) <= limit:
        return text
    prefix = (
        "TRUNCATED: result exceeded the inspection output limit. "
        "Narrow the inspected scope or request fewer include categories.\n"
    )
    return prefix + text[: max(0, limit - len(prefix))]


MutationRecorder = Callable[[str, dict[str, Any], OperationResult], None]


def _resolve_tool_names(
    tools_by_name: dict[str, StructuredTool],
    names: Iterable[str] | None,
) -> tuple[list[str], list[str]]:
    """Split exact tool names into known tools and invalid names."""
    if names is None:
        return [], []

    valid: list[str] = []
    invalid: list[str] = []
    seen_valid: set[str] = set()
    seen_invalid: set[str] = set()
    for raw_name in names:
        name = str(raw_name).strip()
        if not name:
            continue
        if name in tools_by_name:
            if name not in seen_valid:
                valid.append(name)
                seen_valid.add(name)
            continue
        if name not in seen_invalid:
            invalid.append(name)
            seen_invalid.add(name)
    return valid, invalid


def _core_tool_match_payload(
    tool: StructuredTool,
    *,
    core_tool_names: set[str],
) -> dict[str, Any]:
    """Return a compact ``tool_search`` match payload for a core tool."""
    return {
        "name": tool.name,
        "description": _short_docstring(_tool_description_body(tool), limit=1500),
        "tags": sorted(str(tool.name).split("_")),
        "already_core": tool.name in core_tool_names,
        "loaded": True,
    }


def _build_invoker(
    method: Callable[..., Any],
    method_name: str,
    mutation_recorder: MutationRecorder | None = None,
) -> Callable[..., str]:
    """Return the function that actually executes ``method`` for the tool.

    The wrapper captures all exceptions so the LangGraph run does not abort
    on bad agent input; the error string is handed back to the LLM so it
    can correct and retry.
    """
    def invoke_method(**kwargs: Any) -> str:
        try:
            result = method(**kwargs)
        except ValueError as exc:
            return f"ERROR (ValueError): {exc}"
        except TypeError as exc:
            return f"ERROR (TypeError): {exc}"
        except Exception as exc:
            return f"ERROR ({type(exc).__name__}): {exc}"
        if (
            mutation_recorder is not None
            and isinstance(result, OperationResult)
            and result.success
            and result.details.get("changed") is not False
        ):
            mutation_recorder(method_name, dict(kwargs), result)
        return _serialize_tool_result(
            method_name,
            result,
            kwargs=kwargs,
            method=method,
        )

    invoke_method.__name__ = method_name
    return invoke_method


def _normalize_optional_list(value: Any) -> Any:
    """Normalize scalar/list scope inputs for public tool schemas."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _normalize_bar_range(value: Any) -> Any:
    """Normalize bar range input into the tuple shape expected internally."""
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        return value
    if isinstance(value, list) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, str) and "-" in value:
        start_text, end_text = value.split("-", 1)
        return (int(start_text.strip()), int(end_text.strip()))
    raise ValueError(
        "bar_range must be a two-item list/tuple or a string like '3-5'."
    )


def _normalize_tuplet_ratio_input(value: Any) -> Any:
    """Normalize agent tuplet-ratio input into a tuple."""
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, list) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, str) and ":" in value:
        actual, normal = value.split(":", 1)
        return (int(actual.strip()), int(normal.strip()))
    raise ValueError("tuplet_ratio must be a two-item list/tuple or '3:2'.")


def _build_scope_query(
    *,
    parts: Any = None,
    bar_range: Any = None,
    measure_numbers: Any = None,
    voices: Any = None,
) -> dict[str, Any]:
    """Build a structured internal bar-result scope query from tool arguments."""
    scope = {}
    normalized_parts = _normalize_optional_list(parts)
    normalized_measures = _normalize_optional_list(measure_numbers)
    normalized_voices = _normalize_optional_list(voices)
    normalized_range = _normalize_bar_range(bar_range)

    if normalized_parts is not None:
        scope["parts"] = normalized_parts
    if normalized_range is not None:
        scope["bar_range"] = normalized_range
    if normalized_measures is not None:
        scope["measure_numbers"] = [int(value) for value in normalized_measures]
    if normalized_voices is not None:
        scope["voices"] = [
            validate_voice_number(value, "voices entry")
            for value in normalized_voices
        ]

    return {"scope": scope} if scope else {}


def make_search_score_tool(score_state: ScoreSpeak) -> StructuredTool:
    """Build the agent-facing typed semantic score search tool."""
    def search_score(
        parts: Any = None,
        bar_range: Any = None,
        measure_numbers: Any = None,
        voices: Any = None,
        event_sequence: Optional[list[dict[str, Any]]] = None,
        event_kind: Optional[str] = None,
        pitch: Any = None,
        pitch_class: Optional[str] = None,
        duration: Any = None,
        beat: Optional[float] = None,
        tie_status: Optional[str] = None,
        is_grace: Optional[bool] = None,
        dots: Optional[int] = None,
        tuplet_ratio: Any = None,
        chord_mode: str = "exact",
        marking_type: Optional[str] = None,
        marking_value: Optional[str] = None,
        lyric_text: Optional[str] = None,
        span_type: Optional[str] = None,
        span_value: Optional[str] = None,
        structure: Optional[str] = None,
        structure_value: Optional[str] = None,
        time_signature: Optional[str] = None,
        key_signature: Optional[str] = None,
        tempo: Optional[float] = None,
        clef: Optional[str] = None,
        changed_attribute: Optional[str] = None,
        logic: str = "all",
        limit: Optional[int] = 40,
    ) -> str:
        """Search score notation with typed semantic filters.

        Use this to locate bars by scope, event contents, markings, spans,
        structure, active attributes, or voice numbers. key_signature matches
        the bar-level concert key; part-specific written/displayed keys are
        shown by inspect_score_region and inspect_score_attributes. The result
        is a lossy summary with match reasons; use inspect_score_region before
        editing when exact events, beats, or voice separation matter.
        """
        try:
            result = score_state.search_score(
                parts=_normalize_optional_list(parts),
                bar_range=_normalize_bar_range(bar_range),
                measure_numbers=_normalize_optional_list(measure_numbers),
                voices=_normalize_optional_list(voices),
                event_sequence=event_sequence,
                event_kind=event_kind,
                pitch=pitch,
                pitch_class=pitch_class,
                duration=duration,
                beat=beat,
                tie_status=tie_status,
                is_grace=is_grace,
                dots=dots,
                tuplet_ratio=_normalize_tuplet_ratio_input(tuplet_ratio),
                chord_mode=chord_mode,
                marking_type=marking_type,
                marking_value=marking_value,
                lyric_text=lyric_text,
                span_type=span_type,
                span_value=span_value,
                structure=structure,
                structure_value=structure_value,
                time_signature=time_signature,
                key_signature=key_signature,
                tempo=tempo,
                clef=clef,
                changed_attribute=changed_attribute,
                logic=logic,
                limit=limit,
            )
        except ValueError as exc:
            return f"ERROR (ValueError): {exc}"
        except TypeError as exc:
            return f"ERROR (TypeError): {exc}"
        except Exception as exc:
            return f"ERROR ({type(exc).__name__}): {exc}"
        return render_summary_context(
            result,
            empty_message=(
                "No bars matched search_score() filters in the requested "
                "scope. This is a negative search result, not evidence that "
                "matching notation exists elsewhere."
            ),
        )

    return StructuredTool.from_function(
        func=search_score,
        name="search_score",
        description=(
            "search_score(parts=None, bar_range=None, measure_numbers=None, "
            "voices=None, event_sequence=None, event_kind=None, pitch=None, "
            "pitch_class=None, duration=None, beat=None, tie_status=None, "
            "is_grace=None, dots=None, tuplet_ratio=None, marking_type=None, "
            "marking_value=None, lyric_text=None, span_type=None, "
            "span_value=None, structure=None, structure_value=None, "
            "time_signature=None, key_signature=None, tempo=None, clef=None, "
            "changed_attribute=None, logic='all', limit=40)\n\n"
            "Typed semantic search over explicitly supported score fields. "
            "Supported filters: scope via parts, bar_range, measure_numbers, "
            "and voices; events via event_sequence, event_kind ('note', "
            "'rest', 'chord'), pitch, pitch_class, duration, beat, tie_status "
            "('start', 'continue', 'stop', 'none', 'untied'), is_grace, dots, "
            "tuplet_ratio=(actual, normal), and chord_mode ('exact', "
            "'contains'); markings via marking_type ('dynamic', "
            "'articulation', 'ornament', 'fingering', 'lyric', "
            "'text_expression', 'chord_symbol'), marking_value, and "
            "lyric_text; spans via span_type ('hairpin', 'slur', 'ottava', "
            "'glissando', 'pedal') and span_value; structure "
            "via structure ('barline_start', 'barline_end', 'repeat_start', "
            "'repeat_end', 'ending_number', 'rehearsal_mark', 'navigation', "
            "'system_break', 'page_break') and structure_value; attributes via "
            "time_signature, key_signature (bar-level concert key), tempo, "
            "clef, and changed_attribute. logic is 'all' or 'any'. limit caps the "
            "number of matching bars returned to keep broad searches fast; "
            "narrow parts/bar_range/measure_numbers or raise limit when the "
            "summary reports truncation. Returns compact summary context "
            "with match reasons. It cannot directly search arbitrary musical "
            "concepts outside these filters; use inspect_score_region for "
            "exact rows before editing or reasoning about unsupported "
            "concepts."
        ),
    )


def make_inspect_score_region_tool(score_state: ScoreSpeak) -> StructuredTool:
    """Build the always-available exact score inspection tool."""
    def inspect_score_region(
        parts: Any = None,
        bar_range: Any = None,
        measure_numbers: Any = None,
        voices: Any = None,
        include: Optional[list[str]] = None,
    ) -> str:
        """Inspect exact symbolic detail for a scoped score region.

        Core event rows, ties, tuplets, and first-bar-plus-changes attributes
        are always returned, grouped by part and voice. Use this before
        editing multi-voice bars or choosing exact measure, part, voice, and
        beat arguments for surgical note tools.
        Optional channels are
        requested with include, for example ["dynamics", "hairpins", "slurs",
        "structure"] or ["all_current_channels"].
        """
        try:
            query = _build_scope_query(
                parts=parts,
                bar_range=bar_range,
                measure_numbers=measure_numbers,
                voices=voices,
            )
            result = score_state._build_bar_result_set(query)
            exact = render_exact_context(result, include=include)
        except ValueError as exc:
            return f"ERROR (ValueError): {exc}"
        except TypeError as exc:
            return f"ERROR (TypeError): {exc}"
        except Exception as exc:
            return f"ERROR ({type(exc).__name__}): {exc}"
        return _serialize_json_result(exact, limit=_MAX_INSPECTION_RESULT_CHARS)

    return StructuredTool.from_function(
        func=inspect_score_region,
        name="inspect_score_region",
        description=(
            "inspect_score_region(parts=None, bar_range=None, "
            "measure_numbers=None, voices=None, include=None)\n\n"
            "Return exact compact symbolic rows for a scoped score region. "
            "Use this before editing multi-voice bars or deciding exact "
            "measure, part, voice, and beat arguments for surgical "
            "note/rest/chord tools. Mandatory core fields include "
            "events, ties, tuplets, measures, parts/staves, voices, and "
            "first-bar-plus-changes time/concert-key/clef/tempo. Transposing "
            "parts with different displayed written keys include explicit "
            "part notation with concert_key context. Optional include "
            "categories add non-core channels such as dynamics, hairpins, "
            "articulations, slurs, lyrics, text, chord_symbols, ornaments, "
            "technique, spans, markings, structure, or all_current_channels. "
            "Marking rows include beat fields; span and tuplet rows include "
            "beat_range fields for edit-tool locations."
        ),
    )


def make_inspect_score_attributes_tool(score_state: ScoreSpeak) -> StructuredTool:
    """Build the always-available scoped score-attributes inspection tool."""
    def inspect_score_attributes(
        parts: Any = None,
        bar_range: Any = None,
        measure_numbers: Any = None,
        voices: Any = None,
    ) -> str:
        """Inspect active score attributes for a scoped region.

        Returns compact time/concert-key/tempo changes, structure fields, and
        per-part clef/staff/key information without event rows. Use
        inspect_score_region instead when note, rhythm, or voice content
        matters.
        """
        try:
            query = _build_scope_query(
                parts=parts,
                bar_range=bar_range,
                measure_numbers=measure_numbers,
                voices=voices,
            )
            result = score_state._build_bar_result_set(query)
            payload = _extract_attribute_payload(result)
        except ValueError as exc:
            return f"ERROR (ValueError): {exc}"
        except TypeError as exc:
            return f"ERROR (TypeError): {exc}"
        except Exception as exc:
            return f"ERROR ({type(exc).__name__}): {exc}"
        return _serialize_json_result(payload, limit=_MAX_INSPECTION_RESULT_CHARS)

    return StructuredTool.from_function(
        func=inspect_score_attributes,
        name="inspect_score_attributes",
        description=(
            "inspect_score_attributes(parts=None, bar_range=None, "
            "measure_numbers=None, voices=None)\n\n"
            "Return scoped time, concert key, tempo, clef, per-part displayed "
            "key, and structural score attributes without exact event rows. "
            "Use this for notation setup or layout context; use "
            "inspect_score_region when note, rhythm, or voice content matters."
        ),
    )


def _extract_attribute_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Project a bar-result payload down to score attributes only."""
    bars = result.get("bars") if isinstance(result, dict) else []
    payload = {
        "retrieval": "attribute inspection",
        "bars": [],
    }
    if not isinstance(bars, list):
        return payload

    for bar in bars:
        if not isinstance(bar, dict):
            continue
        bar_payload = {
            "measure_number": bar.get("measure_number"),
            "notation": bar.get("notation", {}),
            "parts": [],
        }
        parts = bar.get("parts", [])
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                part_payload = {
                    "part_index": part.get("part_index"),
                    "part_name": part.get("part_name"),
                }
                if part.get("hand") is not None:
                    part_payload["hand"] = part.get("hand")
                if part.get("notation"):
                    part_payload["notation"] = part.get("notation")
                bar_payload["parts"].append(part_payload)
        payload["bars"].append(bar_payload)
    return payload


def make_tool_search_tool(
    catalog: ToolCatalog,
    expansion_requests: ToolExpansionRequests,
    tools_by_name: dict[str, StructuredTool],
    *,
    core_tool_names: set[str],
) -> StructuredTool:
    """Build the always-available tool/capability search tool."""
    def tool_search(
        query: str = "",
        tool_names: Optional[list[str]] = None,
        limit: int = DEFAULT_TOOL_SEARCH_LIMIT,
    ) -> str:
        """Search all available tool capabilities and load returned tools."""
        safe_limit = max(1, min(int(limit), MAX_TOOL_SEARCH_LIMIT))
        valid_names, invalid_names = _resolve_tool_names(tools_by_name, tool_names)
        catalog_names = [
            name
            for name in valid_names
            if name in catalog.names
        ]
        core_names = [
            name
            for name in valid_names
            if name in core_tool_names and name not in catalog.names
        ]

        entries = []
        for name in core_names:
            tool = tools_by_name.get(name)
            if tool is None:
                continue
            entries.append(_core_tool_match_payload(tool, core_tool_names=core_tool_names))
            if len(entries) >= safe_limit:
                break

        for entry in catalog.search(
            query,
            tool_names=catalog_names,
            limit=safe_limit,
        ):
            if len(entries) >= safe_limit:
                break
            loaded = entry.name in tools_by_name
            item = {
                "name": entry.name,
                "description": entry.description,
                "tags": entry.tags,
                "already_core": entry.name in core_tool_names,
                "loaded": loaded,
            }
            entries.append(item)

        loaded_tools = [
            item["name"]
            for item in entries
            if item["loaded"] and not item["already_core"]
        ]
        expansion_requests.add(loaded_tools)

        payload = {
            "query": query,
            "matches": entries,
            "loaded_tools": loaded_tools,
            "invalid_tool_names": invalid_names,
        }
        return _serialize_json_result(payload, limit=_MAX_INSPECTION_RESULT_CHARS)

    return StructuredTool.from_function(
        func=tool_search,
        name="tool_search",
        description=(
            "tool_search(query='', tool_names=None, limit=5)\n\n"
            "Search all ScoreSpeak tools/capabilities by natural language, "
            "return compact names and short descriptions, and load returned "
            "tools so they are available on the next model step. Use "
            "tool_names for exact lookup by name, for example "
            "tool_names=['add_notes'] when adding one or more notes in one "
            "measure, part, and voice."
        ),
    )


def make_tool_from_record(
    score_state: ScoreSpeak,
    record: MethodRecord,
    mutation_recorder: MutationRecorder | None = None,
) -> Optional[StructuredTool]:
    """Wrap a single retrieved method as a LangChain ``StructuredTool``.

    Args:
        score_state: Live ScoreSpeak instance that the tool will mutate.
        record: Method record produced by
            :class:`scorespeak.retrieval.MethodIndex`.

    Returns:
        A :class:`StructuredTool` bound to ``score_state``, or ``None`` if
        the method cannot be introspected or its signature cannot be turned
        into a pydantic schema.  Callers should skip ``None`` results.
    """
    if record.name in AGENT_EXCLUDED_TOOL_NAMES:
        return None

    method = getattr(score_state, record.name, None)
    if method is None or not callable(method):
        logger.warning("Agent tools: method %s not found on ScoreSpeak", record.name)
        return None

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "Agent tools: cannot inspect signature of %s: %s", record.name, exc
        )
        return None

    try:
        type_hints = typing.get_type_hints(method)
    except Exception:
        type_hints = {}

    try:
        args_schema = _build_args_schema(
            record.name,
            signature,
            type_hints=type_hints,
            descriptions=_extract_arg_descriptions(record.docstring),
        )
    except Exception as exc:
        logger.warning(
            "Agent tools: cannot build args schema for %s: %s", record.name, exc
        )
        return None

    signature_text = _format_signature_for_description(signature)
    doc_snippet = _short_docstring(record.docstring)
    description = f"{record.name}{signature_text}"
    if doc_snippet:
        description = f"{description}\n\n{doc_snippet}"

    invoker = _build_invoker(method, record.name, mutation_recorder)

    return StructuredTool.from_function(
        func=invoker,
        name=record.name,
        description=description,
        args_schema=args_schema,
    )


def make_tools_from_records(
    score_state: ScoreSpeak,
    records: list[MethodRecord],
    mutation_recorder: MutationRecorder | None = None,
) -> list[StructuredTool]:
    """Wrap every record in ``records`` as a tool; silently skip failures.

    Duplicate names are de-duplicated (last record wins) so an agent is
    never given two tools with the same name.
    """
    by_name: dict[str, StructuredTool] = {}
    for record in records:
        tool = make_tool_from_record(score_state, record, mutation_recorder)
        if tool is None:
            continue
        by_name[tool.name] = tool
    return list(by_name.values())
