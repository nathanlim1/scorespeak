"""Benchmark edit action registry and application.

Benchmark edit actions are the dataset's stable expected-score construction
contract.  They are intentionally separate from agent-visible tools even when
the initial implementation delegates to a similarly named ``ScoreSpeak`` method.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from music21 import chord as m21chord
from music21 import duration as m21duration
from music21 import note as m21note
from music21 import tie as m21tie

from scorespeak import ScoreSpeak
from scorespeak.music.validation import (
    normalize_duration,
    normalize_pitch,
    validate_voice_number,
)


@dataclass
class BenchmarkEditAction:
    """One precise-edit benchmark edit action."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkEditAction":
        """Create an action from a JSON-decoded mapping."""
        return cls(name=str(data["name"]), args=dict(data.get("args") or {}))

    def to_dict(self) -> dict[str, Any]:
        """Return this action as a JSON-safe dictionary."""
        return asdict(self)


PreciseEditAction = BenchmarkEditAction


@dataclass(frozen=True)
class PreciseEditCase:
    """One public precise-edit benchmark case."""

    public_case_id: str
    base_score_id: str
    base_musicxml_path: str
    prompt: str
    expected_edit_actions: list[BenchmarkEditAction]
    tags: list[str]
    difficulty: str


@dataclass(frozen=True)
class BenchmarkActionParameter:
    """One benchmark action parameter descriptor."""

    name: str
    required: bool
    annotation: str = "Any"
    default: Any = None

    def to_schema(self) -> dict[str, Any]:
        """Return a JSON-safe schema fragment for this parameter."""
        payload = {
            "name": self.name,
            "required": self.required,
            "annotation": self.annotation,
        }
        if not self.required:
            payload["default"] = self.default
        return payload


@dataclass(frozen=True)
class BenchmarkActionSpec:
    """Static benchmark action contract."""

    name: str
    description: str
    parameters: tuple[BenchmarkActionParameter, ...]
    delegate_ignored_names: frozenset[str] = frozenset()

    @property
    def required_names(self) -> set[str]:
        """Return required parameter names."""
        return {parameter.name for parameter in self.parameters if parameter.required}

    @property
    def accepted_names(self) -> set[str]:
        """Return all accepted parameter names."""
        return {parameter.name for parameter in self.parameters}

    def to_schema(self) -> dict[str, Any]:
        """Return a JSON-safe schema for generation and review tooling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [parameter.to_schema() for parameter in self.parameters],
        }


def _parameter(name: str, required: bool, annotation: str = "Any", default: Any = None) -> BenchmarkActionParameter:
    """Create a benchmark action parameter descriptor."""
    return BenchmarkActionParameter(
        name=name,
        required=required,
        annotation=annotation,
        default=default,
    )


def _spec(
    name: str,
    required: tuple[str, ...] = (),
    optional: tuple[tuple[str, Any] | str, ...] = (),
    description: str | None = None,
    delegate_ignored: tuple[str, ...] = (),
) -> BenchmarkActionSpec:
    """Create a static action spec from required and optional names."""
    parameters = [_parameter(parameter, True) for parameter in required]
    for parameter in optional:
        if isinstance(parameter, tuple):
            parameter_name, default = parameter
        else:
            parameter_name, default = parameter, None
        parameters.append(_parameter(parameter_name, False, default=default))
    return BenchmarkActionSpec(
        name=name,
        description=description or f"Apply benchmark edit action {name}.",
        parameters=tuple(parameters),
        delegate_ignored_names=frozenset(delegate_ignored),
    )


ACTION_SPECS: dict[str, BenchmarkActionSpec] = {
    "delete_measure": _spec("delete_measure", ("measure_number",)),
    "delete_measures": _spec("delete_measures", ("start", "end")),
    "insert_measure": _spec("insert_measure", ("before",), (("count", 1),)),
    "add_measures": _spec("add_measures", optional=(("count", 1),)),
    "add_part": _spec(
        "add_part",
        optional=(("name", None), ("instrument", "piano"), ("clef_type", None), ("index", None)),
    ),
    "remove_part": _spec("remove_part", ("part",)),
    "add_chord": _spec(
        "add_chord",
        ("pitches",),
        (("duration", "quarter"), ("measure", None), ("beat", None), ("part", None), ("voice", 1), ("dots", 0)),
    ),
    "add_grace_note": _spec(
        "add_grace_note",
        ("pitch",),
        (
            ("duration", "eighth"),
            ("measure", None),
            ("beat", None),
            ("part", None),
            ("voice", 1),
            ("slash", True),
            ("slur_to_principal", False),
        ),
    ),
    "remove_grace_note": _spec(
        "remove_grace_note",
        ("measure",),
        (("beat", 1.0), ("part", None), ("voice", 1), ("pitch", None)),
    ),
    "add_note": _spec(
        "add_note",
        ("pitch",),
        (("duration", "quarter"), ("measure", None), ("beat", None), ("part", None), ("voice", 1), ("dots", 0)),
    ),
    "remove_note": _spec(
        "remove_note",
        ("measure", "beat"),
        (("part", None), ("voice", 1), ("pitch", None)),
    ),
    "replace_note": _spec(
        "replace_note",
        ("measure", "beat"),
        (("new_pitch", None), ("new_duration", None), ("part", None), ("voice", 1)),
    ),
    "add_rest": _spec(
        "add_rest",
        optional=(("duration", "quarter"), ("measure", None), ("beat", None), ("part", None), ("voice", 1), ("dots", 0)),
    ),
    "add_tie": _spec(
        "add_tie",
        ("measure", "beat"),
        (("part", None), ("voice", 1), ("tie_type", "start")),
    ),
    "remove_tie": _spec("remove_tie", ("measure", "beat"), (("part", None), ("voice", 1))),
    "add_tuplet": _spec(
        "add_tuplet",
        ("pitches_and_durations", "actual_notes", "normal_notes"),
        (("measure", None), ("beat", None), ("part", None), ("voice", 1)),
    ),
    "remove_tuplet": _spec(
        "remove_tuplet",
        ("measure", "beat"),
        (("part", None), ("voice", 1), ("actual_notes", None), ("normal_notes", None)),
    ),
    "set_barline": _spec("set_barline", ("barline_type", "measure_number")),
    "set_clef": _spec("set_clef", ("clef_type", "measure_number"), (("part", None),)),
    "add_coda": _spec("add_coda", ("measure_number",)),
    "add_da_capo": _spec("add_da_capo", ("measure_number",), (("al", None),)),
    "add_dal_segno": _spec("add_dal_segno", ("measure_number",), (("al", None),)),
    "add_fine": _spec("add_fine", ("measure_number",)),
    "set_key_signature": _spec("set_key_signature", ("key_signature", "measure_number"), (("transpose_existing", False),)),
    "remove_navigation_mark": _spec("remove_navigation_mark", ("mark_type", "measure_number")),
    "set_pickup_measure": _spec("set_pickup_measure", ("duration",)),
    "add_repeat": _spec("add_repeat", ("start_measure", "end_measure"), (("times", 2),)),
    "remove_repeat": _spec("remove_repeat", ("start_measure", "end_measure")),
    "add_segno": _spec("add_segno", ("measure_number",)),
    "add_to_coda": _spec("add_to_coda", ("measure_number",)),
    "set_time_signature": _spec("set_time_signature", ("time_signature", "measure_number")),
    "add_articulation": _spec(
        "add_articulation",
        ("articulation_type", "measure_number"),
        (("beat", 1.0), ("part", None), ("voice", 1)),
    ),
    "remove_articulation": _spec(
        "remove_articulation",
        ("articulation_type", "measure_number"),
        (("beat", 1.0), ("part", None), ("voice", 1)),
    ),
    "add_chord_symbol": _spec("add_chord_symbol", ("symbol", "measure_number"), (("beat", 1.0), ("part", None))),
    "remove_chord_symbol": _spec("remove_chord_symbol", ("measure_number",), (("beat", 1.0), ("part", None))),
    "add_dynamic": _spec("add_dynamic", ("level", "measure_number"), (("beat", 1.0), ("part", None))),
    "remove_dynamic": _spec("remove_dynamic", ("measure_number",), (("beat", 1.0), ("part", None))),
    "add_hairpin": _spec(
        "add_hairpin",
        ("hairpin_type", "start_measure", "start_beat", "end_measure", "end_beat"),
        (("part", None),),
    ),
    "remove_hairpin": _spec("remove_hairpin", ("start_measure",), (("start_beat", 1.0), ("part", None))),
    "add_rehearsal_mark": _spec("add_rehearsal_mark", ("text", "measure_number")),
    "remove_rehearsal_mark": _spec("remove_rehearsal_mark", ("measure_number",), (("text", None),)),
    "add_slur": _spec(
        "add_slur",
        ("start_measure", "start_beat", "end_measure", "end_beat"),
        (("part", None), ("voice", 1)),
    ),
    "remove_slur": _spec("remove_slur", ("start_measure",), (("start_beat", 1.0), ("part", None), ("voice", 1))),
    "set_tempo": _spec("set_tempo", ("bpm",), (("measure_number", 1), ("beat", 1.0), ("part", None), ("text", None))),
    "add_text_expression": _spec("add_text_expression", ("text", "measure_number"), (("beat", 1.0), ("part", None))),
    "remove_text_expression": _spec("remove_text_expression", ("measure_number",), (("beat", 1.0), ("part", None), ("text", None))),
    "add_arpeggio": _spec("add_arpeggio", ("measure_number",), (("beat", 1.0), ("part", None), ("voice", 1))),
    "remove_arpeggio": _spec("remove_arpeggio", ("measure_number",), (("beat", 1.0), ("part", None), ("voice", 1))),
    "add_ending_bracket": _spec(
        "add_ending_bracket",
        ("number", "start_measure", "end_measure"),
        (("label", None),),
        delegate_ignored=("label",),
    ),
    "remove_ending_bracket": _spec("remove_ending_bracket", ("number", "start_measure")),
    "add_fingering": _spec(
        "add_fingering",
        ("finger_number", "measure_number"),
        (("beat", 1.0), ("part", None), ("voice", 1), ("substitution", False)),
    ),
    "remove_fingering": _spec("remove_fingering", ("measure_number",), (("beat", 1.0), ("part", None), ("voice", 1))),
    "add_glissando": _spec(
        "add_glissando",
        ("start_measure", "start_beat", "end_measure", "end_beat"),
        (("part", None), ("voice", 1), ("line_type", "wavy"), ("label", None)),
    ),
    "remove_glissando": _spec("remove_glissando", ("start_measure", "start_beat"), (("part", None), ("voice", 1))),
    "add_lyric": _spec(
        "add_lyric",
        ("text", "measure_number"),
        (("beat", 1.0), ("part", None), ("voice", 1), ("lyric_number", 1), ("syllabic", None), ("identifier", None)),
    ),
    "remove_lyric": _spec(
        "remove_lyric",
        ("measure_number",),
        (("beat", 1.0), ("part", None), ("voice", 1), ("lyric_number", 1)),
    ),
    "add_ornament": _spec(
        "add_ornament",
        ("ornament_type", "measure_number"),
        (("beat", 1.0), ("part", None), ("voice", 1), ("tremolo_marks", 1)),
    ),
    "remove_ornament": _spec(
        "remove_ornament",
        ("ornament_type", "measure_number"),
        (("beat", 1.0), ("part", None), ("voice", 1)),
    ),
    "add_ottava": _spec(
        "add_ottava",
        ("ottava_type", "start_measure", "start_beat", "end_measure", "end_beat"),
        (("part", None), ("voice", 1), ("rewrite_pitches", False), ("placement", "above")),
    ),
    "remove_ottava": _spec(
        "remove_ottava",
        ("start_measure", "start_beat"),
        (("part", None), ("voice", 1), ("rewrite_pitches", False)),
    ),
    "add_pedal": _spec("add_pedal", ("start_measure", "start_beat", "end_measure", "end_beat"), (("part", None),)),
    "remove_pedal": _spec("remove_pedal", ("start_measure", "start_beat"), (("part", None),)),
    "set_composer": _spec("set_composer", ("composer",)),
    "add_page_break": _spec("add_page_break", ("measure_number",)),
    "remove_page_break": _spec("remove_page_break", ("measure_number",)),
    "set_subtitle": _spec("set_subtitle", ("subtitle",)),
    "add_system_break": _spec("add_system_break", ("measure_number",)),
    "remove_system_break": _spec("remove_system_break", ("measure_number",)),
    "set_title": _spec("set_title", ("title",)),
    "transpose": _spec("transpose", ("interval",), (("part", None), ("start_measure", None), ("end_measure", None))),
    "transpose_to_concert_pitch": _spec("transpose_to_concert_pitch", optional=(("part", None),)),
    "transpose_to_written_pitch": _spec("transpose_to_written_pitch", optional=(("part", None),)),
}


def benchmark_action_names() -> set[str]:
    """Return all registered benchmark action names."""
    return set(ACTION_SPECS)


def benchmark_action_schemas() -> list[dict[str, Any]]:
    """Return benchmark action schemas in stable name order."""
    return [ACTION_SPECS[name].to_schema() for name in sorted(ACTION_SPECS)]


def benchmark_action_schema_hash() -> str:
    """Return a stable digest for the benchmark action schema."""
    encoded = json.dumps(benchmark_action_schemas(), sort_keys=True, default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def action_to_dict(action: BenchmarkEditAction) -> dict[str, Any]:
    """Return a stable JSON-safe action dict."""
    return asdict(action)


def _benchmark_voice(args: dict[str, Any]) -> int:
    """Return the validated benchmark voice argument."""
    raw_voice = args.get("voice")
    if raw_voice is None:
        raw_voice = 1
    return validate_voice_number(raw_voice)


def _benchmark_v1_normalize_rests(
    score_state: ScoreSpeak,
    part_obj: Any,
    part_idx: int,
    measure_number: int,
    container: Any,
    voice: int,
) -> None:
    """Fill true rhythm gaps while preserving existing rest spelling."""
    measure_obj = score_state._resolve_measure(part_obj, measure_number)
    score_state._normalize_rests_after_rhythm_edit(
        part_obj,
        part_idx,
        measure_obj,
        measure_number,
        container,
        voice,
    )


def _benchmark_v1_refresh_measure_beams(
    score_state: ScoreSpeak,
    part_obj: Any,
    measure_number: int,
) -> None:
    """Recompute beams for one benchmark-edited measure."""
    measure_obj = score_state._resolve_measure(part_obj, measure_number)
    score_state._refresh_measure_beams(measure_obj)


def _benchmark_v1_remove_overlapped_rests(
    score_state: ScoreSpeak,
    container: Any,
    offset: float,
    quarter_length: float,
    exclude: Any = None,
) -> None:
    """Remove rest coverage that conflicts with a benchmark rhythm edit."""
    new_end = offset + quarter_length
    rests = []
    for element in list(container.getElementsByClass(m21note.Rest)):
        if element is exclude:
            continue
        event_range = score_state._rhythm_event_range(container, element)
        if event_range is None:
            continue
        if not score_state._event_ranges_overlap(offset, new_end, event_range):
            continue
        rests.append(element)

    for rest in rests:
        if score_state._is_hidden_rest(rest):
            score_state._replace_rest_with_outside_segments(
                container,
                rest,
                offset,
                new_end,
            )
        else:
            container.remove(rest)


def _benchmark_v1_validate_no_non_rest_overlap(
    score_state: ScoreSpeak,
    container: Any,
    offset: float,
    quarter_length: float,
    measure_number: int,
    beat: float,
    exclude: Any = None,
    action_label: str = "replace",
) -> None:
    """Raise if a benchmark edit overlaps a non-rest event."""
    new_end = offset + quarter_length
    for event_range in score_state._occupied_event_ranges(container, exclude=exclude):
        if event_range["kind"] == "rest":
            continue
        existing_start = float(event_range["offset"])
        existing_end = float(event_range["end"])
        overlaps = (
            offset < existing_end - 1e-9
            and new_end > existing_start + 1e-9
        )
        if overlaps:
            raise ValueError(
                f"Cannot {action_label} event at measure {measure_number}, "
                f"beat {beat}: "
                f"new duration would overlap existing {event_range['label']} "
                f"from beat {event_range['beat']} to {existing_end + 1.0:g}."
            )


def _benchmark_v1_add_note(score_state: ScoreSpeak, args: dict[str, Any]) -> Any:
    """Apply benchmark v1 add_note semantics."""
    voice = _benchmark_voice(args)
    pitch_obj = normalize_pitch(args["pitch"])
    dur_obj = normalize_duration(
        args.get("duration", "quarter"),
        dots=int(args.get("dots") or 0),
    )
    part_obj, part_idx, measure_number, container, offset, beat_pos = (
        _benchmark_v1_insert_context(score_state, args, dur_obj.quarterLength)
    )

    _benchmark_v1_remove_overlapped_rests(
        score_state,
        container,
        offset,
        dur_obj.quarterLength,
    )
    new_note = m21note.Note(pitch=pitch_obj, duration=dur_obj)
    container.insert(offset, new_note)
    _benchmark_v1_normalize_rests(
        score_state,
        part_obj,
        part_idx,
        measure_number,
        container,
        voice,
    )
    _benchmark_v1_refresh_measure_beams(score_state, part_obj, measure_number)
    score_state._refresh_measure_accidentals(part_obj, measure_number)

    warning = score_state._check_instrument_range(part_obj, pitch_obj)
    description = (
        f"Added {pitch_obj.nameWithOctave} {dur_obj.type} note "
        f"at measure {measure_number}, beat {beat_pos}"
    )
    if warning:
        description += f" — {warning}"
    return _benchmark_v1_result(
        description,
        {
            "pitch": pitch_obj.nameWithOctave,
            "duration": dur_obj.type,
            "quarter_length": dur_obj.quarterLength,
            "measure": measure_number,
            "beat": beat_pos,
            "part": part_idx,
            "voice": voice,
            "dots": dur_obj.dots,
            "warning": warning,
        },
    )


def _benchmark_v1_add_rest(score_state: ScoreSpeak, args: dict[str, Any]) -> Any:
    """Apply benchmark v1 add_rest semantics."""
    voice = _benchmark_voice(args)
    dur_obj = normalize_duration(
        args.get("duration", "quarter"),
        dots=int(args.get("dots") or 0),
    )
    part_obj, part_idx, measure_number, container, offset, beat_pos = (
        _benchmark_v1_insert_context(score_state, args, dur_obj.quarterLength)
    )
    _benchmark_v1_validate_no_non_rest_overlap(
        score_state,
        container,
        offset,
        dur_obj.quarterLength,
        measure_number,
        beat_pos,
        action_label="add",
    )

    score_state._remove_rests_in_range_preserving_outside(
        container,
        offset,
        offset + float(dur_obj.quarterLength),
        measure_number,
        part_idx,
        voice,
    )
    rest = m21note.Rest(duration=dur_obj)
    container.insert(offset, rest)
    _benchmark_v1_normalize_rests(
        score_state,
        part_obj,
        part_idx,
        measure_number,
        container,
        voice,
    )
    _benchmark_v1_refresh_measure_beams(score_state, part_obj, measure_number)
    return _benchmark_v1_result(
        f"Added {dur_obj.type} rest at measure {measure_number}, beat {beat_pos}",
        {
            "duration": dur_obj.type,
            "quarter_length": dur_obj.quarterLength,
            "measure": measure_number,
            "beat": beat_pos,
            "part": part_idx,
            "voice": voice,
            "dots": dur_obj.dots,
        },
    )


def _benchmark_v1_add_chord(score_state: ScoreSpeak, args: dict[str, Any]) -> Any:
    """Apply benchmark v1 add_chord semantics."""
    voice = _benchmark_voice(args)
    pitches = args["pitches"]
    if not pitches:
        raise ValueError("A chord requires at least one pitch.")
    pitch_objs = [normalize_pitch(pitch) for pitch in pitches]
    dur_obj = normalize_duration(
        args.get("duration", "quarter"),
        dots=int(args.get("dots") or 0),
    )
    part_obj, part_idx, measure_number, container, offset, beat_pos = (
        _benchmark_v1_insert_context(score_state, args, dur_obj.quarterLength)
    )
    _benchmark_v1_validate_no_non_rest_overlap(
        score_state,
        container,
        offset,
        dur_obj.quarterLength,
        measure_number,
        beat_pos,
        action_label="add",
    )

    _benchmark_v1_remove_overlapped_rests(
        score_state,
        container,
        offset,
        dur_obj.quarterLength,
    )
    warnings = []
    for pitch_obj in pitch_objs:
        warning = score_state._check_instrument_range(part_obj, pitch_obj)
        if warning:
            warnings.append(warning)

    chord_obj = m21chord.Chord(pitch_objs, duration=dur_obj)
    container.insert(offset, chord_obj)
    _benchmark_v1_normalize_rests(
        score_state,
        part_obj,
        part_idx,
        measure_number,
        container,
        voice,
    )
    _benchmark_v1_refresh_measure_beams(score_state, part_obj, measure_number)
    score_state._refresh_measure_accidentals(part_obj, measure_number)

    pitch_names = [pitch_obj.nameWithOctave for pitch_obj in pitch_objs]
    warning_text = "; ".join(warnings) if warnings else None
    description = (
        f"Added chord [{', '.join(pitch_names)}] "
        f"at measure {measure_number}, beat {beat_pos}"
    )
    if warning_text:
        description += f" — {warning_text}"
    return _benchmark_v1_result(
        description,
        {
            "pitches": pitch_names,
            "duration": dur_obj.type,
            "quarter_length": dur_obj.quarterLength,
            "measure": measure_number,
            "beat": beat_pos,
            "part": part_idx,
            "voice": voice,
            "dots": dur_obj.dots,
            "warning": warning_text,
        },
    )


def _benchmark_v1_add_tuplet(score_state: ScoreSpeak, args: dict[str, Any]) -> Any:
    """Apply benchmark v1 add_tuplet semantics."""
    voice = _benchmark_voice(args)
    actual_notes = args["actual_notes"]
    normal_notes = args["normal_notes"]
    if (
        isinstance(actual_notes, bool)
        or isinstance(normal_notes, bool)
        or not isinstance(actual_notes, int)
        or not isinstance(normal_notes, int)
        or actual_notes <= 0
        or normal_notes <= 0
    ):
        raise ValueError("actual_notes and normal_notes must be positive integers.")

    pitches_and_durations = args["pitches_and_durations"]
    if len(pitches_and_durations) != actual_notes:
        raise ValueError(
            f"Expected {actual_notes} notes for this tuplet, "
            f"but received {len(pitches_and_durations)}."
        )

    notes = []
    total_quarter_length = 0.0
    for index, (pitch_in, duration_in) in enumerate(pitches_and_durations):
        pitch_obj = normalize_pitch(pitch_in)
        duration_obj = normalize_duration(duration_in)
        note_obj = m21note.Note(pitch=pitch_obj, duration=duration_obj)
        tuplet_boundary = None
        if index == 0:
            tuplet_boundary = "start"
        elif index == len(pitches_and_durations) - 1:
            tuplet_boundary = "stop"
        tuplet = m21duration.Tuplet(
            numberNotesActual=actual_notes,
            numberNotesNormal=normal_notes,
            type=tuplet_boundary,
            bracket=True,
            placement="above",
            tupletActualShow="number",
            tupletNormalShow=None,
        )
        note_obj.duration.appendTuplet(tuplet)
        total_quarter_length += note_obj.duration.quarterLength
        notes.append(note_obj)

    part_obj, part_idx, measure_number, container, offset, beat_pos = (
        _benchmark_v1_insert_context(
            score_state,
            args,
            total_quarter_length,
        )
    )
    _benchmark_v1_validate_no_non_rest_overlap(
        score_state,
        container,
        offset,
        total_quarter_length,
        measure_number,
        beat_pos,
        action_label="add",
    )

    _benchmark_v1_remove_overlapped_rests(
        score_state,
        container,
        offset,
        total_quarter_length,
    )
    current_offset = offset
    for note_obj in notes:
        container.insert(current_offset, note_obj)
        current_offset += note_obj.duration.quarterLength
    _benchmark_v1_normalize_rests(
        score_state,
        part_obj,
        part_idx,
        measure_number,
        container,
        voice,
    )
    _benchmark_v1_refresh_measure_beams(score_state, part_obj, measure_number)
    score_state._refresh_measure_accidentals(part_obj, measure_number)

    pitch_names = [note_obj.pitch.nameWithOctave for note_obj in notes]
    return _benchmark_v1_result(
        (
            f"Added {actual_notes}:{normal_notes} tuplet "
            f"[{', '.join(pitch_names)}] at measure {measure_number}, "
            f"beat {beat_pos}"
        ),
        {
            "pitches": pitch_names,
            "actual_notes": actual_notes,
            "normal_notes": normal_notes,
            "total_quarter_length": total_quarter_length,
            "measure": measure_number,
            "beat": beat_pos,
            "part": part_idx,
            "voice": voice,
        },
    )


def _benchmark_v1_remove_tuplet(score_state: ScoreSpeak, args: dict[str, Any]) -> Any:
    """Apply benchmark v1 remove_tuplet semantics without public auto-fill."""
    voice = _benchmark_voice(args)
    measure = int(args["measure"])
    beat = float(args["beat"])
    part_obj, part_idx = score_state._resolve_part(args.get("part"))
    measure_obj = score_state._resolve_measure(part_obj, measure)
    container = score_state._get_voice_or_measure(measure_obj, voice)
    start_offset = beat - 1.0

    events = sorted(
        [
            element
            for element in container.getElementsByClass(m21note.GeneralNote)
            if not (
                isinstance(element, m21note.Rest)
                and getattr(element.style, "hideObjectOnPrint", False)
            )
        ],
        key=lambda element: container.elementOffset(element),
    )
    first = None
    found_event_at_offset = False
    for element in events:
        if abs(container.elementOffset(element) - start_offset) < 1e-9:
            found_event_at_offset = True
            if element.duration.tuplets:
                first = element
                break

    if first is None:
        if found_event_at_offset:
            raise ValueError(
                f"No tuplet starts at beat {beat} in measure {measure}, "
                f"voice {voice}; an event exists there but it is not the "
                "start of a tuplet."
            )
        raise ValueError(
            f"No note found at beat {beat} in measure {measure}, voice {voice}."
        )

    tuplet = first.duration.tuplets[0]
    ratio_actual = tuplet.numberNotesActual
    ratio_normal = tuplet.numberNotesNormal
    actual_notes = args.get("actual_notes")
    normal_notes = args.get("normal_notes")
    if actual_notes is not None and ratio_actual != actual_notes:
        raise ValueError(
            f"Tuplet at measure {measure} has {ratio_actual} actual notes, "
            f"not {actual_notes}."
        )
    if normal_notes is not None and ratio_normal != normal_notes:
        raise ValueError(
            f"Tuplet at measure {measure} has {ratio_normal} normal notes, "
            f"not {normal_notes}."
        )

    to_remove = []
    collecting = False
    expected_offset = start_offset
    for element in events:
        element_offset = container.elementOffset(element)
        if not collecting:
            if element is not first:
                continue
            collecting = True
        elif abs(element_offset - expected_offset) > 1e-9:
            if element_offset < expected_offset:
                continue
            break

        if not element.duration.tuplets:
            break
        element_tuplet = element.duration.tuplets[0]
        if (
            element_tuplet.numberNotesActual != ratio_actual
            or element_tuplet.numberNotesNormal != ratio_normal
        ):
            break
        to_remove.append(element)
        expected_offset = element_offset + element.duration.quarterLength
        if len(to_remove) >= ratio_actual:
            break

    if len(to_remove) != ratio_actual:
        raise ValueError(
            f"Could not find the complete {ratio_actual}:{ratio_normal} "
            f"tuplet group starting at measure {measure}, beat {beat}."
        )

    for element in to_remove:
        container.remove(element)
    _benchmark_v1_refresh_measure_beams(score_state, part_obj, measure)
    score_state._refresh_measure_accidentals(part_obj, measure)
    return _benchmark_v1_result(
        f"Removed {ratio_actual}:{ratio_normal} tuplet starting at "
        f"measure {measure}, beat {beat}",
        {
            "actual_notes": ratio_actual,
            "normal_notes": ratio_normal,
            "removed_notes": len(to_remove),
            "measure": measure,
            "beat": beat,
            "part": part_idx,
            "voice": voice,
        },
    )


def _benchmark_v1_insert_context(
    score_state: ScoreSpeak,
    args: dict[str, Any],
    quarter_length: float,
) -> tuple[Any, int, int, Any, float, float]:
    """Return the common insertion context for benchmark v1 note actions."""
    voice = _benchmark_voice(args)
    part_obj, part_idx = score_state._resolve_part(args.get("part"))
    measure_number = score_state._resolve_measure_number(
        part_obj,
        args.get("measure"),
    )
    measure_obj = score_state._resolve_measure(part_obj, measure_number)
    container = score_state._get_voice_or_measure(
        measure_obj,
        voice,
        create=True,
    )
    time_signature = score_state._get_active_time_signature_obj(
        part_obj,
        measure_number,
    )
    capacity = score_state._effective_measure_capacity(
        measure_obj,
        time_signature,
    )
    offset, beat_pos = score_state._resolve_and_validate_beat(
        container,
        args.get("beat"),
        quarter_length,
        time_signature,
        measure_number,
        capacity=capacity,
    )
    return part_obj, part_idx, measure_number, container, offset, beat_pos


def _benchmark_v1_remove_note(score_state: ScoreSpeak, args: dict[str, Any]) -> Any:
    """Apply benchmark v1 remove_note semantics."""
    measure = int(args["measure"])
    beat = float(args["beat"])
    voice = _benchmark_voice(args)
    part_obj, part_idx = score_state._resolve_part(args.get("part"))
    measure_obj = score_state._resolve_measure(part_obj, measure)
    container = score_state._get_voice_or_measure(measure_obj, voice)
    element = score_state._find_element_at_offset(container, beat - 1.0)
    if element is None:
        raise ValueError(f"No note or rest found at beat {beat} in measure {measure}.")

    pitch = args.get("pitch")
    if pitch is not None and isinstance(element, m21chord.Chord):
        refresh_next = score_state._element_tie_continues_to_next_measure(element)
        result = score_state._remove_pitch_from_chord(
            container,
            element,
            pitch,
            measure,
            beat,
            part_idx,
            voice,
        )
        score_state._refresh_changed_measure_accidentals(
            part_obj,
            measure,
            refresh_next=refresh_next,
        )
        return result

    if pitch is not None and isinstance(element, m21note.Note):
        pitch_obj = normalize_pitch(pitch)
        if element.pitch.nameWithOctave != pitch_obj.nameWithOctave:
            raise ValueError(
                f"Expected pitch {pitch_obj.nameWithOctave} but found "
                f"{element.pitch.nameWithOctave} at beat {beat} in measure {measure}."
            )

    description = score_state._describe_element(element)
    refresh_next = score_state._element_tie_continues_to_next_measure(element)
    container.remove(element)
    _benchmark_v1_normalize_rests(
        score_state,
        part_obj,
        part_idx,
        measure,
        container,
        voice,
    )
    _benchmark_v1_refresh_measure_beams(score_state, part_obj, measure)
    score_state._refresh_changed_measure_accidentals(
        part_obj,
        measure,
        refresh_next=refresh_next,
    )
    return _benchmark_v1_result(
        f"Removed {description} from measure {measure}, beat {beat}",
        {
            "measure": measure,
            "beat": beat,
            "part": part_idx,
            "voice": voice,
        },
    )


def _benchmark_v1_replace_note(score_state: ScoreSpeak, args: dict[str, Any]) -> Any:
    """Apply benchmark v1 replace_note semantics."""
    if args.get("new_pitch") is None and args.get("new_duration") is None:
        raise ValueError("At least one of new_pitch or new_duration must be provided.")

    measure = int(args["measure"])
    beat = float(args["beat"])
    voice = _benchmark_voice(args)
    part_obj, part_idx = score_state._resolve_part(args.get("part"))
    measure_obj = score_state._resolve_measure(part_obj, measure)
    container = score_state._get_voice_or_measure(measure_obj, voice)
    offset = beat - 1.0
    element = score_state._find_element_at_offset(container, offset)
    if element is None:
        raise ValueError(f"No note found at beat {beat} in measure {measure}.")
    if not isinstance(element, (m21note.Note, m21chord.Chord, m21note.Rest)):
        raise ValueError(
            f"Element at beat {beat} in measure {measure} is a "
            f"{type(element).__name__}, not a note, chord, or rest."
        )

    changes = []
    new_pitch = args.get("new_pitch")
    new_duration = args.get("new_duration")
    dur_obj = (
        normalize_duration(new_duration)
        if new_duration is not None
        else None
    )
    replaced_rest = isinstance(element, m21note.Rest)

    if replaced_rest and new_pitch is None:
        raise ValueError(
            f"Element at beat {beat} in measure {measure} is a Rest. "
            "Benchmark replace_note requires new_pitch when replacing a rest."
        )

    if dur_obj is not None:
        time_signature = score_state._get_active_time_signature_obj(part_obj, measure)
        capacity = score_state._effective_measure_capacity(
            measure_obj,
            time_signature,
        )
        score_state._validate_event_capacity(
            capacity,
            dur_obj.quarterLength,
            measure,
            beat_position=beat,
            ratio_string=time_signature.ratioString,
        )
        _benchmark_v1_validate_no_non_rest_overlap(
            score_state,
            container,
            offset,
            dur_obj.quarterLength,
            measure,
            beat,
            exclude=element,
        )

    if replaced_rest:
        pitch_obj = normalize_pitch(new_pitch)
        old_duration = element.duration.type
        target_duration = (
            dur_obj
            if dur_obj is not None
            else deepcopy(element.duration)
        )
        _benchmark_v1_remove_overlapped_rests(
            score_state,
            container,
            offset,
            float(target_duration.quarterLength),
        )
        element = m21note.Note(pitch=pitch_obj, duration=target_duration)
        container.insert(offset, element)
        changes.append(f"rest → {pitch_obj.nameWithOctave}")
        if dur_obj is not None:
            changes.append(f"duration {old_duration} → {dur_obj.type}")
    else:
        if new_pitch is not None:
            if isinstance(element, m21chord.Chord):
                raise ValueError(
                    "Cannot replace pitch of a chord directly. "
                    "Use remove_note with a specific pitch, then add_chord."
                )
            pitch_obj = normalize_pitch(new_pitch)
            old_name = element.pitch.nameWithOctave
            element.pitch = pitch_obj
            changes.append(f"pitch {old_name} → {pitch_obj.nameWithOctave}")

        if dur_obj is not None:
            _benchmark_v1_remove_overlapped_rests(
                score_state,
                container,
                offset,
                dur_obj.quarterLength,
                exclude=element,
            )
            old_duration = element.duration.type
            element.duration = dur_obj
            changes.append(f"duration {old_duration} → {dur_obj.type}")

    if replaced_rest or dur_obj is not None:
        _benchmark_v1_normalize_rests(
            score_state,
            part_obj,
            part_idx,
            measure,
            container,
            voice,
        )
        _benchmark_v1_refresh_measure_beams(score_state, part_obj, measure)
    score_state._refresh_changed_measure_accidentals(
        part_obj,
        measure,
        refresh_next=score_state._element_tie_continues_to_next_measure(element),
    )
    return _benchmark_v1_result(
        f"Replaced note at measure {measure}, beat {beat}: {', '.join(changes)}",
        {
            "measure": measure,
            "beat": beat,
            "part": part_idx,
            "voice": voice,
            "changes": changes,
        },
    )


def _benchmark_v1_add_tie(score_state: ScoreSpeak, args: dict[str, Any]) -> Any:
    """Apply benchmark v1 add_tie action with neighbor inference."""
    measure = int(args["measure"])
    beat = float(args["beat"])
    voice = _benchmark_voice(args)
    tie_type = str(args.get("tie_type") or "start")
    if tie_type not in {"start", "stop", "continue"}:
        raise ValueError(
            f"Invalid tie type '{tie_type}'. Must be 'start', 'stop', or 'continue'."
        )

    part_obj, part_idx = score_state._resolve_part(args.get("part"))
    measure_obj = score_state._resolve_measure(part_obj, measure)
    container = score_state._get_voice_or_measure(measure_obj, voice)
    element = score_state._find_element_at_offset(container, beat - 1.0)
    if element is None:
        raise ValueError(f"No note found at beat {beat} in measure {measure}.")
    if isinstance(element, m21note.Rest):
        raise ValueError(
            f"Cannot add a tie to a rest at measure {measure}, beat {beat}."
        )

    if tie_type == "start":
        return score_state.add_tie(
            measure=measure,
            beat=beat,
            part=part_idx,
            voice=voice,
        )

    if tie_type == "stop":
        previous = _benchmark_v1_require_adjacent_same_pitch_tie_event(
            score_state,
            part_obj,
            voice,
            measure,
            beat,
            element,
            "previous",
        )
        return score_state._add_adjacent_tie(
            part_obj,
            part_idx,
            int(previous["measure"]),
            float(previous["beat"]),
            voice,
            previous["element"],
            endpoint_label="previous",
            repair_existing_onward=True,
        )

    previous = _benchmark_v1_require_adjacent_same_pitch_tie_event(
        score_state,
        part_obj,
        voice,
        measure,
        beat,
        element,
        "previous",
    )
    next_event = _benchmark_v1_require_adjacent_same_pitch_tie_event(
        score_state,
        part_obj,
        voice,
        measure,
        beat,
        element,
        "next",
    )
    previous_element = previous["element"]
    next_element = next_event["element"]
    previous_tie = getattr(previous_element, "tie", None)
    previous_element.tie = m21tie.Tie(
        "continue"
        if previous_tie is not None and previous_tie.type == "continue"
        else "start"
    )
    element.tie = m21tie.Tie("continue")
    next_tie = getattr(next_element, "tie", None)
    next_element.tie = m21tie.Tie(
        "continue"
        if next_tie is not None and next_tie.type in {"start", "continue"}
        else "stop"
    )
    score_state._refresh_measure_and_next_accidentals(
        part_obj,
        int(previous["measure"]),
    )
    if int(next_event["measure"]) != int(previous["measure"]):
        score_state._refresh_measure_and_next_accidentals(
            part_obj,
            int(next_event["measure"]),
        )
    return _benchmark_v1_result(
        f"Added continue tie at measure {measure}, beat {beat}",
        {
            "measure": measure,
            "beat": beat,
            "tie_type": "continue",
            "part": part_idx,
            "voice": voice,
            "tied_positions": [
                {"measure": previous["measure"], "beat": previous["beat"]},
                {"measure": measure, "beat": beat},
                {"measure": next_event["measure"], "beat": next_event["beat"]},
            ],
        },
    )


def _benchmark_v1_require_adjacent_same_pitch_tie_event(
    score_state: ScoreSpeak,
    part_obj: Any,
    voice: int,
    measure: int,
    beat: float,
    element: m21note.GeneralNote,
    direction: str,
) -> dict[str, Any]:
    """Return the adjacent same-pitch event or fail benchmark materialization."""
    events = score_state._tie_voice_rhythmic_events(part_obj, voice)
    target_index = score_state._tie_event_index(events, measure, beat, element)
    neighbor_index = target_index + (1 if direction == "next" else -1)
    if neighbor_index < 0 or neighbor_index >= len(events):
        raise ValueError(
            f"Benchmark add_tie {direction} inference failed at measure "
            f"{measure}, beat {beat}: no adjacent rhythmic event exists."
        )

    signature = score_state._tie_pitch_signature(element)
    neighbor = events[neighbor_index]
    neighbor_element = neighbor["element"]
    if isinstance(neighbor_element, m21note.Rest):
        raise ValueError(
            f"Benchmark add_tie {direction} inference failed at measure "
            f"{measure}, beat {beat}: adjacent event is a rest."
        )
    neighbor_signature = score_state._tie_pitch_signature(neighbor_element)
    if neighbor_signature != signature:
        raise ValueError(
            f"Benchmark add_tie {direction} inference failed at measure "
            f"{measure}, beat {beat}: adjacent event has pitch "
            f"{', '.join(neighbor_signature)}, expected {', '.join(signature)}."
        )
    return neighbor


def _benchmark_v1_remove_tie(score_state: ScoreSpeak, args: dict[str, Any]) -> Any:
    """Apply benchmark v1 remove_tie action with chain-level semantics."""
    measure = int(args["measure"])
    beat = float(args["beat"])
    voice = _benchmark_voice(args)
    _part_obj, part_idx = score_state._resolve_part(args.get("part"))
    return score_state.remove_tie(
        start_measure=measure,
        start_beat=beat,
        part=part_idx,
        voice=voice,
    )


def _benchmark_v1_add_fingering(
    score_state: ScoreSpeak,
    args: dict[str, Any],
) -> Any:
    """Apply benchmark v1 add_fingering while accepting legacy substitution."""
    delegate_args = dict(args)
    delegate_args.pop("substitution", None)
    return score_state.add_fingering(**delegate_args)


def _benchmark_v1_result(description: str, details: dict[str, Any]) -> Any:
    """Create an OperationResult-shaped value without importing the dataclass."""
    from scorespeak.types import OperationResult

    return OperationResult(success=True, description=description, details=details)


_BENCHMARK_V1_HANDLERS = {
    "add_chord": _benchmark_v1_add_chord,
    "add_fingering": _benchmark_v1_add_fingering,
    "add_note": _benchmark_v1_add_note,
    "add_rest": _benchmark_v1_add_rest,
    "add_tie": _benchmark_v1_add_tie,
    "add_tuplet": _benchmark_v1_add_tuplet,
    "remove_note": _benchmark_v1_remove_note,
    "remove_tie": _benchmark_v1_remove_tie,
    "remove_tuplet": _benchmark_v1_remove_tuplet,
    "replace_note": _benchmark_v1_replace_note,
}


def apply_benchmark_edit_actions(
    score_state: ScoreSpeak,
    actions: Iterable[BenchmarkEditAction],
) -> list[str]:
    """Apply benchmark edit actions and return result descriptions."""
    results: list[str] = []
    for action in actions:
        spec = ACTION_SPECS.get(action.name)
        if spec is None:
            raise ValueError(f"Unknown benchmark edit action {action.name}")
        extra = sorted(set(action.args) - spec.accepted_names)
        if extra:
            names = ", ".join(extra)
            raise ValueError(f"Unexpected argument(s) for {action.name}: {names}")
        missing = sorted(spec.required_names - set(action.args))
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"Missing required argument(s) for {action.name}: {names}")
        delegate_args = {
            name: value
            for name, value in action.args.items()
            if name not in spec.delegate_ignored_names
        }
        handler = _BENCHMARK_V1_HANDLERS.get(action.name)
        if handler is not None:
            result = handler(score_state, delegate_args)
        else:
            method = getattr(score_state, action.name, None)
            if method is None or not callable(method):
                raise ValueError(
                    f"Benchmark action {action.name} has no ScoreSpeak handler"
                )
            result = method(**delegate_args)
        results.append(getattr(result, "description", str(result)))
    return results


def load_precise_edit_cases(csv_path: str | Path) -> list[PreciseEditCase]:
    """Load public precise-edit cases from ``datasets/precise_edit/cases.csv``."""
    path = Path(csv_path)
    cases: list[PreciseEditCase] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cases.append(_precise_case_from_row(row))
    return cases


def apply_precise_edit_case(
    case: PreciseEditCase,
    *,
    repository_root: str | Path = ".",
) -> tuple[ScoreSpeak, list[str]]:
    """Load a precise-edit base score and apply benchmark edit actions.

    Args:
        case: Public precise-edit case to replay.
        repository_root: Repository root used to resolve the case's relative
            ``base_musicxml_path``.

    Returns:
        The edited ``ScoreSpeak`` object and benchmark action descriptions.
    """
    root = Path(repository_root)
    score_state = ScoreSpeak.from_musicxml(root / case.base_musicxml_path)
    results = apply_benchmark_edit_actions(score_state, case.expected_edit_actions)
    return score_state, results


def apply_precise_edit_actions(
    score_state: ScoreSpeak,
    actions: Iterable[BenchmarkEditAction | dict[str, Any]],
) -> list[str]:
    """Apply public precise-edit benchmark actions to ``score_state``.

    This is the public precise-edit dataset action runner. It intentionally
    uses the separate benchmark edit-action semantics, not the agent-visible
    ``ScoreSpeak`` tool contract.
    """
    return apply_benchmark_edit_actions(
        score_state,
        [_normalize_precise_action(action) for action in actions],
    )


def materialize_precise_edit_musicxml(
    base_musicxml_path: str | Path,
    actions: Iterable[BenchmarkEditAction | dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Apply precise-edit benchmark actions and write expected MusicXML."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    score_state = ScoreSpeak.from_musicxml(base_musicxml_path)
    apply_precise_edit_actions(score_state, actions)
    score_state.to_musicxml(output)
    return output


def _precise_case_from_row(row: dict[str, str]) -> PreciseEditCase:
    """Convert one public precise-edit CSV row into a typed case."""
    actions = [
        BenchmarkEditAction.from_dict(action)
        for action in json.loads(row["expected_edit_actions_json"])
    ]
    tags = json.loads(row["tags_json"])
    return PreciseEditCase(
        public_case_id=row["public_case_id"],
        base_score_id=row["base_score_id"],
        base_musicxml_path=row["base_musicxml_path"],
        prompt=row["prompt"],
        expected_edit_actions=actions,
        tags=tags,
        difficulty=row["difficulty"],
    )


def _normalize_precise_action(
    action: BenchmarkEditAction | dict[str, Any],
) -> BenchmarkEditAction:
    """Return a benchmark edit action from an action object or mapping."""
    if isinstance(action, BenchmarkEditAction):
        return action
    return BenchmarkEditAction.from_dict(action)
