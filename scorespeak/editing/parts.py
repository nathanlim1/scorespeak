"""
Part management operations for ScoreSpeak.

Provides add/remove part operations with instrument specification,
automatic measure synchronization, and state inheritance.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Optional, Union

from music21 import clef as m21clef
from music21 import expressions as m21expressions
from music21 import instrument as m21instrument
from music21 import key as m21key
from music21 import layout as m21layout
from music21 import meter as m21meter
from music21 import note as m21note
from music21 import repeat as m21repeat
from music21 import spanner as m21spanner
from music21 import stream as m21stream
from music21 import tempo as m21tempo

from ..music.pitch_space import (
    concert_key_signature_for_stored_key,
    copy_key_signature,
    default_stored_pitch_space_for_part,
    normalize_instrument_label,
    stored_key_signature_for_concert_key,
)
from ..score.staff_groups import build_part_display_labels, detect_staff_groups
from ..types import OperationResult, PartInfo, ScorePartSpec
from ..music.validation import default_clef_for_instrument, make_clef


_TEXT_NORMALIZATION_PATTERN = re.compile(r"[^a-z0-9]+")

_INSTRUMENT_MAP: dict[str, type] = {
    "piano": m21instrument.Piano,
    "organ": m21instrument.Organ,
    "electric organ": m21instrument.ElectricOrgan,
    "harpsichord": m21instrument.Harpsichord,
    "celesta": m21instrument.Celesta,
    "violin": m21instrument.Violin,
    "viola": m21instrument.Viola,
    "cello": m21instrument.Violoncello,
    "contrabass": m21instrument.Contrabass,
    "double bass": m21instrument.Contrabass,
    "flute": m21instrument.Flute,
    "oboe": m21instrument.Oboe,
    "clarinet": m21instrument.Clarinet,
    "bassoon": m21instrument.Bassoon,
    "trumpet": m21instrument.Trumpet,
    "french horn": m21instrument.Horn,
    "horn": m21instrument.Horn,
    "trombone": m21instrument.Trombone,
    "tuba": m21instrument.Tuba,
    "soprano": m21instrument.Soprano,
    "alto": m21instrument.Alto,
    "tenor": m21instrument.Tenor,
    "baritone": m21instrument.Baritone,
    "bass": m21instrument.Bass,
    "guitar": m21instrument.AcousticGuitar,
    "acoustic guitar": m21instrument.AcousticGuitar,
    "electric guitar": m21instrument.ElectricGuitar,
    "harp": m21instrument.Harp,
    "timpani": m21instrument.Timpani,
    "xylophone": m21instrument.Xylophone,
    "marimba": m21instrument.Marimba,
    "vibraphone": m21instrument.Vibraphone,
    "glockenspiel": m21instrument.Glockenspiel,
    "snare drum": m21instrument.SnareDrum,
    "bass drum": m21instrument.BassDrum,
    "piccolo": m21instrument.Piccolo,
    "english horn": m21instrument.EnglishHorn,
    "alto saxophone": m21instrument.AltoSaxophone,
    "tenor saxophone": m21instrument.TenorSaxophone,
    "baritone saxophone": m21instrument.BaritoneSaxophone,
    "soprano saxophone": m21instrument.SopranoSaxophone,
}

_AUTO_TWO_STAFF_INSTRUMENTS: tuple[type[m21instrument.Instrument], ...] = (
    m21instrument.Piano,
    m21instrument.Harpsichord,
    m21instrument.Celesta,
    m21instrument.Harp,
)
_AUTO_THREE_STAFF_INSTRUMENTS: tuple[type[m21instrument.Instrument], ...] = (
    m21instrument.Organ,
    m21instrument.ElectricOrgan,
)
_AUTO_TWO_STAFF_NAMES = frozenset({"piano", "harpsichord", "celesta", "harp"})
_AUTO_THREE_STAFF_NAMES = frozenset({"organ", "electric organ"})
_GENERIC_TRANSPOSING_PART_NAMES = frozenset({
    "clarinet",
    "horn",
    "french horn",
    "trumpet",
})
_TRANSPOSING_FAMILY_NAMES = frozenset({
    "clarinet",
    "horn",
    "trumpet",
})
_PITCH_NAME_PATTERN = re.compile(r"\b[a-g](?:b|#)?\b", re.IGNORECASE)
_TRAILING_PART_NUMBER_PATTERN = re.compile(r"\s+(?:\d+|[ivx]+)\s*$", re.IGNORECASE)
_SCORE_MARK_CLASSES: tuple[type, ...] = (
    m21tempo.MetronomeMark,
    m21expressions.RehearsalMark,
    m21repeat.Coda,
    m21repeat.Segno,
    m21repeat.Fine,
    m21repeat.DaCapo,
    m21repeat.DaCapoAlFine,
    m21repeat.DaCapoAlCoda,
    m21repeat.DalSegno,
    m21repeat.DalSegnoAlFine,
    m21repeat.DalSegnoAlCoda,
)


@dataclass
class _NormalizedPartSpec:
    """Validated requested logical part."""

    name: str
    instrument: str
    instrument_obj: m21instrument.Instrument
    staff_clefs: tuple[m21clef.Clef, ...]


@dataclass
class _ExistingLogicalPart:
    """Existing score part or staff group considered as one logical part."""

    name: str
    parts: list[m21stream.Part]
    part_indices: list[int]
    instrument_obj: m21instrument.Instrument | None
    group_name: str | None = None


@dataclass
class _ResolvedLogicalPart:
    """Requested logical part resolved to score streams and optional group."""

    spec: _NormalizedPartSpec
    parts: list[m21stream.Part]
    reused_parts: list[m21stream.Part]
    created_parts: list[m21stream.Part]


def resolve_instrument(name: str) -> m21instrument.Instrument:
    """Resolve an instrument name to a music21 Instrument object.

    Tries exact map lookup first, then music21's fromString,
    then falls back to a generic Instrument with the given name.
    """
    normalized_name = normalize_instrument_label(name).strip()
    lookup_name = normalized_name.replace("_", " ").replace("-", " ")
    key = lookup_name.lower()
    if key in _INSTRUMENT_MAP:
        return _INSTRUMENT_MAP[key]()

    try:
        return m21instrument.fromString(lookup_name)
    except Exception:
        pass

    inst = m21instrument.Instrument()
    inst.partName = name
    return inst


def _normalize_part_text(text: object) -> str:
    """Return a lowercase compact key for matching names and instruments."""
    normalized_label = normalize_instrument_label(text)
    normalized = _TEXT_NORMALIZATION_PATTERN.sub(" ", normalized_label.lower())
    return " ".join(normalized.split())


def _instrument_transposition_semitones(
    instrument_obj: m21instrument.Instrument,
) -> int | None:
    """Return an instrument transposition in semitones, if present."""
    interval = getattr(instrument_obj, "transposition", None)
    semitones = getattr(interval, "semitones", None)
    if semitones is None:
        return None
    return int(semitones)


def _display_name_mentions_transposing_pitch(
    display_name: str,
) -> bool:
    """Return whether ``display_name`` embeds a keyed transposing instrument."""
    normalized_display = normalize_instrument_label(display_name).lower()
    compact_display = _TEXT_NORMALIZATION_PATTERN.sub(" ", normalized_display)
    words = compact_display.split()
    for index, word in enumerate(words):
        if word in _TRANSPOSING_FAMILY_NAMES:
            previous_words = words[:index]
            following_words = words[index + 1:]
            if previous_words and _PITCH_NAME_PATTERN.fullmatch(previous_words[0]):
                return True
            if following_words[:1] == ["in"] and len(following_words) > 1:
                return _PITCH_NAME_PATTERN.fullmatch(following_words[1]) is not None
    return False


def _instrument_lookup_name_from_display_name(display_name: str) -> str:
    """Return an instrument lookup label derived from a displayed part name."""
    normalized_name = normalize_instrument_label(display_name).strip()
    return _TRAILING_PART_NUMBER_PATTERN.sub("", normalized_name).strip()


def _effective_score_part_instrument_name(
    instrument_name: str,
    display_name: str | None,
    instrument_obj: m21instrument.Instrument,
) -> str:
    """Return the instrument string to use for a normalized part spec."""
    if display_name is None:
        return instrument_name
    if _normalize_part_text(instrument_name) not in _GENERIC_TRANSPOSING_PART_NAMES:
        return instrument_name

    candidate_name = _instrument_lookup_name_from_display_name(display_name)
    if not candidate_name:
        return instrument_name
    candidate_obj = resolve_instrument(candidate_name)
    if not _instrument_matches(candidate_obj, instrument_obj, instrument_name):
        return instrument_name

    candidate_semitones = _instrument_transposition_semitones(candidate_obj)
    original_semitones = _instrument_transposition_semitones(instrument_obj)
    if (
        type(candidate_obj) is not type(instrument_obj)
        or candidate_semitones != original_semitones
        or _display_name_mentions_transposing_pitch(display_name)
    ):
        return candidate_name
    return instrument_name


def _instrument_label_candidates(
    instrument_obj: m21instrument.Instrument | None,
) -> set[str]:
    """Return normalized labels that identify an instrument object."""
    if instrument_obj is None:
        return set()

    best_name = getattr(instrument_obj, "bestName", None)
    if callable(best_name):
        best_name = best_name()
    labels = {
        getattr(instrument_obj, "partName", None),
        getattr(instrument_obj, "instrumentName", None),
        best_name,
        type(instrument_obj).__name__,
    }
    return {
        normalized
        for label in labels
        if (normalized := _normalize_part_text(label))
    }


def _instrument_matches(
    existing: m21instrument.Instrument | None,
    requested: m21instrument.Instrument,
    requested_name: str,
) -> bool:
    """Return whether an existing instrument is compatible with a request."""
    if existing is None:
        return False
    requested_type = type(requested)
    if type(existing) is requested_type:
        return True
    if requested_type is not m21instrument.Instrument and isinstance(
        existing,
        requested_type,
    ):
        return True

    requested_labels = _instrument_label_candidates(requested)
    requested_labels.add(_normalize_part_text(requested_name))
    existing_labels = _instrument_label_candidates(existing)
    return bool(requested_labels & existing_labels)


def _default_staff_clefs_for_instrument(
    instrument_name: str,
    instrument_obj: m21instrument.Instrument,
) -> tuple[m21clef.Clef, ...]:
    """Return automatic staff clefs for an instrument request."""
    normalized_name = _normalize_part_text(instrument_name)
    if (
        normalized_name in _AUTO_THREE_STAFF_NAMES
        or isinstance(instrument_obj, _AUTO_THREE_STAFF_INSTRUMENTS)
    ):
        return (m21clef.TrebleClef(), m21clef.BassClef(), m21clef.BassClef())
    if (
        normalized_name in _AUTO_TWO_STAFF_NAMES
        or isinstance(instrument_obj, _AUTO_TWO_STAFF_INSTRUMENTS)
    ):
        return (m21clef.TrebleClef(), m21clef.BassClef())
    return (default_clef_for_instrument(instrument_obj),)


def _normalize_score_part_spec(raw_spec: object) -> _NormalizedPartSpec:
    """Validate and normalize one ``set_score_parts`` entry."""
    if isinstance(raw_spec, ScorePartSpec):
        instrument_value = raw_spec.instrument
        name_value = raw_spec.name
    elif isinstance(raw_spec, dict):
        unexpected_keys = set(raw_spec) - {"instrument", "name"}
        if unexpected_keys:
            raise ValueError(
                "Unsupported part spec field(s): "
                f"{', '.join(sorted(str(key) for key in unexpected_keys))}. "
                "Use only 'instrument' and optional 'name'."
            )
        instrument_value = raw_spec.get("instrument")
        name_value = raw_spec.get("name")
    elif hasattr(raw_spec, "model_dump"):
        data = raw_spec.model_dump()
        instrument_value = data.get("instrument")
        name_value = data.get("name")
    else:
        raise ValueError(
            "Each part spec must be a dict or ScorePartSpec with an "
            "'instrument' field and optional 'name'."
        )

    if not isinstance(instrument_value, str) or not instrument_value.strip():
        raise ValueError("Each part spec requires a non-empty instrument string.")
    if name_value is not None and not isinstance(name_value, str):
        raise ValueError("Part spec 'name' must be a string when provided.")

    display_name = name_value.strip() if isinstance(name_value, str) else None
    requested_instrument_name = instrument_value.strip()
    requested_instrument_obj = resolve_instrument(requested_instrument_name)
    instrument_name = _effective_score_part_instrument_name(
        requested_instrument_name,
        display_name,
        requested_instrument_obj,
    )
    instrument_obj = resolve_instrument(instrument_name)
    part_name = _part_display_name(display_name, instrument_name, instrument_obj)
    staff_clefs = _default_staff_clefs_for_instrument(
        instrument_name,
        instrument_obj,
    )
    return _NormalizedPartSpec(
        name=part_name,
        instrument=instrument_name,
        instrument_obj=instrument_obj,
        staff_clefs=staff_clefs,
    )


def _part_display_name(
    name: Optional[str],
    instrument: str,
    inst: m21instrument.Instrument,
) -> str:
    """Return the raw part name to use for a newly created part."""
    return name or inst.partName or instrument.title()


def _normalized_part_display_name(value: object) -> str:
    """Return a comparable display-name key for duplicate part checks."""
    return _TEXT_NORMALIZATION_PATTERN.sub("", str(value or "").casefold())


def _copy_key_signature(
    ks: m21key.KeySignature | m21key.Key,
) -> m21key.KeySignature | m21key.Key:
    """Return a detached key-signature object equivalent to ``ks``."""
    return copy_key_signature(ks)


def _add_empty_measure(
    part: m21stream.Part,
    measure_number: int,
    bar_quarter_length: float,
    clef_obj: m21clef.Clef | None = None,
    time_signature: m21meter.TimeSignature | None = None,
    key_signature: m21key.KeySignature | m21key.Key | None = None,
    tempo: float | None = None,
) -> None:
    """Append one empty measure with optional starting notation attributes."""
    measure = m21stream.Measure(number=measure_number)
    if time_signature is not None:
        measure.timeSignature = m21meter.TimeSignature(time_signature.ratioString)
    if key_signature is not None:
        measure.insert(0, _copy_key_signature(key_signature))
    if tempo is not None:
        measure.insert(0, m21tempo.MetronomeMark(number=tempo))
    if clef_obj is not None:
        measure.insert(0, clef_obj)

    measure.append(m21note.Rest(quarterLength=bar_quarter_length))
    part.append(measure)


def _populate_part_from_reference(
    part: m21stream.Part,
    reference_part: m21stream.Part,
    first_clef: m21clef.Clef,
) -> None:
    """Populate ``part`` with empty measures matching ``reference_part``."""
    reference_measures = sorted(
        reference_part.getElementsByClass(m21stream.Measure),
        key=lambda measure: measure.number,
    )
    for ref_measure in reference_measures:
        ref_time_signatures = ref_measure.getElementsByClass(m21meter.TimeSignature)
        time_signature = ref_time_signatures[0] if ref_time_signatures else None
        active_time_signature = ref_measure.getContextByClass(m21meter.TimeSignature)

        ref_key_signatures = ref_measure.getElementsByClass(m21key.KeySignature)
        key_signature = ref_key_signatures[0] if ref_key_signatures else None
        if key_signature is not None:
            concert_key_signature = concert_key_signature_for_stored_key(
                reference_part,
                key_signature,
            )
            key_signature = stored_key_signature_for_concert_key(
                part,
                concert_key_signature,
            )

        bar_quarter_length = (
            active_time_signature.barDuration.quarterLength
            if active_time_signature
            else 4.0
        )
        clef_obj = None
        if (
            ref_measure.number == 1
            or ref_measure.number == reference_measures[0].number
        ):
            clef_obj = first_clef

        _add_empty_measure(
            part,
            ref_measure.number,
            bar_quarter_length,
            clef_obj=clef_obj,
            time_signature=time_signature,
            key_signature=key_signature,
        )


def _populate_part_from_defaults(
    part: m21stream.Part,
    measures: int,
    time_signature: m21meter.TimeSignature,
    key_signature: m21key.KeySignature | m21key.Key,
    tempo: float | None,
    first_clef: m21clef.Clef,
) -> None:
    """Populate ``part`` with default empty measures for new-score creation."""
    for measure_number in range(1, measures + 1):
        _add_empty_measure(
            part,
            measure_number,
            time_signature.barDuration.quarterLength,
            clef_obj=first_clef if measure_number == 1 else None,
            time_signature=time_signature if measure_number == 1 else None,
            key_signature=key_signature if measure_number == 1 else None,
            tempo=tempo if measure_number == 1 else None,
        )


def _copy_measure_structure_from_reference(
    reference_part: m21stream.Part,
    target_part: m21stream.Part,
) -> None:
    """Copy score-level measure structure from a reference into an empty part."""
    target_by_number = {
        measure.number: measure
        for measure in target_part.getElementsByClass(m21stream.Measure)
    }
    for reference_measure in reference_part.getElementsByClass(m21stream.Measure):
        target_measure = target_by_number.get(reference_measure.number)
        if target_measure is None:
            continue

        if reference_measure.leftBarline is not None:
            target_measure.leftBarline = copy.deepcopy(
                reference_measure.leftBarline
            )
        if reference_measure.rightBarline is not None:
            target_measure.rightBarline = copy.deepcopy(
                reference_measure.rightBarline
            )

        for layout_cls in (m21layout.SystemLayout, m21layout.PageLayout):
            for layout_obj in reference_measure.getElementsByClass(layout_cls):
                target_measure.insert(
                    reference_measure.elementOffset(layout_obj),
                    copy.deepcopy(layout_obj),
                )

    _copy_repeat_brackets_from_reference(reference_part, target_part)


def _copy_repeat_brackets_from_reference(
    reference_part: m21stream.Part,
    target_part: m21stream.Part,
) -> None:
    """Copy repeat ending brackets from one part to another."""
    target_by_number = {
        measure.number: measure
        for measure in target_part.getElementsByClass(m21stream.Measure)
    }
    for repeat_bracket in reference_part.getElementsByClass(
        m21spanner.RepeatBracket
    ):
        target_measures = []
        for element in repeat_bracket.getSpannedElements():
            if not isinstance(element, m21stream.Measure):
                continue
            target_measure = target_by_number.get(element.number)
            if target_measure is None:
                target_measures = []
                break
            target_measures.append(target_measure)
        if not target_measures:
            continue

        copied = m21spanner.RepeatBracket(
            *target_measures,
            number=getattr(repeat_bracket, "number", None),
        )
        copied.overrideDisplay = getattr(repeat_bracket, "overrideDisplay", None)
        target_part.insert(0, copied)


def _make_empty_part_like_score(
    name: str,
    instrument: str,
    first_clef: m21clef.Clef,
    *,
    reference_part: m21stream.Part | None,
    measures: int,
    time_signature: m21meter.TimeSignature,
    key_signature: m21key.KeySignature | m21key.Key,
    tempo: float | None = None,
    part_cls: type[m21stream.Part] = m21stream.Part,
) -> m21stream.Part:
    """Create a detached empty part synchronized with the score structure."""
    inst = resolve_instrument(instrument)
    part = part_cls()
    part.partName = name
    part.insert(0, inst)
    default_stored_pitch_space_for_part(part)

    if reference_part is not None:
        _populate_part_from_reference(part, reference_part, first_clef)
        _copy_measure_structure_from_reference(reference_part, part)
    elif measures > 0:
        _populate_part_from_defaults(
            part,
            measures,
            time_signature,
            stored_key_signature_for_concert_key(part, key_signature),
            tempo,
            first_clef,
        )

    return part


def _make_multi_staff_part(
    name: Optional[str],
    instrument: str,
    staff_clefs: tuple[m21clef.Clef, ...],
    *,
    measures: int = 0,
    time_signature: m21meter.TimeSignature | None = None,
    key_signature: m21key.KeySignature | m21key.Key | None = None,
    tempo: float | None = None,
    reference_part: m21stream.Part | None = None,
) -> tuple[list[m21stream.PartStaff], m21layout.StaffGroup]:
    """Create detached multi-staff streams and a brace group."""
    inst = resolve_instrument(instrument)
    part_name = _part_display_name(name, instrument, inst)
    effective_time_signature = time_signature or m21meter.TimeSignature("4/4")
    effective_key_signature = key_signature or m21key.Key("C", "major")
    parts = [
        _make_empty_part_like_score(
            part_name,
            instrument,
            first_clef,
            reference_part=reference_part,
            measures=measures,
            time_signature=effective_time_signature,
            key_signature=effective_key_signature,
            tempo=tempo if index == 0 else None,
            part_cls=m21stream.PartStaff,
        )
        for index, first_clef in enumerate(staff_clefs)
    ]

    group = m21layout.StaffGroup(parts, name=part_name, symbol="brace")
    return parts, group


def _make_grand_staff(
    name: Optional[str],
    instrument: str,
    *,
    measures: int = 0,
    time_signature: m21meter.TimeSignature | None = None,
    key_signature: m21key.KeySignature | m21key.Key | None = None,
    tempo: float | None = None,
    reference_part: m21stream.Part | None = None,
) -> tuple[list[m21stream.PartStaff], m21layout.StaffGroup]:
    """Create detached RH/LH ``PartStaff`` streams and their brace group."""
    return _make_multi_staff_part(
        name,
        instrument,
        (m21clef.TrebleClef(), m21clef.BassClef()),
        measures=measures,
        time_signature=time_signature,
        key_signature=key_signature,
        tempo=tempo,
        reference_part=reference_part,
    )


def _part_instrument_or_none(
    part: m21stream.Part,
) -> m21instrument.Instrument | None:
    """Return a part's explicit instrument, if present."""
    explicit_instrument = part.getInstrument(returnDefault=False)
    if explicit_instrument is not None:
        return explicit_instrument
    if part.partName:
        return resolve_instrument(part.partName)
    return None


def _existing_logical_parts(score: m21stream.Score) -> list[_ExistingLogicalPart]:
    """Return existing parts grouped into logical score parts."""
    score_parts = list(score.parts)
    staff_groups = detect_staff_groups(score)
    grouped_indices: set[int] = set()
    logical_parts: list[_ExistingLogicalPart] = []

    for group in staff_groups:
        group_parts = [
            score_parts[index]
            for index in group.part_indices
            if 0 <= index < len(score_parts)
        ]
        if not group_parts:
            continue
        grouped_indices.update(group.part_indices)
        logical_parts.append(
            _ExistingLogicalPart(
                name=group.name,
                parts=group_parts,
                part_indices=list(group.part_indices),
                instrument_obj=_part_instrument_or_none(group_parts[0]),
                group_name=group.name,
            )
        )

    for index, part in enumerate(score_parts):
        if index in grouped_indices:
            continue
        logical_parts.append(
            _ExistingLogicalPart(
                name=part.partName or f"Part {index}",
                parts=[part],
                part_indices=[index],
                instrument_obj=_part_instrument_or_none(part),
            )
        )

    logical_parts.sort(key=lambda item: min(item.part_indices))
    return logical_parts


def _match_existing_logical_part(
    spec: _NormalizedPartSpec,
    candidates: list[_ExistingLogicalPart],
    used_candidates: set[int],
) -> tuple[int, _ExistingLogicalPart] | None:
    """Find the first unmatched existing logical part compatible with a spec."""
    requested_name = _normalize_part_text(spec.name)
    for index, candidate in enumerate(candidates):
        if index in used_candidates:
            continue
        if _normalize_part_text(candidate.name) != requested_name:
            continue
        if len(spec.staff_clefs) == 1 and len(candidate.parts) > 1:
            continue
        if not _instrument_matches(
            candidate.instrument_obj,
            spec.instrument_obj,
            spec.instrument,
        ):
            continue
        return index, candidate
    return None


def _score_mark_value(element: object) -> tuple[object, ...]:
    """Return values that identify a score-level mark for de-duplication."""
    if isinstance(element, m21tempo.MetronomeMark):
        return (
            getattr(element, "number", None),
            str(getattr(element, "text", "") or ""),
        )
    if isinstance(element, m21expressions.RehearsalMark):
        return (str(getattr(element, "content", "") or ""),)
    if isinstance(element, m21repeat.Coda):
        return (str(element.getText() or ""),)
    return ()


def _score_mark_key(
    element: object,
    offset: float,
) -> tuple[str, float, tuple[object, ...]]:
    """Return a comparable key for one score-level mark."""
    return (type(element).__name__, round(float(offset), 6), _score_mark_value(element))


def _snapshot_score_level_marks(
    part: m21stream.Part | None,
) -> list[dict[str, Any]]:
    """Capture first-part score-level marks before part replacement."""
    if part is None:
        return []

    snapshots: list[dict[str, Any]] = []
    for measure in part.getElementsByClass(m21stream.Measure):
        if measure.number is None:
            continue
        for element in measure.getElementsByClass(_SCORE_MARK_CLASSES):
            offset = float(measure.elementOffset(element))
            snapshots.append({
                "measure": measure.number,
                "offset": offset,
                "key": _score_mark_key(element, offset),
                "element": copy.deepcopy(element),
            })
    return snapshots


def _measure_has_score_mark(
    measure: m21stream.Measure,
    key: tuple[str, float, tuple[object, ...]],
) -> bool:
    """Return whether a measure already contains a matching score mark."""
    for element in measure.getElementsByClass(_SCORE_MARK_CLASSES):
        offset = float(measure.elementOffset(element))
        if _score_mark_key(element, offset) == key:
            return True
    return False


def _remove_score_mark_keys_from_part(
    part: m21stream.Part,
    keys_by_measure: dict[int, set[tuple[str, float, tuple[object, ...]]]],
) -> int:
    """Remove matching score-level marks from a part and return the count."""
    removed = 0
    for measure in part.getElementsByClass(m21stream.Measure):
        if measure.number not in keys_by_measure:
            continue
        target_keys = keys_by_measure[measure.number]
        for element in list(measure.getElementsByClass(_SCORE_MARK_CLASSES)):
            offset = float(measure.elementOffset(element))
            if _score_mark_key(element, offset) not in target_keys:
                continue
            measure.remove(element)
            removed += 1
    return removed


def _ensure_score_level_marks_on_first_part(
    score: m21stream.Score,
    original_first_part: m21stream.Part | None,
    snapshots: list[dict[str, Any]],
) -> tuple[int, int]:
    """Move first-part score-level marks onto the current first part."""
    score_parts = list(score.parts)
    if not score_parts or original_first_part is None or not snapshots:
        return 0, 0

    new_first_part = score_parts[0]
    if new_first_part is original_first_part:
        return 0, 0

    copied = 0
    keys_by_measure: dict[int, set[tuple[str, float, tuple[object, ...]]]] = {}
    for snapshot in snapshots:
        measure_number = int(snapshot["measure"])
        keys_by_measure.setdefault(measure_number, set()).add(snapshot["key"])
        target_measure = new_first_part.measure(measure_number)
        if target_measure is None:
            continue
        if _measure_has_score_mark(target_measure, snapshot["key"]):
            continue
        target_measure.insert(snapshot["offset"], copy.deepcopy(snapshot["element"]))
        copied += 1

    removed = 0
    if original_first_part in score_parts:
        removed = _remove_score_mark_keys_from_part(
            original_first_part,
            keys_by_measure,
        )

    return copied, removed


def _remove_staff_groups_from_score(score: m21stream.Score) -> int:
    """Remove all existing staff groups from the score."""
    removed = 0
    for group in list(score.recurse().getElementsByClass(m21layout.StaffGroup)):
        active_site = group.activeSite or score
        try:
            active_site.remove(group)
        except Exception:
            try:
                score.remove(group)
            except Exception:
                continue
        removed += 1
    return removed


def _resolve_requested_logical_part(
    spec: _NormalizedPartSpec,
    candidate: _ExistingLogicalPart | None,
    *,
    reference_part: m21stream.Part | None,
    default_time_signature: m21meter.TimeSignature,
) -> _ResolvedLogicalPart:
    """Resolve one requested logical part to reused and newly created staves."""
    parts: list[m21stream.Part] = []
    reused_parts: list[m21stream.Part] = []
    created_parts: list[m21stream.Part] = []
    reference_for_created = candidate.parts[0] if candidate is not None else reference_part
    part_cls = m21stream.PartStaff if len(spec.staff_clefs) > 1 else m21stream.Part

    for index, first_clef in enumerate(spec.staff_clefs):
        if candidate is not None and index < len(candidate.parts):
            part = candidate.parts[index]
            part.partName = spec.name
            parts.append(part)
            reused_parts.append(part)
            continue

        part = _make_empty_part_like_score(
            spec.name,
            spec.instrument,
            first_clef,
            reference_part=reference_for_created,
            measures=1,
            time_signature=default_time_signature,
            key_signature=m21key.Key("C", "major"),
            part_cls=part_cls,
        )
        parts.append(part)
        created_parts.append(part)

    return _ResolvedLogicalPart(
        spec=spec,
        parts=parts,
        reused_parts=reused_parts,
        created_parts=created_parts,
    )


class PartsMixin:
    """Mixin providing part add/remove operations."""

    def set_score_parts(
        self,
        parts: list[ScorePartSpec],
    ) -> OperationResult:
        """Set, initialize, set up, replace, reset, or configure score parts.

        For keyed transposing instruments, put the key/variant in the
        ``instrument`` value itself, for example "Bb clarinet", "Eb horn",
        or "C trumpet"; ``name`` is only the displayed staff label.
        Use this replacement-style tool for a complete requested
        instrumentation, ensemble, roster, lineup, staff list, or part list.
        The requested logical parts are created in the given order. Parts not
        requested are removed. Existing requested parts with the same name and
        compatible instrument are moved to the requested position so their
        musical content is preserved. Clefs and multi-staff layout are fully
        automatic: piano, organ, electric organ, harpsichord, celesta, and
        harp use the appropriate keyboard-style staves.

        Args:
            parts: Ordered list of requested part specs. Each spec has a
                required ``instrument`` string and optional ``name`` string.
                ``instrument`` names the playable instrument and must include
                transposition or variant when relevant, such as "Bb clarinet",
                "Eb horn", "C trumpet", or "bass clarinet". ``name`` is only
                the displayed staff label, such as "Bb Clarinet 1".

        Returns:
            OperationResult describing the resulting score parts.

        Raises:
            ValueError: If ``parts`` is empty or any spec is malformed.
        """
        if not isinstance(parts, list):
            raise ValueError("parts must be a non-empty list of part specs.")
        if not parts:
            raise ValueError("parts must contain at least one part spec.")

        requested_specs = [
            _normalize_score_part_spec(part_spec)
            for part_spec in parts
        ]
        existing_parts = list(self._score.parts)
        original_first_part = existing_parts[0] if existing_parts else None
        original_index_by_id = {
            id(part): index
            for index, part in enumerate(existing_parts)
        }
        score_mark_snapshots = _snapshot_score_level_marks(original_first_part)
        existing_candidates = _existing_logical_parts(self._score)
        used_candidate_indices: set[int] = set()
        default_time_signature = self._get_default_time_signature()

        resolved_parts: list[_ResolvedLogicalPart] = []
        for requested_spec in requested_specs:
            match = _match_existing_logical_part(
                requested_spec,
                existing_candidates,
                used_candidate_indices,
            )
            matched_candidate = None
            if match is not None:
                matched_index, matched_candidate = match
                used_candidate_indices.add(matched_index)

            resolved_parts.append(
                _resolve_requested_logical_part(
                    requested_spec,
                    matched_candidate,
                    reference_part=original_first_part,
                    default_time_signature=default_time_signature,
                )
            )

        ordered_parts = [
            part
            for resolved_part in resolved_parts
            for part in resolved_part.parts
        ]
        ordered_part_ids = {id(part) for part in ordered_parts}
        removed_parts = [
            part
            for part in existing_parts
            if id(part) not in ordered_part_ids
        ]

        removed_staff_groups = _remove_staff_groups_from_score(self._score)
        for existing_part in existing_parts:
            self._score.remove(existing_part)
        for ordered_part in ordered_parts:
            self._score.insert(0, ordered_part)

        created_staff_groups = 0
        for resolved_part in resolved_parts:
            if len(resolved_part.parts) <= 1:
                continue
            staff_group = m21layout.StaffGroup(
                resolved_part.parts,
                name=resolved_part.spec.name,
                symbol="brace",
            )
            self._score.insert(0, staff_group)
            created_staff_groups += 1

        copied_score_marks, removed_moved_score_marks = (
            _ensure_score_level_marks_on_first_part(
                self._score,
                original_first_part,
                score_mark_snapshots,
            )
        )

        labels = build_part_display_labels(self._score)
        final_parts = list(self._score.parts)
        final_logical_parts = []
        for resolved_part in resolved_parts:
            part_indices = [
                final_parts.index(part)
                for part in resolved_part.parts
                if part in final_parts
            ]
            display_names = [
                labels[part_index].display_name
                for part_index in part_indices
                if part_index in labels
            ]
            final_logical_parts.append({
                "name": resolved_part.spec.name,
                "instrument": resolved_part.spec.instrument,
                "part_indices": part_indices,
                "display_names": display_names,
                "staff_count": len(resolved_part.parts),
                "reused_old_indices": [
                    original_index_by_id[id(part)]
                    for part in resolved_part.reused_parts
                    if id(part) in original_index_by_id
                ],
                "created_count": len(resolved_part.created_parts),
            })

        return OperationResult(
            success=True,
            description=(
                "Set score parts to "
                + ", ".join(
                    f"{spec.name} ({spec.instrument})"
                    for spec in requested_specs
                )
            ),
            details={
                "logical_parts": final_logical_parts,
                "part_indices": list(range(len(final_parts))),
                "display_names": [
                    labels[index].display_name
                    for index in range(len(final_parts))
                    if index in labels
                ],
                "created_parts": sum(
                    len(resolved_part.created_parts)
                    for resolved_part in resolved_parts
                ),
                "reused_parts": sum(
                    len(resolved_part.reused_parts)
                    for resolved_part in resolved_parts
                ),
                "removed_parts": len(removed_parts),
                "removed_part_names": [
                    part.partName or f"Part {index}"
                    for index, part in enumerate(removed_parts)
                ],
                "removed_staff_groups": removed_staff_groups,
                "created_staff_groups": created_staff_groups,
                "copied_score_marks_to_first_part": copied_score_marks,
                "removed_moved_score_marks": removed_moved_score_marks,
            },
        )

    def add_part(
        self,
        name: Optional[str] = None,
        instrument: str = "piano",
        clef_type: Optional[str] = None,
        index: Optional[int] = None,
        grand_staff: bool = False,
    ) -> OperationResult:
        """Add a new part to the score.

        The new part is populated with empty measures matching the
        existing score length, inheriting time and key signatures. Set
        ``grand_staff=True`` to add a grouped piano-style RH/LH grand
        staff through this same tool instead of adding a single staff.
        Only use this tool when the user explicitly asks for a new, 
        appended part to the score. To extend an existing part, use 
        ``add_measures`` instead.

        Args:
            name: Display name for the part. Defaults to the instrument name.
            instrument: Instrument name (e.g., "violin", "piano", "flute").
            clef_type: Clef type string; auto-detected from instrument if None.
            index: Position to insert (0-based). None appends at the end.
            grand_staff: When True, insert adjacent RH/LH ``PartStaff``
                streams grouped by a brace. Intended for piano/keyboard
                parts.

        Returns:
            OperationResult with the new part's details.
        """
        inst = resolve_instrument(instrument)
        part_name = _part_display_name(name, instrument, inst)
        normalized_part_name = _normalized_part_display_name(part_name)
        existing_part_names = {
            _normalized_part_display_name(existing_part.partName)
            for existing_part in self._score.parts
            if existing_part.partName
        }
        if (
            not grand_staff
            and normalized_part_name
            and normalized_part_name in existing_part_names
        ):
            raise ValueError(
                f"A part named '{part_name}' already exists. Use that existing "
                "part or choose a distinct new part name."
            )

        if grand_staff:
            existing_parts = list(self._score.parts)
            reference_part = existing_parts[0] if existing_parts else None
            if reference_part is None:
                default_time_signature = self._get_default_time_signature()
                new_parts, staff_group = _make_grand_staff(
                    name,
                    instrument,
                    measures=1,
                    time_signature=default_time_signature,
                    key_signature=m21key.Key("C", "major"),
                )
            else:
                new_parts, staff_group = _make_grand_staff(
                    name,
                    instrument,
                    reference_part=reference_part,
                )

            all_parts = list(self._score.parts)
            target_index = len(all_parts) if index is None else index
            if target_index < 0 or target_index > len(all_parts):
                raise ValueError(
                    f"Part index {target_index} is out of range "
                    f"(0–{len(all_parts)})."
                )

            ordered_parts = (
                all_parts[:target_index] + new_parts + all_parts[target_index:]
            )
            for existing_part in all_parts:
                self._score.remove(existing_part)
            for ordered_part in ordered_parts:
                self._score.insert(0, ordered_part)
            self._score.insert(0, staff_group)

            labels = build_part_display_labels(self._score)
            part_indices = [list(self._score.parts).index(part) for part in new_parts]
            display_names = [
                labels[part_idx].display_name
                for part_idx in part_indices
                if part_idx in labels
            ]
            measure_count = len(
                list(new_parts[0].getElementsByClass(m21stream.Measure))
            )
            return OperationResult(
                success=True,
                description=f"Added grand staff '{part_name}' ({instrument})",
                details={
                    "part_index": part_indices[0],
                    "part_name": new_parts[0].partName,
                    "part_indices": part_indices,
                    "part_names": [part.partName for part in new_parts],
                    "display_names": display_names,
                    "instrument": instrument,
                    "grand_staff": True,
                    "measure_count": measure_count,
                },
            )

        part = m21stream.Part()
        part.partName = part_name
        part.insert(0, inst)
        default_stored_pitch_space_for_part(part)

        if clef_type:
            first_clef = make_clef(clef_type)
        else:
            first_clef = default_clef_for_instrument(inst)

        existing_parts = list(self._score.parts)
        if existing_parts:
            ref_part = existing_parts[0]
            ref_measures = sorted(
                ref_part.getElementsByClass(m21stream.Measure),
                key=lambda m: m.number,
            )
            for ref_m in ref_measures:
                new_m = m21stream.Measure(number=ref_m.number)

                if ref_m.number == 1 or ref_m.number == ref_measures[0].number:
                    new_m.insert(0, first_clef)

                ts = ref_m.getContextByClass(m21meter.TimeSignature)
                ref_ts = ref_m.getElementsByClass(m21meter.TimeSignature)
                if ref_ts:
                    new_m.timeSignature = m21meter.TimeSignature(
                        ref_ts[0].ratioString
                    )

                ref_ks = ref_m.getElementsByClass(m21key.KeySignature)
                if ref_ks:
                    concert_key_signature = concert_key_signature_for_stored_key(
                        ref_part,
                        ref_ks[0],
                    )
                    new_m.insert(
                        0,
                        stored_key_signature_for_concert_key(
                            part,
                            concert_key_signature,
                        ),
                    )

                bar_ql = ts.barDuration.quarterLength if ts else 4.0
                rest = m21note.Rest(quarterLength=bar_ql)
                new_m.append(rest)
                part.append(new_m)
        else:
            m = m21stream.Measure(number=1)
            ts = self._get_default_time_signature()
            m.timeSignature = ts
            m.insert(0, first_clef)
            m.insert(
                0,
                stored_key_signature_for_concert_key(
                    part,
                    m21key.Key("C", "major"),
                ),
            )
            rest = m21note.Rest(quarterLength=ts.barDuration.quarterLength)
            m.append(rest)
            part.append(m)

        all_parts = list(self._score.parts)
        target_index = len(all_parts) if index is None else index
        if target_index < 0 or target_index > len(all_parts):
            raise ValueError(
                f"Part index {target_index} is out of range "
                f"(0–{len(all_parts)})."
            )

        ordered_parts = (
            all_parts[:target_index] + [part] + all_parts[target_index:]
        )
        for existing_part in all_parts:
            self._score.remove(existing_part)
        for ordered_part in ordered_parts:
            self._score.insert(0, ordered_part)

        part_list = list(self._score.parts)
        new_idx = part_list.index(part) if part in part_list else len(part_list) - 1

        return OperationResult(
            success=True,
            description=f"Added part '{part.partName}' ({instrument})",
            details={
                "part_index": new_idx,
                "part_name": part.partName,
                "instrument": instrument,
                "measure_count": len(list(part.getElementsByClass(m21stream.Measure))),
            },
        )

    def remove_part(
        self,
        part: Union[int, str],
    ) -> OperationResult:
        """Remove a part from the score.

        Args:
            part: Part index (0-based) or part name.

        Returns:
            OperationResult with details of the removed part.

        Raises:
            ValueError: If the score would have no parts after removal,
                or if the part identifier is invalid.
        """
        part_obj, part_idx = self._resolve_part(part)
        parts = list(self._score.parts)

        if len(parts) <= 1:
            raise ValueError(
                "Cannot remove the last part from the score. "
                "A score must have at least one part."
            )

        part_name = part_obj.partName or f"Part {part_idx}"
        self._score.remove(part_obj)

        return OperationResult(
            success=True,
            description=f"Removed part '{part_name}' (was index {part_idx})",
            details={
                "removed_part_index": part_idx,
                "removed_part_name": part_name,
                "remaining_parts": len(list(self._score.parts)),
            },
        )

    def get_part_info(
        self,
        part: Optional[Union[int, str]] = None,
    ) -> PartInfo:
        """Get structured information about a part.

        Args:
            part: Part index, name, or None (first part).

        Returns:
            PartInfo with name, instrument, measure count, and a
            ``display_name`` / ``hand`` pair derived from any detected
            grand-staff grouping (``None`` for parts outside a group).
        """
        part_obj, part_idx = self._resolve_part(part)

        inst = part_obj.getInstrument()
        inst_name = inst.partName if inst and inst.partName else "Unknown"

        measure_count = len(
            list(part_obj.getElementsByClass(m21stream.Measure))
        )

        raw_name = part_obj.partName or f"Part {part_idx}"
        labels = build_part_display_labels(self._score)
        label = labels.get(part_idx)
        display_name = label.display_name if label is not None else raw_name
        hand = label.hand if label is not None else None

        return PartInfo(
            index=part_idx,
            name=raw_name,
            instrument=inst_name,
            measure_count=measure_count,
            display_name=display_name,
            hand=hand,
        )

    def list_parts(self) -> list[PartInfo]:
        """List all parts in the score.

        Returns:
            List of ``PartInfo`` objects, one per part, each carrying
            ``display_name`` and ``hand`` fields derived from detected
            brace groups.
        """
        result = []
        labels = build_part_display_labels(self._score)
        for i, part_obj in enumerate(self._score.parts):
            inst = part_obj.getInstrument()
            inst_name = inst.partName if inst and inst.partName else "Unknown"
            measure_count = len(
                list(part_obj.getElementsByClass(m21stream.Measure))
            )
            raw_name = part_obj.partName or f"Part {i}"
            label = labels.get(i)
            display_name = label.display_name if label is not None else raw_name
            hand = label.hand if label is not None else None
            result.append(
                PartInfo(
                    index=i,
                    name=raw_name,
                    instrument=inst_name,
                    measure_count=measure_count,
                    display_name=display_name,
                    hand=hand,
                )
            )
        return result
