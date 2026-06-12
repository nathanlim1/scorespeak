"""
Extended notational markings: lyrics, ornaments, ottava lines, glissandi,
damper pedal, volta (ending) brackets, and arpeggiation.

These map closely to MusicXML elements used by editors such as
`MuseScore <https://github.com/musescore/MuseScore>`_ and are serialized
via music21's MusicXML exporter.
"""

from __future__ import annotations

import re
from typing import Optional, Union

from music21 import articulations as m21articulations
from music21 import chord as m21chord
from music21 import expressions as m21expressions
from music21 import interval as m21interval
from music21 import note as m21note
from music21 import spanner as m21spanner
from music21 import stream as m21stream

from .expression_common import _find_note_at_offset, _validate_beat_in_measure
from ..types import LyricInfo, OperationResult
from ..music.validation import validate_voice_number


_ORNAMENT_MAP: dict[str, type] = {
    "trill": m21expressions.Trill,
    "inverted trill": m21expressions.InvertedTrill,
    "whole step trill": m21expressions.WholeStepTrill,
    "half step trill": m21expressions.HalfStepTrill,
    "turn": m21expressions.Turn,
    "inverted turn": m21expressions.InvertedTurn,
    "mordent": m21expressions.Mordent,
    "inverted mordent": m21expressions.InvertedMordent,
    "whole step mordent": m21expressions.WholeStepMordent,
    "half step mordent": m21expressions.HalfStepMordent,
}

_VALID_SYLLABICS = frozenset(
    {"single", "begin", "middle", "end", "composite", None},
)

_OTTAVA_ALIASES: dict[str, str] = {
    "8va": "8va",
    "8vb": "8vb",
    "ottava alta": "8va",
    "ottava bassa": "8vb",
    "15ma": "15ma",
    "15mb": "15mb",
    "quindicesima": "15ma",
}

_ENDING_ORDINAL_NUMBERS: dict[str, int] = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
    "sixth": 6,
    "6th": 6,
    "seventh": 7,
    "7th": 7,
    "eighth": 8,
    "8th": 8,
    "ninth": 9,
    "9th": 9,
}

_DOTTED_ENDING_NUMBER_PATTERN = re.compile(r"^\s*(\d+)\.\s*$")
_ENDING_NUMBER_TEXT_PATTERN = re.compile(
    r"^\s*(?:\d+|\d+\s*-\s*\d+|\d+(?:\s*,\s*\d+)+)\s*$"
)


def _normalize_ottava_type(raw: str) -> str:
    key = raw.strip().lower()
    if key in _OTTAVA_ALIASES:
        return _OTTAVA_ALIASES[key]
    if key in ("8va", "8vb", "15ma", "15mb"):
        return key
    raise ValueError(
        f"Unknown ottava type '{raw}'. Use '8va', '8vb', '15ma', or '15mb'."
    )


def _ottava_rewrite_interval(
    ottava_type: str,
    *,
    adding: bool,
) -> m21interval.Interval:
    """Return the pitch rewrite interval for adding or removing an ottava."""
    add_semitones = {
        "8va": -12,
        "15ma": -24,
        "8vb": 12,
        "15mb": 24,
    }[_normalize_ottava_type(ottava_type)]
    semitones = add_semitones if adding else -add_semitones
    return m21interval.Interval(semitones)


def _transpose_spanned_pitch_elements(
    elements: list[object],
    interval_obj: m21interval.Interval,
) -> int:
    """Transpose unique note/chord elements and return the affected count."""
    transposed = 0
    seen_ids: set[int] = set()
    for element in elements:
        element_id = id(element)
        if element_id in seen_ids:
            continue
        seen_ids.add(element_id)
        if isinstance(element, (m21note.Note, m21chord.Chord)):
            element.transpose(interval_obj, inPlace=True)
            transposed += 1
    return transposed


def _spanned_measure_numbers(elements: list[object]) -> list[int]:
    """Return sorted unique measure numbers containing spanned elements."""
    measure_numbers: set[int] = set()
    for element in elements:
        if not hasattr(element, "getContextByClass"):
            continue
        measure_obj = element.getContextByClass(m21stream.Measure)
        if not isinstance(measure_obj, m21stream.Measure):
            continue
        if measure_obj.number is None:
            continue
        measure_numbers.add(int(measure_obj.number))
    return sorted(measure_numbers)


def _normalize_ending_number(number: Union[int, str]) -> Union[int, str]:
    """Return a music21-safe ending bracket number."""
    if isinstance(number, int):
        return number

    raw_number = str(number).strip()
    lowered = raw_number.lower().rstrip(".")
    if lowered in _ENDING_ORDINAL_NUMBERS:
        return _ENDING_ORDINAL_NUMBERS[lowered]

    dotted_match = _DOTTED_ENDING_NUMBER_PATTERN.match(raw_number)
    if dotted_match is not None:
        return dotted_match.group(1)

    if _ENDING_NUMBER_TEXT_PATTERN.match(raw_number) is not None:
        return raw_number

    raise ValueError(
        "ending bracket number must be an integer, numeric string, numeric "
        "range such as '1-3', comma-separated numbers such as '1, 2', or "
        "an ordinal such as 'first'."
    )




# ------------------------------------------------------------------
# Module helpers
# ------------------------------------------------------------------


def _pitch_label(el: m21note.GeneralNote) -> str:
    """Return a compact pitch/chord/rest label for lyric summaries."""
    if isinstance(el, m21chord.Chord):
        return "chord"
    if isinstance(el, m21note.Note):
        return el.pitch.nameWithOctave
    return "rest"


def _repeat_bracket_starts_at(
    repeat_bracket: m21spanner.RepeatBracket,
    measure_number: int,
) -> bool:
    """Return whether a repeat bracket's first spanned measure matches."""
    spanned_elements = repeat_bracket.getSpannedElements()
    if not spanned_elements:
        return False

    first_element = spanned_elements[0]
    if isinstance(first_element, m21stream.Measure):
        return first_element.number == measure_number

    measure = first_element.getContextByClass(m21stream.Measure)
    if not isinstance(measure, m21stream.Measure):
        return False
    return measure.number == measure_number


def _iter_voice_streams(
    measure_obj: m21stream.Measure,
) -> list[tuple[int, m21stream.Stream]]:
    """Return voice streams in a measure, defaulting to voice 1."""
    voices = list(measure_obj.voices)
    if not voices:
        return [(1, measure_obj)]
    result: list[tuple[int, m21stream.Stream]] = []
    for v in voices:
        try:
            vid = int(v.id)
        except (TypeError, ValueError):
            vid = 1
        result.append((vid, v))
    return result


def _find_spanner_by_first_anchor(
    part_obj: m21stream.Part,
    spanner_cls: type,
    measure_number: int,
    beat: float,
    voice: int,
) -> Optional[m21spanner.Spanner]:
    """Find a spanner whose first spanned note is at measure/beat in voice."""
    voice = validate_voice_number(voice)
    target_offset = beat - 1.0
    for sp in part_obj.getElementsByClass(spanner_cls):
        els = sp.getSpannedElements()
        if not els:
            continue
        first = els[0]
        m = first.getContextByClass(m21stream.Measure)
        if m is None or m.number != measure_number:
            continue
        parent = first.activeSite
        while parent is not None and not isinstance(
            parent, (m21stream.Voice, m21stream.Measure)
        ):
            parent = parent.activeSite
        if isinstance(parent, m21stream.Voice):
            try:
                pv = int(parent.id)
            except (TypeError, ValueError):
                pv = 1
            if pv != voice:
                continue
        elif voice != 1:
            continue
        try:
            off = m.elementOffset(first)
        except Exception:
            continue
        if abs(off - target_offset) < 1e-9:
            return sp
    return None

__all__ = [name for name in globals() if not name.startswith('__')]
