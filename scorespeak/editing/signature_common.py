"""
Time signature, key signature, clef, and barline operations for ScoreSpeak.

Provides methods for changing time signatures, key signatures, clefs,
barlines, repeats, and pickup (anacrusis) measures.
"""

from __future__ import annotations

import re
from typing import Optional, Union

from music21 import bar as m21bar
from music21 import chord as m21chord
from music21 import clef as m21clef
from music21 import key as m21key
from music21 import meter as m21meter
from music21 import note as m21note
from music21 import repeat as m21repeat
from music21 import stream as m21stream

from ..music.pitch_space import (
    concert_key_signature_for_stored_key,
    copy_key_signature,
    has_marked_local_key_override,
    is_open_key_signature,
    mark_local_key_override,
    mark_open_key_signature,
    stored_key_signature_for_concert_key,
)
from ..types import UNICODE_TO_ASCII_ACCIDENTALS, OperationResult
from ..music.validation import make_clef


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_VALID_BARLINE_TYPES = {
    "regular", "double", "final", "light-heavy", "light-light", "none",
}

_NAVIGATION_MARK_CLASSES: dict[str, type | tuple[type, ...]] = {
    "segno": m21repeat.Segno,
    "fine": m21repeat.Fine,
    "da capo": (m21repeat.DaCapo, m21repeat.DaCapoAlFine, m21repeat.DaCapoAlCoda),
    "dal segno": (m21repeat.DalSegno, m21repeat.DalSegnoAlFine, m21repeat.DalSegnoAlCoda),
}
_CODA_MARK_TYPE = "coda"
_TO_CODA_TEXT = "To Coda"
_TO_CODA_MARK_TYPE = "to coda"


def _normalize_navigation_text(text: object) -> str:
    """Return navigation text normalized for case-insensitive matching."""
    return " ".join(str(text or "").strip().lower().split())


def _is_to_coda_mark(mark: m21repeat.Coda) -> bool:
    """Return whether a Coda object represents a To Coda jump marker."""
    return _normalize_navigation_text(mark.getText()) == _TO_CODA_MARK_TYPE


def _parse_key_signature(key_str: str) -> m21key.KeySignature:
    """Parse a flexible key signature string into a music21 Key or KeySignature.

    Accepts:
        "C", "C major", "A minor", "Bb", "F#m",
        "3" (sharps), "-2" (flats),
        Unicode accidentals like "F♯ minor".

    Returns:
        A music21 Key (with tonic/mode) or KeySignature (integer sharps only).

    Raises:
        ValueError: If the string cannot be parsed.
    """
    text = key_str.strip()
    normalized = text.lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    if normalized in {"open", "atonal", "open atonal", "open/atonal", "none"}:
        return mark_open_key_signature(m21key.KeySignature(0))

    for uc, ac in UNICODE_TO_ASCII_ACCIDENTALS.items():
        text = text.replace(uc, ac)

    try:
        val = int(text)
        return m21key.KeySignature(val)
    except ValueError:
        pass

    if text.endswith("m") and not text.endswith("major") and not text.endswith("minor"):
        tonic = text[:-1].strip()
        return m21key.Key(tonic, "minor")

    parts = text.split()
    if len(parts) == 2:
        tonic, mode = parts
        return m21key.Key(tonic, mode.lower())

    if len(parts) == 1:
        return m21key.Key(parts[0], "major")

    raise ValueError(
        f"Cannot parse key signature '{key_str}'. "
        f"Expected formats: 'C', 'C major', 'A minor', 'Bb', 'F#m', "
        f"or an integer for sharps/flats (e.g. '3', '-2')."
    )


def _direct_sounding_duration(stream_obj: m21stream.Stream) -> float:
    """Return the occupied duration of direct sounding events in a stream."""
    duration = 0.0
    for element in stream_obj:
        if not isinstance(element, (m21note.Note, m21chord.Chord)):
            continue
        if getattr(element.duration, "isGrace", False):
            continue
        offset = float(stream_obj.elementOffset(element))
        duration = max(duration, offset + float(element.quarterLength))
    return duration


def _total_sounding_duration(measure: m21stream.Measure) -> float:
    """Return the maximum occupied duration across measure voices.

    Explicit voices are independent rhythmic timelines, so simultaneous voices
    should not be summed when checking whether a meter can contain them.
    Rests and grace notes do not consume capacity.
    """
    durations = [_direct_sounding_duration(voice) for voice in measure.voices]
    direct_duration = _direct_sounding_duration(measure)
    if direct_duration > 0:
        durations.append(direct_duration)
    return max(durations, default=0.0)


def _measure_has_sounding_content(measure: m21stream.Measure) -> bool:
    """Return whether a measure contains duration-bearing notes or chords."""
    for element in measure.recurse().getElementsByClass(m21note.GeneralNote):
        if not isinstance(element, (m21note.Note, m21chord.Chord)):
            continue
        if getattr(element.duration, "isGrace", False):
            continue
        if float(element.duration.quarterLength) > 1e-9:
            return True
    return False


def _hidden_rests_cover_measure(
    measure: m21stream.Measure,
    quarter_length: float,
) -> bool:
    """Return whether hidden rest ranges fully cover a measure."""
    ranges = []
    for stream_obj in [measure, *list(measure.voices)]:
        for element in stream_obj.getElementsByClass(m21note.Rest):
            if not bool(getattr(element.style, "hideObjectOnPrint", False)):
                continue
            if getattr(element.duration, "isGrace", False):
                continue
            start = float(stream_obj.elementOffset(element))
            end = start + float(element.duration.quarterLength)
            ranges.append((start, end))

    cursor = 0.0
    for start, end in sorted(ranges):
        if start > cursor + 1e-9:
            return False
        cursor = max(cursor, end)
        if cursor >= quarter_length - 1e-9:
            return True
    return quarter_length <= 1e-9


def _collapse_rest_only_measure(
    measure: m21stream.Measure,
    quarter_length: float,
) -> None:
    """Collapse a rest-only measure to one full-measure rest."""
    if _measure_has_sounding_content(measure):
        return

    hidden = _hidden_rests_cover_measure(measure, quarter_length)
    for voice in list(measure.voices):
        measure.remove(voice)
    for element in list(measure.getElementsByClass(m21note.GeneralNote)):
        measure.remove(element)

    rest = m21note.Rest(quarterLength=quarter_length)
    if hidden:
        rest.style.hideObjectOnPrint = True
    measure.append(rest)


def _clear_measure_note_content(measure: m21stream.Measure) -> None:
    """Remove direct note-like elements and explicit voice streams."""
    for voice in list(measure.voices):
        measure.remove(voice)

    for element in list(measure.getElementsByClass(m21note.GeneralNote)):
        measure.remove(element)


# ---------------------------------------------------------------------------
# Mixin class
# ---------------------------------------------------------------------------


__all__ = [name for name in globals() if not name.startswith('__')]
