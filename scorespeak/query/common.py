"""
Bar retrieval operations for ScoreSpeak.

Provides a compact, bar-first content retrieval API over notes, rests,
chords, voices, tuplets, and the full surrounding notation: dynamics,
hairpins, slurs, articulations, ornaments, lyrics, fingerings,
arpeggios, chord symbols, tempo / barline / repeat / ending /
rehearsal / navigation markers, and active time / key / clef context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

from music21 import articulations as m21articulations
from music21 import bar as m21bar
from music21 import chord as m21chord
from music21 import clef as m21clef
from music21 import dynamics as m21dynamics
from music21 import expressions as m21expressions
from music21 import harmony as m21harmony
from music21 import key as m21key
from music21 import layout as m21layout
from music21 import meter as m21meter
from music21 import note as m21note
from music21 import pitch as m21pitch
from music21 import repeat as m21repeat
from music21 import spanner as m21spanner
from music21 import stream as m21stream
from music21 import tempo as m21tempo

from ..music.pitch_space import (
    part_stores_sounding_pitch,
    part_transposition_interval,
    stored_key_signature_for_concert_key,
    stored_pitch_space_label,
)
from ..score.staff_groups import PartLabel, build_part_display_labels
from ..types import (
    ActiveSignatures,
    BarEventRow,
    BarGroup,
    BarNotation,
    BarPart,
    BarQuery,
    BarQueryEvent,
    BarQueryInput,
    BarResultSet,
    BarVoice,
    DurationInput,
    MarkingRow,
    PartNotation,
    PitchInput,
    SpanRow,
    TupletInfo,
    TupletSpanRow,
)
from ..music.validation import normalize_duration, normalize_pitch, validate_voice_number

EVENT_SCHEMA = [
    "kind",
    "beat",
    "pitch",
    "duration",
    "tie_status",
    "is_grace",
    "dots",
    "grace_slash",
    "grace_duration",
]

TUPLET_SCHEMA = [
    "ratio",
    "beat_range",
]

MARKING_SCHEMA = [
    "type",
    "payload",
    "beat",
]

SPAN_SCHEMA = [
    "type",
    "payload",
    "flags",
    "beat_range",
]

BAR_NOTATION_KEYS = [
    "active",
    "changed_here",
    "barline_start",
    "barline_end",
    "repeat_start",
    "repeat_end",
    "ending_number",
    "rehearsal_mark",
    "navigation",
    "system_break",
    "page_break",
]

PART_NOTATION_KEYS = [
    "clef",
    "key",
    "concert_key",
    "key_space",
    "key_role",
    "key_is_transposed",
    "key_label",
]

MATCH_SCHEMA = [
    "channel",
    "detail",
    "part_index",
    "voice",
    "beat",
    "beat_range",
]

SUPPORTED_EVENT_KINDS = {"note", "rest", "chord", "any"}
SUPPORTED_CHORD_MODES = {"exact", "contains"}


# Articulation subclasses that are reported separately as their own
# marking type rather than being rolled up into the generic
# ``articulation`` bucket.
_FINGERING_CLASSES: tuple[type, ...] = (m21articulations.Fingering,)

# Navigation-mark classes (exposed as ``navigation`` on the bar).
_NAVIGATION_MARK_LABELS: tuple[tuple[type, str], ...] = (
    (m21repeat.Segno, "segno"),
    (m21repeat.DaCapoAlFine, "da_capo_al_fine"),
    (m21repeat.DaCapoAlCoda, "da_capo_al_coda"),
    (m21repeat.DaCapo, "da_capo"),
    (m21repeat.DalSegnoAlFine, "dal_segno_al_fine"),
    (m21repeat.DalSegnoAlCoda, "dal_segno_al_coda"),
    (m21repeat.DalSegno, "dal_segno"),
    (m21repeat.Fine, "fine"),
)
_CODA_LABEL = "coda"
_TO_CODA_LABEL = "to_coda"

# Ornament classes we report under the ``ornament`` marking type.
_ORNAMENT_CLASSES: tuple[type, ...] = (
    m21expressions.Trill,
    m21expressions.InvertedTrill,
    m21expressions.Turn,
    m21expressions.InvertedTurn,
    m21expressions.Mordent,
    m21expressions.InvertedMordent,
    m21expressions.Schleifer,
    m21expressions.Shake,
    m21expressions.Tremolo,
)


def _normalize_navigation_text(text: object) -> str:
    """Return navigation text normalized for case-insensitive matching."""
    return " ".join(str(text or "").strip().lower().split())


def _coda_navigation_label(coda: m21repeat.Coda) -> str:
    """Return the bar-retrieval navigation label for a Coda object."""
    if _normalize_navigation_text(coda.getText()) == "to coda":
        return _TO_CODA_LABEL
    return _CODA_LABEL


# Fermata is stored under ``element.expressions`` in music21 but is
# semantically an articulation for our purposes; report it that way.
_FERMATA_CLASSES: tuple[type, ...] = (m21expressions.Fermata,)

# Span types: (music21 class, our label string). Order matters for
# class-hierarchy checks (more specific classes first).
_SPAN_TYPES: tuple[tuple[type, str], ...] = (
    (m21dynamics.Crescendo, "hairpin"),
    (m21dynamics.Diminuendo, "hairpin"),
    (m21dynamics.DynamicWedge, "hairpin"),
    (m21spanner.Slur, "slur"),
    (m21spanner.Ottava, "ottava"),
    (m21spanner.Glissando, "glissando"),
    (m21expressions.PedalMark, "pedal"),
)


@dataclass
class VoiceEvent:
    """Normalized event used internally for matching and output projection.

    ``tie_status`` is one of ``"start"``, ``"continue"``, ``"stop"``, or
    ``None``; the value mirrors ``music21.note.Tie.type`` where present.
    ``offset`` is the 0-based quarter-length offset of the event from
    the start of its measure, retained so measure-level markings
    (dynamics, chord symbols, tempo) can be associated with the nearest
    event by offset.
    """

    kind: str
    beat: float
    duration: float
    part_index: int
    measure_number: int
    voice: int
    tie_status: Optional[str]
    is_grace: bool
    dots: int
    grace_slash: Optional[bool] = None
    grace_duration: Optional[str] = None
    offset: float = 0.0
    pitch: Optional[str] = None
    pitches: Optional[list[str]] = None
    pitch_classes: Optional[list[str]] = None
    tuplets: list[TupletInfo] | None = None
    _source: Optional[m21note.GeneralNote] = None


@dataclass
class ParsedQueryEvent:
    """Validated sequence element for voice-local matching."""

    kind: str
    pitch: Optional[str] = None
    duration: Optional[float] = None
    pitch_classes: Optional[list[str]] = None


@dataclass
class ParsedBarQuery:
    """Validated and normalized bar retrieval query."""

    parts: list[tuple[m21stream.Part, int]]
    measure_numbers: list[int]
    scoped_voices: Optional[set[int]]
    sequence: Optional[list[ParsedQueryEvent]]
    chord_mode: str


@dataclass(frozen=True)
class BarPayloadOptions:
    """Control which optional channels are collected for bar payloads."""

    include_events: bool = True
    include_part_notation: bool = True
    include_tuplets: bool = True
    include_markings: bool = True
    include_spans: bool = True


__all__ = [name for name in globals() if not name.startswith('__')]
