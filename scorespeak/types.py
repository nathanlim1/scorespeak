"""
Type definitions, enums, and constants for the ScoreSpeak framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NotRequired, Optional, TypeAlias, TypedDict, Union


# ---------------------------------------------------------------------------
# Pitch helpers
# ---------------------------------------------------------------------------

UNICODE_TO_ASCII_ACCIDENTALS = {
    "\u266f": "#",   # ♯ → #
    "\u266d": "b",   # ♭ → b
    "\U0001D12A": "##",  # 𝄪 → ##
    "\U0001D12B": "bb",  # 𝄫 → bb
    "\u266e": "",     # ♮ → (natural, strip)
}

DURATION_NAME_TO_QUARTER_LENGTH: dict[str, float] = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "8th": 0.5,
    "16th": 0.25,
    "32nd": 0.125,
    "64th": 0.0625,
    "128th": 0.03125,
    "breve": 8.0,
    "longa": 16.0,
}

DURATION_ALIASES: dict[str, str] = {
    "1": "whole",
    "2": "half",
    "4": "quarter",
    "8": "eighth",
    "16": "16th",
    "32": "32nd",
    "64": "64th",
    "128": "128th",
}

VALID_DYNAMICS = [
    "pppppp", "ppppp", "pppp", "ppp", "pp", "p",
    "mp", "mf",
    "f", "ff", "fff", "ffff", "fffff", "ffffff",
    "fp", "fz", "sf", "sfz", "sfp", "sfpp",
    "sffz", "rfz", "rf",
]


class DynamicLevel(str, Enum):
    """Standard dynamic markings."""
    PPPP = "pppp"
    PPP = "ppp"
    PP = "pp"
    P = "p"
    MP = "mp"
    MF = "mf"
    F = "f"
    FF = "ff"
    FFF = "fff"
    FFFF = "ffff"
    FP = "fp"
    SF = "sf"
    SFZ = "sfz"
    FZ = "fz"
    SFP = "sfp"
    SFPP = "sfpp"
    SFFZ = "sffz"
    RFZ = "rfz"
    RF = "rf"


class ArticulationType(str, Enum):
    """Standard articulation types."""
    STACCATO = "staccato"
    STACCATISSIMO = "staccatissimo"
    ACCENT = "accent"
    STRONG_ACCENT = "strong accent"
    TENUTO = "tenuto"
    FERMATA = "fermata"
    BREATH_MARK = "breath mark"
    CAESURA = "caesura"


class BarlineType(str, Enum):
    """Barline types."""
    REGULAR = "regular"
    DOUBLE = "double"
    FINAL = "final"
    LIGHT_HEAVY = "light-heavy"
    LIGHT_LIGHT = "light-light"
    NONE = "none"


class RepeatDirection(str, Enum):
    """Repeat barline direction."""
    START = "start"
    END = "end"


class ClefType(str, Enum):
    """Common clef types."""
    TREBLE = "treble"
    BASS = "bass"
    ALTO = "alto"
    TENOR = "tenor"
    SOPRANO = "soprano"
    MEZZO_SOPRANO = "mezzo-soprano"
    BARITONE = "baritone"
    PERCUSSION = "percussion"
    TAB = "tab"
    TREBLE_8VB = "treble8vb"
    TREBLE_8VA = "treble8va"
    BASS_8VB = "bass8vb"


class HairpinType(str, Enum):
    """Hairpin / wedge types."""
    CRESCENDO = "crescendo"
    DIMINUENDO = "diminuendo"
    DECRESCENDO = "decrescendo"


# ---------------------------------------------------------------------------
# Result dataclasses — structured return values for every mutation
# ---------------------------------------------------------------------------

@dataclass
class OperationResult:
    """Base result from any ScoreSpeak operation."""
    success: bool
    description: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScorePartSpec:
    """Requested logical score part for instrumentation setup.

    ``instrument`` is required and names the playable instrument, including
    transposition or variant when relevant, such as "Bb clarinet", "Eb horn",
    "C trumpet", or "bass clarinet". ``name`` is the optional displayed part
    name; when omitted, the instrument default is used.
    """
    instrument: str
    name: Optional[str] = None


@dataclass
class TupletInfo:
    """Structured info about one tuplet attached to a note-like event."""
    actual_notes: int
    normal_notes: int


@dataclass
class NoteInfo:
    """Structured info about a note-like event in the score."""
    pitch: str
    octave: int
    duration_type: str
    quarter_length: float
    measure_number: int
    beat: float
    part_index: int
    voice: int
    is_chord: bool = False
    is_rest: bool = False
    is_tied: bool = False
    is_grace: bool = False
    dots: int = 0
    tuplets: list[TupletInfo] = field(default_factory=list)


@dataclass
class MeasureInfo:
    """Structured info about a measure."""
    number: int
    time_signature: str
    key_signature: str
    clef: str
    tempo: Optional[float]
    beat_count: float
    notes_count: int
    rests_count: int


@dataclass
class PartInfo:
    """Structured info about a part.

    ``name`` is the raw ``music21`` ``partName`` (unchanged for backward
    compatibility). ``display_name`` is the agent-facing label that
    disambiguates grand-staff siblings (``"Piano RH"`` / ``"Piano LH"``)
    and defaults to ``name`` when no special label applies. ``hand`` is
    ``"RH" | "LH" | "Pedal" | None``; ``None`` means the part does not
    belong to any detected brace group.
    """
    index: int
    name: str
    instrument: str
    measure_count: int
    display_name: Optional[str] = None
    hand: Optional[str] = None


@dataclass
class LyricInfo:
    """One lyric syllable attached to a note or chord."""
    text: str
    measure_number: int
    beat: float
    part_index: int
    voice: int
    lyric_number: int
    syllabic: Optional[str]
    pitch_or_chord: str


PitchInput = Union[str, int, "music21.pitch.Pitch"]  # type: ignore[name-defined]
DurationInput = Union[str, float]

BarEventRow: TypeAlias = list[Any]
TupletSpanRow: TypeAlias = list[Any]
MarkingRow: TypeAlias = list[Any]
SpanRow: TypeAlias = list[Any]
SearchMatchRow: TypeAlias = list[Any]


class ActiveSignatures(TypedDict):
    """Active time/key/tempo values at one bar.

    ``time`` and ``key`` are always present. ``key`` is the score-level
    concert key; ``concert_key`` is an explicit alias for agent-facing
    payloads. ``tempo`` only appears on the first bar of the retrieved
    scope and on bars where it changes.
    """
    time: str
    key: str
    concert_key: NotRequired[str]
    key_space: NotRequired[str]
    tempo: NotRequired[float]


class BarNotation(TypedDict):
    """Bar-level notation envelope shared across all parts in a bar.

    Only ``active`` is always present; every other key is emitted only
    when applicable.
    """
    active: ActiveSignatures
    changed_here: NotRequired[list[str]]
    barline_start: NotRequired[str]
    barline_end: NotRequired[str]
    repeat_start: NotRequired[bool]
    repeat_end: NotRequired[bool]
    ending_number: NotRequired[str]
    rehearsal_mark: NotRequired[str]
    navigation: NotRequired[list[str]]
    system_break: NotRequired[bool]
    page_break: NotRequired[bool]


class PartNotation(TypedDict):
    """Per-part notation envelope.

    ``clef`` is emitted on scope bar 1 and bars where the clef changes;
    ``key`` is emitted when this part's displayed/stored key differs from
    the score-level concert key. In that case ``concert_key`` and
    ``key_label`` make it explicit whether the displayed key is a transposed
    written key or a local staff key. On other bars the whole ``notation``
    key is omitted from the part payload.
    """
    clef: NotRequired[str]
    key: NotRequired[str]
    concert_key: NotRequired[str]
    key_space: NotRequired[str]
    key_role: NotRequired[str]
    key_is_transposed: NotRequired[bool]
    key_label: NotRequired[str]


class BarVoice(TypedDict):
    """Compact payload for one voice within one returned bar.

    ``markings`` and ``spans`` are only present when the voice has
    point-event or span-style notations in this bar.
    """
    voice: int
    events: list[BarEventRow]
    tuplets: list[TupletSpanRow]
    markings: NotRequired[list[MarkingRow]]
    spans: NotRequired[list[SpanRow]]


class BarPart(TypedDict):
    """Compact payload for one part within one returned bar.

    ``part_name`` is the agent-facing display name (e.g. ``"Piano RH"``
    when the part is the top staff of a brace group). ``hand`` is only
    populated for parts inside a detected grand-staff group and takes
    ``"RH" | "LH" | "Pedal"`` values.
    """
    part_index: int
    part_name: str
    voices: list[BarVoice]
    hand: NotRequired[str]
    notation: NotRequired[PartNotation]


class BarGroup(TypedDict):
    """Compact payload for one returned measure."""
    measure_number: int
    parts: list[BarPart]
    notation: BarNotation
    matches: NotRequired[list[SearchMatchRow]]


class BarResultSet(TypedDict):
    """Top-level result from the internal bar-result projection."""
    event_schema: list[str]
    tuplet_schema: list[str]
    marking_schema: list[str]
    span_schema: list[str]
    bar_notation_keys: list[str]
    part_notation_keys: list[str]
    match_schema: NotRequired[list[str]]
    search_metadata: NotRequired[dict[str, Any]]
    bars: list[BarGroup]


class BarQueryScope(TypedDict):
    """Scope filters for internal bar-result projection."""
    parts: NotRequired[list[Union[int, str]]]
    bar_range: NotRequired[tuple[int, int]]
    measure_numbers: NotRequired[list[int]]
    voices: NotRequired[list[int]]


class BarQueryEvent(TypedDict):
    """One sequence element in an internal event-sequence query."""
    kind: str
    pitch: NotRequired[PitchInput]
    duration: NotRequired[DurationInput]
    pitch_classes: NotRequired[list[str]]


class BarQueryMatch(TypedDict):
    """Pattern matching clause for internal event-sequence queries."""
    sequence: list[BarQueryEvent]


class BarQueryOptions(TypedDict):
    """Optional behavior toggles for internal bar-result projection."""
    chord_mode: NotRequired[str]


class BarQuery(TypedDict):
    """Structured query object for internal bar-result projection."""
    scope: NotRequired[BarQueryScope]
    match: NotRequired[BarQueryMatch]
    options: NotRequired[BarQueryOptions]


BarQueryInput: TypeAlias = BarQuery
