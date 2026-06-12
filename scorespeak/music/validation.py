"""
Music theory validation helpers for the ScoreSpeak framework.

Provides pitch normalization, duration parsing, beat validation,
range checking, and accidental inference from key signatures.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

from music21 import clef as m21clef
from music21 import instrument as m21instrument
from music21 import key as m21key
from music21 import meter as m21meter
from music21 import note as m21note
from music21 import pitch as m21pitch
from music21 import duration as m21duration

from ..types import (
    DURATION_ALIASES,
    DURATION_NAME_TO_QUARTER_LENGTH,
    UNICODE_TO_ASCII_ACCIDENTALS,
    ClefType,
    DurationInput,
    PitchInput,
)


MIN_VOICE_NUMBER = 1
MAX_VOICE_NUMBER = 4


# ---------------------------------------------------------------------------
# Pitch normalization
# ---------------------------------------------------------------------------

_PITCH_PATTERN = re.compile(
    r"^([A-Ga-g])"           # step
    r"([#♯b♭♮-]*)"           # accidentals (zero or more)
    r"(-?\d+)?$"              # optional octave
)


def normalize_pitch(raw: PitchInput) -> m21pitch.Pitch:
    """Convert flexible pitch input to a music21 Pitch object.

    Accepts:
        - String: "C4", "c#4", "C♯4", "Db3", "D♭3", "E4"
        - Integer (MIDI number): 60 → C4
        - music21 Pitch object: returned as-is

    Raises:
        ValueError: If the input cannot be parsed as a valid pitch.
    """
    if isinstance(raw, m21pitch.Pitch):
        return raw

    if isinstance(raw, (int, float)):
        midi_num = int(raw)
        if not 0 <= midi_num <= 127:
            raise ValueError(
                f"MIDI number {midi_num} is out of range (valid: 0–127)."
            )
        p = m21pitch.Pitch()
        p.midi = midi_num
        return p

    if not isinstance(raw, str):
        raise ValueError(
            f"Cannot interpret {type(raw).__name__} as a pitch. "
            f"Expected a string like 'C4', an integer MIDI number, "
            f"or a music21 Pitch object."
        )

    text = raw.strip()
    for unicode_char, ascii_char in UNICODE_TO_ASCII_ACCIDENTALS.items():
        text = text.replace(unicode_char, ascii_char)

    match = _PITCH_PATTERN.match(text)
    if not match:
        raise ValueError(
            f"Cannot parse '{raw}' as a pitch. "
            f"Expected format like 'C4', 'C#4', 'Db3', or a MIDI number."
        )

    step = match.group(1).upper()
    accidental_str = match.group(2)
    octave_str = match.group(3)

    flats = sum(1 for ch in accidental_str if ch in "b♭-")
    sharps = sum(1 for ch in accidental_str if ch in "#♯")
    if flats > 0 and sharps > 0:
        raise ValueError(
            f"Cannot parse '{raw}' as a pitch: "
            f"mixed sharp and flat accidentals are not supported."
        )

    octave_suffix = octave_str if octave_str is not None else ""

    if flats == 0 and sharps == 0:
        name = step + accidental_str + octave_suffix
    elif flats == 1 and sharps == 0:
        name = step + "b" + octave_suffix
    elif flats >= 2 and sharps == 0:
        name = step + ("-" * flats) + octave_suffix
    elif sharps >= 1 and flats == 0:
        name = step + ("#" * sharps) + octave_suffix
    else:
        name = step + accidental_str + octave_suffix

    try:
        return m21pitch.Pitch(name)
    except Exception as exc:
        raise ValueError(
            f"Cannot parse '{raw}' as a pitch: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Duration normalization
# ---------------------------------------------------------------------------

def normalize_duration(raw: DurationInput, dots: int = 0) -> m21duration.Duration:
    """Convert flexible duration input to a music21 Duration.

    Accepts:
        - String name: "quarter", "half", "eighth", "16th", "whole"
        - Numeric alias: "4" → quarter, "8" → eighth, etc.
        - Float/int quarter-length: 1.0 → quarter, 0.5 → eighth
        - Dotted: dots parameter adds augmentation dots

    Raises:
        ValueError: If the input cannot be parsed as a valid duration.
    """
    if isinstance(raw, (int, float)):
        ql = float(raw)
        if ql <= 0:
            raise ValueError(
                f"Duration quarter-length must be positive, got {ql}."
            )
        d = m21duration.Duration(quarterLength=ql)
        if dots:
            d.dots = dots
        return d

    if not isinstance(raw, str):
        raise ValueError(
            f"Cannot interpret {type(raw).__name__} as a duration."
        )

    text = raw.strip().lower()

    if text in DURATION_ALIASES:
        text = DURATION_ALIASES[text]

    if text in DURATION_NAME_TO_QUARTER_LENGTH:
        d = m21duration.Duration(type=text)
        if dots:
            d.dots = dots
        return d

    try:
        ql = float(text)
    except ValueError:
        ql = None
    if ql is not None:
        if ql <= 0:
            raise ValueError(f"Duration must be positive, got {ql}.")
        d = m21duration.Duration(quarterLength=ql)
        if dots:
            d.dots = dots
        return d

    raise ValueError(
        f"Cannot parse '{raw}' as a duration. "
        f"Try entering a quarter-length number (e.g., 1.0)."
    )


# ---------------------------------------------------------------------------
# Beat / measure validation
# ---------------------------------------------------------------------------

def validate_beat_capacity(
    time_sig: m21meter.TimeSignature,
    existing_quarter_lengths: float,
    new_quarter_length: float,
    measure_number: int,
    beat_position: Optional[float] = None,
) -> None:
    """Validate that adding a note won't exceed the measure's beat capacity.

    Raises:
        ValueError: With a musically descriptive error message.
    """
    capacity = time_sig.barDuration.quarterLength
    if beat_position is not None:
        used_after = beat_position - 1.0 + new_quarter_length
    else:
        used_after = existing_quarter_lengths + new_quarter_length

    if used_after > capacity + 1e-9:
        overflow = used_after - capacity
        ts_str = time_sig.ratioString
        if beat_position is not None:
            raise ValueError(
                f"Measure {measure_number} in {ts_str} time can hold "
                f"{capacity} beats, but adding a "
                f"{_quarter_length_to_name(new_quarter_length)} note "
                f"at beat {beat_position} would exceed this by "
                f"{overflow:.4g} beats."
            )
        else:
            raise ValueError(
                f"Measure {measure_number} in {ts_str} time can hold "
                f"{capacity} beats, but it already contains "
                f"{existing_quarter_lengths} beats and adding a "
                f"{_quarter_length_to_name(new_quarter_length)} note "
                f"would exceed this by {overflow:.4g} beats."
            )


def _quarter_length_to_name(ql: float) -> str:
    """Best-effort conversion of a quarter-length to a readable name."""
    for name, length in sorted(
        DURATION_NAME_TO_QUARTER_LENGTH.items(), key=lambda x: -x[1]
    ):
        if abs(ql - length) < 1e-9:
            return name
    return f"{ql}-beat"


# ---------------------------------------------------------------------------
# Voice validation
# ---------------------------------------------------------------------------

def validate_voice_number(value: object, field_name: str = "voice") -> int:
    """Return a supported public voice number or raise a clear error."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be an integer from {MIN_VOICE_NUMBER} to "
            f"{MAX_VOICE_NUMBER}, got {value!r}."
        )
    if value < MIN_VOICE_NUMBER or value > MAX_VOICE_NUMBER:
        raise ValueError(
            f"{field_name} must be between {MIN_VOICE_NUMBER} and "
            f"{MAX_VOICE_NUMBER}, got {value}."
        )
    return value


# ---------------------------------------------------------------------------
# Key signature / accidental helpers
# ---------------------------------------------------------------------------

# Sharps and flats in key signature order
_SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
_FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]


def get_key_accidentals(key_obj: m21key.KeySignature) -> dict[str, str]:
    """Return a dict mapping step names to their accidental in the given key.

    E.g., for B♭ major (2 flats): {"B": "flat", "E": "flat"}
    """
    sharps = key_obj.sharps
    result: dict[str, str] = {}
    if sharps > 0:
        for i in range(min(sharps, 7)):
            result[_SHARP_ORDER[i]] = "sharp"
    elif sharps < 0:
        for i in range(min(-sharps, 7)):
            result[_FLAT_ORDER[i]] = "flat"
    return result


def needs_explicit_accidental(
    pitch_obj: m21pitch.Pitch,
    key_obj: m21key.KeySignature,
) -> bool:
    """Determine if a pitch needs an explicit accidental given the key.

    Returns True if the pitch's accidental differs from what the key implies.
    """
    key_accidentals = get_key_accidentals(key_obj)
    step = pitch_obj.step

    key_acc = key_accidentals.get(step)
    note_acc = pitch_obj.accidental

    if key_acc is None:
        return note_acc is not None and note_acc.name != "natural"
    else:
        if note_acc is None or note_acc.name == "natural":
            return True
        return note_acc.name != key_acc


# ---------------------------------------------------------------------------
# Instrument range validation
# ---------------------------------------------------------------------------

_COMMON_RANGES: dict[str, tuple[str, str]] = {
    "soprano": ("C4", "C6"),
    "mezzo-soprano": ("A3", "A5"),
    "alto": ("F3", "F5"),
    "tenor": ("C3", "C5"),
    "baritone": ("A2", "A4"),
    "bass": ("E2", "E4"),
    "violin": ("G3", "E7"),
    "viola": ("C3", "E6"),
    "cello": ("C2", "C6"),
    "double bass": ("E1", "G4"),
    "contrabass": ("E1", "G4"),
    "flute": ("C4", "D7"),
    "oboe": ("Bb3", "A6"),
    "clarinet": ("D3", "Bb6"),
    "bassoon": ("Bb1", "Eb5"),
    "trumpet": ("F#3", "D6"),
    "french horn": ("B1", "F5"),
    "horn": ("B1", "F5"),
    "trombone": ("E2", "F5"),
    "tuba": ("D1", "F4"),
    "piano": ("A0", "C8"),
}


def validate_pitch_in_range(
    pitch_obj: m21pitch.Pitch,
    instrument_name: str,
    strict: bool = False,
) -> Optional[str]:
    """Check if a pitch is within the typical range for an instrument.

    Returns a warning string if out of range, or None if in range.
    Raises ValueError only if strict=True and out of range.
    """
    name_lower = instrument_name.lower().strip()
    range_tuple = _COMMON_RANGES.get(name_lower)
    if range_tuple is None:
        return None

    low = m21pitch.Pitch(range_tuple[0])
    high = m21pitch.Pitch(range_tuple[1])

    if pitch_obj.midi < low.midi or pitch_obj.midi > high.midi:
        msg = (
            f"The pitch {pitch_obj.nameWithOctave} is outside the typical "
            f"range for {instrument_name} ({range_tuple[0]}–{range_tuple[1]}). "
            f"This may be difficult or impossible to perform."
        )
        if strict:
            raise ValueError(msg)
        return msg

    return None


# ---------------------------------------------------------------------------
# Clef helpers
# ---------------------------------------------------------------------------

_CLEF_MAP: dict[str, type] = {
    "treble": m21clef.TrebleClef,
    "bass": m21clef.BassClef,
    "alto": m21clef.AltoClef,
    "tenor": m21clef.TenorClef,
    "soprano": m21clef.SopranoClef,
    "mezzo-soprano": m21clef.MezzoSopranoClef,
    "baritone": m21clef.CBaritoneClef,
    "percussion": m21clef.PercussionClef,
    "tab": m21clef.TabClef,
    "treble8vb": m21clef.Treble8vbClef,
    "treble8va": m21clef.Treble8vaClef,
    "bass8vb": m21clef.Bass8vbClef,
}

_DEFAULT_CLEF_CONVENTIONS: tuple[
    tuple[tuple[type[m21instrument.Instrument], ...], type[m21clef.Clef]],
    ...
] = (
    (
        (
            m21instrument.Violin,
            m21instrument.Flute,
            m21instrument.Piccolo,
            m21instrument.Oboe,
            m21instrument.EnglishHorn,
            m21instrument.Clarinet,
            m21instrument.Trumpet,
            m21instrument.Horn,
            m21instrument.Soprano,
            m21instrument.Alto,
            m21instrument.SopranoSaxophone,
            m21instrument.AltoSaxophone,
            m21instrument.TenorSaxophone,
            m21instrument.BaritoneSaxophone,
            m21instrument.Piano,
            m21instrument.Harp,
            m21instrument.Xylophone,
            m21instrument.Marimba,
            m21instrument.Vibraphone,
            m21instrument.Glockenspiel,
        ),
        m21clef.TrebleClef,
    ),
    ((m21instrument.Viola,), m21clef.AltoClef),
    (
        (
            m21instrument.Violoncello,
            m21instrument.Contrabass,
            m21instrument.Bassoon,
            m21instrument.Trombone,
            m21instrument.Tuba,
            m21instrument.Timpani,
            m21instrument.Baritone,
            m21instrument.Bass,
        ),
        m21clef.BassClef,
    ),
    (
        (
            m21instrument.AcousticGuitar,
            m21instrument.ElectricGuitar,
            m21instrument.Tenor,
        ),
        m21clef.Treble8vbClef,
    ),
    (
        (
            m21instrument.SnareDrum,
            m21instrument.BassDrum,
            m21instrument.Percussion,
        ),
        m21clef.PercussionClef,
    ),
)


def default_clef_for_instrument(inst: m21instrument.Instrument) -> m21clef.Clef:
    """Choose a reasonable default clef for an instrument.

    Uses common notation conventions for known music21 instrument classes.
    Unknown instruments fall back to treble clef.
    """
    for instrument_types, clef_type in _DEFAULT_CLEF_CONVENTIONS:
        if isinstance(inst, instrument_types):
            return clef_type()

    return m21clef.TrebleClef()


def make_clef(clef_input: str | ClefType) -> m21clef.Clef:
    """Create a music21 Clef from a string or ClefType enum."""
    name = clef_input.value if isinstance(clef_input, ClefType) else clef_input
    name_lower = name.lower().strip()
    clef_cls = _CLEF_MAP.get(name_lower)
    if clef_cls is None:
        raise ValueError(
            f"Unknown clef type '{name}'. Valid types: "
            f"{', '.join(sorted(_CLEF_MAP.keys()))}"
        )
    return clef_cls()
