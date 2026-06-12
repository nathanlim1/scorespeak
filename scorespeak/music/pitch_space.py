"""Helpers for concert and written pitch-space handling."""

from __future__ import annotations

import re
from typing import Optional

from music21 import instrument as m21instrument
from music21 import interval as m21interval
from music21 import key as m21key
from music21 import stream as m21stream


OPEN_KEY_SIGNATURE_LABEL = "open/atonal"
_OPEN_KEY_SIGNATURE_ATTR = "_scorespeak_open_key_signature"
_LOCAL_KEY_OVERRIDE_ATTR = "_scorespeak_local_key_override"
_TEXTUAL_ACCIDENTAL_PATTERN = re.compile(
    r"\b([A-Ga-g])(?:\s*-\s*|\s+)(flat|sharp)\b",
    re.IGNORECASE,
)
_SYMBOL_ACCIDENTAL_PATTERN = re.compile(r"\b([A-Ga-g])\s*([#b])\b")


def normalize_instrument_label(label: object) -> str:
    """Normalize instrument label accidentals before instrument lookup."""
    text = str(label or "")
    text = (
        text.replace("♭", "b")
        .replace("𝄫", "bb")
        .replace("♯", "#")
        .replace("𝄪", "##")
    )

    def replace_textual_accidental(match: re.Match[str]) -> str:
        """Return a compact pitch spelling for a textual accidental match."""
        note_name = match.group(1).upper()
        accidental = match.group(2).lower()
        suffix = "b" if accidental == "flat" else "#"
        return f"{note_name}{suffix}"

    text = _TEXTUAL_ACCIDENTAL_PATTERN.sub(replace_textual_accidental, text)

    def replace_symbol_accidental(match: re.Match[str]) -> str:
        """Return a compact pitch spelling for a symbolic accidental match."""
        note_name = match.group(1).upper()
        accidental = match.group(2)
        return f"{note_name}{accidental}"

    return _SYMBOL_ACCIDENTAL_PATTERN.sub(replace_symbol_accidental, text)


def mark_open_key_signature(
    key_signature: m21key.KeySignature,
) -> m21key.KeySignature:
    """Mark ``key_signature`` as an open/atonal local signature."""
    setattr(key_signature, _OPEN_KEY_SIGNATURE_ATTR, True)
    return key_signature


def is_open_key_signature(key_signature: m21key.KeySignature) -> bool:
    """Return whether ``key_signature`` is a ScoreSpeak open/atonal marker."""
    return bool(getattr(key_signature, _OPEN_KEY_SIGNATURE_ATTR, False))


def mark_local_key_override(part: m21stream.Part) -> None:
    """Mark ``part`` as intentionally carrying a local key-signature timeline."""
    setattr(part, _LOCAL_KEY_OVERRIDE_ATTR, True)


def has_marked_local_key_override(part: m21stream.Part) -> bool:
    """Return whether ``part`` was marked with a local key override."""
    return bool(getattr(part, _LOCAL_KEY_OVERRIDE_ATTR, False))


def copy_key_signature(
    key_signature: m21key.KeySignature,
) -> m21key.KeySignature:
    """Return a detached copy of ``key_signature`` preserving local markers."""
    if is_open_key_signature(key_signature):
        return mark_open_key_signature(m21key.KeySignature(0))
    if isinstance(key_signature, m21key.Key):
        return m21key.Key(key_signature.tonic.name, key_signature.mode)
    return m21key.KeySignature(key_signature.sharps)


def part_transposition_interval(
    part: m21stream.Part,
) -> Optional[m21interval.Interval]:
    """Return the written-to-sounding transposition interval for ``part``."""
    instrument = _first_part_instrument(part)
    if instrument is None:
        return None

    interval = getattr(instrument, "transposition", None)
    if interval is None:
        return None

    semitones = getattr(interval, "semitones", None)
    if semitones is None or int(semitones) == 0:
        return None
    return interval


def part_stores_sounding_pitch(part: m21stream.Part) -> bool:
    """Return whether ``part`` currently stores concert/sounding notation."""
    if part_transposition_interval(part) is None:
        return True
    return getattr(part, "atSoundingPitch", None) is True


def set_part_stores_sounding_pitch(
    part: m21stream.Part,
    stores_sounding: bool,
) -> None:
    """Set the stored pitch-space flag on ``part``."""
    part.atSoundingPitch = bool(stores_sounding)


def default_stored_pitch_space_for_part(part: m21stream.Part) -> None:
    """Initialize ``part`` to conventional storage for its instrument."""
    set_part_stores_sounding_pitch(
        part,
        part_transposition_interval(part) is None,
    )


def stored_pitch_space_label(part: m21stream.Part) -> str:
    """Return a human label for the notation currently stored in ``part``."""
    if part_stores_sounding_pitch(part):
        return "sounding pitch"
    return "written pitch"


def stored_key_signature_for_concert_key(
    part: m21stream.Part,
    concert_key_signature: m21key.KeySignature,
) -> m21key.KeySignature:
    """Return the key signature that should be stored in ``part``."""
    if is_open_key_signature(concert_key_signature):
        return copy_key_signature(concert_key_signature)

    interval = part_transposition_interval(part)
    if interval is None or part_stores_sounding_pitch(part):
        return copy_key_signature(concert_key_signature)

    return _transpose_key_signature(
        concert_key_signature,
        interval.reverse(),
    )


def concert_key_signature_for_stored_key(
    part: m21stream.Part,
    stored_key_signature: m21key.KeySignature,
) -> m21key.KeySignature:
    """Return the concert key represented by ``part``'s stored key."""
    if is_open_key_signature(stored_key_signature):
        return copy_key_signature(stored_key_signature)

    interval = part_transposition_interval(part)
    if interval is None or part_stores_sounding_pitch(part):
        return copy_key_signature(stored_key_signature)

    return _transpose_key_signature(stored_key_signature, interval)


def _first_part_instrument(
    part: m21stream.Part,
) -> Optional[m21instrument.Instrument]:
    """Return the preferred initial instrument found in ``part``."""
    instruments = list(part.getElementsByClass(m21instrument.Instrument))
    if instruments:
        first_offset = _instrument_offset(instruments[0])
        initial_instruments = [
            instrument
            for instrument in instruments
            if _instrument_offset(instrument) == first_offset
        ]
        for instrument in reversed(initial_instruments):
            interval = getattr(instrument, "transposition", None)
            semitones = getattr(interval, "semitones", None)
            if semitones is not None and int(semitones) != 0:
                return instrument
        return initial_instruments[-1]

    instrument = part.getInstrument(returnDefault=False)
    if isinstance(instrument, m21instrument.Instrument):
        return instrument
    return None


def _instrument_offset(instrument: m21instrument.Instrument) -> float:
    """Return an instrument offset suitable for deterministic ordering."""
    try:
        return float(instrument.offset)
    except (TypeError, ValueError):
        return 0.0


def _transpose_key_signature(
    key_signature: m21key.KeySignature,
    interval: m21interval.Interval,
) -> m21key.KeySignature:
    """Return ``key_signature`` transposed by ``interval``."""
    transposed = key_signature.transpose(interval)
    if isinstance(transposed, m21key.KeySignature):
        return transposed
    return m21key.KeySignature(transposed.sharps)
