"""
Dynamics, articulations, hairpins, slurs, text expressions, tempo,
and rehearsal mark operations for ScoreSpeak.

Provides methods for adding and removing expressive markings on a score.
"""

from __future__ import annotations

import re
from typing import Optional, Union

from music21 import articulations as m21articulations
from music21 import duration as m21duration
from music21 import dynamics as m21dynamics
from music21 import expressions as m21expressions
from music21 import harmony as m21harmony
from music21 import note as m21note
from music21 import spanner as m21spanner
from music21 import stream as m21stream
from music21 import tempo as m21tempo

from ..types import (
    VALID_DYNAMICS,
    ArticulationType,
    DynamicLevel,
    HairpinType,
    OperationResult,
)


_ARTICULATION_MAP: dict[str, type] = {
    "staccato": m21articulations.Staccato,
    "staccatissimo": m21articulations.Staccatissimo,
    "accent": m21articulations.Accent,
    "marcato": m21articulations.StrongAccent,
    "strong accent": m21articulations.StrongAccent,
    "tenuto": m21articulations.Tenuto,
    "fermata": m21expressions.Fermata,
    "breath mark": m21articulations.BreathMark,
    "caesura": m21articulations.Caesura,
    "up-bow": m21articulations.UpBow,
    "up bow": m21articulations.UpBow,
    "down-bow": m21articulations.DownBow,
    "down bow": m21articulations.DownBow,
    "harmonic": m21articulations.Harmonic,
    "string harmonic": m21articulations.StringHarmonic,
    "stopped": m21articulations.Stopped,
}

_TEMPO_REFERENT_ALIASES = {
    "whole": "whole",
    "half": "half",
    "quarter": "quarter",
    "eighth": "eighth",
    "sixteenth": "16th",
    "16th": "16th",
    "32nd": "32nd",
    "64th": "64th",
    "128th": "128th",
}
_DOTTED_TEMPO_REFERENT_TYPES = {"half", "quarter"}
_TEMPO_REFERENT_ERROR = (
    "Unsupported tempo referent {referent!r}. Accepted values are whole, half, "
    "quarter, eighth, sixteenth, 16th, 32nd, 64th, 128th, plus note-name "
    "aliases such as 'half note', and dotted half / dotted quarter."
)
_DOTTED_TEXT_ABBREVIATION_RE = re.compile(
    r"^(cresc|decresc|dim|rit|rall|accel|ten)\.{2,}$",
    re.IGNORECASE,
)


def _normalize_text_expression_value(text: str) -> str:
    """Return normalized text-expression content for insertion and matching."""
    normalized = text.strip()
    abbreviation_match = _DOTTED_TEXT_ABBREVIATION_RE.fullmatch(normalized)
    if abbreviation_match is not None:
        return f"{abbreviation_match.group(1)}."
    return normalized


def _find_text_expression_at_offset(
    measure_obj: m21stream.Measure,
    offset: float,
    text: str,
) -> Optional[m21expressions.TextExpression]:
    """Return a matching text expression at a measure offset, if present."""
    for el in measure_obj.getElementsByClass(m21expressions.TextExpression):
        if abs(measure_obj.elementOffset(el) - offset) > 1e-9:
            continue
        content = _normalize_text_expression_value(
            str(getattr(el, "content", "") or "")
        )
        if content == text:
            return el
    return None


def _normalize_tempo_referent(referent: str) -> m21duration.Duration:
    """Return a music21 duration for an accepted tempo beat-unit referent."""
    if not isinstance(referent, str):
        raise ValueError(_TEMPO_REFERENT_ERROR.format(referent=referent))

    normalized = " ".join(
        referent.strip().lower().replace("-", " ").split()
    )
    if not normalized:
        raise ValueError(_TEMPO_REFERENT_ERROR.format(referent=referent))

    tokens = normalized.split()
    if tokens[-1] in {"note", "notes"}:
        normalized = " ".join(tokens[:-1])

    dotted = False
    if normalized.startswith("dotted "):
        dotted = True
        normalized = normalized.removeprefix("dotted ").strip()

    duration_type = _TEMPO_REFERENT_ALIASES.get(normalized)
    if duration_type is None:
        raise ValueError(_TEMPO_REFERENT_ERROR.format(referent=referent))
    if dotted and duration_type not in _DOTTED_TEMPO_REFERENT_TYPES:
        raise ValueError(_TEMPO_REFERENT_ERROR.format(referent=referent))

    duration_obj = m21duration.Duration(duration_type)
    if dotted:
        duration_obj.dots = 1
    return duration_obj


def _tempo_referent_label(referent: m21duration.Duration) -> str:
    """Return the normalized agent-facing label for a tempo referent."""
    base_label = str(referent.type)
    if referent.dots == 1:
        return f"dotted {base_label}"
    return base_label


def _is_expression_articulation(articulation_class: type) -> bool:
    """Return whether an articulation tool class belongs in note expressions."""
    return issubclass(articulation_class, m21expressions.Expression)


def _articulation_allows_rest(type_str: str) -> bool:
    """Return whether the requested articulation can be attached to rests."""
    return type_str == "fermata"


def _articulation_target_label(element: m21note.GeneralNote) -> str:
    """Return a user-facing label for an articulation target element."""
    if isinstance(element, m21note.Rest):
        return "rest"
    if getattr(element, "isChord", False):
        return "chord"
    return "note"


def _has_articulation_marking(
    element: m21note.GeneralNote,
    articulation_class: type,
) -> bool:
    """Return whether an articulation-style marking is already attached."""
    if _is_expression_articulation(articulation_class):
        for expression in element.expressions:
            if isinstance(expression, articulation_class):
                return True
        for articulation in element.articulations:
            if isinstance(articulation, articulation_class):
                return True
        return False

    for articulation in element.articulations:
        if isinstance(articulation, articulation_class):
            return True
    return False


def _add_articulation_marking(
    element: m21note.GeneralNote,
    articulation_class: type,
) -> None:
    """Attach an articulation-style marking to the correct music21 container."""
    marking = articulation_class()
    if isinstance(marking, m21expressions.Fermata):
        marking.type = "upright"
    if _is_expression_articulation(articulation_class):
        element.expressions.append(marking)
        return
    element.articulations.append(marking)


def _remove_articulation_marking(
    element: m21note.GeneralNote,
    articulation_class: type,
) -> bool:
    """Remove an articulation-style marking and return whether one was found."""
    if not _is_expression_articulation(articulation_class):
        original_count = len(element.articulations)
        element.articulations = [
            articulation
            for articulation in element.articulations
            if not isinstance(articulation, articulation_class)
        ]
        return len(element.articulations) != original_count

    original_expression_count = len(element.expressions)
    element.expressions = [
        expression
        for expression in element.expressions
        if not isinstance(expression, articulation_class)
    ]

    original_articulation_count = len(element.articulations)
    element.articulations = [
        articulation
        for articulation in element.articulations
        if not isinstance(articulation, articulation_class)
    ]

    return (
        len(element.expressions) != original_expression_count
        or len(element.articulations) != original_articulation_count
    )


def _find_general_note_at_offset(
    container: m21stream.Stream,
    offset: float,
    include_rests: bool = False,
) -> Optional[m21note.GeneralNote]:
    """Find a note, chord, or optionally rest at a given stream offset."""
    for el in container.getElementsByClass(m21note.GeneralNote):
        if isinstance(el, m21note.Rest) and not include_rests:
            continue
        el_offset = container.elementOffset(el)
        if abs(el_offset - offset) < 1e-9:
            return el
    return None


def _find_dynamic_at_offset(
    measure_obj: m21stream.Measure,
    offset: float,
) -> Optional[m21dynamics.Dynamic]:
    """Return the first dynamic at a measure offset, if one exists."""
    for dynamic in measure_obj.getElementsByClass(m21dynamics.Dynamic):
        if abs(measure_obj.elementOffset(dynamic) - offset) < 1e-9:
            return dynamic
    return None


def _hairpin_type(wedge: m21dynamics.DynamicWedge) -> str:
    """Return the normalized ScoreSpeak hairpin type for a wedge."""
    if isinstance(wedge, m21dynamics.Crescendo):
        return "crescendo"
    return "diminuendo"


def _find_hairpin_starting_at(
    part_obj: m21stream.Part,
    start_measure: int,
    start_offset: float,
) -> Optional[m21dynamics.DynamicWedge]:
    """Return a dynamic wedge whose first anchor matches the target start."""
    for wedge in part_obj.getElementsByClass(m21dynamics.DynamicWedge):
        spanned = wedge.getSpannedElements()
        if not spanned:
            continue
        first_el = spanned[0]
        container = first_el.getContextByClass(m21stream.Measure)
        if container is None:
            continue
        if container.number != start_measure:
            continue
        el_offset = container.elementOffset(first_el)
        if abs(el_offset - start_offset) < 1e-9:
            return wedge
    return None


def _hairpin_matches_request(
    wedge: m21dynamics.DynamicWedge,
    type_str: str,
    end_measure: int,
    end_beat: float,
) -> bool:
    """Return whether an existing wedge satisfies the requested hairpin."""
    if _hairpin_type(wedge) != type_str:
        return False

    recorded_end_measure = getattr(wedge, "scorespeak_end_measure", None)
    recorded_end_beat = getattr(wedge, "scorespeak_end_beat", None)
    if recorded_end_measure is None or recorded_end_beat is None:
        return False

    try:
        same_measure = int(recorded_end_measure) == end_measure
        same_beat = abs(float(recorded_end_beat) - float(end_beat)) < 1e-9
    except (TypeError, ValueError):
        return False
    return same_measure and same_beat




# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _validate_beat_in_measure(
    beat: float,
    time_sig,
    measure_number: int,
) -> None:
    """Validate that a beat position falls within the measure."""
    if beat < 1.0:
        raise ValueError(
            f"Beat position must be at least 1.0 (beats are 1-based), "
            f"got {beat}."
        )
    capacity = time_sig.barDuration.quarterLength
    offset = beat - 1.0
    if offset > capacity - 1e-9:
        raise ValueError(
            f"Beat {beat} is beyond the end of measure {measure_number} "
            f"in {time_sig.ratioString} time "
            f"(max beat: {capacity + 1.0 - 1e-9:.4g})."
        )


def _find_note_at_offset(
    container: m21stream.Stream,
    offset: float,
) -> Optional[m21note.GeneralNote]:
    """Find a note or chord at a given offset in a stream."""
    return _find_general_note_at_offset(container, offset, include_rests=False)


def _find_note_ending_at_offset(
    container: m21stream.Stream,
    offset: float,
) -> Optional[m21note.GeneralNote]:
    """Find the latest visible note, rest, or chord ending at an offset."""
    best = None
    best_start = float("-inf")
    for el in container.getElementsByClass(m21note.GeneralNote):
        if isinstance(el, m21note.Rest):
            if (
                hasattr(el.style, "hideObjectOnPrint")
                and el.style.hideObjectOnPrint
            ):
                continue
        el_offset = float(container.elementOffset(el))
        el_end = el_offset + float(el.quarterLength)
        if abs(el_end - offset) < 1e-9 and el_offset >= best_start:
            best = el
            best_start = el_offset
    return best

__all__ = [name for name in globals() if not name.startswith('__')]
