"""
Note, rest, and chord operations for ScoreSpeak.

Provides methods for adding, removing, replacing, and querying notes,
rests, chords, ties, tuplets, and grace notes in a score.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional, Union

from music21 import beam as m21beam
from music21 import chord as m21chord
from music21 import clef as m21clef
from music21 import duration as m21duration
from music21 import instrument as m21instrument
from music21 import key as m21key
from music21 import meter as m21meter
from music21 import note as m21note
from music21 import pitch as m21pitch
from music21 import spanner as m21spanner
from music21 import stream as m21stream
from music21 import tie as m21tie

from ..types import (
    DurationInput,
    NoteInfo,
    OperationResult,
    PitchInput,
    TupletInfo,
)
from ..music.validation import (
    normalize_duration,
    normalize_pitch,
    validate_voice_number,
    validate_pitch_in_range,
)


_RHYTHM_EPSILON = 1e-9
_ADD_NOTES_REQUIRED_FIELDS = frozenset({"pitch", "beat", "duration", "dots"})
_ADD_NOTES_FORBIDDEN_FIELDS = frozenset({"measure", "part", "voice"})
_ADD_NOTES_ALLOWED_FIELDS = _ADD_NOTES_REQUIRED_FIELDS
_REMOVE_NOTES_REQUIRED_FIELDS = frozenset({"beat"})
_REMOVE_NOTES_FORBIDDEN_FIELDS = frozenset({"measure", "part", "voice"})
_REMOVE_NOTES_ALLOWED_FIELDS = frozenset({"beat", "pitch"})
_STANDARD_DURATION_BASES = (
    4.0,
    2.0,
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
)
_STANDARD_DURATION_VALUES = frozenset(
    round(base * sum(0.5 ** dot for dot in range(dots + 1)), 9)
    for base in _STANDARD_DURATION_BASES
    for dots in range(4)
)
_BEAMABLE_GRACE_DURATION_TYPES = frozenset(
    {
        "eighth",
        "16th",
        "32nd",
        "64th",
        "128th",
        "256th",
        "512th",
        "1024th",
        "2048th",
    }
)


class ChordEditingMixin:
    """Internal mixin for ScoreSpeak note-editing operations."""

    def add_chord(
        self,
        pitches: list[PitchInput],
        duration: DurationInput = "quarter",
        measure: Optional[int] = None,
        beat: Optional[float] = None,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
        dots: int = 0,
    ) -> OperationResult:
        """Add one same-rhythm chord in one rhythmic voice. Use when pitches
        start together and share one duration/stem. Do not change voice for
        chord tones; use separate voices only for independent overlapping
        rhythms.
        This tool auto-completes remaining silent gaps in the target part/measure/voice
        with convenient visible rests, but if a user mentions specific rests,
        check that ``auto_completed_rests`` matches their expected rest spelling and modify the result with 
        ``reshape_rests`` if necessary.

        Args:
            pitches: List of pitch inputs (e.g. ``["C4", "E4", "G4"]``).
            duration: Chord duration — same flexible formats as add_notes item
                durations.
            measure: 1-based measure number.  *None* → last measure.
            beat: 1-based beat position.  *None* → next available beat.
            part: Part identifier.
            voice: 1-based rhythmic timeline inside this part. Defaults to
                voice 1. Use another voice only when this chord belongs to an
                independent overlapping rhythm, not for the chord's pitches.
            dots: Number of augmentation dots.

        Returns:
            OperationResult with details of the added chord.
        """
        return self._add_chord_one(
            pitches,
            duration,
            measure=measure,
            beat=beat,
            part=part,
            voice=voice,
            dots=dots,
            normalize_rests=True,
        )


    def _add_chord_one(
        self,
        pitches: list[PitchInput],
        duration: DurationInput = "quarter",
        measure: Optional[int] = None,
        beat: Optional[float] = None,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
        dots: int = 0,
        normalize_rests: bool = True,
    ) -> OperationResult:
        """Add one same-rhythm chord with optional rest normalization."""
        voice = validate_voice_number(voice)
        pitch_objs = self._normalize_unique_pitches(
            pitches,
            context="A chord",
            minimum_count=2,
        )
        dur_obj = normalize_duration(duration, dots=dots)

        part_obj, part_idx = self._resolve_part(part)
        measure_number = self._resolve_measure_number(part_obj, measure)
        measure_obj = self._resolve_measure(part_obj, measure_number)

        container = self._get_voice_or_measure(measure_obj, voice, create=True)

        ts = self._get_active_time_signature_obj(part_obj, measure_number)
        capacity = self._effective_measure_capacity(measure_obj, ts)
        offset, beat_pos = self._resolve_and_validate_beat(
            container,
            beat,
            dur_obj.quarterLength,
            ts,
            measure_number,
            capacity=capacity,
        )
        replaced_rests = self._replace_overlapped_rests_or_raise(
            container,
            offset,
            dur_obj.quarterLength,
            measure_number,
            beat_pos,
            part_idx=part_idx,
            voice=voice,
        )
        self._validate_no_rhythm_overlap(
            container,
            offset,
            dur_obj.quarterLength,
            measure_number,
            beat_pos,
            voice=voice,
        )

        warnings = []
        for p in pitch_objs:
            w = self._check_instrument_range(part_obj, p)
            if w:
                warnings.append(w)

        chord_obj = m21chord.Chord(pitch_objs, duration=dur_obj)
        container.insert(offset, chord_obj)
        auto_completed_rests: list[dict[str, object]] = []
        if normalize_rests:
            auto_completed_rests = self._normalize_rests_after_rhythm_edit(
                part_obj,
                part_idx,
                measure_obj,
                measure_number,
                container,
                voice,
            )
        self._refresh_measure_beams(measure_obj)
        self._refresh_measure_accidentals(part_obj, measure_number)
        integrity = self._analyze_measure_integrity(
            part_idx,
            measure_number,
            container,
            voice,
            ts,
            capacity=capacity,
        )

        pitch_names = [p.nameWithOctave for p in pitch_objs]
        desc = (
            f"Added chord [{', '.join(pitch_names)}] "
            f"at measure {measure_number}, beat {beat_pos}"
        )
        warning_str = "; ".join(warnings) if warnings else None
        if warning_str:
            desc += f" — {warning_str}"

        return OperationResult(
            success=True,
            description=desc,
            details={
                "pitches": pitch_names,
                "duration": dur_obj.type,
                "quarter_length": dur_obj.quarterLength,
                "measure": measure_number,
                "beat": beat_pos,
                "part": part_idx,
                "voice": voice,
                "dots": dur_obj.dots,
                "warning": warning_str,
                "replaced_rests": replaced_rests,
                "auto_completed_rests": auto_completed_rests,
                "measure_integrity": integrity,
                "repair_hint": self._measure_repair_hint(integrity),
            },
        )


    def add_chord_tones(
        self,
        pitches: list[PitchInput],
        measure: int,
        beat: float,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
    ) -> OperationResult:
        """Add one or more pitches to an existing note or chord event.

        Use this when new pitches should share the exact start, duration, stem,
        and rhythm metadata of an existing note/chord. To create a new chord at
        an empty beat, use ``add_chord`` instead.

        Args:
            pitches: One or more new chord tones. Duplicate pitches and pitches
                already present in the target event are rejected.
            measure: 1-based measure number.
            beat: 1-based beat where an existing note/chord starts.
            part: Part identifier.
            voice: 1-based rhythmic timeline containing the target event.

        Returns:
            OperationResult describing the expanded note/chord.
        """
        voice = validate_voice_number(voice)
        pitch_objs = self._normalize_unique_pitches(
            pitches,
            context="add_chord_tones",
            minimum_count=1,
        )
        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure)
        container = self._get_voice_or_measure(measure_obj, voice)
        offset = beat - 1.0
        element = self._find_element_at_offset(container, offset)
        if element is None:
            spanning = self._find_element_spanning_offset(container, offset)
            if spanning is not None:
                self._raise_interior_beat_target_error(
                    spanning,
                    container,
                    measure,
                    beat,
                    "add chord tones",
                )
            raise ValueError(
                f"No note or chord starts at beat {beat} in measure {measure}, "
                f"voice {voice}. Use add_chord to create a new chord at an "
                "empty beat."
            )
        if isinstance(element, m21note.Rest):
            raise ValueError(
                f"Cannot add chord tones to a rest at measure {measure}, beat "
                f"{beat}, voice {voice}. Use add_chord to replace the rest "
                "with a new chord."
            )
        if not isinstance(element, (m21note.Note, m21chord.Chord)):
            raise ValueError(
                f"Element at beat {beat} in measure {measure} is a "
                f"{type(element).__name__}, not a note or chord."
            )

        existing_pitches = self._element_pitch_names(element)
        new_pitch_names = [pitch_obj.nameWithOctave for pitch_obj in pitch_objs]
        already_present = [
            pitch_name
            for pitch_name in new_pitch_names
            if pitch_name in existing_pitches
        ]
        if already_present:
            raise ValueError(
                f"Pitch(es) already present at measure {measure}, beat {beat}: "
                f"{already_present}."
            )

        combined_pitches = list(self._element_pitches(element)) + pitch_objs
        replacement = self._replace_event_with_pitches(
            container,
            element,
            combined_pitches,
        )
        self._refresh_measure_beams(measure_obj)
        self._refresh_measure_accidentals(part_obj, measure)
        integrity = self._measure_integrity_for_context(
            part_obj,
            part_idx,
            measure,
            container,
            voice,
        )
        warnings = [
            warning
            for pitch_obj in pitch_objs
            if (warning := self._check_instrument_range(part_obj, pitch_obj))
        ]
        warning_text = "; ".join(warnings) if warnings else None
        final_pitches = self._element_pitch_names(replacement)
        description = (
            f"Added chord tone(s) [{', '.join(new_pitch_names)}] to "
            f"measure {measure}, beat {beat}"
        )
        if warning_text:
            description += f" — {warning_text}"

        return OperationResult(
            success=True,
            description=description,
            details={
                "added_pitches": new_pitch_names,
                "pitches": final_pitches,
                "measure": measure,
                "beat": beat,
                "part": part_idx,
                "voice": voice,
                "warning": warning_text,
                "auto_completed_rests": [],
                "measure_integrity": integrity,
                "repair_hint": self._measure_repair_hint(integrity),
            },
        )


    def _remove_pitch_from_chord(
        self,
        container: m21stream.Stream,
        chord_el: m21chord.Chord,
        pitch: PitchInput,
        measure: int,
        beat: float,
        part_idx: int,
        voice: int,
    ) -> OperationResult:
        """Remove a single pitch from a chord, converting if needed."""
        pitch_obj = normalize_pitch(pitch)
        remaining = [
            p for p in chord_el.pitches
            if p.nameWithOctave != pitch_obj.nameWithOctave
        ]

        if len(remaining) == len(chord_el.pitches):
            raise ValueError(
                f"Pitch {pitch_obj.nameWithOctave} not found in the chord "
                f"at beat {beat} in measure {measure}. "
                f"Chord contains: "
                f"{[p.nameWithOctave for p in chord_el.pitches]}"
            )

        if len(remaining) == 0:
            container.remove(chord_el)
            desc = (
                f"Removed last pitch {pitch_obj.nameWithOctave} from chord; "
                f"chord deleted from measure {measure}, beat {beat}"
            )
        elif len(remaining) == 1:
            self._replace_event_with_pitches(container, chord_el, remaining)
            desc = (
                f"Removed {pitch_obj.nameWithOctave} from chord at "
                f"measure {measure}, beat {beat} "
                f"(chord reduced to single note {remaining[0].nameWithOctave})"
            )
        else:
            self._replace_event_with_pitches(container, chord_el, remaining)
            desc = (
                f"Removed {pitch_obj.nameWithOctave} from chord at "
                f"measure {measure}, beat {beat}"
            )

        return OperationResult(
            success=True,
            description=desc,
            details={
                "removed_pitch": pitch_obj.nameWithOctave,
                "measure": measure,
                "beat": beat,
                "part": part_idx,
                "voice": voice,
                "remaining": [p.nameWithOctave for p in remaining],
            },
        )
