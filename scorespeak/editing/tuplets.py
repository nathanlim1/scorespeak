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


class TupletEditingMixin:
    """Internal mixin for ScoreSpeak note-editing operations."""

    def add_tuplet(
        self,
        pitches_and_durations: list[tuple[PitchInput, DurationInput]],
        actual_notes: int,
        normal_notes: int,
        measure: Optional[int] = None,
        beat: Optional[float] = None,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
    ) -> OperationResult:
        """Add one tuplet group in one voice. Durations in
        ``pitches_and_durations`` are written/base durations before the tuplet
        ratio is applied: a 3:2 triplet of three eighth notes in one quarter
        uses ``actual_notes=3``, ``normal_notes=2``, and durations ``0.5`` or
        ``"eighth"`` for each note.

        Use at a known beat/voice. Inspect first when preserving other voices.

        Args:
            pitches_and_durations: List of ``(pitch, duration)`` tuples,
                one per actual tuplet note. Each duration is the written/base
                note value before tupleting, not the already scaled performed
                length. The tool applies ``normal_notes / actual_notes`` to
                each written duration. For three eighth-note triplets in the
                space of one quarter, use ``[("C4", "eighth"), ("D4",
                "eighth"), ("E4", "eighth")]`` or ``[("C4", 0.5), ("D4",
                0.5), ("E4", 0.5)]`` with ``actual_notes=3`` and
                ``normal_notes=2``; do not use ``1/3`` durations.
            actual_notes: Number of listed notes actually played, e.g. 3 for
                a triplet. Must equal ``len(pitches_and_durations)``.
            normal_notes: Number of same-written-value notes whose space the
                tuplet occupies, e.g. 2 for three eighth notes in the time of
                two eighth notes, which is one quarter note of actual time.
            measure: 1-based measure number.  *None* → last measure.
            beat: 1-based beat position.  *None* → next available beat.
            part: Part identifier.
            voice: 1-based rhythmic timeline inside this part. Defaults to
                voice 1. Use another voice only when the tuplet belongs to a
                different simultaneous rhythmic line.

        Returns:
            OperationResult with tuplet details.
        """
        voice = validate_voice_number(voice)
        if (
            isinstance(actual_notes, bool)
            or isinstance(normal_notes, bool)
            or not isinstance(actual_notes, int)
            or not isinstance(normal_notes, int)
            or actual_notes <= 0
            or normal_notes <= 0
        ):
            raise ValueError(
                "actual_notes and normal_notes must be positive integers."
            )
        if len(pitches_and_durations) != actual_notes:
            raise ValueError(
                f"Expected {actual_notes} notes for this tuplet, "
                f"but received {len(pitches_and_durations)}."
            )

        part_obj, part_idx = self._resolve_part(part)
        measure_number = self._resolve_measure_number(part_obj, measure)
        measure_obj = self._resolve_measure(part_obj, measure_number)

        container = self._get_voice_or_measure(measure_obj, voice, create=True)

        notes = []
        for pitch_in, dur_in in pitches_and_durations:
            p = normalize_pitch(pitch_in)
            d = normalize_duration(dur_in)
            n = m21note.Note(pitch=p, duration=d)
            notes.append(n)

        total_ql = 0.0
        for index, n in enumerate(notes):
            tuplet_boundary = None
            if index == 0:
                tuplet_boundary = "start"
            elif index == len(notes) - 1:
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
            n.duration.appendTuplet(tuplet)
            total_ql += n.duration.quarterLength

        ts = self._get_active_time_signature_obj(part_obj, measure_number)
        capacity = self._effective_measure_capacity(measure_obj, ts)

        if beat is not None:
            if beat < 1.0:
                raise ValueError(
                    f"Beat position must be at least 1.0 (beats are "
                    f"1-based), got {beat}."
                )
            offset = beat - 1.0
            beat_pos = beat
            self._validate_event_capacity(
                capacity,
                total_ql,
                measure_number,
                beat_position=beat,
                ratio_string=ts.ratioString,
            )
        else:
            used_ql = self._get_used_quarter_lengths(container)
            offset = used_ql
            beat_pos = offset + 1.0
            self._validate_event_capacity(
                capacity,
                total_ql,
                measure_number,
                existing_quarter_lengths=used_ql,
                ratio_string=ts.ratioString,
            )

        replaced_rests = self._replace_overlapped_rests_or_raise(
            container,
            offset,
            total_ql,
            measure_number,
            beat_pos,
            part_idx=part_idx,
            voice=voice,
        )
        self._validate_no_rhythm_overlap(
            container,
            offset,
            total_ql,
            measure_number,
            beat_pos,
            voice=voice,
        )

        current_offset = offset
        for n in notes:
            container.insert(current_offset, n)
            current_offset += n.duration.quarterLength
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

        pitch_names = [n.pitch.nameWithOctave for n in notes]
        return OperationResult(
            success=True,
            description=(
                f"Added {actual_notes}:{normal_notes} tuplet "
                f"[{', '.join(pitch_names)}] "
                f"at measure {measure_number}, beat {beat_pos}"
            ),
            details={
                "pitches": pitch_names,
                "actual_notes": actual_notes,
                "normal_notes": normal_notes,
                "total_quarter_length": total_ql,
                "measure": measure_number,
                "beat": beat_pos,
                "part": part_idx,
                "voice": voice,
                "replaced_rests": replaced_rests,
                "auto_completed_rests": auto_completed_rests,
                "measure_integrity": integrity,
                "repair_hint": self._measure_repair_hint(integrity),
            },
        )


    def remove_tuplet(
        self,
        measure: int,
        beat: float,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
        actual_notes: Optional[int] = None,
        normal_notes: Optional[int] = None,
    ) -> OperationResult:
        """Remove one tuplet group from one voice.

        Use after inspecting the exact voice and tuplet start beat. Inspect
        first when preserving other voices.

        Args:
            measure: 1-based measure number.
            beat: 1-based beat where the tuplet group begins.
            part: Part identifier.
            voice: 1-based rhythmic timeline inside this part. Defaults to
                voice 1. Inspect first and use another voice only when the
                tuplet belongs to a different simultaneous rhythmic line.
            actual_notes: Optional ratio guard; if provided, the tuplet
                must have this number of actual notes.
            normal_notes: Optional ratio guard; if provided, the tuplet
                must have this number of normal notes.

        Returns:
            OperationResult confirming removal of the tuplet's notes.
        """
        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure)
        container = self._get_voice_or_measure(measure_obj, voice)
        start_offset = beat - 1.0

        events = sorted(
            [
                el
                for el in container.getElementsByClass(m21note.GeneralNote)
                if not (
                    isinstance(el, m21note.Rest)
                    and getattr(el.style, "hideObjectOnPrint", False)
                )
            ],
            key=lambda el: container.elementOffset(el),
        )
        first = None
        found_event_at_offset = False
        for el in events:
            if abs(container.elementOffset(el) - start_offset) < 1e-9:
                found_event_at_offset = True
                if el.duration.tuplets:
                    first = el
                    break

        if first is None:
            if found_event_at_offset:
                raise ValueError(
                    f"No tuplet starts at beat {beat} in measure {measure}, "
                    f"voice {voice}; an event exists there but it is not the "
                    "start of a tuplet. Inspect the measure and retry with the "
                    "tuplet's first beat."
                )
            raise ValueError(
                f"No note found at beat {beat} in measure {measure}, "
                f"voice {voice}. Inspect the measure to find the tuplet's "
                "voice and starting beat."
            )
        if not first.duration.tuplets:
            raise ValueError(
                f"No tuplet starts at beat {beat} in measure {measure}, "
                f"voice {voice}. Inspect the measure and retry with the "
                "tuplet's first event."
            )

        tuplet = first.duration.tuplets[0]
        ratio_actual = tuplet.numberNotesActual
        ratio_normal = tuplet.numberNotesNormal
        if actual_notes is not None and ratio_actual != actual_notes:
            raise ValueError(
                f"Tuplet at measure {measure}, beat {beat} has "
                f"{ratio_actual} actual notes, not {actual_notes}."
            )
        if normal_notes is not None and ratio_normal != normal_notes:
            raise ValueError(
                f"Tuplet at measure {measure}, beat {beat} has "
                f"{ratio_normal} normal notes, not {normal_notes}."
            )

        to_remove = []
        collecting = False
        expected_offset = start_offset
        for el in events:
            el_offset = container.elementOffset(el)
            if not collecting:
                if el is not first:
                    continue
                collecting = True
            elif abs(el_offset - expected_offset) > 1e-9:
                if el_offset < expected_offset:
                    continue
                break

            if not el.duration.tuplets:
                break
            el_tuplet = el.duration.tuplets[0]
            if (
                el_tuplet.numberNotesActual != ratio_actual
                or el_tuplet.numberNotesNormal != ratio_normal
            ):
                break
            to_remove.append(el)
            expected_offset = el_offset + el.duration.quarterLength
            if len(to_remove) >= ratio_actual:
                break

        if len(to_remove) != ratio_actual:
            raise ValueError(
                f"Could not find the complete {ratio_actual}:{ratio_normal} "
                f"tuplet group starting at measure {measure}, beat {beat}."
            )

        for el in to_remove:
            container.remove(el)
        auto_completed_rests = self._normalize_rests_after_rhythm_edit(
            part_obj,
            part_idx,
            measure_obj,
            measure,
            container,
            voice,
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

        return OperationResult(
            success=True,
            description=(
                f"Removed {ratio_actual}:{ratio_normal} tuplet starting at "
                f"measure {measure}, beat {beat}"
            ),
            details={
                "actual_notes": ratio_actual,
                "normal_notes": ratio_normal,
                "removed_notes": len(to_remove),
                "measure": measure,
                "beat": beat,
                "part": part_idx,
                "voice": voice,
                "auto_completed_rests": auto_completed_rests,
                "measure_integrity": integrity,
                "repair_hint": self._measure_repair_hint(integrity),
            },
        )
