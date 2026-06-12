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


class NotesSharedMixin:
    """Internal mixin for ScoreSpeak note-editing operations."""

    def _refresh_measure_beams(
        self,
        measure_obj: m21stream.Measure,
    ) -> None:
        """Recompute default beams and explicit stem directions in one measure."""
        for element in measure_obj.recurse().getElementsByClass(m21note.GeneralNote):
            if getattr(element.duration, "isGrace", False):
                continue
            if isinstance(element, (m21note.Note, m21chord.Chord)):
                element.beams = m21beam.Beams()
        measure_obj.makeBeams(inPlace=True, setStemDirections=False)
        self._refresh_measure_grace_beams(measure_obj)
        self._refresh_measure_stems(measure_obj)


    def _refresh_measure_stems(
        self,
        measure_obj: m21stream.Measure,
    ) -> None:
        """Apply ScoreSpeak's voice-aware stem-direction policy to a measure."""
        containers = self._measure_voice_containers(measure_obj)
        active_voices = {
            voice
            for voice, container in containers
            if (
                1 <= voice <= 4
                and self._stream_has_visible_voice_content(container)
            )
        }
        force_voice_stems = any(voice in active_voices for voice in (2, 3, 4))

        for voice, container in containers:
            if voice < 1 or voice > 4:
                continue
            for element in self._stemmed_elements(container):
                if force_voice_stems:
                    element.stemDirection = "up" if voice in (1, 3) else "down"
                    continue
                if voice == 1:
                    stem_direction = self._pitch_based_stem_direction(element)
                    if stem_direction is not None:
                        element.stemDirection = stem_direction


    def _measure_voice_containers(
        self,
        measure_obj: m21stream.Measure,
    ) -> list[tuple[int, m21stream.Stream]]:
        """Return voice-numbered streams in a measure, including direct voice 1."""
        containers: list[tuple[int, m21stream.Stream]] = []
        for voice_stream in measure_obj.voices:
            containers.append((self._voice_id_from_stream(voice_stream), voice_stream))

        direct_content = self._direct_general_notes(measure_obj)
        if direct_content or not containers:
            containers.append((1, measure_obj))
        return containers


    @staticmethod
    def _voice_id_from_stream(voice_stream: m21stream.Stream) -> int:
        """Resolve a music21 Voice id to a public voice number."""
        voice_id = getattr(voice_stream, "id", None)
        if str(voice_id).isdigit():
            return int(voice_id)
        return 1


    @staticmethod
    def _direct_general_notes(
        stream_obj: m21stream.Stream,
    ) -> list[m21note.GeneralNote]:
        """Return direct note-like children of a stream."""
        return [
            element
            for element in stream_obj
            if isinstance(element, m21note.GeneralNote)
        ]


    def _stream_has_visible_voice_content(
        self,
        stream_obj: m21stream.Stream,
    ) -> bool:
        """Return whether a voice stream has visible note/rest/chord content."""
        for element in self._direct_general_notes(stream_obj):
            if self._is_hidden_rest(element):
                continue
            return True
        return False


    def _stemmed_elements(
        self,
        stream_obj: m21stream.Stream,
    ) -> list[m21note.Note | m21chord.Chord]:
        """Return direct notes/chords whose stems can be assigned."""
        return [
            element
            for element in self._direct_general_notes(stream_obj)
            if isinstance(element, (m21note.Note, m21chord.Chord))
        ]


    @staticmethod
    def _pitch_based_stem_direction(
        element: m21note.Note | m21chord.Chord,
    ) -> str | None:
        """Return the clef-aware stem direction for one note or chord."""
        if isinstance(element, m21note.Note):
            pitches = [element.pitch]
        else:
            pitches = list(element.pitches)
        if not pitches:
            return None

        clef_obj = element.getContextByClass(m21clef.Clef)
        if clef_obj is None:
            clef_obj = m21clef.TrebleClef()
        if len(pitches) == 1:
            return clef_obj.getStemDirectionForPitches(pitches[0])
        return clef_obj.getStemDirectionForPitches(
            pitches,
            extremePitchOnly=True,
        )


    def _resolve_measure_number(
        self,
        part_obj: m21stream.Part,
        measure: Optional[int],
    ) -> int:
        """Resolve *None* to the last measure number."""
        if measure is not None:
            return measure
        measures = list(part_obj.getElementsByClass(m21stream.Measure))
        if not measures:
            raise ValueError(
                "No measures in this part. Use add_measures() first."
            )
        return max(m.number for m in measures)


    def _resolve_and_validate_beat(
        self,
        container: m21stream.Stream,
        beat: Optional[float],
        new_ql: float,
        time_sig: "m21meter.TimeSignature",
        measure_number: int,
        capacity: Optional[float] = None,
    ) -> tuple[float, float]:
        """Convert a 1-based *beat* to a 0-based offset, validating capacity.

        Returns:
            ``(offset, beat_position)`` where *offset* is 0-based and
            *beat_position* is the 1-based value used for messaging.
        """
        if beat is not None:
            if beat < 1.0:
                raise ValueError(
                    f"Beat position must be at least 1.0 (beats are "
                    f"1-based), got {beat}."
                )
            offset = beat - 1.0
            self._validate_event_capacity(
                self._capacity_or_time_signature_bar(capacity, time_sig),
                new_ql,
                measure_number,
                beat_position=beat,
                ratio_string=time_sig.ratioString,
            )
            return offset, beat

        used_ql = self._get_used_quarter_lengths(container)
        offset = used_ql
        beat_pos = offset + 1.0
        self._validate_event_capacity(
            self._capacity_or_time_signature_bar(capacity, time_sig),
            new_ql,
            measure_number,
            existing_quarter_lengths=used_ql,
            ratio_string=time_sig.ratioString,
        )
        return offset, beat_pos


    @staticmethod
    def _capacity_or_time_signature_bar(
        capacity: Optional[float],
        time_signature: m21meter.TimeSignature,
    ) -> float:
        """Return an explicit capacity or the full active meter capacity."""
        if capacity is not None:
            return float(capacity)
        return float(time_signature.barDuration.quarterLength)


    @staticmethod
    def _effective_measure_capacity(
        measure_obj: m21stream.Measure,
        time_signature: m21meter.TimeSignature,
    ) -> float:
        """Return the measure capacity after pickup padding is applied."""
        bar_capacity = float(time_signature.barDuration.quarterLength)
        padding = float(getattr(measure_obj, "paddingLeft", 0.0) or 0.0)
        if padding <= _RHYTHM_EPSILON:
            return bar_capacity
        return max(0.0, bar_capacity - padding)


    @staticmethod
    def _validate_event_capacity(
        capacity: float,
        new_quarter_length: float,
        measure_number: int,
        beat_position: Optional[float] = None,
        existing_quarter_lengths: float = 0.0,
        ratio_string: Optional[str] = None,
    ) -> None:
        """Raise if an event would exceed the effective measure capacity."""
        if beat_position is not None:
            used_after = beat_position - 1.0 + new_quarter_length
        else:
            used_after = existing_quarter_lengths + new_quarter_length

        if used_after <= capacity + _RHYTHM_EPSILON:
            return

        overflow = used_after - capacity
        meter_text = f" in {ratio_string} time" if ratio_string else ""
        if beat_position is not None:
            raise ValueError(
                f"Measure {measure_number}{meter_text} has effective capacity "
                f"{capacity:g} beats, but adding a {new_quarter_length:g}-beat "
                f"event at beat {beat_position} would exceed this by "
                f"{overflow:.4g} beats."
            )
        raise ValueError(
            f"Measure {measure_number}{meter_text} has effective capacity "
            f"{capacity:g} beats, but it already contains "
            f"{existing_quarter_lengths:g} beats and adding a "
            f"{new_quarter_length:g}-beat event would exceed this by "
            f"{overflow:.4g} beats."
        )


    def _measure_integrity_for_context(
        self,
        part_obj: m21stream.Part,
        part_idx: int,
        measure_number: int,
        container: m21stream.Stream,
        voice: int,
    ) -> dict[str, object]:
        """Analyze one measure voice using its active time signature."""
        time_signature = self._get_active_time_signature_obj(
            part_obj,
            measure_number,
        )
        measure_obj = self._resolve_measure(part_obj, measure_number)
        capacity = self._effective_measure_capacity(measure_obj, time_signature)
        return self._analyze_measure_integrity(
            part_idx,
            measure_number,
            container,
            voice,
            time_signature,
            capacity=capacity,
        )


    def _analyze_measure_integrity(
        self,
        part_idx: int,
        measure_number: int,
        container: m21stream.Stream,
        voice: int,
        time_signature: m21meter.TimeSignature,
        capacity: Optional[float] = None,
    ) -> dict[str, object]:
        """Return rhythm occupancy, gap, and overlap diagnostics."""
        if capacity is None:
            capacity = float(time_signature.barDuration.quarterLength)
        ranges = self._occupied_event_ranges(container)
        sorted_ranges = sorted(ranges, key=lambda item: (item["offset"], item["end"]))
        overlaps = self._rhythm_overlaps(sorted_ranges)
        gaps = self._rhythm_gaps(sorted_ranges, capacity)
        overfull_amount = 0.0
        for event_range in sorted_ranges:
            overfull_amount = max(
                overfull_amount,
                float(event_range["end"]) - capacity,
            )
        overfull = overfull_amount > _RHYTHM_EPSILON
        is_complete = not gaps and not overlaps and not overfull
        occupied_quarter_length = sum(
            float(event_range["quarter_length"])
            for event_range in sorted_ranges
        )
        return {
            "measure": measure_number,
            "part": part_idx,
            "voice": voice,
            "capacity_quarter_length": self._clean_rhythm_float(capacity),
            "occupied_quarter_length": self._clean_rhythm_float(
                occupied_quarter_length
            ),
            "event_ranges": sorted_ranges,
            "gaps": gaps,
            "overlaps": overlaps,
            "overfull": overfull,
            "overfull_by": self._clean_rhythm_float(max(0.0, overfull_amount)),
            "is_complete": is_complete,
        }


    def _occupied_event_ranges(
        self,
        container: m21stream.Stream,
        exclude: m21note.GeneralNote | None = None,
    ) -> list[dict[str, object]]:
        """Return non-grace note/rest/chord ranges in a stream."""
        ranges: list[dict[str, object]] = []
        for element in container.getElementsByClass(m21note.GeneralNote):
            if element is exclude:
                continue
            event_range = self._rhythm_event_range(container, element)
            if event_range is not None:
                ranges.append(event_range)
        return ranges


    def _rhythm_event_range(
        self,
        container: m21stream.Stream,
        element: m21note.GeneralNote,
    ) -> dict[str, object] | None:
        """Return the rhythm range for one event, or ``None`` if ignored."""
        if getattr(element.duration, "isGrace", False):
            return None
        offset = float(container.elementOffset(element))
        quarter_length = float(element.duration.quarterLength)
        if quarter_length <= _RHYTHM_EPSILON:
            return None
        end = offset + quarter_length
        return {
            "kind": self._rhythm_event_kind(element),
            "beat": self._clean_rhythm_float(offset + 1.0),
            "offset": self._clean_rhythm_float(offset),
            "end": self._clean_rhythm_float(end),
            "duration": element.duration.type,
            "quarter_length": self._clean_rhythm_float(quarter_length),
            "dots": int(element.duration.dots),
            "label": self._describe_element(element),
            "visibility": self._element_visibility(element),
        }


    @staticmethod
    def _rhythm_event_kind(element: m21note.GeneralNote) -> str:
        """Return a stable kind label for a rhythm event."""
        if isinstance(element, m21chord.Chord):
            return "chord"
        if isinstance(element, m21note.Rest):
            return "rest"
        if isinstance(element, m21note.Note):
            return "note"
        return "event"


    def _rhythm_overlaps(
        self,
        sorted_ranges: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Return pairwise overlaps from sorted rhythm ranges."""
        overlaps: list[dict[str, object]] = []
        if not sorted_ranges:
            return overlaps

        previous = sorted_ranges[0]
        for current in sorted_ranges[1:]:
            previous_end = float(previous["end"])
            current_start = float(current["offset"])
            if current_start < previous_end - _RHYTHM_EPSILON:
                overlaps.append(
                    {
                        "start": self._clean_rhythm_float(current_start),
                        "end": self._clean_rhythm_float(
                            min(previous_end, float(current["end"]))
                        ),
                        "beat": self._clean_rhythm_float(current_start + 1.0),
                        "first": previous,
                        "second": current,
                    }
                )
            if float(current["end"]) > previous_end:
                previous = current
        return overlaps


    def _rhythm_gaps(
        self,
        sorted_ranges: list[dict[str, object]],
        capacity: float,
    ) -> list[dict[str, object]]:
        """Return uncovered ranges in a measure voice."""
        gaps: list[dict[str, object]] = []
        cursor = 0.0
        for event_range in sorted_ranges:
            start = max(0.0, float(event_range["offset"]))
            end = min(capacity, float(event_range["end"]))
            if start > cursor + _RHYTHM_EPSILON:
                gaps.append(self._gap_payload(cursor, start))
            cursor = max(cursor, end)
            if cursor >= capacity - _RHYTHM_EPSILON:
                cursor = capacity
                break
        if cursor < capacity - _RHYTHM_EPSILON:
            gaps.append(self._gap_payload(cursor, capacity))
        return gaps


    def _measure_repair_hint(self, integrity: dict[str, object]) -> str | None:
        """Return a concise repair hint for an integrity payload."""
        if integrity["is_complete"]:
            return None
        measure = integrity["measure"]
        voice = integrity["voice"]
        if integrity["overlaps"] or integrity["overfull"]:
            return (
                f"Measure {measure} voice {voice} has invalid rhythm; fix "
                "overlaps or overfull content before editing visible rest "
                "spelling."
            )
        if integrity["gaps"]:
            return None
        return None


    @staticmethod
    def _clean_rhythm_float(value: float) -> float:
        """Round rhythm floats for stable diagnostics."""
        rounded = round(value, 9)
        if abs(rounded - round(rounded)) <= _RHYTHM_EPSILON:
            return float(round(rounded))
        return rounded


    def _element_visibility(self, element: m21note.GeneralNote) -> str:
        """Return a stable visibility label for a rhythm event."""
        if self._is_hidden_rest(element):
            return "hidden"
        return "visible"


    @staticmethod
    def _is_hidden_rest(element: m21note.GeneralNote) -> bool:
        """Return whether an element is a hidden rest."""
        return (
            isinstance(element, m21note.Rest)
            and hasattr(element.style, "hideObjectOnPrint")
            and bool(element.style.hideObjectOnPrint)
        )


    def _refresh_measure_accidentals(
        self,
        part_obj: m21stream.Part,
        measure_number: int,
    ) -> None:
        """Recompute visible accidentals for one measure in key context."""
        measure_obj = self._resolve_measure(part_obj, measure_number)
        key_signature = self._get_active_key_signature_obj(
            part_obj, measure_number,
        )
        previous_tie_pitch_set = self._previous_measure_tie_pitch_set(
            part_obj, measure_number,
        )
        current_tie_pitch_set = self._measure_tie_continuation_pitch_set(
            measure_obj,
        )
        tie_pitch_set = previous_tie_pitch_set.intersection(current_tie_pitch_set)
        measure_obj.makeAccidentals(
            useKeySignature=key_signature,
            inPlace=True,
            overrideStatus=True,
            cautionaryPitchClass=False,
            cautionaryNotImmediateRepeat=False,
            tiePitchSet=tie_pitch_set or None,
        )


    def _refresh_changed_measure_accidentals(
        self,
        part_obj: m21stream.Part,
        measure_number: int,
        *,
        refresh_next: bool = False,
    ) -> None:
        """Refresh one changed measure and optionally its following measure."""
        if refresh_next:
            self._refresh_measure_and_next_accidentals(part_obj, measure_number)
        else:
            self._refresh_measure_accidentals(part_obj, measure_number)


    def _refresh_measure_and_next_accidentals(
        self,
        part_obj: m21stream.Part,
        measure_number: int,
    ) -> None:
        """Refresh a measure and the following measure when present."""
        self._refresh_measure_accidentals(part_obj, measure_number)
        next_measure_number = self._next_measure_number(part_obj, measure_number)
        if next_measure_number is not None:
            self._refresh_measure_accidentals(part_obj, next_measure_number)


    def _refresh_accidentals_until_next_key_change(
        self,
        part_obj: m21stream.Part,
        start_measure: int,
    ) -> None:
        """Refresh accidentals from a measure through its active key region."""
        for measure_obj in self._sorted_part_measures(part_obj):
            measure_number = measure_obj.number
            if measure_number is None or measure_number < start_measure:
                continue
            if (
                measure_number > start_measure
                and self._measure_has_local_key_signature(measure_obj)
            ):
                break
            self._refresh_measure_accidentals(part_obj, measure_number)


    @staticmethod
    def _sorted_part_measures(
        part_obj: m21stream.Part,
    ) -> list[m21stream.Measure]:
        """Return part measures sorted by numeric measure number."""
        return sorted(
            part_obj.getElementsByClass(m21stream.Measure),
            key=lambda measure: measure.number or 0,
        )


    @staticmethod
    def _measure_has_local_key_signature(
        measure_obj: m21stream.Measure,
    ) -> bool:
        """Return whether a measure contains an explicit key signature."""
        return bool(
            list(measure_obj.getElementsByClass(m21key.KeySignature))
            or list(measure_obj.getElementsByClass(m21key.Key))
        )


    def _find_element_at_offset(
        self,
        container: m21stream.Stream,
        offset: float,
    ) -> Optional[m21note.GeneralNote]:
        """Find a note/rest/chord at *offset*, preferring non-grace notes."""
        matches: list[m21note.GeneralNote] = []
        for el in container.getElementsByClass(m21note.GeneralNote):
            el_offset = container.elementOffset(el)
            if abs(el_offset - offset) < 1e-9:
                if isinstance(el, m21note.Rest):
                    if (
                        hasattr(el.style, "hideObjectOnPrint")
                        and el.style.hideObjectOnPrint
                    ):
                        continue
                matches.append(el)

        non_grace = [
            m for m in matches
            if not getattr(m.duration, "isGrace", False)
        ]
        if non_grace:
            return non_grace[0]
        if matches:
            return matches[0]
        return None


    def _find_element_spanning_offset(
        self,
        container: m21stream.Stream,
        offset: float,
    ) -> Optional[m21note.GeneralNote]:
        """Find a non-grace visible event spanning but not starting at offset."""
        for element in container.getElementsByClass(m21note.GeneralNote):
            if getattr(element.duration, "isGrace", False):
                continue
            if self._is_hidden_rest(element):
                continue
            start = float(container.elementOffset(element))
            end = start + float(element.duration.quarterLength)
            if start + _RHYTHM_EPSILON < offset < end - _RHYTHM_EPSILON:
                return element
        return None


    def _raise_interior_beat_target_error(
        self,
        element: m21note.GeneralNote,
        container: m21stream.Stream,
        measure: int,
        beat: float,
        action: str,
    ) -> None:
        """Raise a clear error for targeting inside an existing event."""
        start = float(container.elementOffset(element))
        end = start + float(element.duration.quarterLength)
        raise ValueError(
            f"Cannot {action} at measure {measure}, beat {beat}: that beat is "
            f"inside existing {self._describe_element(element)} from beat "
            f"{self._clean_rhythm_float(start + 1.0)} to "
            f"{self._clean_rhythm_float(end + 1.0)}. Retry at the event's "
            f"start beat {self._clean_rhythm_float(start + 1.0)}."
        )


    @staticmethod
    def _describe_element(element: m21note.GeneralNote) -> str:
        """Build a short human-readable description of a note/rest/chord."""
        if isinstance(element, m21chord.Chord):
            names = [p.nameWithOctave for p in element.pitches]
            return f"chord [{', '.join(names)}]"
        if isinstance(element, m21note.Note):
            return f"{element.pitch.nameWithOctave} {element.duration.type} note"
        if isinstance(element, m21note.Rest):
            return f"{element.duration.type} rest"
        return "element"
