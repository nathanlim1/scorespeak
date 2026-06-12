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


class TieEditingMixin:
    """Internal mixin for ScoreSpeak note-editing operations."""

    def add_tie(
        self,
        measure: int,
        beat: float,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
    ) -> OperationResult:
        """Tie one note or chord to the immediately following matching event.

        This agent-facing operation behaves like notation software: select the
        note/chord that should start a tie, and ScoreSpeak ties it only to the
        next non-grace rhythmic event in the same part and voice.

        Args:
            measure: 1-based measure containing the selected note/chord.
            beat: Beat where the selected note/chord begins.
            part: Part identifier.
            voice: 1-based rhythmic timeline inside this part. Defaults to
                voice 1. Use another voice only for a different simultaneous
                rhythmic line, not for chord tones.

        Returns:
            OperationResult confirming the tie.

        Raises:
            ValueError: If the selected event is missing, is a rest, has no
                matching adjacent next event, or is followed by a rest or
                different pitch content.
        """
        part_obj, part_idx = self._resolve_part(part)
        start_element = self._resolve_tie_endpoint(
            part_obj,
            measure,
            beat,
            voice,
            "start",
        )
        result = self._add_adjacent_tie(
            part_obj,
            part_idx,
            measure,
            beat,
            voice,
            start_element,
            endpoint_label="start",
        )
        return result


    def remove_tie(
        self,
        start_measure: int,
        start_beat: float,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
        end_measure: Optional[int] = None,
        end_beat: Optional[float] = None,
    ) -> OperationResult:
        """Remove a connected tie chain from one note or chord.

        A single call removes the start/continue/stop markers that form the
        tie connected to the target note. If ``end_measure`` and ``end_beat``
        are provided, removal is limited to the same-pitch chain between the
        two endpoints.

        Args:
            start_measure: 1-based measure containing a tied note/chord.
            start_beat: Beat of the tied note/chord.
            part: Part identifier.
            voice: 1-based rhythmic timeline inside this part. Defaults to
                voice 1. Use another voice only for a different simultaneous
                rhythmic line, not for chord tones.
            end_measure: Optional endpoint measure for a specific tie span.
            end_beat: Optional endpoint beat for a specific tie span.

        Returns:
            OperationResult confirming the tie-chain removal.

        Raises:
            ValueError: If no note/chord is found, the element is a rest,
                or the note/chord has no tie.
        """
        part_obj, part_idx = self._resolve_part(part)
        element = self._resolve_tie_endpoint(
            part_obj,
            start_measure,
            start_beat,
            voice,
            "target",
        )
        if getattr(element, "tie", None) is None:
            raise ValueError(
                f"No tie found at beat {start_beat} in measure {start_measure}, voice "
                f"{voice}. If the tied note is in another voice, inspect the "
                "measure and retry with that voice."
            )

        events = self._tied_voice_events(part_obj, voice)
        target_index = self._tie_event_index(
            events,
            start_measure,
            start_beat,
            element,
        )
        signature = self._tie_pitch_signature(element)

        if end_measure is not None or end_beat is not None:
            if end_measure is None or end_beat is None:
                raise ValueError(
                    "Provide both end_measure and end_beat, or omit both."
                )
            end_element = self._resolve_tie_endpoint(
                part_obj,
                end_measure,
                end_beat,
                voice,
                "end",
            )
            if self._tie_pitch_signature(end_element) != signature:
                raise ValueError(
                    "Tie removal endpoints must have matching pitch content."
                )
            end_index = self._tie_event_index(
                events,
                end_measure,
                end_beat,
                end_element,
            )
            start_index = min(target_index, end_index)
            stop_index = max(target_index, end_index)
        else:
            start_index, stop_index = self._connected_tie_chain_indices(
                events,
                target_index,
                signature,
            )

        removed = []
        for event in events[start_index : stop_index + 1]:
            event_element = event["element"]
            if self._tie_pitch_signature(event_element) != signature:
                continue
            tie_obj = getattr(event_element, "tie", None)
            if tie_obj is None:
                continue
            removed.append({
                "measure": event["measure"],
                "beat": event["beat"],
                "tie_type": tie_obj.type,
            })
            event_element.tie = None

        if not removed:
            raise ValueError(
                f"No connected tie markers found at measure {start_measure}, "
                f"beat {start_beat}."
            )

        self._refresh_measure_and_next_accidentals(part_obj, start_measure)
        last_measure = int(removed[-1]["measure"])
        if last_measure != start_measure:
            self._refresh_measure_and_next_accidentals(part_obj, last_measure)

        return OperationResult(
            success=True,
            description=(
                f"Removed tie chain containing measure {start_measure}, "
                f"beat {start_beat}"
            ),
            details={
                "start_measure": start_measure,
                "start_beat": start_beat,
                "part": part_idx,
                "voice": voice,
                "pitches": list(signature),
                "removed": removed,
            },
        )


    def _resolve_tie_endpoint(
        self,
        part_obj: m21stream.Part,
        measure: int,
        beat: float,
        voice: int,
        endpoint_label: str,
    ) -> m21note.GeneralNote:
        """Return the non-rest note/chord at one tie endpoint."""
        measure_obj = self._resolve_measure(part_obj, measure)
        container = self._get_voice_or_measure(measure_obj, voice)
        element = self._find_element_at_offset(container, beat - 1.0)
        if element is None:
            raise ValueError(
                f"No note found at {endpoint_label} beat {beat} in measure "
                f"{measure}, voice {voice}. Call inspect_score_region to find "
                "the correct voice/beat before editing a tie."
            )
        if isinstance(element, m21note.Rest):
            raise ValueError(
                f"Cannot use a rest at measure {measure}, beat {beat}, voice "
                f"{voice} as a tie {endpoint_label}. Ties require pitched "
                "notes or chords."
            )
        return element


    def _tie_pitch_signature(
        self,
        element: m21note.GeneralNote,
    ) -> tuple[str, ...]:
        """Return a stable pitch signature for a tied note or chord."""
        if isinstance(element, m21note.Note):
            return (element.pitch.nameWithOctave,)
        if isinstance(element, m21chord.Chord):
            return tuple(sorted(pitch.nameWithOctave for pitch in element.pitches))
        raise ValueError(f"Cannot tie {type(element).__name__}; use notes or chords.")


    def _tie_voice_rhythmic_events(
        self,
        part_obj: m21stream.Part,
        voice: int,
    ) -> list[dict[str, Any]]:
        """Return visible non-grace rhythmic events in a part voice."""
        events: list[dict[str, Any]] = []
        for measure_obj in part_obj.getElementsByClass(m21stream.Measure):
            try:
                measure_number = int(measure_obj.number)
                container = self._get_voice_or_measure(measure_obj, voice)
            except (TypeError, ValueError):
                continue
            for element in container.getElementsByClass(m21note.GeneralNote):
                if getattr(element.duration, "isGrace", False):
                    continue
                beat = float(container.elementOffset(element)) + 1.0
                events.append({
                    "measure": measure_number,
                    "beat": beat,
                    "element": element,
                })
        events.sort(key=lambda event: (int(event["measure"]), float(event["beat"])))
        return events


    def _tied_voice_events(
        self,
        part_obj: m21stream.Part,
        voice: int,
    ) -> list[dict[str, Any]]:
        """Return visible note/chord events in a part voice, sorted in score order."""
        events: list[dict[str, Any]] = []
        for measure_obj in part_obj.getElementsByClass(m21stream.Measure):
            try:
                measure_number = int(measure_obj.number)
                container = self._get_voice_or_measure(measure_obj, voice)
            except (TypeError, ValueError):
                continue
            for element in container.getElementsByClass(m21note.GeneralNote):
                if isinstance(element, m21note.Rest):
                    continue
                if getattr(element.duration, "isGrace", False):
                    continue
                beat = float(container.elementOffset(element)) + 1.0
                events.append({
                    "measure": measure_number,
                    "beat": beat,
                    "element": element,
                })
        events.sort(key=lambda event: (int(event["measure"]), float(event["beat"])))
        return events


    def _add_adjacent_tie(
        self,
        part_obj: m21stream.Part,
        part_idx: int,
        measure: int,
        beat: float,
        voice: int,
        start_element: m21note.GeneralNote,
        endpoint_label: str,
        repair_existing_onward: bool = False,
    ) -> OperationResult:
        """Tie one resolved note/chord to the next adjacent matching event."""
        start_signature = self._tie_pitch_signature(start_element)
        start_tie = getattr(start_element, "tie", None)
        if (
            start_tie is not None
            and start_tie.type in {"start", "continue"}
            and not repair_existing_onward
        ):
            return OperationResult(
                success=True,
                description=(
                    f"Tie already exists from measure {measure}, beat {beat}"
                ),
                details={
                    "measure": measure,
                    "beat": beat,
                    "part": part_idx,
                    "voice": voice,
                    "pitches": list(start_signature),
                    "already_tied": True,
                    "tie_type": start_tie.type,
                },
            )

        events = self._tie_voice_rhythmic_events(part_obj, voice)
        start_index = self._tie_event_index(events, measure, beat, start_element)
        next_index = start_index + 1
        if next_index >= len(events):
            raise ValueError(
                f"No following rhythmic event found after {endpoint_label} "
                f"measure {measure}, beat {beat}, voice {voice}."
            )

        next_event = events[next_index]
        next_element = next_event["element"]
        if isinstance(next_element, m21note.Rest):
            raise ValueError(
                f"Cannot tie from measure {measure}, beat {beat} to the next "
                "event because the next event is a rest. Ties require adjacent "
                "matching notes or chords."
            )

        next_signature = self._tie_pitch_signature(next_element)
        if next_signature != start_signature:
            raise ValueError(
                "Ties require adjacent matching pitches in the same part and "
                f"voice; got {', '.join(start_signature)} at the selected "
                f"event and {', '.join(next_signature)} at the next event."
            )

        if start_tie is not None and start_tie.type in {"stop", "continue"}:
            start_element.tie = m21tie.Tie("continue")
        else:
            start_element.tie = m21tie.Tie("start")

        next_tie = getattr(next_element, "tie", None)
        if next_tie is not None and next_tie.type in {"start", "continue"}:
            next_element.tie = m21tie.Tie("continue")
        else:
            next_element.tie = m21tie.Tie("stop")

        next_measure = int(next_event["measure"])
        next_beat = float(next_event["beat"])
        self._refresh_measure_and_next_accidentals(part_obj, measure)
        if next_measure != measure:
            self._refresh_measure_and_next_accidentals(part_obj, next_measure)

        return OperationResult(
            success=True,
            description=(
                f"Added tie from measure {measure} beat {beat} to measure "
                f"{next_measure} beat {next_beat}"
            ),
            details={
                "measure": measure,
                "beat": beat,
                "next_measure": next_measure,
                "next_beat": next_beat,
                "part": part_idx,
                "voice": voice,
                "pitches": list(start_signature),
                "already_tied": False,
                "tied_positions": [
                    {"measure": measure, "beat": float(beat)},
                    {"measure": next_measure, "beat": next_beat},
                ],
            },
        )


    def _tie_event_index(
        self,
        events: list[dict[str, Any]],
        measure: int,
        beat: float,
        element: m21note.GeneralNote,
    ) -> int:
        """Return the event-list index for one resolved tie endpoint."""
        for index, event in enumerate(events):
            if event["element"] is element:
                return index
        raise ValueError(
            f"Could not locate tie event at measure {measure}, beat {beat}."
        )


    def _connected_tie_chain_indices(
        self,
        events: list[dict[str, Any]],
        target_index: int,
        signature: tuple[str, ...],
    ) -> tuple[int, int]:
        """Return the connected same-pitch tie-chain bounds around one event."""
        start_index = target_index
        stop_index = target_index

        while start_index > 0:
            current = events[start_index]["element"]
            previous = events[start_index - 1]["element"]
            current_tie = getattr(current, "tie", None)
            previous_tie = getattr(previous, "tie", None)
            if self._tie_pitch_signature(previous) != signature:
                break
            if current_tie is None or previous_tie is None:
                break
            if current_tie.type not in {"stop", "continue"}:
                break
            if previous_tie.type not in {"start", "continue"}:
                break
            start_index -= 1

        while stop_index + 1 < len(events):
            current = events[stop_index]["element"]
            next_element = events[stop_index + 1]["element"]
            current_tie = getattr(current, "tie", None)
            next_tie = getattr(next_element, "tie", None)
            if self._tie_pitch_signature(next_element) != signature:
                break
            if current_tie is None or next_tie is None:
                break
            if current_tie.type not in {"start", "continue"}:
                break
            if next_tie.type not in {"stop", "continue"}:
                break
            stop_index += 1

        return start_index, stop_index


    def _clear_tie_chains_touching_elements(
        self,
        part_obj: m21stream.Part,
        removed_elements: list[m21note.GeneralNote],
    ) -> list[dict[str, object]]:
        """Clear connected tie markers touching note/chord elements being removed."""
        removed_element_ids = {
            id(element)
            for element in removed_elements
            if not isinstance(element, m21note.Rest)
        }
        if not removed_element_ids:
            return []

        cleared: list[dict[str, object]] = []
        cleared_element_ids: set[int] = set()
        for voice in range(1, 5):
            events = self._tied_voice_events(part_obj, voice)
            for index, event in enumerate(events):
                element = event["element"]
                if id(element) not in removed_element_ids:
                    continue
                tie_obj = getattr(element, "tie", None)
                if tie_obj is None:
                    continue
                signature = self._tie_pitch_signature(element)
                start_index, stop_index = self._connected_tie_chain_indices(
                    events,
                    index,
                    signature,
                )
                for chain_event in events[start_index : stop_index + 1]:
                    chain_element = chain_event["element"]
                    chain_element_id = id(chain_element)
                    if chain_element_id in cleared_element_ids:
                        continue
                    chain_tie = getattr(chain_element, "tie", None)
                    if chain_tie is None:
                        continue
                    if self._tie_pitch_signature(chain_element) != signature:
                        continue
                    cleared_element_ids.add(chain_element_id)
                    cleared.append({
                        "measure": int(chain_event["measure"]),
                        "beat": float(chain_event["beat"]),
                        "tie_type": chain_tie.type,
                        "voice": voice,
                    })
                    chain_element.tie = None
        return cleared


    def _previous_measure_tie_pitch_set(
        self,
        part_obj: m21stream.Part,
        measure_number: int,
    ) -> set[str]:
        """Return pitches tied forward from the preceding measure."""
        previous_measure = self._previous_measure(part_obj, measure_number)
        if previous_measure is None:
            return set()

        tied_pitches: set[str] = set()
        tied_elements = previous_measure.recurse().getElementsByClass(
            m21note.NotRest,
        )
        for element in tied_elements:
            if not self._element_tie_continues_to_next_measure(element):
                continue
            self._add_element_pitches_to_set(element, tied_pitches)
        return tied_pitches


    def _measure_tie_continuation_pitch_set(
        self,
        measure_obj: m21stream.Measure,
    ) -> set[str]:
        """Return pitches marked as continuing a tie from a previous measure."""
        tied_pitches: set[str] = set()
        tied_elements = measure_obj.recurse().getElementsByClass(m21note.NotRest)
        for element in tied_elements:
            if not self._element_tie_continues_from_previous_measure(element):
                continue
            self._add_element_pitches_to_set(element, tied_pitches)
        return tied_pitches


    @staticmethod
    def _add_element_pitches_to_set(
        element: m21note.GeneralNote,
        pitch_set: set[str],
    ) -> None:
        """Add all pitches from a note or chord to ``pitch_set``."""
        if isinstance(element, m21chord.Chord):
            for pitch_obj in element.pitches:
                pitch_set.add(pitch_obj.nameWithOctave)
        elif isinstance(element, m21note.Note):
            pitch_set.add(element.pitch.nameWithOctave)


    def _previous_measure(
        self,
        part_obj: m21stream.Part,
        measure_number: int,
    ) -> Optional[m21stream.Measure]:
        """Return the measure immediately before ``measure_number`` if present."""
        direct_previous = part_obj.measure(measure_number - 1)
        if direct_previous is not None:
            return direct_previous

        previous_measure = None
        for measure_obj in self._sorted_part_measures(part_obj):
            if measure_obj.number is None or measure_obj.number >= measure_number:
                break
            previous_measure = measure_obj
        return previous_measure


    def _next_measure_number(
        self,
        part_obj: m21stream.Part,
        measure_number: int,
    ) -> Optional[int]:
        """Return the next existing measure number after ``measure_number``."""
        direct_next = part_obj.measure(measure_number + 1)
        if direct_next is not None:
            return direct_next.number

        for measure_obj in self._sorted_part_measures(part_obj):
            if measure_obj.number is not None and measure_obj.number > measure_number:
                return measure_obj.number
        return None


    @staticmethod
    def _element_tie_continues_to_next_measure(
        element: m21note.GeneralNote,
    ) -> bool:
        """Return whether an element's tie can affect the following measure."""
        tie_obj = getattr(element, "tie", None)
        return tie_obj is not None and tie_obj.type in {"start", "continue"}


    @staticmethod
    def _element_tie_continues_from_previous_measure(
        element: m21note.GeneralNote,
    ) -> bool:
        """Return whether an element's tie can be affected by the prior measure."""
        tie_obj = getattr(element, "tie", None)
        return tie_obj is not None and tie_obj.type in {"stop", "continue"}
