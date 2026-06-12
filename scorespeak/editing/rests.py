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


class RestEditingMixin:
    """Internal mixin for ScoreSpeak note-editing operations."""

    def add_rest(
        self,
        measure: int,
        beat: float,
        duration: DurationInput,
        part: Union[int, str],
        voice: int = 1,
        dots: int = 0,
    ) -> OperationResult:
        """Make one explicit visible rest at a precise beat and duration.
        Prefer to use ``reshape_rests`` for deliberate rest spelling changes.

        Use this when one rest should be visible at a known location, including
        when a hidden rest should be made visible. If the requested span overlaps
        only rests, the touched rest spelling is locally replaced with the
        requested visible rest and visible rest spelling for any leftover space.

        Args:
            measure: 1-based measure number.
            beat: 1-based beat where the visible rest starts.
            duration: Visible rest duration to add, such as ``"quarter"``.
            part: Part identifier.
            voice: 1-based rhythmic timeline inside this part.
            dots: Number of augmentation dots for ``duration``.

        Returns:
            OperationResult describing inserted and locally refilled rests.
        """
        if measure is None:
            raise ValueError("add_rest requires a measure.")
        if part is None:
            raise ValueError("add_rest requires a part.")
        if beat < 1.0:
            raise ValueError(f"beat must be at least 1.0, got {beat}.")
        if not isinstance(dots, int) or isinstance(dots, bool) or dots < 0:
            raise ValueError("dots must be a non-negative integer.")

        voice = validate_voice_number(voice)
        duration_obj = normalize_duration(duration, dots=dots)
        quarter_length = float(duration_obj.quarterLength)
        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure)
        container = self._get_voice_or_measure(measure_obj, voice, create=True)
        time_signature = self._get_active_time_signature_obj(part_obj, measure)
        capacity = self._effective_measure_capacity(measure_obj, time_signature)
        offset = beat - 1.0
        self._validate_event_capacity(
            capacity,
            quarter_length,
            measure,
            beat_position=beat,
            ratio_string=time_signature.ratioString,
        )
        end_offset = offset + quarter_length
        self._validate_rest_spelling_range_is_silent(
            container,
            offset,
            end_offset,
            measure,
            voice=voice,
        )

        removed_rests = self._remove_rests_overlapping_range(
            container,
            offset,
            end_offset,
            measure,
            part_idx,
            voice,
        )
        requested_payload = {
            "duration": duration_obj.type,
            "quarter_length": self._clean_rhythm_float(quarter_length),
            "dots": int(duration_obj.dots),
        }
        inserted_rests = self._insert_visible_rest_payloads(
            container,
            offset,
            [requested_payload],
            measure,
            part_idx,
            voice,
        )
        refill_payloads = self._visible_rest_payloads_for_removed_remainders(
            removed_rests,
            offset,
            end_offset,
            measure,
            part_idx,
            voice,
            capacity,
        )
        refilled_rests = self._insert_visible_rest_payloads(
            container,
            0.0,
            refill_payloads,
            measure,
            part_idx,
            voice,
        )
        self._refresh_measure_beams(measure_obj)
        integrity = self._analyze_measure_integrity(
            part_idx,
            measure,
            container,
            voice,
            time_signature,
            capacity=capacity,
        )
        mode = self._add_rest_mode(removed_rests, offset, end_offset)

        return OperationResult(
            success=True,
            description=(
                f"Added visible {duration_obj.type} rest in measure {measure}, "
                f"part {part_idx}, voice {voice}, beat {beat}"
            ),
            details={
                "mode": mode,
                "measure": measure,
                "beat": beat,
                "part": part_idx,
                "voice": voice,
                "requested_rest": requested_payload,
                "total_quarter_length": self._clean_rhythm_float(
                    quarter_length
                ),
                "removed_rests": removed_rests,
                "inserted_rests": inserted_rests,
                "refilled_rests": refilled_rests,
                "measure_integrity": integrity,
                "repair_hint": self._measure_repair_hint(integrity),
            },
        )


    def fill_measure_gaps(
        self,
        measure: int,
        part: Union[int, str],
        voice: int = 1,
    ) -> OperationResult:
        """Fill actual uncovered rhythmic gaps in one measure voice with rests.

        Hidden rests already occupy rhythmic time and are not treated as gaps.
        Use ``add_rest`` to make a specific hidden rest visible, and use
        ``reshape_rests`` for deliberate visible-rest spelling changes.

        Args:
            measure: 1-based measure number.
            part: Part identifier.
            voice: 1-based rhythmic timeline inside this part.

        Returns:
            OperationResult describing inserted visible rests, or a successful
            no-op when the measure voice has no actual gaps.
        """
        if measure is None:
            raise ValueError("fill_measure_gaps requires a measure.")
        if part is None:
            raise ValueError("fill_measure_gaps requires a part.")

        voice = validate_voice_number(voice)
        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure)
        container = self._get_voice_or_measure(measure_obj, voice)
        time_signature = self._get_active_time_signature_obj(part_obj, measure)
        capacity = self._effective_measure_capacity(measure_obj, time_signature)
        integrity = self._analyze_measure_integrity(
            part_idx,
            measure,
            container,
            voice,
            time_signature,
            capacity=capacity,
        )
        if integrity["overlaps"] or integrity["overfull"]:
            raise ValueError(
                f"Cannot fill gaps in measure {measure}, voice {voice}: "
                "fix overlaps or overfull content first."
            )

        gaps = list(integrity["gaps"])
        if not gaps:
            return OperationResult(
                success=True,
                description=(
                    f"No rhythmic gaps to fill in measure {measure}, "
                    f"part {part_idx}, voice {voice}"
                ),
                details={
                    "measure": measure,
                    "part": part_idx,
                    "voice": voice,
                    "count": 0,
                    "filled_gaps": [],
                    "inserted_rests": [],
                    "measure_integrity": integrity,
                    "repair_hint": self._measure_repair_hint(integrity),
                },
            )

        rest_payloads = self._rest_payloads_for_gaps(
            gaps,
            measure,
            part_idx,
            voice,
            measure_capacity=capacity,
        )
        inserted_rests = self._insert_visible_rest_payloads(
            container,
            0.0,
            rest_payloads,
            measure,
            part_idx,
            voice,
        )
        self._refresh_measure_beams(measure_obj)
        final_integrity = self._analyze_measure_integrity(
            part_idx,
            measure,
            container,
            voice,
            time_signature,
            capacity=capacity,
        )

        return OperationResult(
            success=True,
            description=(
                f"Filled {len(gaps)} rhythmic gap(s) in measure {measure}, "
                f"part {part_idx}, voice {voice}"
            ),
            details={
                "measure": measure,
                "part": part_idx,
                "voice": voice,
                "count": len(inserted_rests),
                "filled_gaps": gaps,
                "inserted_rests": inserted_rests,
                "measure_integrity": final_integrity,
                "repair_hint": self._measure_repair_hint(final_integrity),
            },
        )


    def reshape_rests(
        self,
        measure: int,
        part: Union[int, str],
        voice: int,
        start_beat: float,
        total_duration: DurationInput,
        rests: list[dict[str, object]],
    ) -> OperationResult:
        """Replace visible rest spelling over an already-silent range.

        Args:
            measure: 1-based measure number.
            part: Part identifier.
            voice: 1-based rhythmic timeline inside this part.
            start_beat: 1-based beat where the silent range starts.
            total_duration: Total duration of the reshaped silent range.
            rests: Rest spelling entries. Each entry must include ``duration``
                and may include ``dots``.

        Returns:
            OperationResult describing removed and inserted visible rests.
        """
        if measure is None:
            raise ValueError("reshape_rests requires a measure.")
        if part is None:
            raise ValueError("reshape_rests requires a part.")
        if start_beat < 1.0:
            raise ValueError(
                f"start_beat must be at least 1.0, got {start_beat}."
            )

        voice = validate_voice_number(voice)
        total_duration_obj = normalize_duration(total_duration)
        rest_payloads = self._normalize_rest_spelling_payload(rests)
        spelled_duration = sum(
            float(rest_payload["quarter_length"])
            for rest_payload in rest_payloads
        )
        if abs(spelled_duration - total_duration_obj.quarterLength) > _RHYTHM_EPSILON:
            raise ValueError(
                "Rest spelling duration does not match total_duration: "
                f"rests total {spelled_duration:g} beats, but "
                f"total_duration is {total_duration_obj.quarterLength:g} beats."
            )

        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure)
        container = self._get_voice_or_measure(measure_obj, voice, create=True)

        ts = self._get_active_time_signature_obj(part_obj, measure)
        capacity = self._effective_measure_capacity(measure_obj, ts)
        offset = start_beat - 1.0
        self._validate_event_capacity(
            capacity,
            total_duration_obj.quarterLength,
            measure,
            beat_position=start_beat,
            ratio_string=ts.ratioString,
        )
        end_offset = offset + float(total_duration_obj.quarterLength)
        self._validate_rest_spelling_range_is_silent(
            container,
            offset,
            end_offset,
            measure,
            voice=voice,
        )

        removed_rests = self._remove_rests_in_range_preserving_outside(
            container,
            offset,
            end_offset,
            measure,
            part_idx,
            voice,
        )
        inserted_rests = self._insert_visible_rest_payloads(
            container,
            offset,
            rest_payloads,
            measure,
            part_idx,
            voice,
        )
        self._refresh_measure_beams(measure_obj)
        integrity = self._analyze_measure_integrity(
            part_idx,
            measure,
            container,
            voice,
            ts,
            capacity=capacity,
        )

        return OperationResult(
            success=True,
            description=(
                f"Reshaped rests in measure {measure}, part {part_idx}, "
                f"voice {voice}, beat {start_beat}"
            ),
            details={
                "measure": measure,
                "start_beat": start_beat,
                "total_duration": total_duration_obj.type,
                "total_quarter_length": total_duration_obj.quarterLength,
                "part": part_idx,
                "voice": voice,
                "removed_rests": removed_rests,
                "inserted_rests": inserted_rests,
                "measure_integrity": integrity,
                "repair_hint": self._measure_repair_hint(integrity),
            },
        )


    def remove_rests(
        self,
        measure: int,
        part: Union[int, str],
        voice: int,
        beat: Optional[float] = None,
    ) -> OperationResult:
        """Remove visible rest notation from one measure voice.

        Args:
            measure: 1-based measure number.
            part: Part identifier.
            voice: 1-based rhythmic timeline inside this part.
            beat: Optional 1-based beat where a visible rest starts. If
                omitted, all visible rests in the selected measure voice are
                removed.

        Returns:
            OperationResult describing the removed visible rests.
        """
        if measure is None:
            raise ValueError("remove_rests requires a measure.")
        if part is None:
            raise ValueError("remove_rests requires a part.")
        if beat is not None and beat < 1.0:
            raise ValueError(f"beat must be at least 1.0, got {beat}.")

        voice = validate_voice_number(voice)
        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure)
        container = self._get_voice_or_measure(measure_obj, voice)

        if beat is None:
            hidden_rests = self._hide_all_visible_rests(
                container,
                measure,
                part_idx,
                voice,
            )
        else:
            hidden_rests = self._hide_visible_rest_at_beat(
                container,
                beat,
                measure,
                part_idx,
                voice,
            )

        if not hidden_rests:
            target = (
                f" at beat {beat}" if beat is not None else ""
            )
            raise ValueError(
                f"No visible rests found{target} in measure {measure}, "
                f"part {part_idx}, voice {voice}."
            )

        self._refresh_measure_beams(measure_obj)
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
                f"Hidden {len(hidden_rests)} visible rest(s) in "
                f"measure {measure}, part {part_idx}, voice {voice}"
            ),
            details={
                "measure": measure,
                "part": part_idx,
                "voice": voice,
                "beat": beat,
                "count": len(hidden_rests),
                "hidden_rests": hidden_rests,
                "measure_integrity": integrity,
                "repair_hint": self._measure_repair_hint(integrity),
            },
        )


    def _gap_payload(self, start: float, end: float) -> dict[str, object]:
        """Return a JSON-safe payload for one rhythm gap."""
        return {
            "beat": self._clean_rhythm_float(start + 1.0),
            "offset": self._clean_rhythm_float(start),
            "end": self._clean_rhythm_float(end),
            "quarter_length": self._clean_rhythm_float(end - start),
        }


    def _rest_payloads_for_gaps(
        self,
        gaps: list[dict[str, object]],
        measure_number: int,
        part_idx: int,
        voice: int,
        measure_capacity: float | None = None,
    ) -> list[dict[str, object]]:
        """Return explicit rest payloads that cover the provided gaps."""
        rests: list[dict[str, object]] = []
        for gap in gaps:
            gap_offset = float(gap["offset"])
            gap_end = float(gap["end"])
            for duration_info in self._rest_duration_payloads_for_gap(
                gap_offset,
                gap_end,
                measure_capacity=measure_capacity,
            ):
                quarter_length = float(duration_info["quarter_length"])
                current_offset = float(duration_info["offset"])
                rests.append(
                    {
                        "kind": "rest",
                        "duration": duration_info["duration"],
                        "quarter_length": self._clean_rhythm_float(
                            quarter_length
                        ),
                        "dots": duration_info["dots"],
                        "measure": measure_number,
                        "part": part_idx,
                        "voice": voice,
                        "beat": self._clean_rhythm_float(current_offset + 1.0),
                        "offset": self._clean_rhythm_float(current_offset),
                        "visibility": "visible",
                    }
                )
        return rests


    def _rest_duration_payloads_for_gap(
        self,
        start_offset: float,
        end_offset: float,
        *,
        measure_capacity: float | None = None,
    ) -> list[dict[str, object]]:
        """Return positioned rest durations for one uncovered range."""
        quarter_length = end_offset - start_offset
        if quarter_length <= _RHYTHM_EPSILON:
            return []

        duration_payloads = self._rest_duration_payloads(quarter_length)
        forward_payloads = self._position_rest_duration_payloads_forward(
            duration_payloads,
            start_offset,
        )
        if self._is_quarter_beat_boundary(start_offset):
            return forward_payloads

        reverse_payloads = self._position_rest_duration_payloads_reverse(
            duration_payloads,
            end_offset,
        )
        if self._is_quarter_beat_boundary(end_offset):
            return reverse_payloads
        if measure_capacity is not None and self._same_rhythm_position(
            end_offset,
            measure_capacity,
        ):
            return reverse_payloads

        if self._rest_spelling_alignment_score(
            reverse_payloads,
        ) < self._rest_spelling_alignment_score(forward_payloads):
            return reverse_payloads
        return forward_payloads


    def _position_rest_duration_payloads_forward(
        self,
        duration_payloads: list[dict[str, object]],
        start_offset: float,
    ) -> list[dict[str, object]]:
        """Position rest durations from the start of a gap."""
        positioned: list[dict[str, object]] = []
        current_offset = start_offset
        for duration_payload in duration_payloads:
            rest_payload = dict(duration_payload)
            rest_payload["offset"] = self._clean_rhythm_float(current_offset)
            positioned.append(rest_payload)
            current_offset += float(duration_payload["quarter_length"])
        return positioned


    def _position_rest_duration_payloads_reverse(
        self,
        duration_payloads: list[dict[str, object]],
        end_offset: float,
    ) -> list[dict[str, object]]:
        """Position rest durations backward from the end of a gap."""
        positioned: list[dict[str, object]] = []
        current_offset = end_offset
        for duration_payload in duration_payloads:
            quarter_length = float(duration_payload["quarter_length"])
            current_offset -= quarter_length
            rest_payload = dict(duration_payload)
            rest_payload["offset"] = self._clean_rhythm_float(current_offset)
            positioned.append(rest_payload)
        return sorted(positioned, key=lambda item: float(item["offset"]))


    def _rest_spelling_alignment_score(
        self,
        rest_payloads: list[dict[str, object]],
    ) -> tuple[float, ...]:
        """Score whether larger rests start close to quarter-beat boundaries."""
        sorted_payloads = sorted(
            rest_payloads,
            key=lambda item: (
                -float(item["quarter_length"]),
                float(item["offset"]),
            ),
        )
        return tuple(
            self._distance_to_quarter_beat(float(payload["offset"]))
            for payload in sorted_payloads
        )


    @staticmethod
    def _is_quarter_beat_boundary(offset: float) -> bool:
        """Return whether an offset falls on a quarter-note beat boundary."""
        return abs(offset - round(offset)) <= _RHYTHM_EPSILON


    @staticmethod
    def _same_rhythm_position(left: float, right: float) -> bool:
        """Return whether two rhythm positions match within rhythm tolerance."""
        return abs(left - right) <= _RHYTHM_EPSILON


    @staticmethod
    def _distance_to_quarter_beat(offset: float) -> float:
        """Return the distance from an offset to the nearest quarter beat."""
        return abs(offset - round(offset))


    def _rest_duration_payloads(
        self,
        quarter_length: float,
    ) -> list[dict[str, object]]:
        """Decompose a gap into simple rest durations."""
        remaining = quarter_length
        payloads: list[dict[str, object]] = []
        for candidate in self._rest_duration_candidates():
            if remaining <= _RHYTHM_EPSILON:
                break
            candidate_ql = float(candidate["quarter_length"])
            while remaining + _RHYTHM_EPSILON >= candidate_ql:
                payloads.append(dict(candidate))
                remaining -= candidate_ql

        if remaining > _RHYTHM_EPSILON:
            duration_obj = m21duration.Duration(quarterLength=remaining)
            payloads.append(
                {
                    "duration": duration_obj.type,
                    "quarter_length": self._clean_rhythm_float(
                        duration_obj.quarterLength
                    ),
                    "dots": int(duration_obj.dots),
                }
            )
        return payloads


    @staticmethod
    def _rest_duration_candidates() -> list[dict[str, object]]:
        """Return simple rest durations sorted from longest to shortest."""
        base_lengths = [
            ("whole", 4.0),
            ("half", 2.0),
            ("quarter", 1.0),
            ("eighth", 0.5),
            ("16th", 0.25),
            ("32nd", 0.125),
            ("64th", 0.0625),
            ("128th", 0.03125),
        ]
        candidates: dict[float, dict[str, object]] = {}
        for duration_name, base_ql in base_lengths:
            for dots in (0, 1, 2):
                multiplier = 1.0
                if dots >= 1:
                    multiplier += 0.5
                if dots >= 2:
                    multiplier += 0.25
                quarter_length = base_ql * multiplier
                candidates[quarter_length] = {
                    "duration": duration_name,
                    "quarter_length": quarter_length,
                    "dots": dots,
                }
        return [
            candidates[quarter_length]
            for quarter_length in sorted(candidates, reverse=True)
        ]


    def _normalize_rest_spelling_payload(
        self,
        rests: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Validate and normalize explicit visible-rest spelling entries."""
        if not isinstance(rests, list) or not rests:
            raise ValueError("reshape_rests requires a non-empty rests list.")

        normalized: list[dict[str, object]] = []
        for index, item in enumerate(rests):
            if hasattr(item, "model_dump"):
                raw_item = item.model_dump()
            else:
                raw_item = item
            if not isinstance(raw_item, dict):
                raise ValueError(
                    f"rests[{index}] must be an object with duration and dots."
                )

            keys = set(raw_item)
            missing = {"duration"} - keys
            if missing:
                raise ValueError(
                    f"rests[{index}] missing required field(s): duration."
                )
            extra = sorted(keys - {"duration", "dots"})
            if extra:
                fields = ", ".join(extra)
                raise ValueError(
                    f"rests[{index}] has unsupported field(s): {fields}."
                )

            dots = raw_item.get("dots", 0)
            if not isinstance(dots, int) or isinstance(dots, bool) or dots < 0:
                raise ValueError(
                    f"rests[{index}] dots must be a non-negative integer."
                )
            duration_obj = normalize_duration(raw_item["duration"], dots=dots)
            normalized.append(
                {
                    "duration": duration_obj.type,
                    "quarter_length": self._clean_rhythm_float(
                        duration_obj.quarterLength
                    ),
                    "dots": int(duration_obj.dots),
                }
            )
        return normalized


    def _validate_rest_spelling_range_is_silent(
        self,
        container: m21stream.Stream,
        offset: float,
        end_offset: float,
        measure_number: int,
        voice: int,
    ) -> None:
        """Raise if a visible rest spelling range overlaps sounding content."""
        for element in container.getElementsByClass(m21note.GeneralNote):
            element_offset = float(container.elementOffset(element))
            if getattr(element.duration, "isGrace", False):
                if (
                    offset <= element_offset + _RHYTHM_EPSILON
                    and element_offset < end_offset - _RHYTHM_EPSILON
                ):
                    raise ValueError(
                        f"Cannot reshape rests in measure {measure_number}, "
                        f"voice {voice}: a grace note is anchored at beat "
                        f"{self._clean_rhythm_float(element_offset + 1.0)}."
                    )
                continue

            event_range = self._rhythm_event_range(container, element)
            if event_range is None:
                continue
            if not self._event_ranges_overlap(offset, end_offset, event_range):
                continue
            if isinstance(element, m21note.Rest):
                continue
            raise ValueError(
                f"Cannot reshape rests in measure {measure_number}, voice "
                f"{voice}: requested range overlaps existing "
                f"{event_range['label']} from beat {event_range['beat']} to "
                f"{self._clean_rhythm_float(float(event_range['end']) + 1.0)}."
            )


    def _insert_visible_rest_payloads(
        self,
        container: m21stream.Stream,
        offset: float,
        rest_payloads: list[dict[str, object]],
        measure_number: int,
        part_idx: int,
        voice: int,
    ) -> list[dict[str, object]]:
        """Insert visible rests from normalized payloads and return summaries."""
        return self._insert_rest_payloads(
            container,
            offset,
            rest_payloads,
            measure_number,
            part_idx,
            voice,
            hidden=False,
        )


    def _insert_rest_payloads(
        self,
        container: m21stream.Stream,
        offset: float,
        rest_payloads: list[dict[str, object]],
        measure_number: int,
        part_idx: int,
        voice: int,
        *,
        hidden: bool,
    ) -> list[dict[str, object]]:
        """Insert rest payloads and return their realized summaries."""
        inserted = []
        current_offset = offset
        for rest_payload in rest_payloads:
            quarter_length = float(rest_payload["quarter_length"])
            insert_offset = float(rest_payload.get("offset", current_offset))
            rest = self._make_rest(
                quarter_length,
                dots=int(rest_payload["dots"]) if "dots" in rest_payload else None,
                hidden=hidden,
            )
            container.insert(insert_offset, rest)
            inserted.append(
                self._rest_event_payload(
                    container,
                    rest,
                    measure_number,
                    part_idx,
                    voice,
                )
            )
            current_offset = insert_offset + quarter_length
        return inserted


    def _remove_rests_in_range_preserving_outside(
        self,
        container: m21stream.Stream,
        offset: float,
        end_offset: float,
        measure_number: int,
        part_idx: int,
        voice: int,
    ) -> list[dict[str, object]]:
        """Remove rest coverage in a range while preserving outside segments."""
        removed = []
        rest_elements = list(container.getElementsByClass(m21note.Rest))
        for element in rest_elements:
            event_range = self._rhythm_event_range(container, element)
            if event_range is None:
                continue
            if not self._event_ranges_overlap(offset, end_offset, event_range):
                continue
            removed.append(
                self._rest_event_payload(
                    container,
                    element,
                    measure_number,
                    part_idx,
                    voice,
                )
            )
            self._replace_rest_with_outside_segments(
                container,
                element,
                offset,
                end_offset,
            )
        return removed


    def _remove_rests_overlapping_range(
        self,
        container: m21stream.Stream,
        offset: float,
        end_offset: float,
        measure_number: int,
        part_idx: int,
        voice: int,
    ) -> list[dict[str, object]]:
        """Remove whole rests that overlap a requested local rest spelling range."""
        removed = []
        rest_elements = list(container.getElementsByClass(m21note.Rest))
        for element in rest_elements:
            event_range = self._rhythm_event_range(container, element)
            if event_range is None:
                continue
            if not self._event_ranges_overlap(offset, end_offset, event_range):
                continue
            removed.append(
                self._rest_event_payload(
                    container,
                    element,
                    measure_number,
                    part_idx,
                    voice,
                )
            )
            container.remove(element)
        return removed


    def _visible_rest_payloads_for_removed_remainders(
        self,
        removed_rests: list[dict[str, object]],
        offset: float,
        end_offset: float,
        measure_number: int,
        part_idx: int,
        voice: int,
        measure_capacity: float,
    ) -> list[dict[str, object]]:
        """Return visible rest payloads for removed-rest space outside a request."""
        remainder_gaps: list[dict[str, object]] = []
        for removed_rest in removed_rests:
            rest_offset = float(removed_rest["offset"])
            rest_end = float(removed_rest["end"])
            left_end = min(offset, rest_end)
            if rest_offset < left_end - _RHYTHM_EPSILON:
                remainder_gaps.append(self._gap_payload(rest_offset, left_end))
            right_start = max(end_offset, rest_offset)
            if right_start < rest_end - _RHYTHM_EPSILON:
                remainder_gaps.append(self._gap_payload(right_start, rest_end))

        remainder_gaps.sort(key=lambda gap: float(gap["offset"]))
        return self._rest_payloads_for_gaps(
            remainder_gaps,
            measure_number,
            part_idx,
            voice,
            measure_capacity=measure_capacity,
        )


    def _add_rest_mode(
        self,
        removed_rests: list[dict[str, object]],
        offset: float,
        end_offset: float,
    ) -> str:
        """Return a stable mode label for an add_rest operation."""
        if not removed_rests:
            return "inserted"
        if len(removed_rests) != 1:
            return "respelled"

        removed_rest = removed_rests[0]
        exact_start = self._same_rhythm_position(float(removed_rest["offset"]), offset)
        exact_end = self._same_rhythm_position(float(removed_rest["end"]), end_offset)
        if (
            exact_start
            and exact_end
            and removed_rest.get("visibility") == "hidden"
        ):
            return "unhidden"
        return "respelled"


    def _hide_all_visible_rests(
        self,
        container: m21stream.Stream,
        measure_number: int,
        part_idx: int,
        voice: int,
    ) -> list[dict[str, object]]:
        """Hide all visible rests in a stream and return summaries."""
        hidden_rests = []
        rest_elements = self._visible_rests(container)
        for rest in rest_elements:
            self._mark_rest_hidden(rest)
            hidden_rests.append(
                self._rest_event_payload(
                    container,
                    rest,
                    measure_number,
                    part_idx,
                    voice,
                )
            )
        return hidden_rests


    def _hide_visible_rest_at_beat(
        self,
        container: m21stream.Stream,
        beat: float,
        measure_number: int,
        part_idx: int,
        voice: int,
    ) -> list[dict[str, object]]:
        """Hide visible rests that start at one beat and return summaries."""
        offset = beat - 1.0
        hidden_rests = []
        for element in self._visible_rests(container):
            if abs(float(container.elementOffset(element)) - offset) > _RHYTHM_EPSILON:
                continue
            self._mark_rest_hidden(element)
            hidden_rests.append(
                self._rest_event_payload(
                    container,
                    element,
                    measure_number,
                    part_idx,
                    voice,
                )
            )
        return hidden_rests


    def _rest_event_payload(
        self,
        container: m21stream.Stream,
        rest: m21note.Rest,
        measure_number: int,
        part_idx: int,
        voice: int,
    ) -> dict[str, object]:
        """Return a JSON-safe payload for one rest event."""
        offset = float(container.elementOffset(rest))
        quarter_length = float(rest.duration.quarterLength)
        return {
            "kind": "rest",
            "duration": rest.duration.type,
            "quarter_length": self._clean_rhythm_float(quarter_length),
            "dots": int(rest.duration.dots),
            "label": self._describe_element(rest),
            "measure": measure_number,
            "part": part_idx,
            "voice": voice,
            "beat": self._clean_rhythm_float(offset + 1.0),
            "offset": self._clean_rhythm_float(offset),
            "end": self._clean_rhythm_float(offset + quarter_length),
            "visibility": self._element_visibility(rest),
        }


    def _visible_rests(
        self,
        container: m21stream.Stream,
    ) -> list[m21note.Rest]:
        """Return non-grace visible rests in one stream."""
        return [
            rest
            for rest in list(container.getElementsByClass(m21note.Rest))
            if not getattr(rest.duration, "isGrace", False)
            and not self._is_hidden_rest(rest)
        ]


    def _make_rest(
        self,
        quarter_length: float,
        *,
        dots: int | None = None,
        hidden: bool = False,
    ) -> m21note.Rest:
        """Create a rest with ScoreSpeak visibility semantics."""
        duration = m21duration.Duration(quarterLength=quarter_length)
        if dots is not None:
            duration.dots = dots
        rest = m21note.Rest(duration=duration)
        if hidden:
            self._mark_rest_hidden(rest)
        return rest


    @staticmethod
    def _mark_rest_hidden(rest: m21note.Rest) -> None:
        """Mark a rest as hidden suppression."""
        rest.style.hideObjectOnPrint = True


    def _replace_rest_with_outside_segments(
        self,
        container: m21stream.Stream,
        rest: m21note.Rest,
        offset: float,
        end_offset: float,
    ) -> None:
        """Remove a rest and keep any non-overlapped portions."""
        rest_offset = float(container.elementOffset(rest))
        rest_end = rest_offset + float(rest.duration.quarterLength)
        hidden = self._is_hidden_rest(rest)
        segments = []
        if rest_offset < offset - _RHYTHM_EPSILON:
            segments.append((rest_offset, offset - rest_offset))
        if end_offset < rest_end - _RHYTHM_EPSILON:
            segments.append((end_offset, rest_end - end_offset))

        container.remove(rest)
        for segment_offset, segment_length in segments:
            segment_rest = self._make_rest(segment_length, hidden=hidden)
            container.insert(segment_offset, segment_rest)


    def _normalize_rests_after_rhythm_edit(
        self,
        part_obj: m21stream.Part,
        part_idx: int,
        measure_obj: m21stream.Measure,
        measure_number: int,
        container: m21stream.Stream,
        voice: int,
    ) -> list[dict[str, object]]:
        """Normalize rests in the affected voice after a rhythm edit."""
        if not self._stream_has_sounding_content(container):
            if not self._measure_has_sounding_content(measure_obj):
                return self._collapse_empty_measure_to_full_rest(
                    part_obj,
                    part_idx,
                    measure_obj,
                )
            self._remove_all_rests_from_stream(container)
            self._remove_empty_voice_if_needed(measure_obj, container)
            return []

        self._remove_visible_rests_from_inactive_streams(measure_obj)
        time_signature = self._get_active_time_signature_obj(
            part_obj,
            measure_number,
        )
        capacity = self._effective_measure_capacity(measure_obj, time_signature)
        ranges = self._occupied_event_ranges(container)
        sorted_ranges = sorted(ranges, key=lambda item: (item["offset"], item["end"]))
        gaps = self._rhythm_gaps(sorted_ranges, capacity)
        rest_payloads = self._rest_payloads_for_gaps(
            gaps,
            measure_number,
            part_idx,
            voice,
            measure_capacity=capacity,
        )
        return self._insert_visible_rest_payloads(
            container,
            0.0,
            rest_payloads,
            measure_number,
            part_idx,
            voice,
        )


    def _collapse_empty_measure_to_full_rest(
        self,
        part_obj: m21stream.Part,
        part_idx: int,
        measure_obj: m21stream.Measure,
    ) -> list[dict[str, object]]:
        """Collapse a measure with no sounding events to one full-measure rest."""
        measure_number = int(measure_obj.number)
        time_signature = self._get_active_time_signature_obj(
            part_obj,
            measure_number,
        )
        capacity = self._effective_measure_capacity(measure_obj, time_signature)
        hidden = self._hidden_rests_cover_range(measure_obj, 0.0, capacity)

        for voice_obj in list(measure_obj.voices):
            measure_obj.remove(voice_obj)
        for element in list(measure_obj.getElementsByClass(m21note.GeneralNote)):
            measure_obj.remove(element)

        rest = self._make_rest(capacity, hidden=hidden)
        measure_obj.insert(0.0, rest)
        if hidden:
            return []
        return [
            self._rest_event_payload(
                measure_obj,
                rest,
                measure_number,
                part_idx,
                1,
            )
        ]


    def _hidden_rests_cover_range(
        self,
        measure_obj: m21stream.Measure,
        offset: float,
        end_offset: float,
    ) -> bool:
        """Return whether hidden rests cover a measure-local range."""
        ranges = []
        for stream_obj in [measure_obj, *list(measure_obj.voices)]:
            for rest in stream_obj.getElementsByClass(m21note.Rest):
                if not self._is_hidden_rest(rest):
                    continue
                if getattr(rest.duration, "isGrace", False):
                    continue
                start = float(stream_obj.elementOffset(rest))
                end = start + float(rest.duration.quarterLength)
                ranges.append((start, end))

        cursor = offset
        for start, end in sorted(ranges):
            if end <= cursor + _RHYTHM_EPSILON:
                continue
            if start > cursor + _RHYTHM_EPSILON:
                return False
            cursor = max(cursor, end)
            if cursor >= end_offset - _RHYTHM_EPSILON:
                return True
        return end_offset <= offset + _RHYTHM_EPSILON


    def _remove_visible_rests_from_inactive_streams(
        self,
        measure_obj: m21stream.Measure,
    ) -> None:
        """Remove visible rests from streams that have no sounding content."""
        for _voice_id, stream_obj in self._measure_voice_containers(measure_obj):
            if self._stream_has_sounding_content(stream_obj):
                continue
            for rest in self._visible_rests(stream_obj):
                stream_obj.remove(rest)
            self._remove_empty_voice_if_needed(measure_obj, stream_obj)


    def _remove_all_rests_from_stream(
        self,
        stream_obj: m21stream.Stream,
    ) -> None:
        """Remove all non-grace rests from a stream."""
        for rest in list(stream_obj.getElementsByClass(m21note.Rest)):
            if getattr(rest.duration, "isGrace", False):
                continue
            stream_obj.remove(rest)


    def _remove_empty_voice_if_needed(
        self,
        measure_obj: m21stream.Measure,
        stream_obj: m21stream.Stream,
    ) -> None:
        """Remove an explicit voice if it no longer contains note-like events."""
        if stream_obj is measure_obj:
            return
        if not isinstance(stream_obj, m21stream.Voice):
            return
        if list(stream_obj.getElementsByClass(m21note.GeneralNote)):
            return
        if stream_obj in measure_obj:
            measure_obj.remove(stream_obj)


    def _stream_has_sounding_content(
        self,
        stream_obj: m21stream.Stream,
    ) -> bool:
        """Return whether a stream has duration-bearing notes or chords."""
        for element in stream_obj.getElementsByClass(m21note.GeneralNote):
            if not isinstance(element, (m21note.Note, m21chord.Chord)):
                continue
            if getattr(element.duration, "isGrace", False):
                continue
            if float(element.duration.quarterLength) > _RHYTHM_EPSILON:
                return True
        return False


    def _measure_has_sounding_content(
        self,
        measure_obj: m21stream.Measure,
    ) -> bool:
        """Return whether a measure has duration-bearing notes or chords."""
        for element in measure_obj.recurse().getElementsByClass(m21note.GeneralNote):
            if not isinstance(element, (m21note.Note, m21chord.Chord)):
                continue
            if getattr(element.duration, "isGrace", False):
                continue
            if float(element.duration.quarterLength) > _RHYTHM_EPSILON:
                return True
        return False
