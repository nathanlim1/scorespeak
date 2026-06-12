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


class NoteEventsMixin:
    """Internal mixin for ScoreSpeak note-editing operations."""

    def add_notes(
        self,
        measure: int,
        part: Union[int, str],
        voice: int,
        notes: list[dict[str, object]],
    ) -> OperationResult:
        """Add one or more notes to one measure, part, and voice. One call is
        locked to a single measure, part, and rhythmic voice.
        If notes must be added to different measures, parts, or voices, make
        multiple ``add_notes`` calls. Each note item must specify its own
        pitch, beat, duration, and dots; note items must not include measure,
        part, or voice fields.
        This tool auto-completes silent gaps with convenient visible rests,
        but if a user mentions specific rests, check that ``auto_completed_rests`` 
        matches their expected rest spelling and modify the result with 
        ``reshape_rests`` if necessary.

        Args:
            measure: 1-based measure number for every note in this call.
            part: Part identifier for every note in this call, either a
                0-based part index or part name.
            voice: 1-based rhythmic timeline for every note in this call.
                Use another voice only when these notes must overlap an
                existing event with an independent rhythm. Same-call notes
                with the same beat, duration, and dots become one chord; use
                add_chord_tones when an existing event should gain pitches.
            notes: Non-empty list of note objects. Each item must include
                pitch, beat, duration, and dots. Pitch accepts string ``"C4"``,
                ``"c#4"``, ``"C♯4"``, MIDI integer 60, or a music21 Pitch
                object. Duration accepts ``"quarter"``, ``"8"``, ``"16th"``,
                float quarter-length, etc. Beat is 1-based. Dots is the
                number of augmentation dots.

        Returns:
            OperationResult with details of all added notes.
        """
        if measure is None:
            raise ValueError("add_notes requires a measure.")
        if part is None:
            raise ValueError("add_notes requires a part.")

        voice = validate_voice_number(voice)
        self._validate_add_notes_payload(notes)
        part_obj, part_idx = self._resolve_part(part)
        self._resolve_measure(part_obj, measure)

        normalized_items = [
            self._normalize_add_notes_item(item, index)
            for index, item in enumerate(notes)
        ]
        grouped_items = self._group_add_notes_items(normalized_items)

        shadow_state = type(self)(deepcopy(self._score))
        added_results: list[OperationResult] = []
        for group in grouped_items:
            first_item = group[0]
            try:
                if len(group) == 1:
                    added_results.append(
                        shadow_state._add_note_one(
                            first_item["pitch"],
                            first_item["duration"],
                            measure=measure,
                            beat=first_item["beat"],
                            part=part,
                            voice=voice,
                            dots=first_item["dots"],
                            normalize_rests=False,
                        )
                    )
                else:
                    added_results.append(
                        shadow_state._add_chord_one(
                            [item["pitch"] for item in group],
                            first_item["duration"],
                            measure=measure,
                            beat=first_item["beat"],
                            part=part,
                            voice=voice,
                            dots=first_item["dots"],
                            normalize_rests=False,
                        )
                    )
            except ValueError as exc:
                raise ValueError(f"notes[{first_item['index']}]: {exc}") from exc
            except TypeError as exc:
                raise TypeError(f"notes[{first_item['index']}]: {exc}") from exc

        shadow_part_obj, shadow_part_idx = shadow_state._resolve_part(part)
        shadow_measure_obj = shadow_state._resolve_measure(shadow_part_obj, measure)
        shadow_container = shadow_state._get_voice_or_measure(
            shadow_measure_obj,
            voice,
        )
        auto_completed_rests = shadow_state._normalize_rests_after_rhythm_edit(
            shadow_part_obj,
            shadow_part_idx,
            shadow_measure_obj,
            measure,
            shadow_container,
            voice,
        )
        shadow_state._refresh_measure_beams(shadow_measure_obj)
        shadow_state._refresh_measure_accidentals(shadow_part_obj, measure)
        final_integrity = shadow_state._measure_integrity_for_context(
            shadow_part_obj,
            shadow_part_idx,
            measure,
            shadow_container,
            voice,
        )
        self._commit_shadow_score(shadow_state)
        note_details = [result.details for result in added_results]
        compact_notes = self._compact_added_note_details(note_details)
        replaced_rests = [
            rest
            for details in note_details
            for rest in details["replaced_rests"]
        ]
        warnings = [
            details["warning"]
            for details in compact_notes
            if details.get("warning")
        ]
        warning_text = "; ".join(warnings) if warnings else None
        description = (
            f"Added {len(compact_notes)} note(s) to measure {measure}, "
            f"part {part_idx}, voice {voice}"
        )
        if warning_text:
            description += f" — {warning_text}"

        return OperationResult(
            success=True,
            description=description,
            details={
                "measure": measure,
                "part": part_idx,
                "voice": voice,
                "count": len(compact_notes),
                "notes": compact_notes,
                "warning": warning_text,
                "replaced_rests": replaced_rests,
                "auto_completed_rests": auto_completed_rests,
                "measure_integrity": final_integrity,
                "repair_hint": self._measure_repair_hint(final_integrity),
            },
        )


    def _add_note_one(
        self,
        pitch: PitchInput,
        duration: DurationInput = "quarter",
        measure: Optional[int] = None,
        beat: Optional[float] = None,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
        dots: int = 0,
        normalize_rests: bool = True,
    ) -> OperationResult:
        """Add one note using the core surgical insertion implementation."""
        voice = validate_voice_number(voice)
        pitch_obj = normalize_pitch(pitch)
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

        new_note = m21note.Note(pitch=pitch_obj, duration=dur_obj)
        container.insert(offset, new_note)
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

        warning = self._check_instrument_range(part_obj, pitch_obj)
        integrity = self._analyze_measure_integrity(
            part_idx,
            measure_number,
            container,
            voice,
            ts,
            capacity=capacity,
        )

        desc = (
            f"Added {pitch_obj.nameWithOctave} {dur_obj.type} note "
            f"at measure {measure_number}, beat {beat_pos}"
        )
        if warning:
            desc += f" — {warning}"

        return OperationResult(
            success=True,
            description=desc,
            details={
                "pitch": pitch_obj.nameWithOctave,
                "duration": dur_obj.type,
                "quarter_length": dur_obj.quarterLength,
                "measure": measure_number,
                "beat": beat_pos,
                "part": part_idx,
                "voice": voice,
                "dots": dur_obj.dots,
                "warning": warning,
                "replaced_rests": replaced_rests,
                "auto_completed_rests": auto_completed_rests,
                "measure_integrity": integrity,
                "repair_hint": self._measure_repair_hint(integrity),
            },
        )


    def _validate_add_notes_payload(self, notes: list[dict[str, object]]) -> None:
        """Validate the outer ``add_notes`` note list."""
        if not isinstance(notes, list) or not notes:
            raise ValueError("add_notes requires a non-empty notes list.")

        for index, item in enumerate(notes):
            self._normalize_add_notes_item(item, index)


    def _normalize_add_notes_item(
        self,
        item: object,
        index: int,
    ) -> dict[str, Any]:
        """Return a validated ``add_notes`` item as a plain dictionary."""
        if hasattr(item, "model_dump"):
            raw_item = item.model_dump()
        else:
            raw_item = item

        if not isinstance(raw_item, dict):
            raise ValueError(
                f"notes[{index}] must be an object with pitch, beat, "
                "duration, and dots."
            )

        keys = set(raw_item)
        forbidden = sorted(keys & _ADD_NOTES_FORBIDDEN_FIELDS)
        if forbidden:
            fields = ", ".join(forbidden)
            raise ValueError(
                f"notes[{index}] may not include {fields}; add_notes is "
                "locked to the top-level measure, part, and voice."
            )

        missing = sorted(_ADD_NOTES_REQUIRED_FIELDS - keys)
        if missing:
            fields = ", ".join(missing)
            raise ValueError(f"notes[{index}] missing required field(s): {fields}.")

        extra = sorted(keys - _ADD_NOTES_ALLOWED_FIELDS)
        if extra:
            fields = ", ".join(extra)
            raise ValueError(f"notes[{index}] has unsupported field(s): {fields}.")

        dots = raw_item["dots"]
        if not isinstance(dots, int) or isinstance(dots, bool) or dots < 0:
            raise ValueError(
                f"notes[{index}] dots must be a non-negative integer."
            )
        duration_obj = normalize_duration(raw_item["duration"], dots=dots)
        quarter_length = round(float(duration_obj.quarterLength), 9)
        if quarter_length not in _STANDARD_DURATION_VALUES:
            raise ValueError(
                f"notes[{index}] duration {raw_item['duration']!r} resolves "
                f"to non-standard quarter length {float(duration_obj.quarterLength):g}. "
                "Use add_tuplet for triplets, sextuplets, and other tuplets "
                "instead of raw fractional durations in add_notes."
            )

        return {
            "pitch": raw_item["pitch"],
            "beat": raw_item["beat"],
            "duration": raw_item["duration"],
            "dots": dots,
            "index": index,
        }


    def _group_add_notes_items(
        self,
        items: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Group same-call note items that should become one chord event."""
        groups: list[list[dict[str, Any]]] = []
        group_by_key: dict[tuple[float, float, int], list[dict[str, Any]]] = {}
        for item in items:
            try:
                beat = float(item["beat"])
                duration_obj = normalize_duration(
                    item["duration"],
                    dots=int(item["dots"]),
                )
            except ValueError as exc:
                raise ValueError(f"notes[{item['index']}]: {exc}") from exc
            except TypeError as exc:
                raise TypeError(f"notes[{item['index']}]: {exc}") from exc

            key = (
                self._clean_rhythm_float(beat),
                self._clean_rhythm_float(duration_obj.quarterLength),
                int(duration_obj.dots),
            )
            group = group_by_key.get(key)
            if group is None:
                group = []
                group_by_key[key] = group
                groups.append(group)
            group.append(item)
        return groups


    def _compact_added_note_details(
        self,
        note_details: list[dict[str, Any]],
    ) -> list[dict[str, object]]:
        """Return per-pitch add details for note and chord insertion results."""
        compact_notes: list[dict[str, object]] = []
        for details in note_details:
            if "pitch" in details:
                compact_notes.append(
                    {
                        "pitch": details["pitch"],
                        "duration": details["duration"],
                        "quarter_length": details["quarter_length"],
                        "measure": details["measure"],
                        "beat": details["beat"],
                        "part": details["part"],
                        "voice": details["voice"],
                        "dots": details["dots"],
                        "warning": details["warning"],
                        "replaced_rests": details["replaced_rests"],
                    }
                )
                continue

            for pitch_name in details["pitches"]:
                compact_notes.append(
                    {
                        "pitch": pitch_name,
                        "duration": details["duration"],
                        "quarter_length": details["quarter_length"],
                        "measure": details["measure"],
                        "beat": details["beat"],
                        "part": details["part"],
                        "voice": details["voice"],
                        "dots": details["dots"],
                        "warning": details["warning"],
                        "replaced_rests": details["replaced_rests"],
                    }
                )
        return compact_notes


    def _validate_remove_notes_payload(
        self,
        notes: list[dict[str, object]],
    ) -> list[dict[str, Any]]:
        """Validate and normalize the outer ``remove_notes`` item list."""
        if not isinstance(notes, list) or not notes:
            raise ValueError("remove_notes requires a non-empty notes list.")
        normalized = [
            self._normalize_remove_notes_item(item, index)
            for index, item in enumerate(notes)
        ]
        self._validate_remove_notes_same_beat_rules(normalized)
        return normalized


    def _normalize_remove_notes_item(
        self,
        item: object,
        index: int,
    ) -> dict[str, Any]:
        """Return a validated ``remove_notes`` item as a plain dictionary."""
        if hasattr(item, "model_dump"):
            raw_item = item.model_dump(exclude_unset=True)
        else:
            raw_item = item

        if not isinstance(raw_item, dict):
            raise ValueError(
                f"notes[{index}] must be an object with beat and optional pitch."
            )

        keys = set(raw_item)
        forbidden = sorted(keys & _REMOVE_NOTES_FORBIDDEN_FIELDS)
        if forbidden:
            fields = ", ".join(forbidden)
            raise ValueError(
                f"notes[{index}] may not include {fields}; remove_notes is "
                "locked to the top-level measure, part, and voice."
            )

        missing = sorted(_REMOVE_NOTES_REQUIRED_FIELDS - keys)
        if missing:
            fields = ", ".join(missing)
            raise ValueError(f"notes[{index}] missing required field(s): {fields}.")

        extra = sorted(keys - _REMOVE_NOTES_ALLOWED_FIELDS)
        if extra:
            fields = ", ".join(extra)
            raise ValueError(f"notes[{index}] has unsupported field(s): {fields}.")

        beat = raw_item["beat"]
        if isinstance(beat, bool) or not isinstance(beat, (int, float)):
            raise TypeError(f"notes[{index}] beat must be a number.")
        if float(beat) < 1.0:
            raise ValueError(
                f"notes[{index}] beat must be at least 1.0, got {beat}."
            )

        return {
            "beat": float(beat),
            "pitch": raw_item.get("pitch"),
            "has_pitch": "pitch" in raw_item and raw_item.get("pitch") is not None,
            "index": index,
        }


    def _validate_remove_notes_same_beat_rules(
        self,
        items: list[dict[str, Any]],
    ) -> None:
        """Reject ambiguous or duplicate same-beat removal items."""
        groups: dict[float, list[dict[str, Any]]] = {}
        for item in items:
            beat_key = self._clean_rhythm_float(float(item["beat"]))
            groups.setdefault(beat_key, []).append(item)

        for beat, group in groups.items():
            has_whole_event = any(not item["has_pitch"] for item in group)
            has_pitch_items = any(item["has_pitch"] for item in group)
            if has_whole_event and has_pitch_items:
                first_index = min(int(item["index"]) for item in group)
                raise ValueError(
                    f"notes[{first_index}] mixes pitchless and pitch-specific "
                    f"removals at beat {beat}; split the request or use only "
                    "one style for that beat."
                )
            if not has_pitch_items:
                if len(group) > 1:
                    first_index = min(int(item["index"]) for item in group)
                    raise ValueError(
                        f"notes[{first_index}] repeats a whole-event removal "
                        f"at beat {beat}."
                    )
                continue

            seen_pitches: set[str] = set()
            for item in group:
                pitch_name = normalize_pitch(item["pitch"]).nameWithOctave
                if pitch_name in seen_pitches:
                    raise ValueError(
                        f"notes[{item['index']}] repeats pitch {pitch_name} "
                        f"at beat {beat}."
                    )
                seen_pitches.add(pitch_name)


    def _group_remove_notes_items(
        self,
        items: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Group removal items by beat while preserving first-seen order."""
        groups: list[list[dict[str, Any]]] = []
        group_by_beat: dict[float, list[dict[str, Any]]] = {}
        for item in items:
            beat_key = self._clean_rhythm_float(float(item["beat"]))
            group = group_by_beat.get(beat_key)
            if group is None:
                group = []
                group_by_beat[beat_key] = group
                groups.append(group)
            group.append(item)
        return groups


    def remove_notes(
        self,
        measure: int,
        part: Union[int, str],
        voice: int,
        notes: list[dict[str, object]],
    ) -> OperationResult:
        """Remove one or more note/chord targets from one scoped voice.
        One call is locked to a single measure, part, and voice. Each item
        specifies a 1-based beat and may optionally specify a pitch. If pitch is
        omitted, the whole note/chord event at that beat is removed. If
        pitch is provided for a chord, only that chord tone is removed; multiple
        pitch-specific items may target the same chord beat. Tuplet events are
        removed only through ``remove_tuplet`` and rests are hidden through
        ``remove_rests``.

        Args:
            measure: 1-based measure number for every removal in this call.
            part: Part identifier for every removal in this call.
            voice: 1-based rhythmic timeline for every removal in this call.
            notes: Non-empty list of removal items. Each item requires ``beat``
                and may include ``pitch``. Items must not include measure, part,
                or voice fields.

        Returns:
            OperationResult with per-target removal summaries and rhythm repair
            diagnostics for the affected voice.
        """
        if measure is None:
            raise ValueError("remove_notes requires a measure.")
        if part is None:
            raise ValueError("remove_notes requires a part.")

        voice = validate_voice_number(voice)
        normalized_items = self._validate_remove_notes_payload(notes)
        grouped_items = self._group_remove_notes_items(normalized_items)

        shadow_state = type(self)(deepcopy(self._score))
        part_obj, part_idx = shadow_state._resolve_part(part)
        measure_obj = shadow_state._resolve_measure(part_obj, measure)
        container = shadow_state._get_voice_or_measure(measure_obj, voice)
        removed: list[dict[str, object]] = []
        refresh_next = False
        spanners_removed = 0
        spanner_anchors_removed = 0
        ties_removed = 0

        for group in grouped_items:
            try:
                group_result = shadow_state._remove_notes_group(
                    part_obj,
                    container,
                    group,
                    measure,
                    part_idx,
                    voice,
                )
            except ValueError as exc:
                raise ValueError(f"notes[{group[0]['index']}]: {exc}") from exc
            except TypeError as exc:
                raise TypeError(f"notes[{group[0]['index']}]: {exc}") from exc
            removed.extend(group_result["removed"])
            refresh_next = refresh_next or bool(group_result["refresh_next"])
            spanners_removed += int(group_result.get("spanners_removed") or 0)
            spanner_anchors_removed += int(
                group_result.get("spanner_anchors_removed") or 0
            )
            ties_removed += int(group_result.get("ties_removed") or 0)

        shadow_state._refresh_measure_beams(measure_obj)
        shadow_state._refresh_changed_measure_accidentals(
            part_obj,
            measure,
            refresh_next=refresh_next,
        )
        auto_completed_rests = shadow_state._normalize_rests_after_rhythm_edit(
            part_obj,
            part_idx,
            measure_obj,
            measure,
            container,
            voice,
        )
        shadow_state._refresh_measure_beams(measure_obj)
        integrity = shadow_state._measure_integrity_for_context(
            part_obj,
            part_idx,
            measure,
            container,
            voice,
        )
        self._commit_shadow_score(shadow_state)

        return OperationResult(
            success=True,
            description=(
                f"Removed {len(removed)} note/chord target(s) from "
                f"measure {measure}, part {part_idx}, voice {voice}"
            ),
            details={
                "measure": measure,
                "part": part_idx,
                "voice": voice,
                "count": len(removed),
                "removed": removed,
                "spanners_removed": spanners_removed,
                "spanner_anchors_removed": spanner_anchors_removed,
                "ties_removed": ties_removed,
                "auto_completed_rests": auto_completed_rests,
                "measure_integrity": integrity,
                "repair_hint": self._measure_repair_hint(integrity),
            },
        )


    def replace_note(
        self,
        measure: int,
        beat: float,
        new_pitch: Optional[PitchInput] = None,
        new_duration: Optional[DurationInput] = None,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
    ) -> OperationResult:
        """Replace one note or chord event in one voice. Use for local pitch or
        duration changes at a known beat/voice. If rhythm changes affect other
        events, rests, or voices, inspect first and use the surgical note,
        rest, chord, remove, and replace tools.

        Args:
            measure: 1-based measure number.
            beat: 1-based beat position.
            new_pitch: Replacement pitch (or *None* to keep existing).
            new_duration: Replacement duration (or *None* to keep existing).
            part: Part identifier.
            voice: 1-based rhythmic timeline inside this part. Defaults to
                voice 1. Inspect first and use another voice only when the
                target belongs to a different simultaneous rhythmic line.

        Returns:
            OperationResult describing the changes made.

        Raises:
            ValueError: If no note is found, or neither pitch nor
                duration is provided.
        """
        if new_pitch is None and new_duration is None:
            raise ValueError(
                "At least one of new_pitch or new_duration must be provided."
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
                    "replace",
                )
            raise ValueError(
                f"No note found at beat {beat} in measure {measure}, "
                f"voice {voice}. Call inspect_score_region for this measure to "
                "find the correct voice/beat."
            )
        if not isinstance(element, (m21note.Note, m21chord.Chord)):
            raise ValueError(
                f"Element at beat {beat} in measure {measure} is a "
                f"{type(element).__name__}, not a note or chord. Use "
                "remove_rests or reshape_rests for visible rest notation."
            )

        changes = []
        replaced_rests: list[dict[str, object]] = []
        duration_changed = False

        if new_pitch is not None:
            if isinstance(element, m21chord.Chord):
                raise ValueError(
                    "Cannot replace pitch of a chord directly. "
                    "Use remove_notes with a specific pitch to remove one "
                    "chord member, or add_chord_tones to add pitches."
                )
            pitch_obj = normalize_pitch(new_pitch)
            if pitch_obj.octave is None and element.pitch.octave is not None:
                pitch_obj.octave = element.pitch.octave
            old_name = element.pitch.nameWithOctave
            element.pitch = pitch_obj
            changes.append(f"pitch {old_name} → {pitch_obj.nameWithOctave}")

        if new_duration is not None:
            if element.duration.tuplets:
                raise ValueError(
                    f"Cannot change duration of a tuplet event at measure "
                    f"{measure}, beat {beat}, voice {voice}. Replace the "
                    "whole tuplet group instead."
                )
            dur_obj = normalize_duration(new_duration)
            ts = self._get_active_time_signature_obj(part_obj, measure)
            capacity = self._effective_measure_capacity(measure_obj, ts)
            self._validate_event_capacity(
                capacity,
                dur_obj.quarterLength,
                measure,
                beat_position=beat,
                ratio_string=ts.ratioString,
            )
            replaced_rests = self._replace_overlapped_rests_or_raise(
                container,
                offset,
                dur_obj.quarterLength,
                measure,
                beat,
                part_idx=part_idx,
                voice=voice,
                exclude=element,
            )
            self._validate_no_rhythm_overlap(
                container,
                offset,
                dur_obj.quarterLength,
                measure,
                beat,
                exclude=element,
                voice=voice,
            )
            old_dur = element.duration.type
            element.duration = dur_obj
            changes.append(f"duration {old_dur} → {dur_obj.type}")
            duration_changed = True

        auto_completed_rests: list[dict[str, object]] = []
        if duration_changed:
            auto_completed_rests = self._normalize_rests_after_rhythm_edit(
                part_obj,
                part_idx,
                measure_obj,
                measure,
                container,
                voice,
            )
        self._refresh_measure_beams(measure_obj)
        self._refresh_changed_measure_accidentals(
            part_obj,
            measure,
            refresh_next=self._element_tie_continues_to_next_measure(element),
        )
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
                f"Replaced note at measure {measure}, beat {beat}: "
                f"{', '.join(changes)}"
            ),
            details={
                "measure": measure,
                "beat": beat,
                "part": part_idx,
                "voice": voice,
                "changes": changes,
                "replaced_rests": replaced_rests,
                "auto_completed_rests": auto_completed_rests,
                "measure_integrity": integrity,
                "repair_hint": self._measure_repair_hint(integrity),
            },
        )


    def get_notes(
        self,
        measure: Optional[int] = None,
        part: Optional[Union[int, str]] = None,
        voice: Optional[int] = None,
    ) -> list[NoteInfo]:
        """Query notes, rests, and chords as simple records. Before editing,
        prefer ``inspect_score_region`` for multi-voice bars because it returns
        exact events grouped by part and voice.

        Args:
            measure: 1-based measure number. *None* → all measures.
            part: Part identifier.  *None* → all parts.
            voice: 1-based voice number.  *None* → all voices.

        Returns:
            List of NoteInfo objects describing each element found,
            including attached tuplet metadata when present.
        """
        if voice is not None:
            voice = validate_voice_number(voice)

        targets = self._resolve_parts_or_all(part)
        results: list[NoteInfo] = []

        for part_obj, part_idx in targets:
            if measure is not None:
                measure_objs = [self._resolve_measure(part_obj, measure)]
            else:
                measure_objs = sorted(
                    part_obj.getElementsByClass(m21stream.Measure),
                    key=lambda m: m.number,
                )

            for m_obj in measure_objs:
                self._collect_notes_from_measure(
                    m_obj, part_idx, voice, results,
                )

        return results


    def _normalize_unique_pitches(
        self,
        pitches: list[PitchInput],
        *,
        context: str,
        minimum_count: int,
    ) -> list[m21pitch.Pitch]:
        """Normalize a pitch list and reject duplicates or too few pitches."""
        if not isinstance(pitches, list) or len(pitches) < minimum_count:
            plural = "pitch" if minimum_count == 1 else "unique pitches"
            raise ValueError(f"{context} requires at least {minimum_count} {plural}.")

        normalized = [normalize_pitch(pitch) for pitch in pitches]
        seen: set[str] = set()
        duplicates: list[str] = []
        for pitch_obj in normalized:
            pitch_name = pitch_obj.nameWithOctave
            if pitch_name in seen and pitch_name not in duplicates:
                duplicates.append(pitch_name)
            seen.add(pitch_name)
        if duplicates:
            raise ValueError(
                f"{context} contains duplicate pitch(es): {duplicates}."
            )
        if len(seen) < minimum_count:
            plural = "pitch" if minimum_count == 1 else "unique pitches"
            raise ValueError(f"{context} requires at least {minimum_count} {plural}.")
        return normalized


    def _remove_notes_group(
        self,
        part_obj: m21stream.Part,
        container: m21stream.Stream,
        group: list[dict[str, Any]],
        measure: int,
        part_idx: int,
        voice: int,
    ) -> dict[str, object]:
        """Remove one beat-group from a shadow score and return summaries."""
        beat = float(group[0]["beat"])
        offset = beat - 1.0
        element = self._find_element_at_offset(container, offset)
        if element is None:
            spanning = self._find_element_spanning_offset(container, offset)
            if spanning is not None:
                if spanning.duration.tuplets:
                    start_beat = self._clean_rhythm_float(
                        container.elementOffset(spanning) + 1.0,
                    )
                    raise ValueError(
                        f"Target at measure {measure}, beat {beat}, voice "
                        f"{voice} is inside a tuplet member that starts at "
                        f"beat {start_beat}. Use remove_tuplet for tuplet "
                        "groups."
                    )
                self._raise_interior_beat_target_error(
                    spanning,
                    container,
                    measure,
                    beat,
                    "remove",
                )
            raise ValueError(
                f"No note, rest, or chord starts at beat {beat} in measure "
                f"{measure}, voice {voice}. If the target is in another layer, "
                "inspect_score_region for this measure and retry with the "
                "correct voice."
            )

        if getattr(element.duration, "isGrace", False):
            raise ValueError(
                f"Target at measure {measure}, beat {beat}, voice {voice} is "
                "a grace note. Use remove_grace_note instead."
            )
        if element.duration.tuplets:
            raise ValueError(
                f"Target at measure {measure}, beat {beat}, voice {voice} is "
                "inside a tuplet. Use remove_tuplet for tuplet groups."
            )
        if isinstance(element, m21note.Rest):
            raise ValueError(
                f"Target at measure {measure}, beat {beat}, voice {voice} is "
                "a rest. Use remove_rests to hide rest notation or "
                "reshape_rests to change visible rest spelling."
            )

        refresh_next = self._element_tie_continues_to_next_measure(element)
        whole_event = not group[0]["has_pitch"]
        if whole_event:
            desc = self._describe_element(element)
            cleared_ties = self._clear_tie_chains_touching_elements(
                part_obj,
                [element],
            )
            spanners_removed, anchors_removed = self._remove_dependent_spanners(
                part_obj,
                [element],
            )
            container.remove(element)
            return {
                "refresh_next": refresh_next,
                "spanners_removed": spanners_removed,
                "spanner_anchors_removed": anchors_removed,
                "ties_removed": len(cleared_ties),
                "removed": [
                    {
                        "beat": beat,
                        "pitch": None,
                        "kind": self._rhythm_event_kind(element),
                        "label": desc,
                        "measure": measure,
                        "part": part_idx,
                        "voice": voice,
                    }
                ],
            }

        pitch_names = [
            normalize_pitch(item["pitch"]).nameWithOctave
            for item in group
        ]
        if isinstance(element, m21note.Rest):
            raise ValueError(
                f"Cannot remove pitch(es) {pitch_names} from a rest at "
                f"measure {measure}, beat {beat}, voice {voice}."
            )
        if isinstance(element, m21note.Note):
            note_pitch = element.pitch.nameWithOctave
            if len(pitch_names) != 1:
                raise ValueError(
                    f"Multiple pitch-specific removals at beat {beat} require "
                    f"a chord, but found single note {note_pitch} in measure "
                    f"{measure}, voice {voice}."
                )
            if pitch_names != [note_pitch]:
                raise ValueError(
                    f"Expected pitch {pitch_names[0]} but found {note_pitch} "
                    f"at beat {beat} in measure {measure}, voice {voice}."
                )
            cleared_ties = self._clear_tie_chains_touching_elements(
                part_obj,
                [element],
            )
            spanners_removed, anchors_removed = self._remove_dependent_spanners(
                part_obj,
                [element],
            )
            container.remove(element)
            return {
                "refresh_next": refresh_next,
                "spanners_removed": spanners_removed,
                "spanner_anchors_removed": anchors_removed,
                "ties_removed": len(cleared_ties),
                "removed": [
                    {
                        "beat": beat,
                        "pitch": note_pitch,
                        "kind": "note",
                        "label": self._describe_element(element),
                        "measure": measure,
                        "part": part_idx,
                        "voice": voice,
                    }
                ],
            }
        if not isinstance(element, m21chord.Chord):
            raise ValueError(
                f"Element at beat {beat} in measure {measure} is a "
                f"{type(element).__name__}, not a note, rest, or chord."
            )

        existing_names = self._element_pitch_names(element)
        missing = [
            pitch_name
            for pitch_name in pitch_names
            if pitch_name not in existing_names
        ]
        if missing:
            raise ValueError(
                f"Pitch(es) {missing} not found in the chord at beat {beat} "
                f"in measure {measure}. Chord contains: {existing_names}"
            )
        remaining = [
            pitch_obj
            for pitch_obj in self._element_pitches(element)
            if pitch_obj.nameWithOctave not in set(pitch_names)
        ]
        if remaining:
            cleared_ties = self._clear_tie_chains_touching_elements(
                part_obj,
                [element],
            )
            self._replace_event_with_pitches(container, element, remaining)
            spanners_removed = 0
            anchors_removed = 0
        else:
            cleared_ties = self._clear_tie_chains_touching_elements(
                part_obj,
                [element],
            )
            spanners_removed, anchors_removed = self._remove_dependent_spanners(
                part_obj,
                [element],
            )
            container.remove(element)

        return {
            "refresh_next": refresh_next,
            "spanners_removed": spanners_removed,
            "spanner_anchors_removed": anchors_removed,
            "ties_removed": len(cleared_ties),
            "removed": [
                {
                    "beat": beat,
                    "pitch": pitch_name,
                    "kind": "chord_tone",
                    "label": f"chord tone {pitch_name}",
                    "measure": measure,
                    "part": part_idx,
                    "voice": voice,
                }
                for pitch_name in pitch_names
            ],
        }


    @staticmethod
    def _element_pitches(
        element: m21note.Note | m21chord.Chord,
    ) -> list[m21pitch.Pitch]:
        """Return a note/chord's pitches as a list."""
        if isinstance(element, m21chord.Chord):
            return list(element.pitches)
        return [element.pitch]


    def _element_pitch_names(
        self,
        element: m21note.Note | m21chord.Chord,
    ) -> list[str]:
        """Return a note/chord's pitch names with octaves."""
        return [pitch_obj.nameWithOctave for pitch_obj in self._element_pitches(element)]


    def _replace_event_with_pitches(
        self,
        container: m21stream.Stream,
        source: m21note.Note | m21chord.Chord,
        pitches: list[m21pitch.Pitch],
    ) -> m21note.Note | m21chord.Chord:
        """Replace a note/chord with a note or chord preserving metadata."""
        if len(pitches) == 1:
            replacement: m21note.Note | m21chord.Chord = m21note.Note(
                pitch=pitches[0],
                duration=deepcopy(source.duration),
            )
        else:
            replacement = m21chord.Chord(
                pitches,
                duration=deepcopy(source.duration),
            )
        self._copy_general_note_metadata(source, replacement)
        container.replace(source, replacement)
        return replacement


    def _copy_general_note_metadata(
        self,
        source: m21note.GeneralNote,
        target: m21note.GeneralNote,
    ) -> None:
        """Copy common note/chord metadata when replacing music21 objects."""
        target.tie = deepcopy(getattr(source, "tie", None))
        for attr_name in (
            "articulations",
            "expressions",
            "lyrics",
            "style",
            "beams",
        ):
            if hasattr(source, attr_name) and hasattr(target, attr_name):
                setattr(target, attr_name, deepcopy(getattr(source, attr_name)))
        for attr_name in (
            "stemDirection",
            "notehead",
            "noteheadFill",
            "noteheadParenthesis",
            "volume",
        ):
            if hasattr(source, attr_name) and hasattr(target, attr_name):
                setattr(target, attr_name, deepcopy(getattr(source, attr_name)))


    def _replace_overlapped_rests_or_raise(
        self,
        container: m21stream.Stream,
        offset: float,
        quarter_length: float,
        measure_number: int,
        beat: float,
        part_idx: int | None = None,
        voice: int | None = None,
        exclude: m21note.GeneralNote | None = None,
    ) -> list[dict[str, object]]:
        """Remove or trim rests overlapped by a proposed sounding event."""
        new_end = offset + quarter_length
        rest_elements: list[tuple[m21note.Rest, bool]] = []
        replaced_rests: list[dict[str, object]] = []
        for element in container.getElementsByClass(m21note.GeneralNote):
            if element is exclude:
                continue
            event_range = self._rhythm_event_range(container, element)
            if event_range is None:
                continue
            if not self._event_ranges_overlap(offset, new_end, event_range):
                continue
            if not isinstance(element, m21note.Rest):
                self._raise_rhythm_overlap_error(
                    offset,
                    new_end,
                    event_range,
                    measure_number,
                    beat,
                    voice,
                )
            hidden = self._is_hidden_rest(element)
            rest_elements.append((element, hidden))
            rest_payload = dict(event_range)
            rest_payload["measure"] = measure_number
            if part_idx is not None:
                rest_payload["part"] = part_idx
            if voice is not None:
                rest_payload["voice"] = voice
            replaced_rests.append(rest_payload)

        for rest, hidden in rest_elements:
            if hidden:
                self._replace_rest_with_outside_segments(
                    container,
                    rest,
                    offset,
                    new_end,
                )
            else:
                container.remove(rest)
        return replaced_rests


    def _validate_no_rhythm_overlap(
        self,
        container: m21stream.Stream,
        offset: float,
        quarter_length: float,
        measure_number: int,
        beat: float,
        exclude: m21note.GeneralNote | None = None,
        voice: int | None = None,
    ) -> None:
        """Raise if a proposed event range overlaps an existing event."""
        new_end = offset + quarter_length
        for event_range in self._occupied_event_ranges(container, exclude=exclude):
            if self._event_ranges_overlap(offset, new_end, event_range):
                self._raise_rhythm_overlap_error(
                    offset,
                    new_end,
                    event_range,
                    measure_number,
                    beat,
                    voice,
                )


    def _raise_rhythm_overlap_error(
        self,
        offset: float,
        new_end: float,
        event_range: dict[str, object],
        measure_number: int,
        beat: float,
        voice: int | None,
    ) -> None:
        """Raise the standard error for a proposed event overlap."""
        existing_end = float(event_range["end"])
        suggestion = self._rhythm_overlap_suggestion(
            offset,
            new_end,
            event_range,
            voice,
        )
        raise ValueError(
            f"Cannot place event at measure {measure_number}, beat "
            f"{beat}: it would overlap existing {event_range['label']} "
            f"from beat {event_range['beat']} to "
            f"{self._clean_rhythm_float(existing_end + 1.0)}. "
            f"{suggestion}"
        )


    @staticmethod
    def _event_ranges_overlap(
        offset: float,
        new_end: float,
        event_range: dict[str, object],
    ) -> bool:
        """Return whether a proposed range overlaps an existing event range."""
        existing_start = float(event_range["offset"])
        existing_end = float(event_range["end"])
        return (
            offset < existing_end - _RHYTHM_EPSILON
            and new_end > existing_start + _RHYTHM_EPSILON
        )


    def _rhythm_overlap_suggestion(
        self,
        offset: float,
        new_end: float,
        event_range: dict[str, object],
        voice: int | None,
    ) -> str:
        """Return a concrete suggestion for resolving one rhythm overlap."""
        alternative_voice = 2 if voice != 2 else 3
        current_voice = (
            f"voice={voice}" if voice is not None else "the current voice"
        )
        same_start = (
            abs(offset - float(event_range["offset"])) <= _RHYTHM_EPSILON
        )
        same_end = (
            abs(new_end - float(event_range["end"])) <= _RHYTHM_EPSILON
        )

        suggestions = [
            f"If this should sound simultaneously in another rhythmic line, "
            f"retry in a different voice, for example voice={alternative_voice} "
            f"instead of {current_voice}.",
        ]
        if same_start and same_end:
            suggestions.append(
                "If the existing event should gain chord tones, use "
                "add_chord_tones. If this is a brand-new same-rhythm chord, "
                "use one add_notes call with all same-beat notes or use "
                "add_chord."
            )
        suggestions.append(
            "If you intended to change the existing event, use replace_note or "
            "remove_notes first. Use add_notes again only after the target "
            "voice has room for the new note."
        )
        return " ".join(suggestions)


    def _check_instrument_range(
        self,
        part_obj: m21stream.Part,
        pitch_obj: m21pitch.Pitch,
    ) -> Optional[str]:
        """Return a warning string if the pitch is outside instrument range."""
        instruments = list(
            part_obj.getElementsByClass(m21instrument.Instrument)
        )
        if instruments:
            inst = instruments[0]
            inst_name = inst.partName or getattr(inst, "instrumentName", "") or ""
            if inst_name:
                return validate_pitch_in_range(pitch_obj, inst_name)
        return None


    def _collect_notes_from_measure(
        self,
        measure_obj: m21stream.Measure,
        part_idx: int,
        voice_filter: Optional[int],
        results: list[NoteInfo],
    ) -> None:
        """Dispatch note collection across voices within a measure."""
        m_num = measure_obj.number
        voices = list(measure_obj.voices)

        if voices:
            voice_ids_seen: set[int] = set()
            for v in voices:
                voice_id = int(v.id) if str(v.id).isdigit() else 1
                voice_ids_seen.add(voice_id)
                if voice_filter is not None and voice_id != voice_filter:
                    continue
                self._collect_notes_from_stream(
                    v, m_num, part_idx, voice_id, results,
                )
            # Direct children not inside any Voice are treated as voice 1.
            if 1 not in voice_ids_seen:
                if voice_filter is None or voice_filter == 1:
                    self._collect_notes_from_stream(
                        measure_obj, m_num, part_idx, 1, results,
                    )
        else:
            if voice_filter is not None and voice_filter != 1:
                return
            self._collect_notes_from_stream(
                measure_obj, m_num, part_idx, 1, results,
            )


    def _extract_tuplet_info(
        self,
        general_note: m21note.GeneralNote,
    ) -> list[TupletInfo]:
        """Return tuplet metadata attached to a note-like event."""
        tuplets: list[TupletInfo] = []
        for tuplet in general_note.duration.tuplets:
            tuplets.append(TupletInfo(
                actual_notes=int(tuplet.numberNotesActual),
                normal_notes=int(tuplet.numberNotesNormal),
            ))
        return tuplets


    def _collect_notes_from_stream(
        self,
        stream_obj: m21stream.Stream,
        measure_number: int,
        part_idx: int,
        voice_id: int,
        results: list[NoteInfo],
    ) -> None:
        """Walk a stream's GeneralNote elements and append NoteInfo records."""
        for el in stream_obj.getElementsByClass(m21note.GeneralNote):
            el_offset = stream_obj.elementOffset(el)
            is_grace = getattr(el.duration, "isGrace", False)
            tuplets = self._extract_tuplet_info(el)

            if isinstance(el, m21note.Rest):
                if (
                    hasattr(el.style, "hideObjectOnPrint")
                    and el.style.hideObjectOnPrint
                ):
                    continue
                results.append(NoteInfo(
                    pitch="rest",
                    octave=0,
                    duration_type=el.duration.type,
                    quarter_length=el.duration.quarterLength,
                    measure_number=measure_number,
                    beat=el_offset + 1.0,
                    part_index=part_idx,
                    voice=voice_id,
                    is_rest=True,
                    dots=el.duration.dots,
                    tuplets=list(tuplets),
                ))
            elif isinstance(el, m21chord.Chord):
                for p in el.pitches:
                    results.append(NoteInfo(
                        pitch=p.nameWithOctave,
                        octave=p.octave if p.octave is not None else 0,
                        duration_type=el.duration.type,
                        quarter_length=el.duration.quarterLength,
                        measure_number=measure_number,
                        beat=el_offset + 1.0,
                        part_index=part_idx,
                        voice=voice_id,
                        is_chord=True,
                        is_tied=el.tie is not None,
                        is_grace=is_grace,
                        dots=el.duration.dots,
                        tuplets=list(tuplets),
                    ))
            elif isinstance(el, m21note.Note):
                results.append(NoteInfo(
                    pitch=el.pitch.nameWithOctave,
                    octave=el.pitch.octave if el.pitch.octave is not None else 0,
                    duration_type=el.duration.type,
                    quarter_length=el.duration.quarterLength,
                    measure_number=measure_number,
                    beat=el_offset + 1.0,
                    part_index=part_idx,
                    voice=voice_id,
                    is_tied=el.tie is not None,
                    is_grace=is_grace,
                    dots=el.duration.dots,
                    tuplets=list(tuplets),
                ))
