"""
Measure management operations for ScoreSpeak.

Provides add, insert, delete operations on measures with automatic
state inheritance (implied continuity) and correct offset management.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from music21 import bar as m21bar
from music21 import dynamics as m21dynamics
from music21 import expressions as m21expressions
from music21 import harmony as m21harmony
from music21 import meter as m21meter
from music21 import note as m21note
from music21 import spanner as m21spanner
from music21 import stream as m21stream

from ..types import MeasureInfo, OperationResult
from ..music.validation import validate_voice_number


_REGULAR_BARLINE_TYPES = {"regular", "normal", None}
_RHYTHM_EPSILON = 1e-9


@dataclass
class _MeasureContentSnapshot:
    """Cloned source-measure contents ready to insert into a target measure."""

    source_measure: int
    target_measure: int
    elements: list[tuple[float, Any]] = field(default_factory=list)
    element_map: dict[int, Any] = field(default_factory=dict)
    visible_events: int = 0
    local_markings: int = 0
    anchors: int = 0
    stripped_ties: int = 0


@dataclass
class _CopyPartPlan:
    """Validated measure-copy work for one source/target part pair."""

    source_part: m21stream.Part
    source_part_index: int
    target_part: m21stream.Part
    target_part_index: int
    source_measures: list[m21stream.Measure]
    target_measures: list[m21stream.Measure]


class MeasuresMixin:
    """Mixin providing measure add/delete/insert operations."""

    def add_measures(
        self,
        count: int = 1,
    ) -> OperationResult:
        """Append new empty measures to the end of every part.

        New measures are filled with whole-measure rests and inherit the
        active time signature through implied continuity.

        Args:
            count: Number of measures to add (default 1).

        Returns:
            OperationResult with details of added measures.
        """
        if count < 1:
            raise ValueError(f"Count must be at least 1, got {count}.")

        targets = self._resolve_parts_or_all()
        added = []
        moved_barlines = []

        for part_obj, part_idx in targets:
            existing = list(part_obj.getElementsByClass(m21stream.Measure))
            last_num = existing[-1].number if existing else 0

            if existing:
                active_ts = self._get_active_time_signature_obj(
                    part_obj, last_num
                )
            else:
                active_ts = self._get_default_time_signature()

            bar_ql = active_ts.barDuration.quarterLength

            carried_barline_type = self._detach_terminating_barline(
                existing[-1] if existing else None
            )

            last_new_measure = None
            for i in range(count):
                new_num = last_num + 1 + i
                m = m21stream.Measure(number=new_num)
                rest = m21note.Rest(quarterLength=bar_ql)
                m.append(rest)
                part_obj.append(m)
                added.append({"part": part_idx, "measure": new_num})
                last_new_measure = m

            if carried_barline_type is not None and last_new_measure is not None:
                last_new_measure.rightBarline = m21bar.Barline(
                    type=carried_barline_type
                )
                moved_barlines.append({
                    "part": part_idx,
                    "barline_type": carried_barline_type,
                    "from_measure": last_num,
                    "to_measure": last_new_measure.number,
                })

        description = (
            f"Added {count} measure(s) to all parts"
        )
        details: dict = {
            "measures_added": added,
            "time_signature": active_ts.ratioString,
        }
        if moved_barlines:
            details["barlines_moved"] = moved_barlines
        return OperationResult(
            success=True,
            description=description,
            details=details,
        )

    def insert_measure(
        self,
        before: int,
        count: int = 1,
    ) -> OperationResult:
        """Insert new empty measures before a given measure number in every part.

        Subsequent measures are renumbered. The new measures inherit
        the active time signature at the insertion point.

        Args:
            before: 1-based measure number to insert before.
            count: Number of measures to insert (default 1).

        Returns:
            OperationResult with details of inserted measures.
        """
        if count < 1:
            raise ValueError(f"Count must be at least 1, got {count}.")
        if before < 1:
            raise ValueError(
                f"Measure number must be at least 1, got {before}."
            )

        targets = self._resolve_parts_or_all()
        inserted = []
        moved_barlines = []

        for part_obj, part_idx in targets:
            measures = sorted(
                part_obj.getElementsByClass(m21stream.Measure),
                key=lambda m: m.number,
            )
            max_num = measures[-1].number if measures else 0

            if before > max_num + 1:
                raise ValueError(
                    f"Cannot insert before measure {before}: part {part_idx} "
                    f"only has {max_num} measures. Use add_measures() to "
                    f"append to the end, or insert before measure "
                    f"{max_num + 1}."
                )

            active_ts = self._get_active_time_signature_for_insert(
                part_obj, before, measures
            )
            bar_ql = active_ts.barDuration.quarterLength

            append_beyond_end = measures and before == max_num + 1
            carried_barline_type = None
            if append_beyond_end:
                carried_barline_type = self._detach_terminating_barline(
                    measures[-1]
                )

            for m in measures:
                if m.number >= before:
                    m.number += count

            last_new_measure = None
            for i in range(count):
                new_num = before + i
                m = m21stream.Measure(number=new_num)
                rest = m21note.Rest(quarterLength=bar_ql)
                m.append(rest)
                inserted.append({"part": part_idx, "measure": new_num})

                insert_offset = self._compute_measure_offset(
                    part_obj, new_num, measures, count, bar_ql
                )
                part_obj.insert(insert_offset, m)
                last_new_measure = m

            if carried_barline_type is not None and last_new_measure is not None:
                last_new_measure.rightBarline = m21bar.Barline(
                    type=carried_barline_type
                )
                moved_barlines.append({
                    "part": part_idx,
                    "barline_type": carried_barline_type,
                    "from_measure": max_num,
                    "to_measure": last_new_measure.number,
                })

            self._rebuild_measure_offsets(part_obj)

        details: dict = {
            "measures_inserted": inserted,
            "measures_renumbered_from": before + count,
        }
        if moved_barlines:
            details["barlines_moved"] = moved_barlines
        return OperationResult(
            success=True,
            description=f"Inserted {count} measure(s) before measure {before}",
            details=details,
        )

    def delete_measure(
        self,
        measure_number: int,
    ) -> OperationResult:
        """Delete one measure from the score structure. Use only when the bar
        itself should be removed from every part. To keep the measure and
        replace contents, use note/rest tools.

        Args:
            measure_number: 1-based number of the measure to delete.

        Returns:
            OperationResult with details of the deletion.
        """
        return self.delete_measures(measure_number, measure_number)

    def delete_measures(
        self,
        start: int,
        end: int,
    ) -> OperationResult:
        """Delete a range of measures from the score structure. Use only when
        the bars themselves should be removed from every part. To keep measures
        and replace contents, use note/rest tools.

        Args:
            start: 1-based number of the first measure to delete.
            end: 1-based number of the last measure to delete (inclusive).

        Returns:
            OperationResult with details.
        """
        if start < 1:
            raise ValueError(
                f"Start measure must be at least 1, got {start}."
            )
        if end < start:
            raise ValueError(
                f"End measure ({end}) must be >= start measure ({start})."
            )

        targets = self._resolve_parts_or_all()
        deleted_count = end - start + 1
        deleted_measure_numbers = set(range(start, end + 1))
        deleted = []
        removed_spanners = []
        removed_spanner_anchors = 0
        removed_ties = []
        shifted_spanner_measure_attrs = 0

        for part_obj, part_idx in targets:
            measures = sorted(
                part_obj.getElementsByClass(m21stream.Measure),
                key=lambda m: m.number,
            )
            max_num = measures[-1].number if measures else 0

            if start > max_num:
                raise ValueError(
                    f"Cannot delete measure {start}: part {part_idx} "
                    f"only has {max_num} measures."
                )
            if end > max_num:
                raise ValueError(
                    f"Cannot delete through measure {end}: part {part_idx} "
                    f"only has {max_num} measures."
                )

            to_remove = [m for m in measures if start <= m.number <= end]
            removed_elements = []
            for measure_obj in to_remove:
                removed_elements.extend(self._measure_general_notes(measure_obj))
            part_removed_ties = self._clear_tie_chains_touching_elements(
                part_obj,
                removed_elements,
            )
            for tie_detail in part_removed_ties:
                tie_detail["part"] = part_idx
                tie_detail["reason"] = "measure_deleted"
                removed_ties.append(tie_detail)
            part_removed_spanners, part_removed_anchors = (
                self._remove_musical_spanners_touching_measures(
                    part_obj,
                    deleted_measure_numbers,
                    part_idx,
                    reason="measure_deleted",
                )
            )
            removed_spanners.extend(part_removed_spanners)
            removed_spanner_anchors += part_removed_anchors

            for m in to_remove:
                part_obj.remove(m)
                deleted.append({"part": part_idx, "measure": m.number})

            for m in part_obj.getElementsByClass(m21stream.Measure):
                if m.number > end:
                    m.number -= deleted_count

            shifted_spanner_measure_attrs += (
                self._shift_scorespeak_measure_attrs_after_deletion(
                    part_obj,
                    end,
                    deleted_count,
                )
            )
            self._rebuild_measure_offsets(part_obj)
            if part_removed_ties:
                for measure_obj in self._sorted_part_measures(part_obj):
                    if measure_obj.number is not None:
                        self._refresh_measure_accidentals(
                            part_obj,
                            int(measure_obj.number),
                        )

        details = {"measures_deleted": deleted}
        if removed_spanners:
            details["removed_spanners"] = removed_spanners
            details["removed_spanner_anchors"] = removed_spanner_anchors
        if removed_ties:
            details["removed_ties"] = removed_ties
        if shifted_spanner_measure_attrs:
            details["shifted_spanner_measure_attrs"] = (
                shifted_spanner_measure_attrs
            )

        return OperationResult(
            success=True,
            description=(
                f"Deleted measure(s) {start}–{end}"
                if start != end
                else f"Deleted measure {start}"
            ),
            details=details,
        )

    def clear_measures(
        self,
        start: int,
        end: Optional[int] = None,
        part: Optional[Union[int, str]] = None,
        voice: Optional[int] = None,
        all_voices: bool = False,
    ) -> OperationResult:
        """Clear rhythmic content from existing measures without deleting bars.

        Cleared measures keep structural notation such as time/key/clef/tempo
        changes, barlines, repeats, endings, layout breaks, and measure-level
        text or marks. Notes, rests, chords, voices, lyrics, and note-attached
        notations are removed, then each cleared measure receives a visible
        full-measure rest.

        Args:
            start: 1-based number of the first measure to clear.
            end: 1-based number of the last measure to clear, inclusive. If
                omitted, only ``start`` is cleared.
            part: Optional part index or part name. If omitted, clears the
                measure range in every part.
            voice: Optional 1-based rhythmic voice to clear. Use this for
                multi-voice bars when only one line should be replaced.
            all_voices: Set to True to intentionally clear every voice in a
                multi-voice measure. Multi-voice measures require either
                ``voice`` or ``all_voices=True``.

        Returns:
            OperationResult with details about cleared measures.
        """
        if start < 1:
            raise ValueError(
                f"Start measure must be at least 1, got {start}."
            )

        resolved_end = start if end is None else end
        if resolved_end < start:
            raise ValueError(
                f"End measure ({resolved_end}) must be >= start measure "
                f"({start})."
            )

        if voice is not None and all_voices:
            raise ValueError("Use either voice or all_voices=True, not both.")
        resolved_voice = validate_voice_number(voice) if voice is not None else None

        targets = self._validated_clear_measure_targets(
            start,
            resolved_end,
            part,
        )
        self._validate_clear_measure_voice_scope(
            targets,
            resolved_voice,
            all_voices,
        )

        cleared = []
        for part_obj, part_idx, measures in targets:
            for measure_obj in measures:
                measure_number = int(measure_obj.number)
                if resolved_voice is None:
                    clear_details = self._clear_measure_rhythmic_content(
                        part_obj,
                        measure_obj,
                    )
                    rest_container: m21stream.Stream = measure_obj
                else:
                    (
                        clear_details,
                        rest_container,
                    ) = self._clear_measure_voice_rhythmic_content(
                        part_obj,
                        measure_obj,
                        resolved_voice,
                    )
                rest_length = self._full_measure_rest_quarter_length(
                    part_obj,
                    measure_obj,
                )
                rest = m21note.Rest(quarterLength=rest_length)
                rest_container.insert(0.0, rest)
                if clear_details["ties_removed"]:
                    for refreshed_measure in self._sorted_part_measures(part_obj):
                        if refreshed_measure.number is not None:
                            self._refresh_measure_accidentals(
                                part_obj,
                                int(refreshed_measure.number),
                            )
                clear_details.update({
                    "part": part_idx,
                    "measure": measure_number,
                    "rest_quarter_length": (
                        self._clean_measure_duration(rest_length)
                    ),
                })
                cleared.append(clear_details)

        description = (
            f"Cleared measure {start}"
            if resolved_end == start
            else f"Cleared measures {start}-{resolved_end}"
        )
        if part is not None:
            description += f" in part {targets[0][1]}"
        if resolved_voice is not None:
            description += f", voice {resolved_voice}"

        return OperationResult(
            success=True,
            description=description,
            details={
                "start": start,
                "end": resolved_end,
                "parts": [
                    part_idx
                    for _part_obj, part_idx, _measures in targets
                ],
                "voice": resolved_voice,
                "all_voices": all_voices,
                "measures_cleared": cleared,
            },
        )

    def copy_measure_contents(
        self,
        source_start: int,
        target_start: int,
        count: int = 1,
        source_part: Optional[Union[int, str]] = None,
        target_part: Optional[Union[int, str]] = None,
    ) -> OperationResult:
        """Copy measure contents into target measures, replacing target music.

        The copy preserves the target measures' structural notation such as
        measure numbers, active signatures, barlines, endings, layout breaks,
        pickup padding, and navigation marks. It replaces all musical contents:
        notes, rests, chords, voices, lyrics, note-attached notation, local
        dynamics/text/chord symbols, and fully contained musical spanners of the copied part.

        Args:
            source_start: 1-based first source measure number.
            target_start: 1-based first target measure number.
            count: Number of consecutive measures to copy.
            source_part: Optional source part index or name. If omitted with
                ``target_part``, corresponding measures are copied in all parts.
            target_part: Optional target part index or name. If omitted while
                ``source_part`` is given, the target is the same part.

        Returns:
            OperationResult with copied, skipped, and removed span details.
        """
        shadow_state = type(self)(deepcopy(self._score))
        result = shadow_state._copy_measure_contents_in_place(
            source_start=source_start,
            target_start=target_start,
            count=count,
            source_part=source_part,
            target_part=target_part,
        )
        self._commit_shadow_score(shadow_state)
        return result

    def get_measure_info(
        self,
        measure_number: int,
        part: Optional[Union[int, str]] = None,
    ) -> MeasureInfo:
        """Get structured information about a specific measure.

        Args:
            measure_number: 1-based measure number.
            part: Part index, name, or None (defaults to first part).

        Returns:
            MeasureInfo with time signature, key, notes count, etc.
        """
        part_obj, _ = self._resolve_part(part)
        m = self._resolve_measure(part_obj, measure_number)

        ts = self._get_active_time_signature_obj(part_obj, measure_number)
        ts_str = ts.ratioString

        ks = self._get_active_key_signature_obj(part_obj, measure_number)
        ks_str = self._format_key_signature(ks)

        cl = self._get_active_clef_obj(part_obj, measure_number)
        cl_str = cl.name if cl else "Treble Clef"

        tempo_val = self.get_active_tempo(measure_number, part)

        notes_count = len(m.flatten().notes)
        rests_count = len([
            el for el in m.flatten().notesAndRests
            if isinstance(el, m21note.Rest)
        ])

        beat_count = ts.barDuration.quarterLength if ts else 4.0

        return MeasureInfo(
            number=measure_number,
            time_signature=ts_str,
            key_signature=ks_str,
            clef=cl_str,
            tempo=tempo_val,
            beat_count=beat_count,
            notes_count=notes_count,
            rests_count=rests_count,
        )

    # ------------------------------------------------------------------
    # Internal helpers for measure offset management
    # ------------------------------------------------------------------

    def _copy_measure_contents_in_place(
        self,
        source_start: int,
        target_start: int,
        count: int,
        source_part: Optional[Union[int, str]],
        target_part: Optional[Union[int, str]],
    ) -> OperationResult:
        """Copy measure contents inside this state after all validation."""
        plans = self._validated_copy_measure_plans(
            source_start,
            target_start,
            count,
            source_part,
            target_part,
        )

        copied_measures = []
        copied_spanners = []
        skipped_spanners = []
        removed_target_spanners = []
        part_pairs = []
        source_end = source_start + count - 1
        target_end = target_start + count - 1

        for plan in plans:
            part_pairs.append({
                "source_part": plan.source_part_index,
                "target_part": plan.target_part_index,
            })
            source_measure_numbers = {
                int(measure_obj.number)
                for measure_obj in plan.source_measures
            }
            target_measure_numbers = {
                int(measure_obj.number)
                for measure_obj in plan.target_measures
            }
            source_to_target_measure = {
                int(source_measure.number): int(target_measure.number)
                for source_measure, target_measure in zip(
                    plan.source_measures,
                    plan.target_measures,
                )
            }
            source_spanners, skipped = self._source_spanners_for_copy(
                plan.source_part,
                source_measure_numbers,
                plan.source_part_index,
            )
            skipped_spanners.extend(skipped)
            required_anchor_ids = self._required_anchor_ids(source_spanners)
            boundary_tie_ids = self._boundary_tie_element_ids(
                plan.source_measures,
            )

            snapshots = []
            element_map = {}
            for source_measure, target_measure in zip(
                plan.source_measures,
                plan.target_measures,
            ):
                snapshot = self._snapshot_measure_contents(
                    source_measure,
                    int(target_measure.number),
                    required_anchor_ids,
                    boundary_tie_ids,
                )
                snapshots.append(snapshot)
                element_map.update(snapshot.element_map)

            removed_spanners, target_anchor_count = (
                self._remove_musical_spanners_touching_measures(
                    plan.target_part,
                    target_measure_numbers,
                    plan.target_part_index,
                )
            )
            removed_target_spanners.extend(removed_spanners)

            for target_measure, snapshot in zip(
                plan.target_measures,
                snapshots,
            ):
                clear_details = self._clear_measure_copy_target_content(
                    target_measure,
                )
                for offset, element in snapshot.elements:
                    target_measure.insert(offset, element)

                rest_added = False
                rest_length = None
                if snapshot.visible_events == 0:
                    rest_added = True
                    rest_length = self._full_measure_rest_quarter_length(
                        plan.target_part,
                        target_measure,
                    )
                    rest = m21note.Rest(quarterLength=rest_length)
                    target_measure.insert(0.0, rest)

                target_measure_number = int(target_measure.number)
                self._refresh_copied_measure_notation(
                    plan.target_part,
                    target_measure,
                    target_measure_number,
                )

                measure_details = {
                    "source_part": plan.source_part_index,
                    "source_measure": snapshot.source_measure,
                    "target_part": plan.target_part_index,
                    "target_measure": target_measure_number,
                    "copied_events": snapshot.visible_events,
                    "copied_local_markings": snapshot.local_markings,
                    "copied_spanner_anchors": snapshot.anchors,
                    "stripped_ties": snapshot.stripped_ties,
                    "rest_added": rest_added,
                    "removed_events": clear_details["events_removed"],
                    "removed_voices": clear_details["voices_removed"],
                    "removed_local_markings": clear_details[
                        "local_markings_removed"
                    ],
                    "removed_spanner_anchors": (
                        clear_details["anchors_removed"]
                    ),
                }
                if rest_length is not None:
                    measure_details["rest_quarter_length"] = (
                        self._clean_measure_duration(rest_length)
                    )
                copied_measures.append(measure_details)

            for spanner_obj in source_spanners:
                spanner_clone = self._clone_musical_spanner(
                    spanner_obj,
                    element_map,
                    source_to_target_measure,
                )
                plan.target_part.insert(0, spanner_clone)
                copied_spanners.append(
                    self._spanner_detail(
                        spanner_clone,
                        plan.target_part_index,
                        reason=None,
                    )
                )

            if target_anchor_count:
                part_pairs[-1]["removed_target_spanner_anchors"] = (
                    target_anchor_count
                )

        description = (
            f"Copied measure {source_start} to measure {target_start}"
            if count == 1
            else (
                f"Copied measures {source_start}-{source_end} to "
                f"measures {target_start}-{target_end}"
            )
        )

        return OperationResult(
            success=True,
            description=description,
            details={
                "source_start": source_start,
                "source_end": source_end,
                "target_start": target_start,
                "target_end": target_end,
                "count": count,
                "replaced": True,
                "parts": part_pairs,
                "measures_copied": copied_measures,
                "copied_spanners": copied_spanners,
                "skipped_spanners": skipped_spanners,
                "removed_target_spanners": removed_target_spanners,
            },
        )

    def _validated_copy_measure_plans(
        self,
        source_start: int,
        target_start: int,
        count: int,
        source_part: Optional[Union[int, str]],
        target_part: Optional[Union[int, str]],
    ) -> list[_CopyPartPlan]:
        """Return fully validated copy plans without mutating the score."""
        if source_start < 1:
            raise ValueError(
                f"Source measure must be at least 1, got {source_start}."
            )
        if target_start < 1:
            raise ValueError(
                f"Target measure must be at least 1, got {target_start}."
            )
        if count < 1:
            raise ValueError(f"Count must be at least 1, got {count}.")
        if source_part is None and target_part is not None:
            raise ValueError(
                "target_part cannot be provided without source_part. "
                "Specify both parts for cross-part copy, or omit both to "
                "copy corresponding measures in all parts."
            )

        source_end = source_start + count - 1
        target_end = target_start + count - 1

        if source_part is None and target_part is None:
            source_targets = self._resolve_parts_or_all()
            target_targets = source_targets
        else:
            source_obj, source_idx = self._resolve_part(source_part)
            if target_part is None:
                target_obj, target_idx = source_obj, source_idx
            else:
                target_obj, target_idx = self._resolve_part(target_part)
            source_targets = [(source_obj, source_idx)]
            target_targets = [(target_obj, target_idx)]

        plans = []
        for (source_obj, source_idx), (target_obj, target_idx) in zip(
            source_targets,
            target_targets,
        ):
            source_measures = self._validated_measure_range(
                source_obj,
                source_idx,
                source_start,
                source_end,
                role="source",
            )
            target_measures = self._validated_measure_range(
                target_obj,
                target_idx,
                target_start,
                target_end,
                role="target",
            )
            self._validate_copy_measure_durations(
                source_obj,
                source_measures,
                target_obj,
                target_measures,
                source_idx,
                target_idx,
            )
            plans.append(
                _CopyPartPlan(
                    source_part=source_obj,
                    source_part_index=source_idx,
                    target_part=target_obj,
                    target_part_index=target_idx,
                    source_measures=source_measures,
                    target_measures=target_measures,
                )
            )
        return plans

    def _validated_measure_range(
        self,
        part_obj: m21stream.Part,
        part_idx: int,
        start: int,
        end: int,
        *,
        role: str,
    ) -> list[m21stream.Measure]:
        """Resolve and validate an inclusive measure range in one part."""
        max_num = self._get_measure_count(part_obj)
        if start > max_num:
            raise ValueError(
                f"Cannot copy {role} measure {start}: part {part_idx} "
                f"only has {max_num} measures."
            )
        if end > max_num:
            raise ValueError(
                f"Cannot copy through {role} measure {end}: part {part_idx} "
                f"only has {max_num} measures."
            )
        return [
            self._resolve_measure(part_obj, measure_number)
            for measure_number in range(start, end + 1)
        ]

    def _validate_copy_measure_durations(
        self,
        source_part: m21stream.Part,
        source_measures: list[m21stream.Measure],
        target_part: m21stream.Part,
        target_measures: list[m21stream.Measure],
        source_part_idx: int,
        target_part_idx: int,
    ) -> None:
        """Reject source/target measure pairs with different effective lengths."""
        for source_measure, target_measure in zip(
            source_measures,
            target_measures,
        ):
            source_length = self._full_measure_rest_quarter_length(
                source_part,
                source_measure,
            )
            target_length = self._full_measure_rest_quarter_length(
                target_part,
                target_measure,
            )
            if abs(source_length - target_length) <= _RHYTHM_EPSILON:
                continue
            raise ValueError(
                "Cannot copy measure contents with mismatched effective "
                f"durations: source part {source_part_idx} measure "
                f"{source_measure.number} has "
                f"{self._clean_measure_duration(source_length)} quarter "
                f"beats, but target part {target_part_idx} measure "
                f"{target_measure.number} has "
                f"{self._clean_measure_duration(target_length)}."
            )

    def _source_spanners_for_copy(
        self,
        part_obj: m21stream.Part,
        source_measure_numbers: set[int],
        part_idx: int,
    ) -> tuple[list[m21spanner.Spanner], list[dict[str, object]]]:
        """Split source musical spanners into copyable and skipped groups."""
        copied = []
        skipped = []
        for spanner_obj in part_obj.getElementsByClass(m21spanner.Spanner):
            if not self._is_musical_copy_spanner(spanner_obj):
                continue
            spanner_numbers = self._spanner_copy_measure_numbers(spanner_obj)
            touches_source = any(
                number in source_measure_numbers
                for number in spanner_numbers
            )
            if not touches_source:
                continue
            fully_contained = (
                bool(spanner_numbers)
                and all(
                    number in source_measure_numbers
                    for number in spanner_numbers
                )
            )
            if fully_contained:
                copied.append(spanner_obj)
                continue
            skipped.append(
                self._spanner_detail(
                    spanner_obj,
                    part_idx,
                    reason="crosses_outside_source_range",
                )
            )
        return copied, skipped

    def _required_anchor_ids(
        self,
        spanners: list[m21spanner.Spanner],
    ) -> set[int]:
        """Return IDs of source anchors needed by copied spanners."""
        anchor_ids = set()
        for spanner_obj in spanners:
            for element in spanner_obj.getSpannedElements():
                if isinstance(element, m21spanner.SpannerAnchor):
                    anchor_ids.add(id(element))
        return anchor_ids

    def _snapshot_measure_contents(
        self,
        source_measure: m21stream.Measure,
        target_measure_number: int,
        required_anchor_ids: set[int],
        boundary_tie_ids: set[int],
    ) -> _MeasureContentSnapshot:
        """Clone copyable direct contents from a source measure."""
        snapshot = _MeasureContentSnapshot(
            source_measure=int(source_measure.number),
            target_measure=target_measure_number,
        )
        for element in source_measure:
            if isinstance(element, m21stream.Voice):
                self._snapshot_voice(
                    source_measure,
                    element,
                    snapshot,
                    required_anchor_ids,
                    boundary_tie_ids,
                )
                continue
            if self._is_copyable_local_marking(element):
                offset = float(source_measure.elementOffset(element))
                snapshot.elements.append((offset, deepcopy(element)))
                snapshot.local_markings += 1
                continue
            if isinstance(element, m21note.GeneralNote):
                self._snapshot_general_note(
                    source_measure,
                    element,
                    snapshot,
                    boundary_tie_ids,
                )
                continue
            if (
                isinstance(element, m21spanner.SpannerAnchor)
                and id(element) in required_anchor_ids
            ):
                offset = float(source_measure.elementOffset(element))
                clone = deepcopy(element)
                snapshot.elements.append((offset, clone))
                snapshot.element_map[id(element)] = clone
                snapshot.anchors += 1
        return snapshot

    def _snapshot_voice(
        self,
        source_measure: m21stream.Measure,
        voice_obj: m21stream.Voice,
        snapshot: _MeasureContentSnapshot,
        required_anchor_ids: set[int],
        boundary_tie_ids: set[int],
    ) -> None:
        """Clone a source voice when it contains copyable musical content."""
        if not self._stream_has_copyable_content(voice_obj, required_anchor_ids):
            return

        voice_clone = deepcopy(voice_obj)
        offset = float(source_measure.elementOffset(voice_obj))
        snapshot.elements.append((offset, voice_clone))
        old_elements = self._mappable_stream_elements(voice_obj)
        new_elements = self._mappable_stream_elements(voice_clone)
        for old_element, new_element in zip(old_elements, new_elements):
            snapshot.element_map[id(old_element)] = new_element
            if isinstance(old_element, m21note.GeneralNote):
                snapshot.visible_events += 1
                if id(old_element) in boundary_tie_ids:
                    snapshot.stripped_ties += self._strip_tie(new_element)
            elif isinstance(old_element, m21spanner.SpannerAnchor):
                snapshot.anchors += 1

    def _snapshot_general_note(
        self,
        source_measure: m21stream.Measure,
        element: m21note.GeneralNote,
        snapshot: _MeasureContentSnapshot,
        boundary_tie_ids: set[int],
    ) -> None:
        """Clone one direct note-like source element."""
        offset = float(source_measure.elementOffset(element))
        clone = deepcopy(element)
        snapshot.elements.append((offset, clone))
        snapshot.element_map[id(element)] = clone
        snapshot.visible_events += 1
        if id(element) in boundary_tie_ids:
            snapshot.stripped_ties += self._strip_tie(clone)

    def _remove_musical_spanners_touching_measures(
        self,
        part_obj: m21stream.Part,
        measure_numbers: set[int],
        part_idx: int,
        reason: str = "target_contents_replaced",
    ) -> tuple[list[dict[str, object]], int]:
        """Remove musical spanners that touch a set of measures."""
        spanners_to_remove = []
        for spanner_obj in list(part_obj.getElementsByClass(m21spanner.Spanner)):
            if not self._is_musical_copy_spanner(spanner_obj):
                continue
            if any(
                number in measure_numbers
                for number in self._spanner_copy_measure_numbers(spanner_obj)
            ):
                spanners_to_remove.append(spanner_obj)

        details = []
        anchors_removed = 0
        removed_anchor_ids = set()
        for spanner_obj in spanners_to_remove:
            details.append(
                self._spanner_detail(
                    spanner_obj,
                    part_idx,
                    reason=reason,
                )
            )
            for element in spanner_obj.getSpannedElements():
                if not isinstance(element, m21spanner.SpannerAnchor):
                    continue
                anchor_id = id(element)
                if anchor_id in removed_anchor_ids:
                    continue
                removed_anchor_ids.add(anchor_id)
                active_site = getattr(element, "activeSite", None)
                if active_site is None:
                    continue
                active_site.remove(element)
                anchors_removed += 1
            part_obj.remove(spanner_obj)
        return details, anchors_removed

    def _clear_measure_copy_target_content(
        self,
        measure_obj: m21stream.Measure,
    ) -> dict[str, int]:
        """Remove target musical contents while preserving structure."""
        removed_elements = [
            element
            for element in self._measure_general_notes(measure_obj)
            if not self._is_copyable_local_marking(element)
        ]
        visible_events_removed = sum(
            1
            for element in removed_elements
            if isinstance(element, m21note.GeneralNote)
        )
        voices_removed = len(list(measure_obj.voices))
        local_markings_removed = 0
        anchors_removed = 0

        for voice_obj in list(measure_obj.voices):
            measure_obj.remove(voice_obj)

        for element in list(measure_obj):
            if self._is_copyable_local_marking(element):
                measure_obj.remove(element)
                local_markings_removed += 1
                continue
            if isinstance(element, m21note.GeneralNote):
                measure_obj.remove(element)
                continue
            if isinstance(element, m21spanner.SpannerAnchor):
                measure_obj.remove(element)
                anchors_removed += 1

        return {
            "events_removed": visible_events_removed,
            "voices_removed": voices_removed,
            "local_markings_removed": local_markings_removed,
            "anchors_removed": anchors_removed,
        }

    def _clone_musical_spanner(
        self,
        spanner_obj: m21spanner.Spanner,
        element_map: dict[int, Any],
        source_to_target_measure: dict[int, int],
    ) -> m21spanner.Spanner:
        """Clone one fully contained source spanner with remapped endpoints."""
        mapped_elements = []
        for old_element in spanner_obj.getSpannedElements():
            new_element = element_map.get(id(old_element))
            if new_element is None:
                raise ValueError(
                    "Cannot copy spanner because one of its endpoints was "
                    "not copied into the target measures."
                )
            mapped_elements.append(new_element)

        spanner_clone = deepcopy(spanner_obj)
        spanner_clone.spannerStorage.clear()
        spanner_clone.addSpannedElements(*mapped_elements)
        self._remap_scorespeak_measure_attrs(
            spanner_clone,
            source_to_target_measure,
        )
        return spanner_clone

    def _remap_scorespeak_measure_attrs(
        self,
        spanner_obj: m21spanner.Spanner,
        source_to_target_measure: dict[int, int],
    ) -> None:
        """Remap custom logical measure attrs on a copied spanner in place."""
        for attr_name, measure_number in self._scorespeak_measure_attrs(
            spanner_obj,
        ).items():
            if measure_number not in source_to_target_measure:
                continue
            setattr(
                spanner_obj,
                attr_name,
                source_to_target_measure[measure_number],
            )

    def _shift_scorespeak_measure_attrs_after_deletion(
        self,
        part_obj: m21stream.Part,
        end: int,
        deleted_count: int,
    ) -> int:
        """Shift custom spanner measure attrs after a deleted range."""
        shifted = 0
        for spanner_obj in list(part_obj.getElementsByClass(m21spanner.Spanner)):
            if not self._is_musical_copy_spanner(spanner_obj):
                continue
            for attr_name, measure_number in self._scorespeak_measure_attrs(
                spanner_obj,
            ).items():
                if measure_number <= end:
                    continue
                setattr(
                    spanner_obj,
                    attr_name,
                    measure_number - deleted_count,
                )
                shifted += 1
        return shifted

    def _refresh_copied_measure_notation(
        self,
        part_obj: m21stream.Part,
        measure_obj: m21stream.Measure,
        measure_number: int,
    ) -> None:
        """Refresh derived notation after copied contents are inserted."""
        self._refresh_measure_beams(measure_obj)
        self._refresh_measure_stems(measure_obj)
        self._refresh_measure_and_next_accidentals(part_obj, measure_number)

    def _is_copyable_local_marking(self, element: object) -> bool:
        """Return whether a direct measure element is copied as local music."""
        if isinstance(element, m21dynamics.Dynamic):
            return True
        if isinstance(element, m21harmony.ChordSymbol):
            return True
        return (
            isinstance(element, m21expressions.TextExpression)
            and not isinstance(element, m21expressions.RehearsalMark)
        )

    def _is_musical_copy_spanner(
        self,
        spanner_obj: m21spanner.Spanner,
    ) -> bool:
        """Return whether a spanner belongs to copied musical contents."""
        for element in spanner_obj.getSpannedElements():
            if self._is_copyable_local_marking(element):
                continue
            if isinstance(
                element,
                (m21note.GeneralNote, m21spanner.SpannerAnchor),
            ):
                return True
        return False

    def _spanner_measure_numbers(
        self,
        spanner_obj: m21spanner.Spanner,
    ) -> list[int | None]:
        """Return measure numbers for a spanner's endpoints."""
        return [
            self._element_measure_number(element)
            for element in spanner_obj.getSpannedElements()
        ]

    def _spanner_copy_measure_numbers(
        self,
        spanner_obj: m21spanner.Spanner,
    ) -> list[int]:
        """Return physical and logical measures relevant to copy semantics."""
        measures = [
            number
            for number in self._spanner_measure_numbers(spanner_obj)
            if number is not None
        ]
        measures.extend(self._scorespeak_measure_attrs(spanner_obj).values())
        return list(dict.fromkeys(measures))

    def _scorespeak_measure_attrs(
        self,
        spanner_obj: m21spanner.Spanner,
    ) -> dict[str, int]:
        """Return custom ``scorespeak_*_measure`` attrs with int values."""
        attrs: dict[str, int] = {}
        for attr_name, value in vars(spanner_obj).items():
            if not self._is_scorespeak_measure_attr_name(attr_name):
                continue
            measure_number = self._coerce_scorespeak_measure_number(value)
            if measure_number is None:
                continue
            attrs[attr_name] = measure_number
        return attrs

    @staticmethod
    def _is_scorespeak_measure_attr_name(attr_name: str) -> bool:
        """Return whether an attribute stores ScoreSpeak measure metadata."""
        return attr_name.startswith("scorespeak_") and attr_name.endswith(
            "_measure"
        )

    @staticmethod
    def _coerce_scorespeak_measure_number(value: object) -> int | None:
        """Return a positive integer measure number from an int-like value."""
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float) and value.is_integer():
            number = int(value)
            return number if number > 0 else None
        if isinstance(value, str):
            try:
                number = int(value.strip())
            except ValueError:
                return None
            return number if number > 0 else None
        return None

    def _spanner_detail(
        self,
        spanner_obj: m21spanner.Spanner,
        part_idx: int,
        reason: str | None,
    ) -> dict[str, object]:
        """Return a compact payload describing a copied/skipped/removed spanner."""
        measures = [
            number
            for number in self._spanner_copy_measure_numbers(spanner_obj)
        ]
        detail = {
            "part": part_idx,
            "type": type(spanner_obj).__name__,
            "measures": list(dict.fromkeys(measures)),
        }
        if reason is not None:
            detail["reason"] = reason
        return detail

    def _element_measure_number(self, element: object) -> int | None:
        """Return the containing measure number for an element, if known."""
        measure_obj = None
        if hasattr(element, "getContextByClass"):
            measure_obj = element.getContextByClass(m21stream.Measure)
        if not isinstance(measure_obj, m21stream.Measure):
            return None
        if measure_obj.number is None:
            return None
        return int(measure_obj.number)

    def _element_offset_in_measure(self, element: object) -> float:
        """Return an element offset relative to its containing measure."""
        measure_obj = None
        if hasattr(element, "getContextByClass"):
            measure_obj = element.getContextByClass(m21stream.Measure)
        if isinstance(measure_obj, m21stream.Measure):
            try:
                return float(element.getOffsetInHierarchy(measure_obj))
            except Exception:
                pass

        active_site = getattr(element, "activeSite", None)
        if active_site is not None:
            try:
                return float(active_site.elementOffset(element))
            except Exception:
                pass
        return 0.0

    def _boundary_tie_element_ids(
        self,
        source_measures: list[m21stream.Measure],
    ) -> set[int]:
        """Return tied source elements whose matching tie endpoint is outside."""
        tied_events = []
        for measure_obj in source_measures:
            measure_number = int(measure_obj.number)
            for element in measure_obj.recurse().getElementsByClass(
                m21note.NotRest,
            ):
                tie_obj = getattr(element, "tie", None)
                if tie_obj is None:
                    continue
                tied_events.append({
                    "id": id(element),
                    "measure": measure_number,
                    "offset": self._element_offset_in_measure(element),
                    "tie_type": tie_obj.type,
                    "pitches": self._element_pitch_signature(element),
                })

        boundary_ids = set()
        for event in tied_events:
            tie_type = event["tie_type"]
            needs_previous = tie_type in {"stop", "continue"}
            needs_next = tie_type in {"start", "continue"}
            has_previous = self._has_tie_neighbor(
                event,
                tied_events,
                before=True,
            )
            has_next = self._has_tie_neighbor(
                event,
                tied_events,
                before=False,
            )
            if (needs_previous and not has_previous) or (
                needs_next and not has_next
            ):
                boundary_ids.add(event["id"])
        return boundary_ids

    def _has_tie_neighbor(
        self,
        event: dict[str, object],
        tied_events: list[dict[str, object]],
        *,
        before: bool,
    ) -> bool:
        """Return whether a tied event has a matching neighbor in range."""
        valid_neighbor_types = (
            {"start", "continue"} if before else {"stop", "continue"}
        )
        for candidate in tied_events:
            if candidate["id"] == event["id"]:
                continue
            if candidate["pitches"] != event["pitches"]:
                continue
            if candidate["tie_type"] not in valid_neighbor_types:
                continue
            candidate_position = (
                candidate["measure"],
                candidate["offset"],
            )
            event_position = (event["measure"], event["offset"])
            if before and candidate_position < event_position:
                return True
            if not before and candidate_position > event_position:
                return True
        return False

    def _element_pitch_signature(
        self,
        element: object,
    ) -> tuple[str, ...]:
        """Return a stable pitch signature for a note or chord-like element."""
        pitches = getattr(element, "pitches", None)
        if pitches is not None:
            return tuple(
                sorted(pitch_obj.nameWithOctave for pitch_obj in pitches)
            )
        pitch_obj = getattr(element, "pitch", None)
        if pitch_obj is not None:
            return (pitch_obj.nameWithOctave,)
        return ()

    def _strip_tie(self, element: object) -> int:
        """Remove a tie from a cloned note/chord and return a removal count."""
        if getattr(element, "tie", None) is None:
            return 0
        element.tie = None
        return 1

    def _stream_has_copyable_content(
        self,
        stream_obj: m21stream.Stream,
        required_anchor_ids: set[int],
    ) -> bool:
        """Return whether a stream contains content worth copying."""
        for element in stream_obj.recurse():
            if isinstance(element, m21note.GeneralNote):
                return True
            if (
                isinstance(element, m21spanner.SpannerAnchor)
                and id(element) in required_anchor_ids
            ):
                return True
        return False

    def _mappable_stream_elements(
        self,
        stream_obj: m21stream.Stream,
    ) -> list[object]:
        """Return copied-stream elements that can anchor ties or spanners."""
        elements = []
        for element in stream_obj.recurse():
            if isinstance(element, m21note.GeneralNote):
                elements.append(element)
                continue
            if isinstance(element, m21spanner.SpannerAnchor):
                elements.append(element)
        return elements

    def _validated_clear_measure_targets(
        self,
        start: int,
        end: int,
        part: Optional[Union[int, str]],
    ) -> list[tuple[m21stream.Part, int, list[m21stream.Measure]]]:
        """Resolve and validate every measure before clearing anything."""
        targets = self._resolve_parts_or_all(part)
        validated = []
        for part_obj, part_idx in targets:
            measures = []
            max_num = self._get_measure_count(part_obj)
            if start > max_num:
                raise ValueError(
                    f"Cannot clear measure {start}: part {part_idx} "
                    f"only has {max_num} measures."
                )
            if end > max_num:
                raise ValueError(
                    f"Cannot clear through measure {end}: part {part_idx} "
                    f"only has {max_num} measures."
                )
            for measure_number in range(start, end + 1):
                measures.append(self._resolve_measure(part_obj, measure_number))
            validated.append((part_obj, part_idx, measures))
        return validated

    def _validate_clear_measure_voice_scope(
        self,
        targets: list[tuple[m21stream.Part, int, list[m21stream.Measure]]],
        voice: Optional[int],
        all_voices: bool,
    ) -> None:
        """Reject broad clears of multi-voice measures unless explicitly scoped."""
        if voice is not None or all_voices:
            return

        for _part_obj, part_idx, measures in targets:
            for measure_obj in measures:
                if not self._measure_has_multiple_rhythmic_voices(measure_obj):
                    continue
                raise ValueError(
                    f"Measure {measure_obj.number} in part {part_idx} has "
                    "multiple rhythmic voices. Provide voice=<number> to "
                    "clear one line, or all_voices=True to intentionally "
                    "clear every voice."
                )

    def _measure_has_multiple_rhythmic_voices(
        self,
        measure_obj: m21stream.Measure,
    ) -> bool:
        """Return whether a measure has more than one rhythmic voice stream."""
        rhythmic_voice_count = 0
        if list(measure_obj.getElementsByClass(m21note.GeneralNote)):
            rhythmic_voice_count += 1

        for voice_obj in measure_obj.voices:
            if self._stream_general_notes(voice_obj):
                rhythmic_voice_count += 1
            if rhythmic_voice_count > 1:
                return True
        return False

    def _clear_measure_rhythmic_content(
        self,
        part_obj: m21stream.Part,
        measure_obj: m21stream.Measure,
    ) -> dict[str, int]:
        """Remove note-like content and dependent note-anchored spanners."""
        removed_elements = self._measure_general_notes(measure_obj)
        removed_ties = self._clear_tie_chains_touching_elements(
            part_obj,
            removed_elements,
        )
        spanners_removed, anchors_removed = self._remove_dependent_spanners(
            part_obj,
            removed_elements,
        )

        voices_removed = len(list(measure_obj.voices))
        for voice_obj in list(measure_obj.voices):
            measure_obj.remove(voice_obj)

        for element in list(measure_obj.getElementsByClass(m21note.GeneralNote)):
            measure_obj.remove(element)

        visible_events_removed = sum(
            1
            for element in removed_elements
            if isinstance(element, m21note.GeneralNote)
        )

        return {
            "events_removed": visible_events_removed,
            "voices_removed": voices_removed,
            "spanners_removed": spanners_removed,
            "anchors_removed": anchors_removed,
            "ties_removed": len(removed_ties),
        }

    def _clear_measure_voice_rhythmic_content(
        self,
        part_obj: m21stream.Part,
        measure_obj: m21stream.Measure,
        voice: int,
    ) -> tuple[dict[str, int], m21stream.Stream]:
        """Remove note-like content from one voice and return its container."""
        container = self._resolve_clear_voice_container(measure_obj, voice)
        removed_elements = self._stream_general_notes(container)
        removed_ties = self._clear_tie_chains_touching_elements(
            part_obj,
            removed_elements,
        )
        spanners_removed, anchors_removed = self._remove_dependent_spanners(
            part_obj,
            removed_elements,
        )

        for element in list(container.getElementsByClass(m21note.GeneralNote)):
            container.remove(element)

        for anchor in list(container.getElementsByClass(m21spanner.SpannerAnchor)):
            container.remove(anchor)
            anchors_removed += 1

        visible_events_removed = sum(
            1
            for element in removed_elements
            if isinstance(element, m21note.GeneralNote)
        )

        return {
            "events_removed": visible_events_removed,
            "voices_removed": 0,
            "spanners_removed": spanners_removed,
            "anchors_removed": anchors_removed,
            "ties_removed": len(removed_ties),
            "voice": voice,
        }, container

    def _resolve_clear_voice_container(
        self,
        measure_obj: m21stream.Measure,
        voice: int,
    ) -> m21stream.Stream:
        """Return the stream that should receive a single-voice clear."""
        voices = list(measure_obj.voices)
        if not voices:
            if voice == 1:
                return measure_obj
            raise ValueError(
                f"Voice {voice} not found in measure {measure_obj.number}. "
                "Only voice 1 exists in this single-voice measure."
            )

        for voice_obj in voices:
            if str(voice_obj.id) == str(voice):
                return voice_obj

        voice_ids = [voice_obj.id for voice_obj in voices]
        raise ValueError(
            f"Voice {voice} not found in measure {measure_obj.number}. "
            f"Available voices: {voice_ids}."
        )

    def _measure_general_notes(
        self,
        measure_obj: m21stream.Measure,
    ) -> list[m21note.GeneralNote]:
        """Return all note/rest/chord-like elements in a measure."""
        return self._stream_general_notes(measure_obj)

    def _stream_general_notes(
        self,
        stream_obj: m21stream.Stream,
    ) -> list[m21note.GeneralNote]:
        """Return all note/rest/chord-like elements in a stream."""
        elements = []
        seen_ids = set()
        for candidate_stream in (stream_obj, stream_obj.recurse()):
            for element in candidate_stream.getElementsByClass(m21note.GeneralNote):
                element_id = id(element)
                if element_id in seen_ids:
                    continue
                seen_ids.add(element_id)
                elements.append(element)
        return elements

    def _remove_dependent_spanners(
        self,
        part_obj: m21stream.Part,
        removed_elements: list[m21note.GeneralNote],
    ) -> tuple[int, int]:
        """Remove spanners that point at note-like elements being deleted."""
        removed_element_ids = {id(element) for element in removed_elements}
        spanners_to_remove = []
        anchors_to_remove = []

        for spanner_obj in list(part_obj.getElementsByClass(m21spanner.Spanner)):
            spanned_elements = list(spanner_obj.getSpannedElements())
            touches_removed_element = any(
                id(element) in removed_element_ids
                for element in spanned_elements
            )
            if not touches_removed_element:
                continue
            spanners_to_remove.append(spanner_obj)
            for element in spanned_elements:
                if isinstance(element, m21spanner.SpannerAnchor):
                    anchors_to_remove.append(element)

        for spanner_obj in spanners_to_remove:
            part_obj.remove(spanner_obj)

        removed_anchor_ids = set()
        anchors_removed = 0
        for anchor in anchors_to_remove:
            anchor_id = id(anchor)
            if anchor_id in removed_anchor_ids:
                continue
            removed_anchor_ids.add(anchor_id)
            active_site = getattr(anchor, "activeSite", None)
            if active_site is None:
                continue
            active_site.remove(anchor)
            anchors_removed += 1

        return len(spanners_to_remove), anchors_removed

    def _full_measure_rest_quarter_length(
        self,
        part_obj: m21stream.Part,
        measure_obj: m21stream.Measure,
    ) -> float:
        """Return the visible-bar capacity for a full-measure rest."""
        time_signature = self._get_active_time_signature_obj(
            part_obj,
            int(measure_obj.number),
        )
        bar_length = float(time_signature.barDuration.quarterLength)
        padding_left = float(getattr(measure_obj, "paddingLeft", 0.0) or 0.0)
        return max(0.0, bar_length - padding_left)

    @staticmethod
    def _clean_measure_duration(value: float) -> float:
        """Round duration floats for stable result payloads."""
        rounded = round(value, 9)
        if abs(rounded - round(rounded)) <= 1e-9:
            return float(round(rounded))
        return rounded

    def _detach_terminating_barline(
        self,
        measure: Optional[m21stream.Measure],
    ) -> Optional[str]:
        """Detach and return a measure's right barline if it terminates a piece.

        A "terminating" barline is any right barline that is not a plain
        regular bar and is not a repeat (repeats are scoped to their
        section and should stay put). When the last measure of the piece
        carries such a barline (commonly a ``final`` or ``double`` bar),
        we want that marker to follow the real end of the piece when new
        measures are appended.

        Args:
            measure: The measure whose right barline to inspect. If None
                or if the barline is regular/absent/a repeat, no change
                is made and None is returned.

        Returns:
            The barline ``type`` string if a terminating barline was
            detached and reset to a regular barline, otherwise None.
        """
        if measure is None:
            return None

        right = measure.rightBarline
        if right is None:
            return None
        if isinstance(right, m21bar.Repeat):
            return None

        bar_type = getattr(right, "type", None)
        if bar_type in _REGULAR_BARLINE_TYPES:
            return None

        measure.rightBarline = m21bar.Barline(type="regular")
        return bar_type

    def _rebuild_measure_offsets(self, part_obj: m21stream.Part) -> None:
        """Recalculate all measure offsets to be sequential."""
        measures = sorted(
            part_obj.getElementsByClass(m21stream.Measure),
            key=lambda m: m.number,
        )
        offset = 0.0
        for m in measures:
            current = part_obj.elementOffset(m)
            if abs(current - offset) > 1e-9:
                part_obj.setElementOffset(m, offset)
            ts = m.getContextByClass(m21meter.TimeSignature)
            if ts is not None:
                offset += ts.barDuration.quarterLength
            else:
                offset += 4.0

    def _get_active_time_signature_for_insert(
        self,
        part_obj: m21stream.Part,
        before: int,
        measures: list,
    ) -> m21meter.TimeSignature:
        """Get the active time signature just before an insertion point."""
        if before == 1:
            return self._get_default_time_signature()

        prev_measures = [m for m in measures if m.number < before]
        if prev_measures:
            last_prev = max(prev_measures, key=lambda m: m.number)
            ts = last_prev.getContextByClass(m21meter.TimeSignature)
            if ts is not None:
                return ts

        return self._get_default_time_signature()

    def _compute_measure_offset(
        self,
        part_obj: m21stream.Part,
        measure_number: int,
        existing_measures: list,
        insert_count: int,
        bar_ql: float,
    ) -> float:
        """Compute the offset where a new measure should be inserted."""
        prev_measures = [
            m for m in part_obj.getElementsByClass(m21stream.Measure)
            if m.number < measure_number
        ]
        if not prev_measures:
            return 0.0

        last_prev = max(prev_measures, key=lambda m: m.number)
        last_offset = part_obj.elementOffset(last_prev)
        ts = last_prev.getContextByClass(m21meter.TimeSignature)
        prev_ql = ts.barDuration.quarterLength if ts else 4.0
        return last_offset + prev_ql

    def _get_default_time_signature(self) -> m21meter.TimeSignature:
        """Get the score-level default time signature."""
        for part_obj in self._score.parts:
            measures = list(part_obj.getElementsByClass(m21stream.Measure))
            if measures:
                ts = measures[0].getContextByClass(m21meter.TimeSignature)
                if ts is not None:
                    return ts
        return m21meter.TimeSignature("4/4")
