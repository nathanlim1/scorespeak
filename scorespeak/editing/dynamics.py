"""Internal expression-editing implementation slice."""

from __future__ import annotations

from .expression_common import *


class DynamicsEditingMixin:
    """Internal mixin for ScoreSpeak expression operations."""

    def _validate_hairpin_target(
        self,
        part_obj: m21stream.Part,
        start_measure: int,
        start_beat: float,
        end_measure: int,
        end_beat: float,
    ) -> tuple[m21stream.Measure, m21stream.Measure]:
        """Resolve hairpin endpoint measures and validate endpoint beats."""
        start_m = self._resolve_measure(part_obj, start_measure)
        end_m = self._resolve_measure(part_obj, end_measure)
        ts_start = self._get_active_time_signature_obj(part_obj, start_measure)
        ts_end = self._get_active_time_signature_obj(part_obj, end_measure)
        _validate_beat_in_measure(start_beat, ts_start, start_measure)
        _validate_beat_in_measure(end_beat, ts_end, end_measure)
        start_offset = start_beat - 1.0
        end_offset = end_beat - 1.0
        start_anchor = _find_general_note_at_offset(
            start_m,
            start_offset,
            include_rests=True,
        )
        if start_anchor is None:
            raise ValueError(
                f"No note, chord, or visible rest found at hairpin start "
                f"beat {start_beat} in measure {start_measure}. Inspect the "
                "score and use the exact beat where the hairpin begins."
            )
        end_anchor = _find_general_note_at_offset(
            end_m,
            end_offset,
            include_rests=True,
        )
        if end_anchor is None:
            raise ValueError(
                f"No note, chord, or visible rest found at hairpin end "
                f"beat {end_beat} in measure {end_measure}. The end beat is "
                "the non-inclusive arrival event, so use the exact beat of "
                "the note/rest/chord the hairpin leads into."
            )
        return start_m, end_m


    def _insert_hairpin_one(
        self,
        type_str: str,
        part_obj: m21stream.Part,
        start_measure: int,
        start_beat: float,
        end_measure: int,
        end_beat: float,
        start_m: m21stream.Measure,
        end_m: m21stream.Measure,
    ) -> None:
        """Insert one already validated hairpin into one part."""
        start_offset = start_beat - 1.0
        end_offset = end_beat - 1.0

        start_anchor = _find_note_at_offset(start_m, start_offset)
        if start_anchor is None:
            start_anchor = _find_general_note_at_offset(
                start_m,
                start_offset,
                include_rests=True,
            )

        end_anchor_measure = end_m
        end_anchor_offset = end_offset
        if abs(end_offset) < 1e-9 and end_measure > start_measure:
            end_anchor_measure = self._resolve_measure(part_obj, end_measure - 1)
            try:
                end_anchor_offset = float(
                    end_anchor_measure.barDuration.quarterLength
                )
            except Exception:
                previous_ts = self._get_active_time_signature_obj(
                    part_obj,
                    end_measure - 1,
                )
                end_anchor_offset = previous_ts.barDuration.quarterLength

        end_anchor = _find_note_ending_at_offset(
            end_anchor_measure,
            end_anchor_offset,
        )
        if end_anchor is start_anchor:
            end_anchor = m21spanner.SpannerAnchor()
            end_anchor_measure.insert(end_anchor_offset, end_anchor)

        if end_anchor is None:
            end_anchor = _find_general_note_at_offset(
                end_m,
                end_offset,
                include_rests=True,
            )
        if end_anchor is None:
            raise ValueError(
                f"No valid hairpin endpoint found at measure {end_measure}, "
                f"beat {end_beat}."
            )

        if type_str == "crescendo":
            hairpin = m21dynamics.Crescendo()
        else:
            hairpin = m21dynamics.Diminuendo()
        hairpin.scorespeak_start_measure = start_measure
        hairpin.scorespeak_start_beat = float(start_beat)
        hairpin.scorespeak_end_measure = end_measure
        hairpin.scorespeak_end_beat = float(end_beat)

        hairpin.addSpannedElements(start_anchor, end_anchor)
        part_obj.insert(0, hairpin)


    def add_dynamic(
        self,
        level: Union[str, DynamicLevel],
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
    ) -> OperationResult:
        """Add dynamic marking(s) at a measure and beat; only one dynamic can
        exist at each resolved position.

        To replace a dynamic, remove the existing dynamic first.

        Args:
            level: Dynamic level such as ``"pp"``, ``"mf"``, ``"ff"``
                or a DynamicLevel enum member.
            measure_number: 1-based measure number.
            beat: 1-based beat position (default 1.0).
            part: Part index or name, or None for ALL parts.

        Returns:
            OperationResult confirming the dynamic was placed.
        """
        level_str = level.value if isinstance(level, DynamicLevel) else str(level).strip().lower()
        if level_str not in VALID_DYNAMICS:
            raise ValueError(
                f"'{level_str}' is not a valid dynamic level. "
                f"Valid dynamics: {', '.join(VALID_DYNAMICS)}"
            )

        offset = beat - 1.0

        if part is not None:
            part_obj, part_idx = self._resolve_part(part)
            measure_obj = self._resolve_measure(part_obj, measure_number)
            ts = self._get_active_time_signature_obj(part_obj, measure_number)
            _validate_beat_in_measure(beat, ts, measure_number)

            existing = _find_dynamic_at_offset(measure_obj, offset)
            if existing is not None:
                raise ValueError(
                    f"A {existing.value} dynamic already exists at measure "
                    f"{measure_number}, beat {beat} in part {part_idx}. "
                    "Remove the existing dynamic first with remove_dynamic "
                    "before adding a new dynamic."
                )

            dyn = m21dynamics.Dynamic(level_str)
            measure_obj.insert(offset, dyn)

            return OperationResult(
                success=True,
                description=(
                    f"Added {level_str} dynamic at measure {measure_number}, "
                    f"beat {beat}"
                ),
                details={
                    "level": level_str,
                    "measure": measure_number,
                    "beat": beat,
                    "part": part_idx,
                },
            )

        targets = self._resolve_parts_or_all(part)
        insertions = []
        skipped_parts = []
        for part_obj, part_idx in targets:
            measure_obj = self._resolve_measure(part_obj, measure_number)
            ts = self._get_active_time_signature_obj(part_obj, measure_number)
            _validate_beat_in_measure(beat, ts, measure_number)

            existing = _find_dynamic_at_offset(measure_obj, offset)
            if existing is not None:
                if existing.value == level_str:
                    skipped_parts.append(part_idx)
                    continue
                raise ValueError(
                    f"A {existing.value} dynamic already exists at measure "
                    f"{measure_number}, beat {beat} in part {part_idx}. "
                    "Remove the existing dynamic first with remove_dynamic "
                    "before adding a new dynamic."
                )
            insertions.append((measure_obj, part_idx))

        for measure_obj, _part_idx in insertions:
            measure_obj.insert(offset, m21dynamics.Dynamic(level_str))

        added_parts = [part_idx for _measure_obj, part_idx in insertions]
        target_parts = [part_idx for _part_obj, part_idx in targets]
        if added_parts:
            description = (
                f"Added {level_str} dynamic at measure {measure_number}, "
                f"beat {beat} in parts {added_parts}"
            )
            if skipped_parts:
                description += f"; already present in parts {skipped_parts}"
        else:
            description = (
                f"{level_str} dynamic already present at measure "
                f"{measure_number}, beat {beat} in parts {skipped_parts}"
            )

        return OperationResult(
            success=True,
            description=description,
            details={
                "level": level_str,
                "measure": measure_number,
                "beat": beat,
                "parts": target_parts,
                "added_parts": added_parts,
                "skipped_parts": skipped_parts,
            },
        )


    def remove_dynamic(
        self,
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
    ) -> OperationResult:
        """Remove dynamic marking(s) at a specific beat position. If part is
        None, remove dynamics from every part that has one there, skip parts
        without one, and fail only if no parts have a dynamic there. Explicit
        part removal remains strict.

        Args:
            measure_number: 1-based measure number.
            beat: 1-based beat position (default 1.0).
            part: Part index or name, or None for all parts.

        Returns:
            OperationResult confirming removal.
        """
        offset = beat - 1.0

        if part is not None:
            part_obj, part_idx = self._resolve_part(part)
            measure_obj = self._resolve_measure(part_obj, measure_number)
            found = _find_dynamic_at_offset(measure_obj, offset)

            if found is None:
                raise ValueError(
                    f"No dynamic found at beat {beat} in "
                    f"measure {measure_number}."
                )

            level_str = found.value
            measure_obj.remove(found)

            return OperationResult(
                success=True,
                description=(
                    f"Removed {level_str} dynamic from measure "
                    f"{measure_number}, beat {beat}"
                ),
                details={
                    "level": level_str,
                    "measure": measure_number,
                    "beat": beat,
                    "part": part_idx,
                },
            )

        targets = self._resolve_parts_or_all(part)
        removals = []
        skipped_parts = []
        levels_by_part = {}
        for part_obj, part_idx in targets:
            measure_obj = self._resolve_measure(part_obj, measure_number)
            found = _find_dynamic_at_offset(measure_obj, offset)
            if found is None:
                skipped_parts.append(part_idx)
                continue
            levels_by_part[part_idx] = found.value
            removals.append((measure_obj, found, part_idx))

        if not removals:
            raise ValueError(
                f"No dynamic found at beat {beat} in measure {measure_number}."
            )

        for measure_obj, found, _part_idx in removals:
            measure_obj.remove(found)

        removed_parts = [part_idx for _measure_obj, _found, part_idx in removals]
        unique_levels = sorted(set(levels_by_part.values()))
        details = {
            "measure": measure_number,
            "beat": beat,
            "parts": [part_idx for _part_obj, part_idx in targets],
            "removed_parts": removed_parts,
            "skipped_parts": skipped_parts,
            "levels_by_part": levels_by_part,
        }
        if len(unique_levels) == 1:
            details["level"] = unique_levels[0]

        return OperationResult(
            success=True,
            description=(
                f"Removed dynamics from measure {measure_number}, "
                f"beat {beat} in parts {removed_parts}"
            ),
            details=details,
        )


    def add_hairpin(
        self,
        hairpin_type: Union[str, HairpinType],
        start_measure: int,
        start_beat: float,
        end_measure: int,
        end_beat: float,
        part: Optional[Union[int, str]] = None,
    ) -> OperationResult:
        """Add a crescendo or diminuendo hairpin over ``[start, end)``: the end beat is the non-inclusive arrival.

        A dynamic or note at the same ``end_measure``/``end_beat`` appears
        after the hairpin.

        Args:
            hairpin_type: ``"crescendo"``, ``"diminuendo"``, or
                ``"decrescendo"`` (alias for diminuendo), or HairpinType enum.
            start_measure: 1-based measure where the hairpin begins.
            start_beat: Beat where the hairpin begins.
            end_measure: 1-based measure containing the hairpin arrival.
            end_beat: Non-inclusive arrival beat where the hairpin stops.
            part: Part index or name, or None for all parts.

        Returns:
            OperationResult confirming the hairpin was placed.
        """
        type_str = hairpin_type.value if isinstance(hairpin_type, HairpinType) else str(hairpin_type).strip().lower()
        if type_str == "decrescendo":
            type_str = "diminuendo"
        if type_str not in ("crescendo", "diminuendo"):
            raise ValueError(
                f"Invalid hairpin type '{type_str}'. "
                f"Must be 'crescendo' or 'diminuendo'."
            )

        start_offset = start_beat - 1.0

        if part is not None:
            part_obj, part_idx = self._resolve_part(part)
            start_m, end_m = self._validate_hairpin_target(
                part_obj,
                start_measure,
                start_beat,
                end_measure,
                end_beat,
            )
            self._insert_hairpin_one(
                type_str,
                part_obj,
                start_measure,
                start_beat,
                end_measure,
                end_beat,
                start_m,
                end_m,
            )

            return OperationResult(
                success=True,
                description=(
                    f"Added {type_str} hairpin from measure {start_measure} "
                    f"beat {start_beat} to measure {end_measure} beat "
                    f"{end_beat}"
                ),
                details={
                    "type": type_str,
                    "start_measure": start_measure,
                    "start_beat": start_beat,
                    "end_measure": end_measure,
                    "end_beat": end_beat,
                    "part": part_idx,
                },
            )

        targets = self._resolve_parts_or_all(part)
        insertions = []
        skipped_parts = []
        for part_obj, part_idx in targets:
            start_m, end_m = self._validate_hairpin_target(
                part_obj,
                start_measure,
                start_beat,
                end_measure,
                end_beat,
            )
            existing = _find_hairpin_starting_at(
                part_obj,
                start_measure,
                start_offset,
            )
            if existing is not None:
                if _hairpin_matches_request(
                    existing,
                    type_str,
                    end_measure,
                    end_beat,
                ):
                    skipped_parts.append(part_idx)
                    continue
                existing_type = _hairpin_type(existing)
                raise ValueError(
                    f"A {existing_type} hairpin already starts at measure "
                    f"{start_measure}, beat {start_beat} in part {part_idx}. "
                    "Remove the existing hairpin first with remove_hairpin "
                    "before adding a new hairpin."
                )
            insertions.append((part_obj, part_idx, start_m, end_m))

        for part_obj, _part_idx, start_m, end_m in insertions:
            self._insert_hairpin_one(
                type_str,
                part_obj,
                start_measure,
                start_beat,
                end_measure,
                end_beat,
                start_m,
                end_m,
            )

        added_parts = [
            part_idx
            for _part_obj, part_idx, _start_m, _end_m in insertions
        ]
        target_parts = [part_idx for _part_obj, part_idx in targets]
        if added_parts:
            description = (
                f"Added {type_str} hairpin from measure {start_measure} "
                f"beat {start_beat} to measure {end_measure} beat "
                f"{end_beat} in parts {added_parts}"
            )
            if skipped_parts:
                description += f"; already present in parts {skipped_parts}"
        else:
            description = (
                f"{type_str.capitalize()} hairpin already present from "
                f"measure {start_measure} beat {start_beat} to measure "
                f"{end_measure} beat {end_beat} in parts {skipped_parts}"
            )

        return OperationResult(
            success=True,
            description=description,
            details={
                "type": type_str,
                "start_measure": start_measure,
                "start_beat": start_beat,
                "end_measure": end_measure,
                "end_beat": end_beat,
                "parts": target_parts,
                "added_parts": added_parts,
                "skipped_parts": skipped_parts,
            },
        )


    def remove_hairpin(
        self,
        start_measure: int,
        start_beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
    ) -> OperationResult:
        """Remove hairpin(s) that start at the given position. If part is
        None, remove hairpins from every part that has one there, skip parts
        without one, and fail only if no parts have a hairpin there. Explicit
        part removal remains strict.

        Args:
            start_measure: 1-based measure where the hairpin begins.
            start_beat: Beat where the hairpin begins (default 1.0).
            part: Part index or name, or None for all parts.

        Returns:
            OperationResult confirming removal.
        """
        start_offset = start_beat - 1.0

        if part is not None:
            part_obj, part_idx = self._resolve_part(part)
            self._resolve_measure(part_obj, start_measure)
            found = _find_hairpin_starting_at(
                part_obj,
                start_measure,
                start_offset,
            )

            if found is None:
                raise ValueError(
                    f"No hairpin found starting at beat {start_beat} "
                    f"in measure {start_measure}."
                )

            type_str = _hairpin_type(found)
            part_obj.remove(found)

            return OperationResult(
                success=True,
                description=(
                    f"Removed {type_str} hairpin starting at "
                    f"measure {start_measure}, beat {start_beat}"
                ),
                details={
                    "type": type_str,
                    "start_measure": start_measure,
                    "start_beat": start_beat,
                    "part": part_idx,
                },
            )

        targets = self._resolve_parts_or_all(part)
        removals = []
        skipped_parts = []
        types_by_part = {}
        for part_obj, part_idx in targets:
            self._resolve_measure(part_obj, start_measure)
            found = _find_hairpin_starting_at(
                part_obj,
                start_measure,
                start_offset,
            )
            if found is None:
                skipped_parts.append(part_idx)
                continue
            type_str = _hairpin_type(found)
            types_by_part[part_idx] = type_str
            removals.append((part_obj, found, part_idx))

        if not removals:
            raise ValueError(
                f"No hairpin found starting at beat {start_beat} "
                f"in measure {start_measure}."
            )

        for part_obj, found, _part_idx in removals:
            part_obj.remove(found)

        removed_parts = [part_idx for _part_obj, _found, part_idx in removals]
        unique_types = sorted(set(types_by_part.values()))
        details = {
            "start_measure": start_measure,
            "start_beat": start_beat,
            "parts": [part_idx for _part_obj, part_idx in targets],
            "removed_parts": removed_parts,
            "skipped_parts": skipped_parts,
            "types_by_part": types_by_part,
        }
        if len(unique_types) == 1:
            details["type"] = unique_types[0]

        return OperationResult(
            success=True,
            description=(
                f"Removed hairpins starting at measure {start_measure}, "
                f"beat {start_beat} in parts {removed_parts}"
            ),
            details=details,
        )
