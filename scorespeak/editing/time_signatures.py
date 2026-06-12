"""Internal signature-editing implementation slice."""

from __future__ import annotations

from .signature_common import *


class TimeSignatureEditingMixin:
    """Internal mixin for ScoreSpeak signature operations."""

    def set_time_signature(
        self,
        time_signature: str,
        measure_number: int,
    ) -> OperationResult:
        """Change the time signature at a specific measure.

        The new time signature takes effect at the given measure and
        propagates forward (implied continuity) until another time
        signature is encountered. If the requested value matches the
        inherited value from before the measure, any local change at the
        measure is removed so the timeline inherits cleanly.

        Args:
            time_signature: Time signature string, e.g. "4/4", "3/4", "6/8".
            measure_number: 1-based measure number where the change occurs.

        Returns:
            OperationResult describing the outcome.

        Raises:
            ValueError: If the time signature string is invalid, the measure
                does not exist, or existing notes do not fit in the new meter.
        """
        try:
            new_ts = m21meter.TimeSignature(time_signature)
        except Exception as exc:
            raise ValueError(
                f"'{time_signature}' is not a valid time signature. "
                f"Expected a format like '4/4', '3/4', or '6/8'."
            ) from exc

        targets = self._resolve_parts_or_all()
        edit_plans = self._time_signature_edit_plans(
            targets,
            new_ts,
            measure_number,
        )
        validation_errors = self._time_signature_region_validation_errors(
            edit_plans,
            new_ts,
        )
        if validation_errors:
            raise ValueError(
                self._format_time_signature_region_validation_error(
                    time_signature,
                    measure_number,
                    validation_errors,
                )
            )

        changed_parts: list[int] = []
        part_actions: list[dict[str, object]] = []
        total_validated_measures = 0
        total_normalized_measures = 0
        total_removed_overflow_rests = 0
        total_auto_completed_rests = 0

        for edit_plan in edit_plans:
            part_obj = edit_plan["part_obj"]
            part_idx = int(edit_plan["part"])
            m = edit_plan["measure"]
            active_before = edit_plan["active_before"]
            inherited_before = edit_plan["inherited_before"]
            effective_change = bool(edit_plan["effective_change"])
            pre_removed = self._canonicalize_time_signatures_from(
                part_obj,
                measure_number,
            )

            edit_changed = False
            action = str(edit_plan["action"])
            if effective_change:
                self._remove_local_time_signatures(m)
                if (
                    inherited_before is not None
                    and self._time_signatures_equal(new_ts, inherited_before)
                ):
                    action = "removed_local_change"
                else:
                    m.timeSignature = self._copy_time_signature(new_ts)
                    action = "set"
                edit_changed = True
            elif pre_removed and action == "already_active":
                action = "canonicalized"

            post_removed = self._canonicalize_time_signatures_from(
                part_obj,
                measure_number,
            )
            normalization = self._empty_time_signature_normalization_summary()
            if edit_changed:
                normalization = self._normalize_time_signature_region_rests(
                    part_obj,
                    part_idx,
                    edit_plan["region_measures"],
                    new_ts,
                )
                self._rebuild_measure_offsets_from(part_obj, measure_number)

            total_validated_measures += int(edit_plan["validated_measures"])
            total_normalized_measures += int(normalization["normalized_measures"])
            total_removed_overflow_rests += int(
                normalization["removed_overflow_rests"]
            )
            total_auto_completed_rests += int(normalization["auto_completed_rests"])
            active_after = self._get_active_time_signature_obj(
                part_obj,
                measure_number,
            )
            part_changed = bool(edit_changed or pre_removed or post_removed)
            if part_changed:
                changed_parts.append(part_idx)
            part_actions.append({
                "part": part_idx,
                "action": action,
                "changed": part_changed,
                "active_before": active_before.ratioString,
                "active_after": active_after.ratioString,
                "inherited_before": (
                    inherited_before.ratioString
                    if inherited_before is not None
                    else None
                ),
                "removed_redundant": pre_removed + post_removed,
                "validated_measures": int(edit_plan["validated_measures"]),
                "normalized_measures": int(normalization["normalized_measures"]),
                "removed_overflow_rests": int(
                    normalization["removed_overflow_rests"]
                ),
                "auto_completed_rests": int(
                    normalization["auto_completed_rests"]
                ),
            })

        changed = bool(changed_parts)
        action = self._combined_signature_action(part_actions)
        if changed:
            description = (
                f"Set time signature timeline to {time_signature} "
                f"at measure {measure_number}"
            )
        else:
            description = (
                f"Time signature is already {time_signature} at measure "
                f"{measure_number}; no score change made"
            )
        return OperationResult(
            success=True,
            description=description,
            details={
                "time_signature": time_signature,
                "measure": measure_number,
                "parts": changed_parts,
                "changed": changed,
                "action": action,
                "active_before": self._common_part_action_value(
                    part_actions,
                    "active_before",
                ),
                "active_after": self._common_part_action_value(
                    part_actions,
                    "active_after",
                ),
                "inherited_before": self._common_part_action_value(
                    part_actions,
                    "inherited_before",
                ),
                "validated_measures": total_validated_measures,
                "normalized_measures": total_normalized_measures,
                "removed_overflow_rests": total_removed_overflow_rests,
                "auto_completed_rests": total_auto_completed_rests,
                "part_actions": part_actions,
            },
        )


    def _collapse_rest_only_measures_to_active_meters(
        self,
        part_obj: m21stream.Part,
    ) -> None:
        """Keep rest-only measures aligned with active time signatures."""
        measures = sorted(
            part_obj.getElementsByClass(m21stream.Measure),
            key=lambda measure: measure.number,
        )
        for measure in measures:
            active_time_signature = self._get_active_time_signature_obj(
                part_obj,
                measure.number,
            )
            _collapse_rest_only_measure(
                measure,
                active_time_signature.barDuration.quarterLength,
            )


    def _collapse_rest_only_measures_to_active_meter_region(
        self,
        part_obj: m21stream.Part,
        start_measure: int,
        active_time_signature: m21meter.TimeSignature,
    ) -> None:
        """Resize rest-only measures in the affected meter region."""
        next_change = self._next_local_time_signature_measure(
            part_obj,
            start_measure,
        )
        for measure in self._sorted_measures_for_signature_scan(part_obj):
            measure_number = measure.number
            if measure_number is None or measure_number < start_measure:
                continue
            if next_change is not None and measure_number >= next_change:
                break
            _collapse_rest_only_measure(
                measure,
                active_time_signature.barDuration.quarterLength,
            )


    def _rebuild_measure_offsets_from(
        self,
        part_obj: m21stream.Part,
        start_measure: int,
    ) -> None:
        """Recalculate measure offsets from ``start_measure`` onward."""
        measures = self._sorted_measures_for_signature_scan(part_obj)
        start_obj = part_obj.measure(start_measure)
        if start_obj is None:
            return

        offset = float(part_obj.elementOffset(start_obj))
        active_time_signature = self._get_active_time_signature_obj(
            part_obj,
            start_measure,
        )
        for measure in measures:
            measure_number = measure.number
            if measure_number is None or measure_number < start_measure:
                continue

            local_time_signature = self._first_local_time_signature(measure)
            if local_time_signature is not None:
                active_time_signature = local_time_signature

            current_offset = float(part_obj.elementOffset(measure))
            if abs(current_offset - offset) > 1e-9:
                part_obj.setElementOffset(measure, offset)
            offset += float(active_time_signature.barDuration.quarterLength)


    def _time_signature_edit_plans(
        self,
        targets: list[tuple[m21stream.Part, int]],
        new_time_signature: m21meter.TimeSignature,
        measure_number: int,
    ) -> list[dict[str, object]]:
        """Return non-mutating per-part edit plans for a meter change."""
        plans: list[dict[str, object]] = []
        for part_obj, part_idx in targets:
            measure = self._resolve_measure(part_obj, measure_number)
            active_before = self._get_active_time_signature_obj(
                part_obj,
                measure_number,
            )
            inherited_before = self._get_time_signature_before_measure(
                part_obj,
                measure_number,
            )
            local_time_signatures = self._local_time_signatures(measure)
            effective_change = not self._time_signatures_equal(
                active_before,
                new_time_signature,
            )
            action = "already_active"
            if effective_change:
                if (
                    inherited_before is not None
                    and self._time_signatures_equal(
                        new_time_signature,
                        inherited_before,
                    )
                ):
                    action = "removed_local_change"
                else:
                    action = "set"
            elif (
                local_time_signatures
                and inherited_before is not None
                and self._time_signatures_equal(active_before, inherited_before)
            ):
                action = "removed_redundant_local_change"

            region_measures: list[m21stream.Measure] = []
            if effective_change:
                region_measures = self._time_signature_region_measures_after_change(
                    part_obj,
                    measure_number,
                    new_time_signature,
                )

            plans.append({
                "part_obj": part_obj,
                "part": part_idx,
                "measure": measure,
                "active_before": active_before,
                "inherited_before": inherited_before,
                "effective_change": effective_change,
                "action": action,
                "region_measures": region_measures,
                "validated_measures": len(region_measures),
            })
        return plans


    def _time_signature_region_measures_after_change(
        self,
        part_obj: m21stream.Part,
        start_measure: int,
        active_time_signature: m21meter.TimeSignature,
    ) -> list[m21stream.Measure]:
        """Return measures affected by a planned effective meter change."""
        redundant_measure_numbers = self._redundant_time_signature_measure_numbers(
            part_obj,
            start_measure,
        )
        region_measures: list[m21stream.Measure] = []
        for measure in self._sorted_measures_for_signature_scan(part_obj):
            measure_number = measure.number
            if measure_number is None or measure_number < start_measure:
                continue

            local_time_signature = self._first_local_time_signature(measure)
            if measure_number > start_measure and local_time_signature is not None:
                if int(measure_number) in redundant_measure_numbers:
                    region_measures.append(measure)
                    continue
                if self._time_signatures_equal(
                    local_time_signature,
                    active_time_signature,
                ):
                    region_measures.append(measure)
                    continue
                break

            region_measures.append(measure)
        return region_measures


    def _redundant_time_signature_measure_numbers(
        self,
        part_obj: m21stream.Part,
        start_measure: int,
    ) -> set[int]:
        """Return local meter measures canonicalization would remove."""
        previous_time_signature = self._get_time_signature_before_measure(
            part_obj,
            start_measure,
        )
        redundant_measure_numbers: set[int] = set()
        for measure in self._sorted_measures_for_signature_scan(part_obj):
            measure_number = measure.number
            if measure_number is None or measure_number < start_measure:
                continue

            local_time_signature = self._first_local_time_signature(measure)
            if local_time_signature is None:
                continue
            if (
                previous_time_signature is not None
                and self._time_signatures_equal(
                    local_time_signature,
                    previous_time_signature,
                )
            ):
                redundant_measure_numbers.add(int(measure_number))
                continue
            previous_time_signature = local_time_signature
        return redundant_measure_numbers


    @staticmethod
    def _first_local_time_signature(
        measure: m21stream.Measure,
    ) -> Optional[m21meter.TimeSignature]:
        """Return the first explicit time signature on a measure."""
        local_time_signatures = list(
            measure.getElementsByClass(m21meter.TimeSignature)
        )
        if not local_time_signatures:
            return None
        return local_time_signatures[0]


    def _time_signature_region_validation_errors(
        self,
        edit_plans: list[dict[str, object]],
        new_time_signature: m21meter.TimeSignature,
    ) -> list[dict[str, object]]:
        """Return sounding-content overflow errors for all edit plans."""
        errors: list[dict[str, object]] = []
        for edit_plan in edit_plans:
            if not bool(edit_plan["effective_change"]):
                continue
            part_idx = int(edit_plan["part"])
            for measure in edit_plan["region_measures"]:
                measure_number = int(measure.number)
                capacity = self._time_signature_effective_capacity(
                    measure,
                    new_time_signature,
                )
                for voice, container in self._measure_voice_containers(measure):
                    for event_range in self._sounding_event_ranges(container):
                        end_offset = float(event_range["end"])
                        if end_offset <= capacity + 1e-9:
                            continue
                        errors.append({
                            "part": part_idx,
                            "measure": measure_number,
                            "voice": voice,
                            "label": event_range["label"],
                            "beat": event_range["beat"],
                            "ending_beat": self._clean_rhythm_float(
                                end_offset + 1.0,
                            ),
                            "used_beats": self._clean_rhythm_float(end_offset),
                            "capacity": self._clean_rhythm_float(capacity),
                        })
        return errors


    def _format_time_signature_region_validation_error(
        self,
        time_signature: str,
        measure_number: int,
        errors: list[dict[str, object]],
    ) -> str:
        """Return a compact region-validation failure message."""
        rendered_errors = []
        for error in errors[:8]:
            rendered_errors.append(
                "part {part}, measure {measure}, voice {voice} contains "
                "{used_beats:g} beats of music ({label} starts at beat {beat}) "
                "but {time_signature} allows {capacity:g}".format(
                    time_signature=time_signature,
                    **error,
                )
            )
        if len(errors) > 8:
            rendered_errors.append(f"...and {len(errors) - 8} more")
        return (
            f"Cannot set time signature to {time_signature} at measure "
            f"{measure_number}: affected region has sounding content beyond "
            f"the new meter. " + "; ".join(rendered_errors)
        )


    @staticmethod
    def _time_signature_effective_capacity(
        measure: m21stream.Measure,
        time_signature: m21meter.TimeSignature,
    ) -> float:
        """Return meter capacity after pickup padding is applied."""
        bar_capacity = float(time_signature.barDuration.quarterLength)
        padding = float(getattr(measure, "paddingLeft", 0.0) or 0.0)
        if padding <= 1e-9:
            return bar_capacity
        return max(0.0, bar_capacity - padding)


    def _sounding_event_ranges(
        self,
        container: m21stream.Stream,
    ) -> list[dict[str, object]]:
        """Return non-grace note/chord ranges in a stream."""
        ranges: list[dict[str, object]] = []
        for element in container.getElementsByClass(m21note.GeneralNote):
            if not isinstance(element, (m21note.Note, m21chord.Chord)):
                continue
            event_range = self._rhythm_event_range(container, element)
            if event_range is not None:
                ranges.append(event_range)
        return ranges


    @staticmethod
    def _empty_time_signature_normalization_summary() -> dict[str, int]:
        """Return an empty meter-normalization summary."""
        return {
            "normalized_measures": 0,
            "removed_overflow_rests": 0,
            "auto_completed_rests": 0,
        }


    def _normalize_time_signature_region_rests(
        self,
        part_obj: m21stream.Part,
        part_idx: int,
        measures: list[m21stream.Measure],
        active_time_signature: m21meter.TimeSignature,
    ) -> dict[str, int]:
        """Normalize rest spelling across an affected meter region."""
        summary = self._empty_time_signature_normalization_summary()
        for measure in measures:
            measure_summary = self._normalize_time_signature_measure_rests(
                part_obj,
                part_idx,
                measure,
                active_time_signature,
            )
            self._add_time_signature_normalization_summary(
                summary,
                measure_summary,
            )
        return summary


    def _normalize_time_signature_measure_rests(
        self,
        part_obj: m21stream.Part,
        part_idx: int,
        measure: m21stream.Measure,
        active_time_signature: m21meter.TimeSignature,
    ) -> dict[str, int]:
        """Normalize rest spelling in one measure after a meter change."""
        summary = self._empty_time_signature_normalization_summary()
        measure_number = int(measure.number)
        capacity = self._time_signature_effective_capacity(
            measure,
            active_time_signature,
        )
        if not self._measure_has_sounding_content(measure):
            summary["removed_overflow_rests"] = (
                self._visible_overflow_rest_count(measure, capacity)
            )
            hidden = self._collapse_time_signature_rest_only_measure(
                measure,
                capacity,
            )
            summary["auto_completed_rests"] = 0 if hidden else 1
            summary["normalized_measures"] = 1
            return summary

        measure_changed = False
        for voice, container in self._measure_voice_containers(measure):
            stream_summary = self._normalize_time_signature_stream_rests(
                measure,
                measure_number,
                part_idx,
                voice,
                container,
                capacity,
            )
            if self._time_signature_normalization_changed(stream_summary):
                measure_changed = True
            self._add_time_signature_normalization_summary(
                summary,
                stream_summary,
            )

        if measure_changed:
            summary["normalized_measures"] = 1
        return summary


    @staticmethod
    def _collapse_time_signature_rest_only_measure(
        measure: m21stream.Measure,
        capacity: float,
    ) -> bool:
        """Collapse a rest-only measure and return whether it stays hidden."""
        hidden = _hidden_rests_cover_measure(measure, capacity)
        for voice in list(measure.voices):
            measure.remove(voice)
        for element in list(measure.getElementsByClass(m21note.GeneralNote)):
            measure.remove(element)

        rest = m21note.Rest(quarterLength=capacity)
        if hidden:
            rest.style.hideObjectOnPrint = True
        measure.append(rest)
        return hidden


    def _normalize_time_signature_stream_rests(
        self,
        measure: m21stream.Measure,
        measure_number: int,
        part_idx: int,
        voice: int,
        container: m21stream.Stream,
        capacity: float,
    ) -> dict[str, int]:
        """Normalize rest spelling in one measure voice or direct stream."""
        summary = self._empty_time_signature_normalization_summary()
        summary["removed_overflow_rests"] = self._visible_overflow_rest_count(
            container,
            capacity,
        )
        visible_rests = self._visible_rests(container)
        for rest in visible_rests:
            container.remove(rest)

        trimmed_hidden_rests = self._trim_hidden_rests_to_capacity(
            container,
            capacity,
        )
        if not self._stream_has_sounding_content(container):
            self._remove_empty_voice_if_needed(measure, container)
            return summary

        ranges = sorted(
            self._sounding_event_ranges(container),
            key=lambda item: (item["offset"], item["end"]),
        )
        gaps = self._rhythm_gaps(ranges, capacity)
        visible_gaps = self._time_signature_gaps_without_hidden_rests(
            container,
            gaps,
            capacity,
        )
        rest_payloads = self._rest_payloads_for_gaps(
            visible_gaps,
            measure_number,
            part_idx,
            voice,
            measure_capacity=capacity,
        )
        inserted_rests = self._insert_visible_rest_payloads(
            container,
            0.0,
            rest_payloads,
            measure_number,
            part_idx,
            voice,
        )
        summary["auto_completed_rests"] = len(inserted_rests)
        if visible_rests or trimmed_hidden_rests:
            summary["normalized_measures"] = 1
        return summary


    def _visible_overflow_rest_count(
        self,
        stream_obj: m21stream.Stream,
        capacity: float,
    ) -> int:
        """Return visible rests that extend beyond ``capacity``."""
        count = 0
        for rest in self._visible_rests(stream_obj):
            start = float(stream_obj.elementOffset(rest))
            end = start + float(rest.duration.quarterLength)
            if end > capacity + 1e-9:
                count += 1
        return count


    def _trim_hidden_rests_to_capacity(
        self,
        stream_obj: m21stream.Stream,
        capacity: float,
    ) -> int:
        """Remove or shorten hidden rests that extend beyond ``capacity``."""
        changed_count = 0
        for rest in list(stream_obj.getElementsByClass(m21note.Rest)):
            if not self._is_hidden_rest(rest):
                continue
            if getattr(rest.duration, "isGrace", False):
                continue
            start = float(stream_obj.elementOffset(rest))
            end = start + float(rest.duration.quarterLength)
            if start >= capacity - 1e-9:
                stream_obj.remove(rest)
                changed_count += 1
                continue
            if end <= capacity + 1e-9:
                continue
            rest.duration.quarterLength = max(0.0, capacity - start)
            changed_count += 1
        return changed_count


    def _time_signature_gaps_without_hidden_rests(
        self,
        stream_obj: m21stream.Stream,
        gaps: list[dict[str, object]],
        capacity: float,
    ) -> list[dict[str, object]]:
        """Return gap segments not already covered by hidden rests."""
        hidden_ranges = self._hidden_rest_ranges_in_stream(stream_obj, capacity)
        visible_gaps: list[dict[str, object]] = []
        for gap in gaps:
            gap_segments = self._subtract_ranges_from_segment(
                float(gap["offset"]),
                float(gap["end"]),
                hidden_ranges,
            )
            for start, end in gap_segments:
                visible_gaps.append(self._gap_payload(start, end))
        return visible_gaps


    def _hidden_rest_ranges_in_stream(
        self,
        stream_obj: m21stream.Stream,
        capacity: float,
    ) -> list[tuple[float, float]]:
        """Return hidden rest ranges clipped to one measure capacity."""
        ranges: list[tuple[float, float]] = []
        for rest in stream_obj.getElementsByClass(m21note.Rest):
            if not self._is_hidden_rest(rest):
                continue
            if getattr(rest.duration, "isGrace", False):
                continue
            start = float(stream_obj.elementOffset(rest))
            end = start + float(rest.duration.quarterLength)
            clipped_start = max(0.0, start)
            clipped_end = min(capacity, end)
            if clipped_end > clipped_start + 1e-9:
                ranges.append((clipped_start, clipped_end))
        return sorted(ranges)


    @staticmethod
    def _subtract_ranges_from_segment(
        start: float,
        end: float,
        ranges: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Subtract ranges from one segment and return remaining segments."""
        segments: list[tuple[float, float]] = []
        cursor = start
        for range_start, range_end in ranges:
            if range_end <= cursor + 1e-9:
                continue
            if range_start >= end - 1e-9:
                break
            clipped_start = max(start, range_start)
            clipped_end = min(end, range_end)
            if clipped_start > cursor + 1e-9:
                segments.append((cursor, clipped_start))
            cursor = max(cursor, clipped_end)
            if cursor >= end - 1e-9:
                break
        if cursor < end - 1e-9:
            segments.append((cursor, end))
        return segments


    @staticmethod
    def _add_time_signature_normalization_summary(
        summary: dict[str, int],
        addition: dict[str, int],
    ) -> None:
        """Add normalization counts from ``addition`` into ``summary``."""
        for key in (
            "normalized_measures",
            "removed_overflow_rests",
            "auto_completed_rests",
        ):
            summary[key] += int(addition.get(key, 0))


    @staticmethod
    def _time_signature_normalization_changed(
        summary: dict[str, int],
    ) -> bool:
        """Return whether a stream normalization summary changed notation."""
        return any(int(value) > 0 for value in summary.values())


    @staticmethod
    def _copy_time_signature(
        time_signature: m21meter.TimeSignature,
    ) -> m21meter.TimeSignature:
        """Return a fresh time signature with the same displayed ratio."""
        return m21meter.TimeSignature(time_signature.ratioString)


    @staticmethod
    def _time_signatures_equal(
        left: m21meter.TimeSignature,
        right: m21meter.TimeSignature,
    ) -> bool:
        """Return whether two time signatures have the same effective ratio."""
        return left.ratioString == right.ratioString


    @staticmethod
    def _local_time_signatures(
        measure: m21stream.Measure,
    ) -> list[m21meter.TimeSignature]:
        """Return explicit time signatures stored directly on ``measure``."""
        return list(measure.getElementsByClass(m21meter.TimeSignature))


    def _remove_local_time_signatures(
        self,
        measure: m21stream.Measure,
    ) -> int:
        """Remove explicit time signatures from ``measure`` and count them."""
        local_time_signatures = self._local_time_signatures(measure)
        for time_signature in local_time_signatures:
            measure.remove(time_signature)
        return len(local_time_signatures)


    def _canonicalize_time_signatures_from(
        self,
        part_obj: m21stream.Part,
        start_measure: int,
    ) -> int:
        """Remove redundant local time signatures from ``start_measure`` on."""
        previous_time_signature = self._get_time_signature_before_measure(
            part_obj,
            start_measure,
        )
        removed_count = 0
        for measure in self._sorted_measures_for_signature_scan(part_obj):
            measure_number = measure.number
            if measure_number is None or measure_number < start_measure:
                continue

            local_time_signatures = self._local_time_signatures(measure)
            if not local_time_signatures:
                continue

            effective_time_signature = local_time_signatures[0]
            if (
                previous_time_signature is not None
                and self._time_signatures_equal(
                    effective_time_signature,
                    previous_time_signature,
                )
            ):
                removed_count += self._remove_local_time_signatures(measure)
                continue

            for extra_time_signature in local_time_signatures[1:]:
                measure.remove(extra_time_signature)
                removed_count += 1
            previous_time_signature = effective_time_signature
        return removed_count


    def _get_time_signature_before_measure(
        self,
        part_obj: m21stream.Part,
        measure_number: int,
    ) -> Optional[m21meter.TimeSignature]:
        """Return the active time signature before ``measure_number``."""
        previous_measure = self._previous_signature_measure(part_obj, measure_number)
        if previous_measure is None or previous_measure.number is None:
            return None
        return self._get_active_time_signature_obj(part_obj, previous_measure.number)


    def _next_local_time_signature_measure(
        self,
        part_obj: m21stream.Part,
        start_measure: int,
    ) -> Optional[int]:
        """Return the next measure after ``start_measure`` with a local meter."""
        for measure in self._sorted_measures_for_signature_scan(part_obj):
            measure_number = measure.number
            if measure_number is None or measure_number <= start_measure:
                continue
            if self._local_time_signatures(measure):
                return int(measure_number)
        return None
