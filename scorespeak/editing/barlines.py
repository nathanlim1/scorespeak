"""Internal signature-editing implementation slice."""

from __future__ import annotations

from .signature_common import *


class BarlineEditingMixin:
    """Internal mixin for ScoreSpeak signature operations."""

    def set_clef(
        self,
        clef_type: str,
        measure_number: int,
        part: Optional[Union[int, str]] = None,
    ) -> OperationResult:
        """Change the clef at a specific measure.

        Args:
            clef_type: Clef name such as "treble", "bass", "alto", "tenor".
            measure_number: 1-based measure number.
            part: Part index (0-based), part name, or None for all parts.

        Returns:
            OperationResult describing the outcome.

        Raises:
            ValueError: If the clef type is unrecognized or the measure
                does not exist.
        """
        new_clef = make_clef(clef_type)
        targets = self._resolve_parts_or_all(part)
        changed_parts = []

        for part_obj, part_idx in targets:
            m = self._resolve_measure(part_obj, measure_number)

            existing_clefs = list(m.getElementsByClass(m21clef.Clef))
            for old_clef in existing_clefs:
                m.remove(old_clef)

            m.insert(0, make_clef(clef_type))
            changed_parts.append(part_idx)

        return OperationResult(
            success=True,
            description=(
                f"Set clef to {clef_type} at measure {measure_number}"
            ),
            details={
                "clef": clef_type,
                "measure": measure_number,
                "parts": changed_parts,
            },
        )


    def set_barline(
        self,
        barline_type: str,
        measure_number: int,
    ) -> OperationResult:
        """Set the barline at the right edge (END) of a measure. To change the
        left side of a measure, set the right barline of the previous measure
        instead.

        Args:
            barline_type: Barline style such as "double", "final",
                "light-light", "light-heavy", "regular", or "none".
                Repeat barlines are managed with add_repeat/remove_repeat.
            measure_number: 1-based measure number whose right edge should
                receive this barline.

        Returns:
            OperationResult describing the outcome.

        Raises:
            ValueError: If the barline type is invalid, or the measure does
                not exist.
        """
        normalized_type = barline_type.strip().lower()
        if normalized_type in {
            "repeat-start",
            "start-repeat",
            "repeat-end",
            "end-repeat",
            "repeat-both",
        } or normalized_type.startswith("repeat"):
            raise ValueError(
                "Repeat barlines are structural repeats. Use add_repeat() "
                "or remove_repeat() instead of set_barline()."
            )

        if normalized_type not in _VALID_BARLINE_TYPES:
            raise ValueError(
                f"Unknown barline type '{barline_type}'. "
                f"Valid types: {', '.join(sorted(_VALID_BARLINE_TYPES))}."
            )

        targets = self._resolve_parts_or_all()
        changed_parts = []

        for part_obj, part_idx in targets:
            m = self._resolve_measure(part_obj, measure_number)

            barline_obj = m21bar.Barline(type=normalized_type)
            m.rightBarline = barline_obj

            changed_parts.append(part_idx)

        return OperationResult(
            success=True,
            description=(
                f"Set right-edge barline of measure {measure_number} to "
                f"{barline_type}."
            ),
            details={
                "barline_type": barline_type,
                "side": "right",
                "measure": measure_number,
                "parts": changed_parts,
            },
        )


    def add_repeat(
        self,
        start_measure: int,
        end_measure: int,
        times: int = 2,
    ) -> OperationResult:
        """Add repeat barlines around a range of measures.

        Places a repeat-start barline at the beginning of start_measure
        and a repeat-end barline at the end of end_measure.

        Args:
            start_measure: 1-based measure number for the repeat start.
            end_measure: 1-based measure number for the repeat end.
            times: Number of times to play the repeated section (default 2).

        Returns:
            OperationResult describing the outcome.

        Raises:
            ValueError: If measures are invalid or start > end.
        """
        if start_measure > end_measure:
            raise ValueError(
                f"Repeat start measure ({start_measure}) must not be "
                f"after end measure ({end_measure})."
            )
        if times < 2:
            raise ValueError(
                f"Repeat times must be at least 2, got {times}."
            )

        targets = self._resolve_parts_or_all()
        changed_parts = []

        for part_obj, part_idx in targets:
            start_m = self._resolve_measure(part_obj, start_measure)
            end_m = self._resolve_measure(part_obj, end_measure)

            start_barline = m21bar.Repeat(direction="start")
            start_m.leftBarline = start_barline

            end_barline = m21bar.Repeat(direction="end", times=times)
            end_m.rightBarline = end_barline

            changed_parts.append(part_idx)

        return OperationResult(
            success=True,
            description=(
                f"Added repeat from measure {start_measure} to "
                f"{end_measure} ({times}x)"
            ),
            details={
                "start_measure": start_measure,
                "end_measure": end_measure,
                "times": times,
                "parts": changed_parts,
            },
        )


    def remove_repeat(
        self,
        start_measure: int,
        end_measure: int,
    ) -> OperationResult:
        """Remove repeat barlines around a range of measures.

        Args:
            start_measure: 1-based measure number for the repeat start.
            end_measure: 1-based measure number for the repeat end.

        Returns:
            OperationResult describing the removed repeat barlines.
        """
        if start_measure > end_measure:
            raise ValueError(
                f"Repeat start measure ({start_measure}) must not be "
                f"after end measure ({end_measure})."
            )

        targets = self._resolve_parts_or_all()
        changed_parts = []
        removed_barlines = []

        for part_obj, part_idx in targets:
            start_m = self._resolve_measure(part_obj, start_measure)
            end_m = self._resolve_measure(part_obj, end_measure)
            left = start_m.leftBarline
            right = end_m.rightBarline

            missing = []
            if isinstance(left, m21bar.Repeat) and left.direction == "start":
                pass
            else:
                missing.append("start")

            if isinstance(right, m21bar.Repeat) and right.direction == "end":
                pass
            else:
                missing.append("end")

            if missing:
                raise ValueError(
                    f"No complete repeat from measure {start_measure} to "
                    f"{end_measure} in part {part_idx}; missing "
                    f"{', '.join(missing)} repeat barline."
                )

            start_m.leftBarline = None
            end_m.rightBarline = None
            changed_parts.append(part_idx)
            removed_barlines.append({
                "part": part_idx,
                "barlines": ["start", "end"],
            })

        return OperationResult(
            success=True,
            description=(
                f"Removed repeat from measure {start_measure} to "
                f"{end_measure}"
            ),
            details={
                "start_measure": start_measure,
                "end_measure": end_measure,
                "parts": changed_parts,
                "removed_barlines": removed_barlines,
            },
        )


    def set_pickup_measure(
        self,
        duration: float,
    ) -> OperationResult:
        """Convert measure 1 into a pickup (anacrusis) measure.

        Adjusts the first measure so that its effective length matches the
        given duration.  Existing content (notes/rests) is replaced by a
        single rest of the specified duration.

        Args:
            duration: Duration of the pickup in quarter-note lengths
                (e.g. 1.0 for a quarter-note pickup, 2.0 for a half-note).

        Returns:
            OperationResult describing the outcome.

        Raises:
            ValueError: If the duration is non-positive, exceeds the
                time signature's bar length, or measure 1 doesn't exist.
        """
        if duration <= 0:
            raise ValueError(
                f"Pickup duration must be positive, got {duration}."
            )

        targets = self._resolve_parts_or_all()
        changed_parts = []

        for part_obj, part_idx in targets:
            m = self._resolve_measure(part_obj, 1)
            ts = self._get_active_time_signature_obj(part_obj, 1)
            bar_capacity = ts.barDuration.quarterLength

            if duration > bar_capacity + 1e-9:
                raise ValueError(
                    f"Pickup duration ({duration} beats) exceeds the "
                    f"measure's capacity in {ts.ratioString} time "
                    f"({bar_capacity} beats)."
                )

            _clear_measure_note_content(m)

            pickup_rest = m21note.Rest(quarterLength=duration)
            m.insert(0, pickup_rest)

            padding = bar_capacity - duration
            m.paddingLeft = max(0.0, padding)

            changed_parts.append(part_idx)

        return OperationResult(
            success=True,
            description=(
                f"Set measure 1 as a pickup measure "
                f"({duration} quarter-note beats)"
            ),
            details={
                "duration": duration,
                "parts": changed_parts,
            },
        )
