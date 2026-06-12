"""Internal marking-editing implementation slice."""

from __future__ import annotations

from .marking_common import *


class EndingBracketEditingMixin:
    """Internal mixin for ScoreSpeak marking operations."""

    def add_ending_bracket(
        self,
        number: Union[int, str],
        start_measure: int,
        end_measure: int,
    ) -> OperationResult:
        """Add a repeat ending (volta) bracket spanning measure numbers.

        This matches MusicXML ``ending`` / MuseScore volta brackets. It spans
        whole measures (not individual beats within a bar).
        """
        if end_measure < start_measure:
            raise ValueError(
                f"end_measure ({end_measure}) must be >= "
                f"start_measure ({start_measure})."
            )

        normalized_number = _normalize_ending_number(number)

        spans: list[tuple[m21stream.Part, int, list[m21stream.Measure]]] = []
        for part_obj, part_idx in self._resolve_parts_or_all():
            measures = sorted(
                part_obj.getElementsByClass(m21stream.Measure),
                key=lambda mm: mm.number,
            )
            span = [
                m
                for m in measures
                if start_measure <= m.number <= end_measure
            ]
            if len(span) != (end_measure - start_measure + 1):
                missing = [
                    n
                    for n in range(start_measure, end_measure + 1)
                    if not any(m.number == n for m in measures)
                ]
                raise ValueError(
                    f"Cannot span measures {start_measure}–{end_measure} "
                    f"in part {part_idx}: missing measure(s) {missing}."
                )
            spans.append((part_obj, part_idx, span))

        changed_parts = []
        for part_obj, part_idx, span in spans:
            rb = m21spanner.RepeatBracket(
                *span,
                number=normalized_number,
            )
            part_obj.insert(0, rb)
            changed_parts.append(part_idx)

        return OperationResult(
            success=True,
            description=(
                f"Added ending bracket #{normalized_number} over measures "
                f"{start_measure}–{end_measure}"
            ),
            details={
                "number": normalized_number,
                "start_measure": start_measure,
                "end_measure": end_measure,
                "parts": changed_parts,
            },
        )


    def remove_ending_bracket(
        self,
        number: Union[int, str],
        start_measure: int,
    ) -> OperationResult:
        """Remove a RepeatBracket with this ending number starting at a measure."""
        normalized_number = _normalize_ending_number(number)
        target_num = str(normalized_number)
        found: list[tuple[m21stream.Part, int, m21spanner.RepeatBracket]] = []

        for part_obj, part_idx in self._resolve_parts_or_all():
            for sp in part_obj.getElementsByClass(m21spanner.RepeatBracket):
                sn = getattr(sp, "number", None)
                if sn is None:
                    continue
                if str(sn) != target_num:
                    continue
                if _repeat_bracket_starts_at(sp, start_measure):
                    found.append((part_obj, part_idx, sp))

        if not found:
            raise ValueError(
                f"No ending bracket #{normalized_number} starting at measure "
                f"{start_measure}."
            )

        changed_parts = []
        for part_obj, part_idx, repeat_bracket in found:
            part_obj.remove(repeat_bracket)
            if part_idx not in changed_parts:
                changed_parts.append(part_idx)

        return OperationResult(
            success=True,
            description=(
                f"Removed ending bracket #{normalized_number} from measure "
                f"{start_measure}"
            ),
            details={
                "number": normalized_number,
                "start_measure": start_measure,
                "parts": changed_parts,
            },
        )
