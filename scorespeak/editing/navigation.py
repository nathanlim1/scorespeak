"""Internal signature-editing implementation slice."""

from __future__ import annotations

from .signature_common import *


class NavigationMarkEditingMixin:
    """Internal mixin for ScoreSpeak signature operations."""

    def add_coda(
        self,
        measure_number: int,
    ) -> OperationResult:
        """Insert a coda sign at the beginning of a measure.

        Args:
            measure_number: 1-based measure number.

        Returns:
            OperationResult describing the outcome.

        Raises:
            ValueError: If the measure does not exist.
        """
        part_obj, _ = self._resolve_part(None)
        m = self._resolve_measure(part_obj, measure_number)
        m.insert(0, m21repeat.Coda())
        return OperationResult(
            success=True,
            description=f"Added coda sign at measure {measure_number}",
            details={
                "mark_type": "coda",
                "measure": measure_number,
            },
        )


    def add_segno(
        self,
        measure_number: int,
    ) -> OperationResult:
        """Insert a segno sign at the beginning of a measure.

        Args:
            measure_number: 1-based measure number.

        Returns:
            OperationResult describing the outcome.

        Raises:
            ValueError: If the measure does not exist.
        """
        part_obj, _ = self._resolve_part(None)
        m = self._resolve_measure(part_obj, measure_number)
        m.insert(0, m21repeat.Segno())
        return OperationResult(
            success=True,
            description=f"Added segno sign at measure {measure_number}",
            details={
                "mark_type": "segno",
                "measure": measure_number,
            },
        )


    def add_to_coda(
        self,
        measure_number: int,
    ) -> OperationResult:
        """Insert a To Coda direction at a measure.

        In a D.C./D.S. al Coda roadmap, this marks where playback jumps to the
        destination coda section on the repeat pass.

        Args:
            measure_number: 1-based measure number.

        Returns:
            OperationResult describing the outcome.

        Raises:
            ValueError: If the measure does not exist.
        """
        part_obj, _ = self._resolve_part(None)
        m = self._resolve_measure(part_obj, measure_number)
        m.insert(0, m21repeat.Coda(_TO_CODA_TEXT))
        return OperationResult(
            success=True,
            description=f"Added To Coda direction at measure {measure_number}",
            details={
                "mark_type": _TO_CODA_MARK_TYPE,
                "measure": measure_number,
            },
        )


    def add_fine(
        self,
        measure_number: int,
    ) -> OperationResult:
        """Insert a Fine marker at the RIGHT SIDE (end) of a measure.

        In a D.C./D.S. al Fine roadmap, this marks where playback ends on the
        repeat pass.

        Args:
            measure_number: 1-based measure number.

        Returns:
            OperationResult describing the outcome.

        Raises:
            ValueError: If the measure does not exist.
        """
        part_obj, _ = self._resolve_part(None)
        m = self._resolve_measure(part_obj, measure_number)
        m.insert(0, m21repeat.Fine())
        return OperationResult(
            success=True,
            description=f"Added Fine marker at the end of measure {measure_number}",
            details={
                "mark_type": "fine",
                "measure": measure_number,
            },
        )


    def add_da_capo(
        self,
        measure_number: int,
        al: Optional[str] = None,
    ) -> OperationResult:
        """Insert a Da Capo (D.C.) direction at the RIGHT SIDE (end) of a measure.

        Args:
            measure_number: 1-based measure number.
            al: Optional qualifier — None for plain D.C., ``"fine"`` for
                D.C. al Fine, ``"coda"`` for D.C. al Coda.

        Returns:
            OperationResult describing the outcome.

        Raises:
            ValueError: If the measure does not exist or *al* is invalid.
        """
        _DC_MAP: dict[str | None, type] = {
            None: m21repeat.DaCapo,
            "fine": m21repeat.DaCapoAlFine,
            "coda": m21repeat.DaCapoAlCoda,
        }
        if al is not None:
            al = al.lower()
        if al not in _DC_MAP:
            raise ValueError(
                f"Invalid 'al' value '{al}' for Da Capo. "
                f"Expected None, 'fine', or 'coda'."
            )
        part_obj, _ = self._resolve_part(None)
        m = self._resolve_measure(part_obj, measure_number)
        m.insert(0, _DC_MAP[al]())
        mark_label = "da capo" if al is None else f"da capo al {al}"
        return OperationResult(
            success=True,
            description=f"Added {mark_label} at the end ofmeasure {measure_number}",
            details={
                "mark_type": "da capo",
                "measure": measure_number,
                "al": al,
            },
        )


    def add_dal_segno(
        self,
        measure_number: int,
        al: Optional[str] = None,
    ) -> OperationResult:
        """Insert a Dal Segno (D.S.) direction at the RIGHT SIDE (end) of a measure.

        Args:
            measure_number: 1-based measure number.
            al: Optional qualifier — None for plain D.S., ``"fine"`` for
                D.S. al Fine, ``"coda"`` for D.S. al Coda.

        Returns:
            OperationResult describing the outcome.

        Raises:
            ValueError: If the measure does not exist or *al* is invalid.
        """
        _DS_MAP: dict[str | None, type] = {
            None: m21repeat.DalSegno,
            "fine": m21repeat.DalSegnoAlFine,
            "coda": m21repeat.DalSegnoAlCoda,
        }
        if al is not None:
            al = al.lower()
        if al not in _DS_MAP:
            raise ValueError(
                f"Invalid 'al' value '{al}' for Dal Segno. "
                f"Expected None, 'fine', or 'coda'."
            )
        part_obj, _ = self._resolve_part(None)
        m = self._resolve_measure(part_obj, measure_number)
        m.insert(0, _DS_MAP[al]())
        mark_label = "dal segno" if al is None else f"dal segno al {al}"
        return OperationResult(
            success=True,
            description=f"Added {mark_label} at the end of measure {measure_number}",
            details={
                "mark_type": "dal segno",
                "measure": measure_number,
                "al": al,
            },
        )


    def remove_navigation_mark(
        self,
        mark_type: str,
        measure_number: int,
    ) -> OperationResult:
        """Remove matching navigation marks from a measure across all parts.

        Args:
            mark_type: One of ``"coda"``, ``"segno"``, ``"to coda"``,
                ``"fine"``, ``"da capo"``, ``"dal segno"``
                (case-insensitive).
            measure_number: 1-based measure number.

        Returns:
            OperationResult describing the outcome.

        Raises:
            ValueError: If *mark_type* is unrecognized or no matching
                mark exists in the measure.
        """
        normalized = mark_type.strip().lower().replace("_", " ")
        valid_types = sorted(
            [*_NAVIGATION_MARK_CLASSES, _CODA_MARK_TYPE, _TO_CODA_MARK_TYPE]
        )
        if (
            normalized not in _NAVIGATION_MARK_CLASSES
            and normalized not in {_CODA_MARK_TYPE, _TO_CODA_MARK_TYPE}
        ):
            raise ValueError(
                f"Unknown navigation mark type '{mark_type}'. "
                f"Valid types: {', '.join(valid_types)}."
            )

        targets = self._resolve_parts_or_all()
        removed_count = 0

        if normalized in {_CODA_MARK_TYPE, _TO_CODA_MARK_TYPE}:
            wants_to_coda = normalized == _TO_CODA_MARK_TYPE
            for part_obj, _ in targets:
                m = self._resolve_measure(part_obj, measure_number)
                marks = [
                    el
                    for el in m.recurse().getElementsByClass(m21repeat.Coda)
                    if _is_to_coda_mark(el) == wants_to_coda
                ]
                for mark in marks:
                    m.remove(mark)
                removed_count += len(marks)
        else:
            target_classes = _NAVIGATION_MARK_CLASSES[normalized]
            if isinstance(target_classes, type):
                target_classes = (target_classes,)

            for part_obj, _ in targets:
                m = self._resolve_measure(part_obj, measure_number)
                marks = [el for el in m.recurse() if isinstance(el, target_classes)]
                for mark in marks:
                    m.remove(mark)
                removed_count += len(marks)

        if removed_count:
            return OperationResult(
                success=True,
                description=(
                    f"Removed {removed_count} {normalized} mark(s) "
                    f"from measure {measure_number}"
                ),
                details={
                    "mark_type": normalized,
                    "measure": measure_number,
                    "removed_count": removed_count,
                },
            )

        raise ValueError(
            f"No '{normalized}' mark found in measure {measure_number}."
        )
