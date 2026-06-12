"""Internal expression-editing implementation slice."""

from __future__ import annotations

from .expression_common import *


class TextExpressionEditingMixin:
    """Internal mixin for ScoreSpeak expression operations."""

    def add_text_expression(
        self,
        text: str,
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
    ) -> OperationResult:
        """Add a text expression at a specific position.

        Args:
            text: Expression text such as ``"dolce"``, ``"con brio"``,
                ``"rit."``. For conventional dotted abbreviations such as
                ``"cresc."``, preserve the single written period.
            measure_number: 1-based measure number.
            beat: 1-based beat position (default 1.0).
            part: Part index, name, or None for first part.

        Returns:
            OperationResult confirming the text expression was placed.
        """
        if not text or not text.strip():
            raise ValueError("Text expression cannot be empty.")
        normalized_text = _normalize_text_expression_value(text)

        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure_number)
        ts = self._get_active_time_signature_obj(part_obj, measure_number)
        _validate_beat_in_measure(beat, ts, measure_number)

        offset = beat - 1.0
        if _find_text_expression_at_offset(measure_obj, offset, normalized_text):
            raise ValueError(
                f"Text expression '{normalized_text}' already exists at "
                f"beat {beat} in measure {measure_number}."
            )

        te = m21expressions.TextExpression(normalized_text)
        measure_obj.insert(offset, te)

        return OperationResult(
            success=True,
            description=(
                f"Added text expression '{normalized_text}' at "
                f"measure {measure_number}, beat {beat}"
            ),
            details={
                "text": normalized_text,
                "measure": measure_number,
                "beat": beat,
                "part": part_idx,
            },
        )


    def remove_text_expression(
        self,
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
        text: Optional[str] = None,
    ) -> OperationResult:
        """Remove a text expression at a specific position.

        Args:
            measure_number: 1-based measure number.
            beat: 1-based beat position (default 1.0).
            part: Part index, name, or None for first part.
            text: Optional text guard when multiple expressions share a beat.

        Returns:
            OperationResult confirming the text expression removal.
        """
        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure_number)
        offset = beat - 1.0
        expected_text = (
            _normalize_text_expression_value(text)
            if text is not None
            else None
        )

        found = None
        for el in measure_obj.getElementsByClass(m21expressions.TextExpression):
            if abs(measure_obj.elementOffset(el) - offset) > 1e-9:
                continue
            content = _normalize_text_expression_value(
                str(getattr(el, "content", "") or "")
            )
            if expected_text is not None and content != expected_text:
                continue
            found = el
            break

        if found is None:
            qualifier = f" matching '{expected_text}'" if expected_text else ""
            raise ValueError(
                f"No text expression{qualifier} found at beat {beat} "
                f"in measure {measure_number}."
            )

        removed_text = str(getattr(found, "content", ""))
        measure_obj.remove(found)

        return OperationResult(
            success=True,
            description=(
                f"Removed text expression '{removed_text}' from "
                f"measure {measure_number}, beat {beat}"
            ),
            details={
                "text": removed_text,
                "measure": measure_number,
                "beat": beat,
                "part": part_idx,
            },
        )


    def set_tempo(
        self,
        bpm: float,
        measure_number: int = 1,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
        text: Optional[str] = None,
        referent: str = "quarter",
    ) -> OperationResult:
        """Set or change the tempo at a specific position.

        If a tempo marking already exists at the exact position it is
        replaced; otherwise a new one is inserted.

        Args:
            bpm: Tempo in beats per minute.
            measure_number: 1-based measure number (default 1).
            beat: 1-based beat position (default 1.0).
            part: Part index, name, or None for first part.
            text: Optional tempo text such as ``"Allegro"``, ``"Andante"``.
            referent: Note value that ``bpm`` maps to. Accepted values are
                ``"whole"``, ``"half"``, ``"quarter"``, ``"eighth"``,
                ``"sixteenth"``, ``"16th"``, ``"32nd"``, ``"64th"``,
                ``"128th"``, note-name aliases such as ``"half note"``,
                and dotted values ``"dotted half"`` / ``"dotted half note"``
                or ``"dotted quarter"`` / ``"dotted quarter note"``.

        Returns:
            OperationResult confirming the tempo change.
        """
        if bpm <= 0:
            raise ValueError(
                f"Tempo must be a positive number, got {bpm} BPM."
            )

        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure_number)
        ts = self._get_active_time_signature_obj(part_obj, measure_number)
        _validate_beat_in_measure(beat, ts, measure_number)

        offset = beat - 1.0
        referent_duration = _normalize_tempo_referent(referent)
        referent_label = _tempo_referent_label(referent_duration)

        existing = None
        for el in measure_obj.getElementsByClass(m21tempo.MetronomeMark):
            if abs(measure_obj.elementOffset(el) - offset) < 1e-9:
                existing = el
                break

        if existing is not None:
            measure_obj.remove(existing)

        kwargs = {"number": bpm, "referent": referent_duration}
        if text is not None:
            kwargs["text"] = text
        mm = m21tempo.MetronomeMark(**kwargs)
        measure_obj.insert(offset, mm)

        desc = f"Set tempo to {referent_label} = {bpm} BPM"
        if text:
            desc += f" ({text})"
        desc += f" at measure {measure_number}, beat {beat}"

        return OperationResult(
            success=True,
            description=desc,
            details={
                "bpm": bpm,
                "text": text,
                "referent": referent_label,
                "measure": measure_number,
                "beat": beat,
                "part": part_idx,
                "replaced": existing is not None,
            },
        )


    def add_rehearsal_mark(
        self,
        text: str,
        measure_number: int,
    ) -> OperationResult:
        """Add a rehearsal mark at the beginning of a measure.

        Args:
            text: Rehearsal mark text such as ``"A"``, ``"B"``, ``"1"``.
            measure_number: 1-based measure number.

        Returns:
            OperationResult confirming the rehearsal mark was placed.
        """
        if not text or not text.strip():
            raise ValueError("Rehearsal mark text cannot be empty.")

        part_obj, _ = self._resolve_part(None)
        measure_obj = self._resolve_measure(part_obj, measure_number)

        rm = m21expressions.RehearsalMark(text.strip())
        measure_obj.insert(0, rm)

        return OperationResult(
            success=True,
            description=(
                f"Added rehearsal mark '{text.strip()}' at "
                f"measure {measure_number}"
            ),
            details={
                "text": text.strip(),
                "measure": measure_number,
            },
        )


    def remove_rehearsal_mark(
        self,
        measure_number: int,
        text: Optional[str] = None,
    ) -> OperationResult:
        """Remove a rehearsal mark from the beginning of a measure.

        Args:
            measure_number: 1-based measure number.
            text: Optional text guard when multiple marks share a measure.

        Returns:
            OperationResult confirming the rehearsal mark removal.
        """
        expected_text = text.strip() if text is not None else None

        targets = self._resolve_parts_or_all()
        found: list[tuple[m21stream.Measure, m21expressions.RehearsalMark]] = []
        for part_obj, _ in targets:
            measure_obj = self._resolve_measure(part_obj, measure_number)
            for el in measure_obj.getElementsByClass(m21expressions.RehearsalMark):
                mark_text = str(getattr(el, "content", ""))
                if expected_text is not None and mark_text != expected_text:
                    continue
                found.append((measure_obj, el))

        if not found:
            qualifier = f" matching '{expected_text}'" if expected_text else ""
            raise ValueError(
                f"No rehearsal mark{qualifier} found in "
                f"measure {measure_number}."
            )

        removed_texts = [str(getattr(mark, "content", "")) for _, mark in found]
        for measure_obj, mark in found:
            measure_obj.remove(mark)

        return OperationResult(
            success=True,
            description=(
                f"Removed {len(found)} rehearsal mark(s) from "
                f"measure {measure_number}"
            ),
            details={
                "text": removed_texts[0],
                "removed_texts": removed_texts,
                "measure": measure_number,
                "removed_count": len(found),
            },
        )


    def add_chord_symbol(
        self,
        symbol: str,
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
    ) -> OperationResult:
        """Place a chord symbol at a specific beat position.

        Args:
            symbol: Standard chord notation string, e.g. ``"Cmaj7"``,
                ``"Am"``, ``"G7"``, ``"Dm7b5"``, ``"F#m7"``, ``"Bb"``.
            measure_number: 1-based measure number.
            beat: 1-based beat position (default 1.0).
            part: Part index, name, or None for first part.

        Returns:
            OperationResult confirming the chord symbol was placed.

        Raises:
            ValueError: If the chord string is unparseable, the measure
                does not exist, or the beat is out of range.
        """
        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure_number)
        ts = self._get_active_time_signature_obj(part_obj, measure_number)
        _validate_beat_in_measure(beat, ts, measure_number)

        try:
            cs = m21harmony.ChordSymbol(symbol)
        except Exception as exc:
            raise ValueError(
                f"Cannot parse chord symbol '{symbol}'. "
                f"Expected formats like 'C', 'Cm', 'C7', 'Cmaj7', "
                f"'Dm7b5', 'F#m7', 'Bb'."
            ) from exc

        offset = beat - 1.0
        measure_obj.insert(offset, cs)

        return OperationResult(
            success=True,
            description=(
                f"Added chord symbol '{symbol}' at measure "
                f"{measure_number}, beat {beat}"
            ),
            details={
                "chord_symbol": symbol,
                "measure": measure_number,
                "beat": beat,
                "part": part_idx,
            },
        )


    def remove_chord_symbol(
        self,
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
    ) -> OperationResult:
        """Remove the first chord symbol at the given beat.

        Args:
            measure_number: 1-based measure number.
            beat: 1-based beat position (default 1.0).
            part: Part index, name, or None for first part.

        Returns:
            OperationResult confirming removal.

        Raises:
            ValueError: If no chord symbol exists at the position.
        """
        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure_number)
        offset = beat - 1.0

        for el in measure_obj.getElementsByClass(m21harmony.ChordSymbol):
            if abs(measure_obj.elementOffset(el) - offset) < 1e-9:
                measure_obj.remove(el)
                return OperationResult(
                    success=True,
                    description=(
                        f"Removed chord symbol from measure "
                        f"{measure_number}, beat {beat}"
                    ),
                    details={
                        "measure": measure_number,
                        "beat": beat,
                        "part": part_idx,
                    },
                )

        raise ValueError(
            f"No chord symbol found at measure {measure_number}, "
            f"beat {beat}."
        )


    def get_chord_symbols(
        self,
        measure_number: Optional[int] = None,
        part: Optional[Union[int, str]] = None,
    ) -> list[dict]:
        """Return all chord symbols as dicts.

        Args:
            measure_number: If provided, filter to this measure only.
            part: Part index, name, or None for first part.

        Returns:
            List of dicts with ``symbol``, ``measure_number``, ``beat``,
            and ``part_index`` keys.
        """
        part_obj, part_idx = self._resolve_part(part)
        results: list[dict] = []

        measures = sorted(
            part_obj.getElementsByClass(m21stream.Measure),
            key=lambda m: m.number,
        )

        for m in measures:
            if measure_number is not None and m.number != measure_number:
                continue
            for el in m.getElementsByClass(m21harmony.ChordSymbol):
                offset = m.elementOffset(el)
                results.append({
                    "symbol": el.figure,
                    "measure_number": m.number,
                    "beat": offset + 1.0,
                    "part_index": part_idx,
                })

        return results
