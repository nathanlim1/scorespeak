"""Internal bar-retrieval implementation slice."""

from __future__ import annotations

from .common import *


class BarNotationMixin:
    """Internal mixin for ScoreSpeak bar retrieval."""

    def _build_bar_notations(
        self,
        parts: list[tuple[m21stream.Part, int]],
        measure_numbers: list[int],
    ) -> dict[int, BarNotation]:
        """Build the per-bar notation dict for every scoped measure."""
        notations: dict[int, BarNotation] = {}
        if not measure_numbers:
            return notations

        scope_first = min(measure_numbers)
        reference_part = parts[0][0] if parts else None

        prev_time: Optional[str] = None
        prev_key: Optional[str] = None
        prev_tempo: Optional[float] = None

        sorted_measures = sorted(measure_numbers)
        for measure_number in sorted_measures:
            measures_in_parts = [
                (part_obj, part_obj.measure(measure_number))
                for part_obj, _ in parts
            ]

            time_str: Optional[str] = None
            key_str: Optional[str] = None
            tempo_value: Optional[float] = None
            if reference_part is not None:
                try:
                    ts_obj = self._get_active_time_signature_obj(
                        reference_part, measure_number
                    )
                    time_str = ts_obj.ratioString
                except Exception:
                    time_str = None
                try:
                    ks_obj = self._get_active_concert_key_signature_obj(
                        measure_number,
                    )
                    key_str = self._format_key_signature(ks_obj)
                except Exception:
                    key_str = None
                try:
                    tempo_value = self.get_active_tempo(measure_number)
                except Exception:
                    tempo_value = None

            active: ActiveSignatures = {
                "time": time_str if time_str is not None else "",
                "key": key_str if key_str is not None else "",
            }
            if key_str is not None:
                active["concert_key"] = key_str
                active["key_space"] = "concert"

            is_first_bar = measure_number == scope_first
            changed: list[str] = []

            if is_first_bar:
                if tempo_value is not None:
                    active["tempo"] = float(tempo_value)
            else:
                if time_str != prev_time and time_str is not None:
                    changed.append("time")
                if key_str != prev_key and key_str is not None:
                    changed.append("key")
                if tempo_value != prev_tempo:
                    if tempo_value is not None:
                        active["tempo"] = float(tempo_value)
                        changed.append("tempo")

            notation: BarNotation = {"active": active}
            if changed:
                notation["changed_here"] = changed

            self._populate_structural_notation_fields(
                notation,
                measures_in_parts,
            )

            notations[measure_number] = notation

            prev_time = time_str
            prev_key = key_str
            prev_tempo = tempo_value

        return notations


    def _populate_structural_notation_fields(
        self,
        notation: BarNotation,
        measures_in_parts: list[
            tuple[m21stream.Part, Optional[m21stream.Measure]]
        ],
    ) -> None:
        """Fill barline, repeat, ending, rehearsal, and navigation keys."""
        barline_start: Optional[str] = None
        barline_end: Optional[str] = None
        repeat_start = False
        repeat_end = False
        rehearsal_mark: Optional[str] = None
        navigation: list[str] = []
        ending_number: Optional[str] = None
        system_break = False
        page_break = False

        for part_obj, measure_obj in measures_in_parts:
            if measure_obj is None:
                continue

            left = getattr(measure_obj, "leftBarline", None)
            right = getattr(measure_obj, "rightBarline", None)

            if isinstance(left, m21bar.Repeat):
                if getattr(left, "direction", None) == "start":
                    repeat_start = True
                if barline_start is None:
                    barline_start = left.type
            elif isinstance(left, m21bar.Barline):
                if barline_start is None and left.type != "regular":
                    barline_start = left.type

            if isinstance(right, m21bar.Repeat):
                if getattr(right, "direction", None) == "end":
                    repeat_end = True
                if barline_end is None:
                    barline_end = right.type
            elif isinstance(right, m21bar.Barline):
                if barline_end is None and right.type != "regular":
                    barline_end = right.type

            if rehearsal_mark is None:
                for rm in measure_obj.getElementsByClass(
                    m21expressions.RehearsalMark
                ):
                    text = str(getattr(rm, "content", "") or "").strip()
                    if text:
                        rehearsal_mark = text
                        break

            for nav_cls, label in _NAVIGATION_MARK_LABELS:
                if label in navigation:
                    continue
                hits = list(measure_obj.getElementsByClass(nav_cls))
                if hits:
                    navigation.append(label)

            for coda in measure_obj.getElementsByClass(m21repeat.Coda):
                label = _coda_navigation_label(coda)
                if label not in navigation:
                    navigation.append(label)

            if not system_break:
                system_break = any(
                    bool(getattr(layout_obj, "isNew", False))
                    for layout_obj in measure_obj.getElementsByClass(
                        m21layout.SystemLayout
                    )
                )

            if not page_break:
                page_break = any(
                    bool(getattr(layout_obj, "isNew", False))
                    for layout_obj in measure_obj.getElementsByClass(
                        m21layout.PageLayout
                    )
                )

            if ending_number is None:
                measure_number = int(measure_obj.number)
                for bracket in part_obj.getElementsByClass(m21spanner.RepeatBracket):
                    spanned = bracket.getSpannedElements()
                    if not spanned:
                        continue
                    numbers = [
                        int(el.number)
                        for el in spanned
                        if isinstance(el, m21stream.Measure)
                        and getattr(el, "number", None) is not None
                    ]
                    if measure_number in numbers:
                        ending_number = str(getattr(bracket, "number", "") or "")
                        break

        if barline_start and barline_start != "regular":
            notation["barline_start"] = barline_start
        if barline_end and barline_end != "regular":
            notation["barline_end"] = barline_end
        if repeat_start:
            notation["repeat_start"] = True
        if repeat_end:
            notation["repeat_end"] = True
        if ending_number:
            notation["ending_number"] = ending_number
        if rehearsal_mark:
            notation["rehearsal_mark"] = rehearsal_mark
        if navigation:
            notation["navigation"] = navigation
        if system_break:
            notation["system_break"] = True
        if page_break:
            notation["page_break"] = True


    def _build_part_notation(
        self,
        part_obj: m21stream.Part,
        measure_obj: Optional[m21stream.Measure],
        measure_number: int,
        scope_measures: list[int],
    ) -> PartNotation:
        """Return part-specific clef and key info when relevant."""
        notation: PartNotation = {}
        if not scope_measures:
            return notation

        scope_first = min(scope_measures)

        try:
            active_clef = self._get_active_clef_obj(part_obj, measure_number)
        except Exception:
            return notation

        clef_label = self._format_clef(active_clef)
        if clef_label is None:
            return notation

        if measure_number == scope_first:
            notation["clef"] = clef_label
            if measure_obj is not None:
                self._populate_part_key_notation(
                    notation,
                    part_obj,
                    measure_obj,
                    measure_number,
                    scope_first,
                )
            return notation

        if measure_obj is None:
            return notation

        declared_clefs = list(measure_obj.getElementsByClass(m21clef.Clef))
        if declared_clefs:
            notation["clef"] = clef_label
        self._populate_part_key_notation(
            notation,
            part_obj,
            measure_obj,
            measure_number,
            scope_first,
        )
        return notation


    def _populate_part_key_notation(
        self,
        notation: PartNotation,
        part_obj: m21stream.Part,
        measure_obj: Optional[m21stream.Measure],
        measure_number: int,
        scope_first: int,
    ) -> None:
        """Add part key notation when it differs from the concert key."""
        try:
            active_part_key = self._get_active_key_signature_obj(
                part_obj,
                measure_number,
            )
            active_concert_key = self._get_active_concert_key_signature_obj(
                measure_number,
            )
        except Exception:
            return

        if self._key_signatures_equal(active_part_key, active_concert_key):
            return
        if measure_number != scope_first and measure_obj is not None:
            declared_keys = list(
                measure_obj.getElementsByClass(m21key.KeySignature)
            )
            if not declared_keys:
                return

        self._set_part_key_notation_fields(
            notation,
            part_obj,
            active_part_key,
            active_concert_key,
        )


    def _set_part_key_notation_fields(
        self,
        notation: PartNotation,
        part_obj: m21stream.Part,
        active_part_key: m21key.KeySignature,
        active_concert_key: m21key.KeySignature,
    ) -> None:
        """Populate explicit part-key metadata for agent-facing payloads."""
        part_key_label = self._format_key_signature(active_part_key)
        concert_key_label = self._format_key_signature(active_concert_key)
        has_transposition = part_transposition_interval(part_obj) is not None
        stores_sounding = part_stores_sounding_pitch(part_obj)
        expected_part_key = stored_key_signature_for_concert_key(
            part_obj,
            active_concert_key,
        )
        is_expected_transposed_key = (
            has_transposition
            and not stores_sounding
            and self._key_signatures_equal(active_part_key, expected_part_key)
        )

        if is_expected_transposed_key:
            key_space = "written pitch"
            key_role = "transposed_written_key"
            key_label = (
                f"written key: {part_key_label} "
                f"(concert key: {concert_key_label})"
            )
        else:
            key_space = (
                stored_pitch_space_label(part_obj)
                if has_transposition
                else "concert pitch"
            )
            key_role = "local_staff_key"
            key_label = (
                f"local key: {part_key_label} "
                f"(concert key: {concert_key_label})"
            )

        notation["key"] = part_key_label
        notation["concert_key"] = concert_key_label
        notation["key_space"] = key_space
        notation["key_role"] = key_role
        notation["key_is_transposed"] = is_expected_transposed_key
        notation["key_label"] = key_label


    @staticmethod
    def _format_clef(clef_obj: m21clef.Clef) -> Optional[str]:
        """Return a short human label for a music21 Clef."""
        if clef_obj is None:
            return None
        sign = getattr(clef_obj, "sign", None)
        line = getattr(clef_obj, "line", None)
        octave_change = int(getattr(clef_obj, "octaveChange", 0) or 0)

        base_labels = {
            ("G", 2): "treble",
            ("F", 4): "bass",
            ("C", 3): "alto",
            ("C", 4): "tenor",
            ("C", 1): "soprano",
            ("C", 2): "mezzo-soprano",
            ("F", 3): "baritone",
            ("percussion", None): "percussion",
            ("TAB", None): "tab",
        }

        key = (sign, line)
        label = base_labels.get(key)
        if label is None:
            if sign and line:
                label = f"{sign}{line}"
            elif sign:
                label = str(sign).lower()
            else:
                return None

        if label == "treble" and octave_change == -1:
            return "treble8vb"
        if label == "treble" and octave_change == 1:
            return "treble8va"
        if label == "bass" and octave_change == -1:
            return "bass8vb"
        return label
