"""Internal bar-retrieval implementation slice."""

from __future__ import annotations

from .common import *


class BarMarkingPayloadMixin:
    """Internal mixin for ScoreSpeak bar retrieval."""

    def _collect_voice_markings(
        self,
        measure_obj: m21stream.Measure,
        stream_obj: m21stream.Stream,
        events: list[VoiceEvent],
        voice_id: int,
    ) -> list[MarkingRow]:
        """Collect point-event notations attached to events in this voice.

        Per-note attachments (articulations, ornaments, lyrics,
        fingerings, arpeggios, fermatas) are read from each event's
        source ``GeneralNote``. Measure-level markings (dynamics,
        chord symbols, text expressions) are reported only on voice 1
        at their actual measure beat.
        """
        if not events:
            return []

        rows: list[MarkingRow] = []

        for event in events:
            source = event._source
            if source is None:
                continue
            rows.extend(
                self._extract_point_markings_from_note(
                    source,
                    event.beat,
                )
            )

        if voice_id == 1:
            rows.extend(
                self._collect_measure_level_markings(measure_obj)
            )

        rows.sort(key=lambda row: (row[2], row[0]))
        return rows


    def _extract_point_markings_from_note(
        self,
        general_note: m21note.GeneralNote,
        beat: float,
    ) -> list[MarkingRow]:
        """Extract every marking attached directly to one GeneralNote."""
        rows: list[MarkingRow] = []

        for articulation in getattr(general_note, "articulations", []) or []:
            if isinstance(articulation, _FINGERING_CLASSES):
                rows.append([
                    "fingering",
                    self._format_fingering_payload(articulation),
                    float(beat),
                ])
                continue
            rows.append([
                "articulation",
                self._format_articulation_payload(articulation),
                float(beat),
            ])

        for expression in getattr(general_note, "expressions", []) or []:
            if isinstance(expression, m21expressions.TextExpression):
                content = self._format_text_expression_payload(expression)
                if content:
                    rows.append([
                        "text_expression",
                        content,
                        float(beat),
                    ])
                continue
            if isinstance(expression, _FERMATA_CLASSES):
                rows.append([
                    "articulation",
                    "fermata",
                    float(beat),
                ])
                continue
            if isinstance(expression, m21expressions.ArpeggioMark):
                rows.append([
                    "arpeggio",
                    self._format_arpeggio_payload(expression),
                    float(beat),
                ])
                continue
            if isinstance(expression, _ORNAMENT_CLASSES):
                rows.append([
                    "ornament",
                    self._format_ornament_payload(expression),
                    float(beat),
                ])
                continue

        for lyric in getattr(general_note, "lyrics", []) or []:
            rows.append([
                "lyric",
                self._format_lyric_payload(lyric),
                float(beat),
            ])

        return rows


    def _collect_measure_level_markings(
        self,
        measure_obj: m21stream.Measure,
    ) -> list[MarkingRow]:
        """Collect measure-attached dynamics, chord symbols, and text."""
        rows: list[MarkingRow] = []

        for dyn in measure_obj.getElementsByClass(m21dynamics.Dynamic):
            offset = float(measure_obj.elementOffset(dyn))
            rows.append(["dynamic", dyn.value, offset + 1.0])

        for symbol in measure_obj.getElementsByClass(m21harmony.ChordSymbol):
            offset = float(measure_obj.elementOffset(symbol))
            figure = getattr(symbol, "figure", None) or str(symbol)
            rows.append(["chord_symbol", figure, offset + 1.0])

        for text_el in measure_obj.getElementsByClass(
            m21expressions.TextExpression
        ):
            try:
                offset = float(measure_obj.elementOffset(text_el))
            except Exception:
                continue
            content = self._format_text_expression_payload(text_el)
            if not content:
                continue
            rows.append(["text_expression", content, offset + 1.0])

        return rows


    @staticmethod
    def _format_text_expression_payload(expression: object) -> str:
        """Return the compact text-expression content."""
        return str(getattr(expression, "content", "") or "").strip()


    @staticmethod
    def _format_articulation_payload(articulation: object) -> str:
        """Return a short label for an articulation instance."""
        display = getattr(articulation, "name", None)
        if display:
            return str(display).strip().lower()
        return type(articulation).__name__.lower()


    @staticmethod
    def _format_ornament_payload(ornament: object) -> object:
        """Return a compact label (or dict for tremolo) for an ornament."""
        if isinstance(ornament, m21expressions.Tremolo):
            marks = getattr(ornament, "numberOfMarks", None)
            if marks is not None:
                return {"type": "tremolo", "marks": int(marks)}
            return "tremolo"
        display = getattr(ornament, "name", None)
        if display:
            return str(display).strip().lower()
        return type(ornament).__name__.lower()


    @staticmethod
    def _format_fingering_payload(fingering: object) -> object:
        """Return a compact payload for a fingering articulation."""
        value = getattr(fingering, "fingerNumber", None)
        substitution = bool(getattr(fingering, "substitution", False))
        if not substitution:
            return value if value is not None else ""
        return {"finger": value, "substitution": True}


    @staticmethod
    def _format_arpeggio_payload(arpeggio: object) -> object:
        """Return a compact payload for an ArpeggioMark."""
        arp_type = getattr(arpeggio, "type", None)
        if arp_type:
            return {"direction": str(arp_type)}
        return "arpeggio"


    @staticmethod
    def _format_lyric_payload(lyric: object) -> dict:
        """Return a compact payload for a Lyric."""
        text = str(getattr(lyric, "text", "") or "")
        payload: dict = {"text": text}
        number = getattr(lyric, "number", None)
        if number is not None and int(number) != 1:
            payload["number"] = int(number)
        syllabic = getattr(lyric, "syllabic", None)
        if syllabic:
            payload["syllabic"] = str(syllabic)
        return payload


    @staticmethod
    def _event_index_near_offset(
        events: list[VoiceEvent],
        offset: float,
    ) -> Optional[int]:
        """Return the 1-based index of the event at/before ``offset``.

        Returns ``None`` if the voice has no events. If every event
        starts after ``offset`` (marking orphaned before any note), the
        first event's index (``1``) is returned so the marking anchors
        at the top of the voice.
        """
        if not events:
            return None

        best: Optional[int] = None
        for index, event in enumerate(events, start=1):
            if event.offset <= offset + 1e-9:
                best = index
            else:
                break

        if best is None:
            return 1
        return best


    def _collect_voice_spans(
        self,
        part_obj: m21stream.Part,
        events: list[VoiceEvent],
        measure_number: int,
        voice_id: int,
        scope_measures: list[int],
    ) -> list[SpanRow]:
        """Collect spanner notations that touch this bar / voice.

        Each span is truncated to the current bar and tagged with a
        flags string: ``"L"`` if it began in an earlier bar (or before
        the retrieved scope), ``"R"`` if it continues into a later bar
        (or past the scope), ``"LR"`` for both, empty otherwise.
        """
        if not events:
            return []

        rows: list[SpanRow] = []
        seen_spanner_ids: set[int] = set()
        scope_first = min(scope_measures) if scope_measures else measure_number
        scope_last = max(scope_measures) if scope_measures else measure_number

        for span_cls, type_label in _SPAN_TYPES:
            for spanner in part_obj.getElementsByClass(span_cls):
                spanner_id = id(spanner)
                if spanner_id in seen_spanner_ids:
                    continue
                seen_spanner_ids.add(spanner_id)
                row = self._span_row_for_bar(
                    spanner,
                    type_label,
                    events,
                    measure_number,
                    voice_id,
                    scope_first,
                    scope_last,
                )
                if row is not None:
                    rows.append(row)

        rows.sort(key=lambda row: (row[3][0], row[3][1], row[0]))
        return rows


    def _span_row_for_bar(
        self,
        spanner: m21spanner.Spanner,
        type_label: str,
        events: list[VoiceEvent],
        measure_number: int,
        voice_id: int,
        scope_first: int,
        scope_last: int,
    ) -> Optional[SpanRow]:
        """Return one per-bar span row if ``spanner`` touches this bar."""
        spanned = spanner.getSpannedElements()
        if len(spanned) < 2:
            return None

        first_el = spanned[0]
        last_el = spanned[-1]

        first_measure = first_el.getContextByClass(m21stream.Measure)
        last_measure = last_el.getContextByClass(m21stream.Measure)
        if first_measure is None or last_measure is None:
            return None

        physical_start_measure = int(first_measure.number)
        physical_end_measure = int(last_measure.number)
        start_measure = self._spanner_int_attr(
            spanner,
            "scorespeak_start_measure",
            physical_start_measure,
        )
        end_measure = self._spanner_int_attr(
            spanner,
            "scorespeak_end_measure",
            physical_end_measure,
        )
        if measure_number < start_measure or measure_number > end_measure:
            return None

        if self._span_voice_id(first_el) != voice_id:
            return None

        start_index: int
        logical_start_beat = self._spanner_float_attr(
            spanner,
            "scorespeak_start_beat",
        )
        start_beat: float
        if measure_number == physical_start_measure:
            try:
                start_offset = float(first_measure.elementOffset(first_el))
            except Exception:
                return None
            maybe_index = self._event_index_near_offset(events, start_offset)
            if maybe_index is None:
                return None
            start_index = maybe_index
            start_beat = start_offset + 1.0
        else:
            start_index = 1
            start_beat = self._event_beat_range(
                events,
                start_index,
                start_index,
            )[0]
        if measure_number == start_measure and logical_start_beat is not None:
            start_beat = logical_start_beat

        end_index: int
        end_beat: Optional[float] = None
        ends_at_measure_right_edge = False
        logical_end_beat = self._spanner_float_attr(
            spanner,
            "scorespeak_end_beat",
        )
        logical_end_offset = None
        if logical_end_beat is not None:
            logical_end_offset = logical_end_beat - 1.0
        if measure_number == physical_end_measure:
            try:
                physical_end_offset = float(last_measure.elementOffset(last_el))
            except Exception:
                return None
            end_extent = physical_end_offset + self._element_quarter_length(last_el)
            try:
                bar_duration = float(last_measure.barDuration.quarterLength)
            except Exception:
                bar_duration = 0.0
            ends_at_measure_right_edge = (
                bar_duration > 0.0 and end_extent >= bar_duration - 1e-9
            )
            if measure_number < end_measure:
                end_index = len(events)
            else:
                end_offset = physical_end_offset
                if measure_number == end_measure and logical_end_offset is not None:
                    end_offset = logical_end_offset
                maybe_end = self._event_index_near_offset(events, end_offset)
                if maybe_end is None:
                    end_index = len(events)
                else:
                    end_index = maybe_end
                end_beat = end_offset + 1.0
        elif measure_number == end_measure:
            end_offset = 0.0
            if logical_end_offset is not None:
                end_offset = logical_end_offset
            maybe_end = self._event_index_near_offset(events, end_offset)
            if maybe_end is None:
                end_index = 1
            else:
                end_index = maybe_end
            end_beat = end_offset + 1.0
        else:
            end_index = len(events)

        if end_index < start_index:
            end_index = start_index

        if end_beat is None:
            end_beat = self._event_beat_range(events, end_index, end_index)[0]

        flags_parts: list[str] = []
        if measure_number > start_measure or start_measure < scope_first:
            flags_parts.append("L")
        if (
            measure_number < end_measure
            or end_measure > scope_last
            or ends_at_measure_right_edge
        ):
            flags_parts.append("R")
        flags = "".join(flags_parts)

        payload = self._format_span_payload(spanner, type_label)
        return [
            type_label,
            payload,
            flags,
            [float(start_beat), float(end_beat)],
        ]


    @staticmethod
    def _spanner_int_attr(
        spanner: m21spanner.Spanner,
        attr_name: str,
        default: int,
    ) -> int:
        """Return an integer spanner attribute or a fallback value."""
        value = getattr(spanner, attr_name, None)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


    @staticmethod
    def _spanner_float_attr(
        spanner: m21spanner.Spanner,
        attr_name: str,
    ) -> Optional[float]:
        """Return a float spanner attribute when it can be parsed."""
        value = getattr(spanner, attr_name, None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


    @staticmethod
    def _element_quarter_length(element: object) -> float:
        """Return an element duration in quarter lengths if available."""
        value = getattr(element, "quarterLength", 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


    @staticmethod
    def _span_voice_id(element: object) -> int:
        """Return the voice id containing ``element`` (default 1)."""
        parent = getattr(element, "activeSite", None)
        while parent is not None and not isinstance(
            parent, (m21stream.Voice, m21stream.Measure)
        ):
            parent = getattr(parent, "activeSite", None)
        if isinstance(parent, m21stream.Voice):
            raw_id = getattr(parent, "id", None)
            try:
                return int(raw_id)
            except (TypeError, ValueError):
                return 1
        return 1


    @staticmethod
    def _format_span_payload(
        spanner: m21spanner.Spanner,
        type_label: str,
    ) -> object:
        """Return a compact payload string or dict for a spanner."""
        if type_label == "hairpin":
            if isinstance(spanner, m21dynamics.Crescendo):
                return "crescendo"
            if isinstance(spanner, m21dynamics.Diminuendo):
                return "diminuendo"
            return "hairpin"
        if type_label == "slur":
            return ""
        if type_label == "ottava":
            return getattr(spanner, "type", None) or "8va"
        if type_label == "glissando":
            line_type = getattr(spanner, "lineType", None)
            label = getattr(spanner, "label", None)
            if label:
                return {"line_type": line_type, "label": label}
            return line_type or "wavy"
        if type_label == "pedal":
            return ""
        return ""
