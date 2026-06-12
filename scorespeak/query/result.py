"""Internal bar-retrieval implementation slice."""

from __future__ import annotations

from .common import *


class BarRetrievalResultMixin:
    """Internal mixin for ScoreSpeak bar retrieval."""

    def _empty_bar_result_set(self) -> BarResultSet:
        """Return an empty bar payload with the normal shared schemas."""
        return {
            "event_schema": list(EVENT_SCHEMA),
            "tuplet_schema": list(TUPLET_SCHEMA),
            "marking_schema": list(MARKING_SCHEMA),
            "span_schema": list(SPAN_SCHEMA),
            "bar_notation_keys": list(BAR_NOTATION_KEYS),
            "part_notation_keys": list(PART_NOTATION_KEYS),
            "bars": [],
        }


    def _build_bar_result_set(
        self,
        query: Optional[BarQueryInput] = None,
    ) -> BarResultSet:
        """Build compact bar payloads for the given structured query.

        Args:
            query: Optional structured query dict with `scope`, `match`,
                and `options` sections.

        Returns:
            Compact `BarResultSet` with shared schemas and bar payloads.
        """
        parsed_query = self._parse_bar_query(query)
        bar_parts_by_measure, voice_sequences = self._build_bar_payloads(
            parsed_query.parts,
            parsed_query.measure_numbers,
            parsed_query.scoped_voices,
        )

        if parsed_query.sequence is None:
            qualifying_measures = set(parsed_query.measure_numbers)
        else:
            qualifying_measures = self._find_qualifying_measures(
                voice_sequences,
                parsed_query.sequence,
                parsed_query.chord_mode,
            )

        qualifying_measure_numbers = [
            measure_number
            for measure_number in parsed_query.measure_numbers
            if measure_number in qualifying_measures
        ]

        bar_notations = self._build_bar_notations(
            parsed_query.parts,
            qualifying_measure_numbers,
        )

        bars: list[BarGroup] = []
        for measure_number in qualifying_measure_numbers:
            bar_group: BarGroup = {
                "measure_number": measure_number,
                "parts": bar_parts_by_measure[measure_number],
                "notation": bar_notations[measure_number],
            }
            bars.append(bar_group)

        return {
            "event_schema": list(EVENT_SCHEMA),
            "tuplet_schema": list(TUPLET_SCHEMA),
            "marking_schema": list(MARKING_SCHEMA),
            "span_schema": list(SPAN_SCHEMA),
            "bar_notation_keys": list(BAR_NOTATION_KEYS),
            "part_notation_keys": list(PART_NOTATION_KEYS),
            "bars": bars,
        }


    def search_score(
        self,
        *,
        parts: Optional[list[Union[int, str]]] = None,
        bar_range: Optional[tuple[int, int]] = None,
        measure_numbers: Optional[list[int]] = None,
        voices: Optional[list[int]] = None,
        event_sequence: Optional[list[BarQueryEvent]] = None,
        event_kind: Optional[str] = None,
        pitch: Optional[PitchInput] = None,
        pitch_class: Optional[str] = None,
        duration: Optional[DurationInput] = None,
        beat: Optional[float] = None,
        tie_status: Optional[str] = None,
        is_grace: Optional[bool] = None,
        dots: Optional[int] = None,
        tuplet_ratio: Optional[tuple[int, int]] = None,
        chord_mode: str = "exact",
        marking_type: Optional[str] = None,
        marking_value: Optional[str] = None,
        lyric_text: Optional[str] = None,
        span_type: Optional[str] = None,
        span_value: Optional[str] = None,
        structure: Optional[str] = None,
        structure_value: Optional[str] = None,
        time_signature: Optional[str] = None,
        key_signature: Optional[str] = None,
        tempo: Optional[float] = None,
        clef: Optional[str] = None,
        changed_attribute: Optional[str] = None,
        logic: str = "all",
        limit: Optional[int] = None,
    ) -> BarResultSet:
        """Search the score with explicit semantic filters.

        Args:
            parts: Optional part indices or names.
            bar_range: Optional inclusive ``(start, end)`` measure range.
            measure_numbers: Optional disconnected measure list.
            voices: Optional voice numbers.
            event_sequence: Optional ordered note/rest/chord sequence using
                the same structured event shape as internal bar queries.
            event_kind: Single event kind: ``note``, ``rest``, or ``chord``.
            pitch: Pitch-with-octave to match on notes or chord members.
            pitch_class: Pitch class to match on notes or chord members.
            duration: Duration to match, as a name or quarter length.
            beat: 1-based event beat to match.
            tie_status: Tie status to match: ``start``, ``continue``,
                ``stop``, or ``none``.
            is_grace: Whether matching events must be grace notes.
            dots: Required dot count.
            tuplet_ratio: Required ``(actual, normal)`` tuplet ratio.
            chord_mode: Event-sequence chord matching mode, ``exact`` or
                ``contains``.
            marking_type: Point marking type, such as ``dynamic`` or
                ``articulation``.
            marking_value: Text/value to find in a marking payload.
            lyric_text: Convenience filter for lyric text.
            span_type: Span type, such as ``slur`` or ``hairpin``.
            span_value: Text/value to find in a span payload.
            structure: Structural field, such as ``repeat_end`` or
                ``ending_number``.
            structure_value: Value to find in a structural field.
            time_signature: Active time signature to match.
            key_signature: Active key string to match.
            tempo: Active tempo to match.
            clef: Clef value present at the scoped bar/part.
            changed_attribute: Attribute name present in ``changed_here``.
            logic: ``"all"`` requires every supplied filter family to match;
                ``"any"`` accepts bars matching at least one family.
            limit: Optional maximum number of matching bars to hydrate and
                return. ``None`` returns all matches.

        Returns:
            A compact ``BarResultSet`` containing qualifying bars plus
            ``matches`` rows explaining why each bar matched.
        """
        normalized_logic = logic.strip().lower()
        if normalized_logic not in {"all", "any"}:
            raise ValueError("logic must be either 'all' or 'any'.")
        if chord_mode not in SUPPORTED_CHORD_MODES:
            raise ValueError(
                f"chord_mode must be one of {sorted(SUPPORTED_CHORD_MODES)}."
            )
        normalized_limit = self._normalize_search_limit(limit)

        scope_query = self._search_scope_query(
            parts=parts,
            bar_range=bar_range,
            measure_numbers=measure_numbers,
            voices=voices,
        )
        criteria = self._search_criteria_names(
            event_sequence=event_sequence,
            event_kind=event_kind,
            pitch=pitch,
            pitch_class=pitch_class,
            duration=duration,
            beat=beat,
            tie_status=tie_status,
            is_grace=is_grace,
            dots=dots,
            tuplet_ratio=tuplet_ratio,
            marking_type=marking_type,
            marking_value=marking_value,
            lyric_text=lyric_text,
            span_type=span_type,
            span_value=span_value,
            structure=structure,
            structure_value=structure_value,
            time_signature=time_signature,
            key_signature=key_signature,
            tempo=tempo,
            clef=clef,
            changed_attribute=changed_attribute,
        )

        parsed_query = self._parse_bar_query(scope_query)
        if not criteria:
            return self._full_search_result_for_measures(
                parts=parts,
                voices=voices,
                parsed_query=parsed_query,
                limit=normalized_limit,
            )

        parsed_sequence = self._parse_match_sequence(
            {"sequence": event_sequence}
            if event_sequence is not None
            else None
        )
        payload_options = self._search_payload_options(
            event_sequence=event_sequence,
            event_kind=event_kind,
            pitch=pitch,
            pitch_class=pitch_class,
            duration=duration,
            beat=beat,
            tie_status=tie_status,
            is_grace=is_grace,
            dots=dots,
            tuplet_ratio=tuplet_ratio,
            marking_type=marking_type,
            marking_value=marking_value,
            lyric_text=lyric_text,
            span_type=span_type,
            span_value=span_value,
            clef=clef,
        )
        include_bar_notation = self._has_structure_criteria(
            structure,
            structure_value,
        ) or self._has_attribute_criteria(
            time_signature,
            key_signature,
            tempo,
            None,
            changed_attribute,
        )
        scan_result, voice_sequences = self._build_search_scan_result(
            parsed_query,
            payload_options=payload_options,
            include_bar_notation=include_bar_notation,
        )
        sequence_measure_numbers = (
            self._find_qualifying_measures(
                voice_sequences,
                parsed_sequence,
                chord_mode,
            )
            if parsed_sequence is not None
            else set()
        )

        matches_by_measure: dict[int, list[list[Any]]] = {}
        matching_measure_numbers: list[int] = []
        for bar in scan_result["bars"]:
            matches_by_criterion: dict[str, list[list[Any]]] = {}
            measure_number = int(bar.get("measure_number", 0))

            if event_sequence is not None:
                if measure_number in sequence_measure_numbers:
                    matches_by_criterion["sequence"] = [[
                        "sequence",
                        "event_sequence",
                        None,
                        None,
                        None,
                        None,
                    ]]
                else:
                    matches_by_criterion["sequence"] = []

            event_matches = self._event_search_matches(
                bar,
                event_kind=event_kind,
                pitch=pitch,
                pitch_class=pitch_class,
                duration=duration,
                beat=beat,
                tie_status=tie_status,
                is_grace=is_grace,
                dots=dots,
                tuplet_ratio=tuplet_ratio,
            )
            if self._has_event_criteria(
                event_kind, pitch, pitch_class, duration, beat,
                tie_status, is_grace, dots, tuplet_ratio,
            ):
                matches_by_criterion["event"] = event_matches

            marking_matches = self._marking_search_matches(
                bar,
                marking_type=marking_type,
                marking_value=marking_value,
                lyric_text=lyric_text,
            )
            if self._has_marking_criteria(marking_type, marking_value, lyric_text):
                matches_by_criterion["marking"] = marking_matches

            span_matches = self._span_search_matches(
                bar,
                span_type=span_type,
                span_value=span_value,
            )
            if self._has_span_criteria(span_type, span_value):
                matches_by_criterion["span"] = span_matches

            structure_matches = self._structure_search_matches(
                bar,
                structure=structure,
                structure_value=structure_value,
            )
            if self._has_structure_criteria(structure, structure_value):
                matches_by_criterion["structure"] = structure_matches

            attribute_matches = self._attribute_search_matches(
                bar,
                time_signature=time_signature,
                key_signature=key_signature,
                tempo=tempo,
                clef=clef,
                changed_attribute=changed_attribute,
            )
            if self._has_attribute_criteria(
                time_signature, key_signature, tempo, clef, changed_attribute
            ):
                matches_by_criterion["attribute"] = attribute_matches

            criterion_matches = [
                matches_by_criterion.get(name, [])
                for name in criteria
            ]
            if normalized_logic == "all":
                qualifies = all(bool(matches) for matches in criterion_matches)
            else:
                qualifies = any(bool(matches) for matches in criterion_matches)
            if not qualifies:
                continue

            combined_matches: list[list[Any]] = []
            for matches in criterion_matches:
                combined_matches.extend(matches)
            matches_by_measure[measure_number] = combined_matches
            matching_measure_numbers.append(measure_number)

        result = self._hydrate_search_matches(
            parts=parts,
            voices=voices,
            matching_measure_numbers=matching_measure_numbers,
            matches_by_measure=matches_by_measure,
            limit=normalized_limit,
        )
        result["match_schema"] = list(MATCH_SCHEMA)
        return result


    def _hydrate_search_matches(
        self,
        *,
        parts: Optional[list[Union[int, str]]],
        voices: Optional[list[int]],
        matching_measure_numbers: list[int],
        matches_by_measure: dict[int, list[list[Any]]],
        limit: Optional[int],
    ) -> BarResultSet:
        """Hydrate matching measure numbers into the public search payload."""
        returned_measure_numbers = self._limited_measure_numbers(
            matching_measure_numbers,
            limit,
        )
        result: BarResultSet = {
            "event_schema": list(EVENT_SCHEMA),
            "tuplet_schema": list(TUPLET_SCHEMA),
            "marking_schema": list(MARKING_SCHEMA),
            "span_schema": list(SPAN_SCHEMA),
            "bar_notation_keys": list(BAR_NOTATION_KEYS),
            "part_notation_keys": list(PART_NOTATION_KEYS),
            "bars": [],
        }
        self._attach_search_metadata(
            result,
            total_matches=len(matching_measure_numbers),
            returned_matches=len(returned_measure_numbers),
            limit=limit,
        )
        if not returned_measure_numbers:
            return result

        hydrate_query = self._search_scope_query(
            parts=parts,
            bar_range=None,
            measure_numbers=returned_measure_numbers,
            voices=voices,
        )
        hydrated = self._build_bar_result_set(hydrate_query)
        for bar in hydrated["bars"]:
            measure_number = int(bar.get("measure_number", 0))
            bar["matches"] = matches_by_measure.get(measure_number, [])
            result["bars"].append(bar)
        return result


    def _full_search_result_for_measures(
        self,
        *,
        parts: Optional[list[Union[int, str]]],
        voices: Optional[list[int]],
        parsed_query: ParsedBarQuery,
        limit: Optional[int],
    ) -> BarResultSet:
        """Return an unfiltered bar result, optionally limited by measure."""
        returned_measure_numbers = self._limited_measure_numbers(
            parsed_query.measure_numbers,
            limit,
        )
        query = self._search_scope_query(
            parts=parts,
            bar_range=None,
            measure_numbers=returned_measure_numbers,
            voices=voices,
        )
        if returned_measure_numbers:
            result = self._build_bar_result_set(query)
        else:
            result = self._empty_bar_result_set()
        self._attach_search_metadata(
            result,
            total_matches=len(parsed_query.measure_numbers),
            returned_matches=len(returned_measure_numbers),
            limit=limit,
        )
        return result


    @staticmethod
    def _normalize_search_limit(limit: Optional[int]) -> Optional[int]:
        """Validate an optional search result limit."""
        if limit is None:
            return None
        normalized = int(limit)
        if normalized < 1:
            raise ValueError("limit must be >= 1 when provided.")
        return normalized


    @staticmethod
    def _limited_measure_numbers(
        measure_numbers: list[int],
        limit: Optional[int],
    ) -> list[int]:
        """Return measure numbers capped by ``limit`` while preserving order."""
        if limit is None:
            return list(measure_numbers)
        return list(measure_numbers[:limit])


    @staticmethod
    def _attach_search_metadata(
        result: BarResultSet,
        *,
        total_matches: int,
        returned_matches: int,
        limit: Optional[int],
    ) -> None:
        """Attach result-limit metadata when a limit was requested."""
        if limit is None:
            return
        result["search_metadata"] = {
            "limit": limit,
            "total_matches": total_matches,
            "returned_matches": returned_matches,
            "truncated": returned_matches < total_matches,
        }


    def _search_scope_query(
        self,
        *,
        parts: Optional[list[Union[int, str]]],
        bar_range: Optional[tuple[int, int]],
        measure_numbers: Optional[list[int]],
        voices: Optional[list[int]],
    ) -> BarQuery:
        """Build a structured scope query for semantic score search."""
        scope: dict[str, object] = {}
        if parts is not None:
            scope["parts"] = parts
        if bar_range is not None:
            scope["bar_range"] = bar_range
        if measure_numbers is not None:
            scope["measure_numbers"] = measure_numbers
        if voices is not None:
            scope["voices"] = voices
        return {"scope": scope} if scope else {}


    def _search_payload_options(self, **criteria: Any) -> BarPayloadOptions:
        """Return payload-channel options needed by active search filters."""
        include_events = (
            criteria.get("event_sequence") is not None
            or self._has_event_criteria(
                criteria.get("event_kind"),
                criteria.get("pitch"),
                criteria.get("pitch_class"),
                criteria.get("duration"),
                criteria.get("beat"),
                criteria.get("tie_status"),
                criteria.get("is_grace"),
                criteria.get("dots"),
                criteria.get("tuplet_ratio"),
            )
        )
        include_markings = self._has_marking_criteria(
            criteria.get("marking_type"),
            criteria.get("marking_value"),
            criteria.get("lyric_text"),
        )
        include_spans = self._has_span_criteria(
            criteria.get("span_type"),
            criteria.get("span_value"),
        )
        include_tuplets = criteria.get("tuplet_ratio") is not None
        include_part_notation = criteria.get("clef") is not None
        return BarPayloadOptions(
            include_events=include_events or include_markings or include_spans,
            include_part_notation=include_part_notation,
            include_tuplets=include_tuplets,
            include_markings=include_markings,
            include_spans=include_spans,
        )


    def _build_search_scan_result(
        self,
        parsed_query: ParsedBarQuery,
        *,
        payload_options: BarPayloadOptions,
        include_bar_notation: bool,
    ) -> tuple[BarResultSet, dict[tuple[int, int], list[VoiceEvent]]]:
        """Build the minimum payload needed to evaluate search filters."""
        needs_part_payload = (
            payload_options.include_events
            or payload_options.include_part_notation
            or payload_options.include_tuplets
            or payload_options.include_markings
            or payload_options.include_spans
        )
        bar_parts_by_measure: dict[int, list[BarPart]]
        voice_sequences: dict[tuple[int, int], list[VoiceEvent]]
        if needs_part_payload:
            bar_parts_by_measure, voice_sequences = self._build_bar_payloads(
                parsed_query.parts,
                parsed_query.measure_numbers,
                parsed_query.scoped_voices,
                options=payload_options,
            )
        else:
            bar_parts_by_measure = {
                measure_number: []
                for measure_number in parsed_query.measure_numbers
            }
            voice_sequences = {}

        if include_bar_notation:
            bar_notations = self._build_bar_notations(
                parsed_query.parts,
                parsed_query.measure_numbers,
            )
        else:
            bar_notations = {
                measure_number: {}
                for measure_number in parsed_query.measure_numbers
            }

        bars: list[BarGroup] = []
        for measure_number in parsed_query.measure_numbers:
            bars.append({
                "measure_number": measure_number,
                "parts": bar_parts_by_measure[measure_number],
                "notation": bar_notations[measure_number],
            })

        return {
            "event_schema": list(EVENT_SCHEMA),
            "tuplet_schema": list(TUPLET_SCHEMA),
            "marking_schema": list(MARKING_SCHEMA),
            "span_schema": list(SPAN_SCHEMA),
            "bar_notation_keys": list(BAR_NOTATION_KEYS),
            "part_notation_keys": list(PART_NOTATION_KEYS),
            "bars": bars,
        }, voice_sequences


    def _search_criteria_names(self, **criteria: Any) -> list[str]:
        """Return the active search criterion families in evaluation order."""
        names: list[str] = []
        if criteria.get("event_sequence") is not None:
            names.append("sequence")
        if self._has_event_criteria(
            criteria.get("event_kind"),
            criteria.get("pitch"),
            criteria.get("pitch_class"),
            criteria.get("duration"),
            criteria.get("beat"),
            criteria.get("tie_status"),
            criteria.get("is_grace"),
            criteria.get("dots"),
            criteria.get("tuplet_ratio"),
        ):
            names.append("event")
        if self._has_marking_criteria(
            criteria.get("marking_type"),
            criteria.get("marking_value"),
            criteria.get("lyric_text"),
        ):
            names.append("marking")
        if self._has_span_criteria(
            criteria.get("span_type"),
            criteria.get("span_value"),
        ):
            names.append("span")
        if self._has_structure_criteria(
            criteria.get("structure"),
            criteria.get("structure_value"),
        ):
            names.append("structure")
        if self._has_attribute_criteria(
            criteria.get("time_signature"),
            criteria.get("key_signature"),
            criteria.get("tempo"),
            criteria.get("clef"),
            criteria.get("changed_attribute"),
        ):
            names.append("attribute")
        return names
