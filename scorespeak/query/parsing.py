"""Internal bar-retrieval implementation slice."""

from __future__ import annotations

from .common import *


class BarQueryParsingMixin:
    """Internal mixin for ScoreSpeak bar retrieval."""

    def _parse_bar_query(
        self,
        query: Optional[BarQueryInput],
    ) -> ParsedBarQuery:
        """Validate and normalize an internal structured bar-result query."""
        if not list(self._score.parts):
            raise ValueError(
                "Score is empty: no parts available for bar-result projection."
            )

        if query is None:
            query_dict: BarQuery = {}
        elif isinstance(query, dict):
            query_dict = query
        else:
            raise TypeError(
                "query must be a dict with optional 'scope', 'match', and "
                "'options' sections."
            )
        self._validate_bar_query_keys(query_dict)

        scope = query_dict.get("scope", {})
        if scope is None:
            scope = {}
        if not isinstance(scope, dict):
            raise TypeError("query.scope must be a dict when provided.")

        options = query_dict.get("options", {})
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise TypeError("query.options must be a dict when provided.")

        parts = self._resolve_scope_parts(scope.get("parts"))
        measure_numbers = self._resolve_scope_measures(
            parts,
            scope.get("bar_range"),
            scope.get("measure_numbers"),
        )
        scoped_voices = self._resolve_scope_voices(scope.get("voices"))

        chord_mode = options.get("chord_mode", "exact")
        if chord_mode not in SUPPORTED_CHORD_MODES:
            raise ValueError(
                f"query.options.chord_mode must be one of "
                f"{sorted(SUPPORTED_CHORD_MODES)}, got {chord_mode!r}."
            )

        match_section = query_dict.get("match")
        sequence = self._parse_match_sequence(match_section)

        return ParsedBarQuery(
            parts=parts,
            measure_numbers=measure_numbers,
            scoped_voices=scoped_voices,
            sequence=sequence,
            chord_mode=chord_mode,
        )


    @staticmethod
    def _validate_bar_query_keys(query_dict: BarQuery) -> None:
        """Reject unsupported structured bar query fields."""
        allowed_top = {"scope", "match", "options"}
        extra_top = sorted(set(query_dict) - allowed_top)
        if extra_top:
            raise ValueError(
                "query contains unsupported top-level field(s): "
                + ", ".join(extra_top)
            )

        scope = query_dict.get("scope")
        if isinstance(scope, dict):
            extra_scope = sorted(
                set(scope) - {"parts", "bar_range", "measure_numbers", "voices"}
            )
            if extra_scope:
                raise ValueError(
                    "query.scope contains unsupported field(s): "
                    + ", ".join(extra_scope)
                )

        match = query_dict.get("match")
        if isinstance(match, dict):
            extra_match = sorted(set(match) - {"sequence"})
            if extra_match:
                raise ValueError(
                    "query.match contains unsupported field(s): "
                    + ", ".join(extra_match)
                )

        options = query_dict.get("options")
        if isinstance(options, dict):
            extra_options = sorted(set(options) - {"chord_mode"})
            if extra_options:
                raise ValueError(
                    "query.options contains unsupported field(s): "
                    + ", ".join(extra_options)
                )


    def _resolve_scope_parts(
        self,
        part_ids: Optional[list[Union[int, str]]],
    ) -> list[tuple[m21stream.Part, int]]:
        """Resolve the scoped parts for bar-result projection."""
        if part_ids is None:
            parts = list(self._score.parts)
            if not parts:
                raise ValueError(
                    "Score is empty: no parts available for bar-result projection."
                )
            return [(part_obj, idx) for idx, part_obj in enumerate(parts)]

        if not isinstance(part_ids, list):
            raise TypeError("query.scope.parts must be a list when provided.")
        if not part_ids:
            raise ValueError("query.scope.parts must not be empty.")

        resolved_by_index: dict[int, tuple[m21stream.Part, int]] = {}
        for part_id in part_ids:
            try:
                part_obj, part_idx = self._resolve_part(part_id)
            except ValueError as exc:
                raise ValueError(
                    f"query.scope.parts contains an unresolvable part: {part_id!r}."
                ) from exc
            resolved_by_index[part_idx] = (part_obj, part_idx)

        return [resolved_by_index[idx] for idx in sorted(resolved_by_index)]


    def _resolve_scope_measures(
        self,
        parts: list[tuple[m21stream.Part, int]],
        bar_range: Optional[tuple[int, int]],
        measure_numbers: Optional[list[int]],
    ) -> list[int]:
        """Resolve and clamp the scoped measure numbers."""
        if bar_range is not None and measure_numbers is not None:
            raise ValueError(
                "query.scope may not provide both 'bar_range' and "
                "'measure_numbers'."
            )

        max_measure = max(self._get_measure_count(part_obj) for part_obj, _ in parts)
        if max_measure <= 0:
            return []

        if measure_numbers is not None:
            if not isinstance(measure_numbers, list):
                raise TypeError(
                    "query.scope.measure_numbers must be a list of integers."
                )
            if not measure_numbers:
                raise ValueError("query.scope.measure_numbers must not be empty.")

            clamped_measure_numbers: set[int] = set()
            for measure_number in measure_numbers:
                if not isinstance(measure_number, int):
                    raise TypeError(
                        "query.scope.measure_numbers entries must be integers."
                    )
                if measure_number < 1:
                    raise ValueError(
                        "query.scope.measure_numbers entries must be >= 1."
                    )
                if measure_number > max_measure:
                    continue
                clamped_measure_numbers.add(measure_number)

            return sorted(clamped_measure_numbers)

        if bar_range is not None:
            if (
                not isinstance(bar_range, tuple)
                or len(bar_range) != 2
                or not all(isinstance(value, int) for value in bar_range)
            ):
                raise TypeError(
                    "query.scope.bar_range must be a tuple of two integers."
                )
            start, end = bar_range
            if start > end:
                raise ValueError(
                    f"query.scope.bar_range is invalid: start {start} > end {end}."
                )
        else:
            start, end = 1, max(self._get_measure_count(part_obj) for part_obj, _ in parts)

        clamped_start = max(1, start)
        clamped_end = min(end, max_measure)
        if clamped_end < clamped_start:
            return []
        return list(range(clamped_start, clamped_end + 1))


    def _resolve_scope_voices(
        self,
        voices: Optional[list[int]],
    ) -> Optional[set[int]]:
        """Resolve the scoped voices for bar-result projection."""
        if voices is None:
            return None
        if not isinstance(voices, list):
            raise TypeError("query.scope.voices must be a list when provided.")
        if not voices:
            raise ValueError("query.scope.voices must not be empty.")

        resolved: set[int] = set()
        for voice in voices:
            resolved.add(validate_voice_number(voice, "query.scope.voices entry"))
        return resolved


    def _parse_match_sequence(
        self,
        match_section: Optional[object],
    ) -> Optional[list[ParsedQueryEvent]]:
        """Validate and normalize the optional match sequence."""
        if match_section is None:
            return None
        if not isinstance(match_section, dict):
            raise TypeError("query.match must be a dict when provided.")

        sequence = match_section.get("sequence")
        if sequence is None:
            return None
        if not isinstance(sequence, list):
            raise TypeError("query.match.sequence must be a list when provided.")
        if not sequence:
            raise ValueError("query.match.sequence must not be empty.")

        return [
            self._parse_query_event(event, idx)
            for idx, event in enumerate(sequence)
        ]


    def _parse_query_event(
        self,
        event: BarQueryEvent,
        index: int,
    ) -> ParsedQueryEvent:
        """Validate and normalize one query sequence element."""
        if not isinstance(event, dict):
            raise TypeError(
                f"query.match.sequence[{index}] must be a dict."
            )

        kind = event.get("kind")
        if not isinstance(kind, str):
            raise ValueError(
                f"query.match.sequence[{index}].kind must be a string."
            )
        if kind not in SUPPORTED_EVENT_KINDS:
            raise ValueError(
                f"query.match.sequence[{index}].kind must be one of "
                f"{sorted(SUPPORTED_EVENT_KINDS)}, got {kind!r}."
            )

        duration_value = event.get("duration")
        duration: Optional[float] = None
        if duration_value is not None:
            try:
                duration = float(normalize_duration(duration_value).quarterLength)
            except ValueError as exc:
                raise ValueError(
                    f"query.match.sequence[{index}].duration is invalid: "
                    f"{duration_value!r}."
                ) from exc

        if kind == "note":
            if "pitch" not in event:
                raise ValueError(
                    f"query.match.sequence[{index}] note events require 'pitch'."
                )
            try:
                pitch = self._format_pitch_with_octave(event["pitch"])
            except ValueError as exc:
                raise ValueError(
                    f"query.match.sequence[{index}].pitch is invalid: "
                    f"{event['pitch']!r}."
                ) from exc
            return ParsedQueryEvent(kind="note", pitch=pitch, duration=duration)

        if kind == "rest":
            return ParsedQueryEvent(kind="rest", duration=duration)

        if kind == "chord":
            pitch_classes_raw = event.get("pitch_classes")
            if not isinstance(pitch_classes_raw, list) or not pitch_classes_raw:
                raise ValueError(
                    f"query.match.sequence[{index}] chord events require a "
                    f"non-empty 'pitch_classes' list."
                )
            pitch_classes = [
                self._normalize_pitch_class(value, f"query.match.sequence[{index}].pitch_classes")
                for value in pitch_classes_raw
            ]
            return ParsedQueryEvent(
                kind="chord",
                pitch_classes=self._unique_preserving_order(pitch_classes),
                duration=duration,
            )

        return ParsedQueryEvent(kind="any", duration=duration)
