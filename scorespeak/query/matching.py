"""Internal bar-retrieval implementation slice."""

from __future__ import annotations

from .common import *


class BarSearchMatchingMixin:
    """Internal mixin for ScoreSpeak bar retrieval."""

    @staticmethod
    def _has_event_criteria(*values: object) -> bool:
        """Return True when any event predicate is active."""
        return any(value is not None for value in values)


    @staticmethod
    def _has_marking_criteria(
        marking_type: Optional[str],
        marking_value: Optional[str],
        lyric_text: Optional[str],
    ) -> bool:
        """Return True when any marking predicate is active."""
        return any(value is not None for value in (marking_type, marking_value, lyric_text))


    @staticmethod
    def _has_span_criteria(
        span_type: Optional[str],
        span_value: Optional[str],
    ) -> bool:
        """Return True when any span predicate is active."""
        return any(value is not None for value in (span_type, span_value))


    @staticmethod
    def _has_structure_criteria(
        structure: Optional[str],
        structure_value: Optional[str],
    ) -> bool:
        """Return True when any structure predicate is active."""
        return any(value is not None for value in (structure, structure_value))


    @staticmethod
    def _has_attribute_criteria(
        time_signature: Optional[str],
        key_signature: Optional[str],
        tempo: Optional[float],
        clef: Optional[str],
        changed_attribute: Optional[str],
    ) -> bool:
        """Return True when any attribute predicate is active."""
        return any(
            value is not None
            for value in (time_signature, key_signature, tempo, clef, changed_attribute)
        )


    def _event_search_matches(
        self,
        bar: BarGroup,
        *,
        event_kind: Optional[str],
        pitch: Optional[PitchInput],
        pitch_class: Optional[str],
        duration: Optional[DurationInput],
        beat: Optional[float],
        tie_status: Optional[str],
        is_grace: Optional[bool],
        dots: Optional[int],
        tuplet_ratio: Optional[tuple[int, int]],
    ) -> list[list[Any]]:
        """Return event match rows for one bar."""
        if not self._has_event_criteria(
            event_kind, pitch, pitch_class, duration, beat,
            tie_status, is_grace, dots, tuplet_ratio,
        ):
            return []

        normalized_kind = event_kind.strip().lower() if event_kind else None
        if normalized_kind is not None and normalized_kind not in {"note", "rest", "chord"}:
            raise ValueError("event_kind must be 'note', 'rest', or 'chord'.")

        normalized_pitch = self._format_pitch_with_octave(pitch) if pitch is not None else None
        normalized_pitch_class = (
            self._normalize_pitch_class(pitch_class, "pitch_class")
            if pitch_class is not None
            else None
        )
        normalized_duration = (
            float(normalize_duration(duration).quarterLength)
            if duration is not None
            else None
        )
        normalized_tie = self._normalize_tie_status(tie_status)
        normalized_ratio = self._normalize_tuplet_ratio(tuplet_ratio)

        matches: list[list[Any]] = []
        detail = self._detail_string(
            "event",
            {
                "kind": normalized_kind,
                "pitch": normalized_pitch,
                "pitch_class": normalized_pitch_class,
                "duration": normalized_duration,
                "beat": beat,
                "tie": normalized_tie,
                "grace": is_grace,
                "dots": dots,
                "tuplet": normalized_ratio,
            },
        )

        for part in bar.get("parts", []):
            part_index = part.get("part_index")
            for voice in part.get("voices", []):
                voice_number = voice.get("voice")
                tuplet_map = self._search_tuplet_ratio_map(voice)
                for event_index, row in enumerate(voice.get("events", []), start=1):
                    if not isinstance(row, list):
                        continue
                    if not self._event_row_matches(
                        row,
                        event_index,
                        tuplet_map,
                        event_kind=normalized_kind,
                        pitch=normalized_pitch,
                        pitch_class=normalized_pitch_class,
                        duration=normalized_duration,
                        beat=beat,
                        tie_status=normalized_tie,
                        is_grace=is_grace,
                        dots=dots,
                        tuplet_ratio=normalized_ratio,
                    ):
                        continue
                    matches.append([
                        "event",
                        detail,
                        part_index,
                        voice_number,
                        row[1],
                        None,
                    ])
        return matches


    def _event_row_matches(
        self,
        row: list[Any],
        event_index: int,
        tuplet_map: dict[int, tuple[int, int]],
        *,
        event_kind: Optional[str],
        pitch: Optional[str],
        pitch_class: Optional[str],
        duration: Optional[float],
        beat: Optional[float],
        tie_status: Optional[str],
        is_grace: Optional[bool],
        dots: Optional[int],
        tuplet_ratio: Optional[tuple[int, int]],
    ) -> bool:
        """Return True when one compact event row satisfies event filters."""
        if len(row) < len(EVENT_SCHEMA):
            return False
        kind = str(row[0])
        if event_kind is not None and kind != event_kind:
            return False
        if beat is not None and abs(float(row[1]) - float(beat)) > 1e-9:
            return False
        if pitch is not None and not self._event_row_has_pitch(row, pitch):
            return False
        if pitch_class is not None and not self._event_row_has_pitch_class(row, pitch_class):
            return False
        if duration is not None and abs(float(row[3]) - duration) > 1e-9:
            return False
        if tie_status is not None:
            current = row[4] if row[4] is not None else "none"
            if str(current).lower() != tie_status:
                return False
        if is_grace is not None and bool(row[5]) != bool(is_grace):
            return False
        if dots is not None and int(row[6]) != int(dots):
            return False
        if tuplet_ratio is not None and tuplet_map.get(event_index) != tuplet_ratio:
            return False
        return True


    @staticmethod
    def _event_row_has_pitch(row: list[Any], pitch: str) -> bool:
        """Return True when an event row contains ``pitch``."""
        payload = row[2]
        if isinstance(payload, list):
            return pitch in {str(item) for item in payload}
        return str(payload) == pitch


    def _event_row_has_pitch_class(self, row: list[Any], pitch_class: str) -> bool:
        """Return True when an event row contains ``pitch_class``."""
        payload = row[2]
        pitches = payload if isinstance(payload, list) else [payload]
        for raw_pitch in pitches:
            if raw_pitch in (None, ""):
                continue
            try:
                candidate = self._normalize_pitch_class(raw_pitch, "event pitch")
            except ValueError:
                continue
            if candidate == pitch_class:
                return True
        return False


    def _marking_search_matches(
        self,
        bar: BarGroup,
        *,
        marking_type: Optional[str],
        marking_value: Optional[str],
        lyric_text: Optional[str],
    ) -> list[list[Any]]:
        """Return point-marking match rows for one bar."""
        if not self._has_marking_criteria(marking_type, marking_value, lyric_text):
            return []
        normalized_type = (
            "lyric" if lyric_text is not None and marking_type is None
            else self._normalize_optional_token(marking_type)
        )
        expected_value = lyric_text if lyric_text is not None else marking_value
        detail = self._detail_string(
            "marking",
            {"type": normalized_type, "value": expected_value},
        )
        matches: list[list[Any]] = []
        for part in bar.get("parts", []):
            part_index = part.get("part_index")
            for voice in part.get("voices", []):
                voice_number = voice.get("voice")
                for row in voice.get("markings", []):
                    if not isinstance(row, list) or len(row) < 3:
                        continue
                    row_type, payload, row_beat = row[:3]
                    if normalized_type is not None and str(row_type) != normalized_type:
                        continue
                    if expected_value is not None and not self._payload_matches(payload, expected_value):
                        continue
                    matches.append([
                        "marking",
                        detail,
                        part_index,
                        voice_number,
                        row_beat,
                        None,
                    ])
        return matches


    def _span_search_matches(
        self,
        bar: BarGroup,
        *,
        span_type: Optional[str],
        span_value: Optional[str],
    ) -> list[list[Any]]:
        """Return span match rows for one bar."""
        if not self._has_span_criteria(span_type, span_value):
            return []
        normalized_type = self._normalize_optional_token(span_type)
        detail = self._detail_string(
            "span",
            {"type": normalized_type, "value": span_value},
        )
        matches: list[list[Any]] = []
        for part in bar.get("parts", []):
            part_index = part.get("part_index")
            for voice in part.get("voices", []):
                voice_number = voice.get("voice")
                for row in voice.get("spans", []):
                    if not isinstance(row, list) or len(row) < 4:
                        continue
                    row_type, payload, _flags, beat_range = row[:4]
                    if normalized_type is not None and str(row_type) != normalized_type:
                        continue
                    if span_value is not None and not self._payload_matches(payload, span_value):
                        continue
                    matches.append([
                        "span",
                        detail,
                        part_index,
                        voice_number,
                        None,
                        beat_range,
                    ])
        return matches


    def _structure_search_matches(
        self,
        bar: BarGroup,
        *,
        structure: Optional[str],
        structure_value: Optional[str],
    ) -> list[list[Any]]:
        """Return structural-notation match rows for one bar."""
        if not self._has_structure_criteria(structure, structure_value):
            return []
        notation = bar.get("notation", {})
        if not isinstance(notation, dict):
            return []
        structural_keys = set(BAR_NOTATION_KEYS) - {"active", "changed_here"}
        key = self._normalize_optional_token(structure)
        if key is not None and key not in structural_keys:
            raise ValueError(
                f"structure must be one of {sorted(structural_keys)}, got {structure!r}."
            )
        candidate_keys = [key] if key is not None else sorted(structural_keys)
        matches: list[list[Any]] = []
        for candidate in candidate_keys:
            value = notation.get(candidate)
            if value in (None, False, "", []):
                continue
            if structure_value is not None and not self._payload_matches(value, structure_value):
                continue
            detail = self._detail_string(
                "structure",
                {"field": candidate, "value": structure_value if structure_value else value},
            )
            matches.append(["structure", detail, None, None, None, None])
        return matches


    def _attribute_search_matches(
        self,
        bar: BarGroup,
        *,
        time_signature: Optional[str],
        key_signature: Optional[str],
        tempo: Optional[float],
        clef: Optional[str],
        changed_attribute: Optional[str],
    ) -> list[list[Any]]:
        """Return attribute match rows for one bar."""
        if not self._has_attribute_criteria(
            time_signature, key_signature, tempo, clef, changed_attribute
        ):
            return []
        matches: list[list[Any]] = []
        notation = bar.get("notation", {})
        active = notation.get("active", {}) if isinstance(notation, dict) else {}
        changed = notation.get("changed_here", []) if isinstance(notation, dict) else []
        if time_signature is not None:
            if active.get("time") != time_signature:
                return []
            matches.append(["attribute", f"time={time_signature}", None, None, None, None])
        if key_signature is not None:
            if not self._payload_matches(active.get("key"), key_signature):
                return []
            matches.append(
                [
                    "attribute",
                    f"concert_key={key_signature}",
                    None,
                    None,
                    None,
                    None,
                ]
            )
        if tempo is not None:
            if active.get("tempo") is None:
                return []
            if abs(float(active["tempo"]) - float(tempo)) >= 1e-9:
                return []
            matches.append(["attribute", f"tempo={tempo:g}", None, None, None, None])
        if changed_attribute is not None:
            normalized_changed = self._normalize_optional_token(changed_attribute)
            if normalized_changed not in {str(item) for item in changed}:
                return []
            matches.append(["attribute", f"changed={normalized_changed}", None, None, None, None])
        if clef is not None:
            normalized_clef = self._normalize_optional_token(clef)
            clef_matches = []
            for part in bar.get("parts", []):
                notation_part = part.get("notation", {})
                if not isinstance(notation_part, dict):
                    continue
                if notation_part.get("clef") != normalized_clef:
                    continue
                clef_matches.append([
                    "attribute",
                    f"clef={normalized_clef}",
                    part.get("part_index"),
                    None,
                    None,
                    None,
                ])
            if not clef_matches:
                return []
            matches.extend(clef_matches)
        return matches


    @staticmethod
    def _normalize_optional_token(value: Optional[str]) -> Optional[str]:
        """Normalize optional search labels to internal underscore tokens."""
        if value is None:
            return None
        return value.strip().lower().replace("-", "_").replace(" ", "_")


    @staticmethod
    def _normalize_tie_status(value: Optional[str]) -> Optional[str]:
        """Normalize optional tie status values."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized == "untied":
            normalized = "none"
        if normalized not in {"start", "continue", "stop", "none"}:
            raise ValueError(
                "tie_status must be 'start', 'continue', 'stop', 'none', or 'untied'."
            )
        return normalized


    @staticmethod
    def _normalize_tuplet_ratio(
        value: Optional[tuple[int, int]],
    ) -> Optional[tuple[int, int]]:
        """Validate and normalize an optional tuplet ratio."""
        if value is None:
            return None
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("tuplet_ratio must be a two-item tuple or list.")
        actual, normal = int(value[0]), int(value[1])
        return actual, normal


    @staticmethod
    def _search_tuplet_ratio_map(voice: dict[str, Any]) -> dict[int, tuple[int, int]]:
        """Map compact event indices to tuplet ratios for search."""
        ratios: dict[int, tuple[int, int]] = {}
        for row in voice.get("tuplets", []):
            if not isinstance(row, list) or len(row) < 2:
                continue
            ratio = row[0]
            beat_range = row[1]
            if (
                not isinstance(ratio, list)
                or len(ratio) != 2
                or not isinstance(beat_range, list)
                or len(beat_range) != 2
            ):
                continue
            ratio_tuple = (int(ratio[0]), int(ratio[1]))
            start_beat = float(beat_range[0])
            end_beat = float(beat_range[1])
            for index, event in enumerate(voice.get("events", []), start=1):
                if not isinstance(event, list) or len(event) < 2:
                    continue
                event_beat = float(event[1])
                if start_beat - 1e-9 <= event_beat <= end_beat + 1e-9:
                    ratios[index] = ratio_tuple
        return ratios


    @staticmethod
    def _payload_matches(payload: Any, expected: object) -> bool:
        """Return True when ``expected`` is found in a compact payload."""
        expected_text = str(expected).strip().lower()
        if expected_text == "":
            return True
        return expected_text in BarSearchMatchingMixin._payload_text(payload).lower()


    @staticmethod
    def _payload_text(payload: Any) -> str:
        """Flatten a scalar/list/dict payload for case-insensitive search."""
        if payload is None:
            return ""
        if isinstance(payload, dict):
            return " ".join(
                BarSearchMatchingMixin._payload_text(value)
                for value in payload.values()
            )
        if isinstance(payload, list):
            return " ".join(BarSearchMatchingMixin._payload_text(value) for value in payload)
        return str(payload)


    @staticmethod
    def _detail_string(prefix: str, values: dict[str, object]) -> str:
        """Format search predicate details for match rows."""
        fragments = []
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, tuple):
                value_text = ":".join(str(item) for item in value)
            else:
                value_text = str(value)
            fragments.append(f"{key}={value_text}")
        if not fragments:
            return prefix
        return prefix + ":" + ",".join(fragments)
