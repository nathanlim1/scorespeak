"""Internal bar-retrieval implementation slice."""

from __future__ import annotations

from .common import *


class BarPayloadAssemblyMixin:
    """Internal mixin for ScoreSpeak bar retrieval."""

    def _build_bar_payloads(
        self,
        parts: list[tuple[m21stream.Part, int]],
        measure_numbers: list[int],
        scoped_voices: Optional[set[int]],
        options: Optional[BarPayloadOptions] = None,
    ) -> tuple[dict[int, list[BarPart]], dict[tuple[int, int], list[VoiceEvent]]]:
        """Build compact bar payloads and voice-local event sequences."""
        payload_options = options or BarPayloadOptions()
        bar_parts_by_measure: dict[int, list[BarPart]] = {
            measure_number: [] for measure_number in measure_numbers
        }
        voice_sequences: dict[tuple[int, int], list[VoiceEvent]] = {}

        display_labels = build_part_display_labels(self._score)

        for measure_number in measure_numbers:
            for part_obj, part_idx in parts:
                measure_obj = part_obj.measure(measure_number)
                part_payload, part_sequences = self._build_part_payload(
                    part_obj,
                    part_idx,
                    measure_obj,
                    measure_number,
                    scoped_voices,
                    measure_numbers,
                    display_labels.get(part_idx),
                    payload_options,
                )
                bar_parts_by_measure[measure_number].append(part_payload)

                for voice_id, events in part_sequences.items():
                    key = (part_idx, voice_id)
                    voice_sequences.setdefault(key, []).extend(events)

        return bar_parts_by_measure, voice_sequences


    def _build_part_payload(
        self,
        part_obj: m21stream.Part,
        part_idx: int,
        measure_obj: Optional[m21stream.Measure],
        measure_number: int,
        scoped_voices: Optional[set[int]],
        scope_measures: list[int],
        display_label: Optional[PartLabel] = None,
        options: Optional[BarPayloadOptions] = None,
    ) -> tuple[BarPart, dict[int, list[VoiceEvent]]]:
        """Build one compact part payload for one measure.

        ``display_label`` carries the grand-staff-aware display name and
        ``hand`` (``"RH" | "LH" | "Pedal"``). When ``None`` or when the
        part is not in a detected group, the payload falls back to the
        raw ``part.partName`` and omits the ``hand`` key entirely.
        """
        raw_name = part_obj.partName or f"Part {part_idx}"
        if display_label is not None:
            part_name = display_label.display_name
            hand = display_label.hand
        else:
            part_name = raw_name
            hand = None
        payload_options = options or BarPayloadOptions()
        if payload_options.include_part_notation:
            part_notation = self._build_part_notation(
                part_obj,
                measure_obj,
                measure_number,
                scope_measures,
            )
        else:
            part_notation = {}
        if measure_obj is None:
            part_payload: BarPart = {
                "part_index": part_idx,
                "part_name": part_name,
                "voices": [],
            }
            if hand:
                part_payload["hand"] = hand
            if part_notation:
                part_payload["notation"] = part_notation
            return part_payload, {}
        if not payload_options.include_events:
            part_payload = {
                "part_index": part_idx,
                "part_name": part_name,
                "voices": [],
            }
            if hand:
                part_payload["hand"] = hand
            if part_notation:
                part_payload["notation"] = part_notation
            return part_payload, {}

        voice_entries: list[tuple[int, m21stream.Stream]] = []
        voices = list(measure_obj.voices)
        if voices:
            seen_voice_ids: set[int] = set()
            for voice_stream in voices:
                voice_id = self._resolve_voice_id(voice_stream)
                seen_voice_ids.add(voice_id)
                if scoped_voices is not None and voice_id not in scoped_voices:
                    continue
                voice_entries.append((voice_id, voice_stream))
            if 1 not in seen_voice_ids:
                if scoped_voices is None or 1 in scoped_voices:
                    voice_entries.append((1, measure_obj))
        else:
            if scoped_voices is None or 1 in scoped_voices:
                voice_entries.append((1, measure_obj))

        voice_entries.sort(key=lambda item: item[0])

        voice_payloads: list[BarVoice] = []
        voice_sequences: dict[int, list[VoiceEvent]] = {}
        for voice_id, stream_obj in voice_entries:
            events = self._collect_voice_events(
                stream_obj,
                measure_number,
                part_idx,
                voice_id,
            )
            tuplet_rows = (
                self._build_tuplet_rows(events)
                if payload_options.include_tuplets
                else []
            )
            marking_rows = (
                self._collect_voice_markings(
                    measure_obj,
                    stream_obj,
                    events,
                    voice_id,
                )
                if payload_options.include_markings
                else []
            )
            span_rows = (
                self._collect_voice_spans(
                    part_obj,
                    events,
                    measure_number,
                    voice_id,
                    scope_measures,
                )
                if payload_options.include_spans
                else []
            )
            has_content = (
                events or tuplet_rows or marking_rows or span_rows
            )
            if not has_content:
                continue
            voice_sequences[voice_id] = events
            voice_payload: BarVoice = {
                "voice": voice_id,
                "events": [self._voice_event_to_row(event) for event in events],
                "tuplets": tuplet_rows,
            }
            if marking_rows:
                voice_payload["markings"] = marking_rows
            if span_rows:
                voice_payload["spans"] = span_rows
            voice_payloads.append(voice_payload)

        part_payload = {
            "part_index": part_idx,
            "part_name": part_name,
            "voices": voice_payloads,
        }
        if hand:
            part_payload["hand"] = hand
        if part_notation:
            part_payload["notation"] = part_notation
        return part_payload, voice_sequences


    def _collect_voice_events(
        self,
        stream_obj: m21stream.Stream,
        measure_number: int,
        part_idx: int,
        voice_id: int,
    ) -> list[VoiceEvent]:
        """Collect normalized events from one voice or measure stream."""
        events: list[VoiceEvent] = []
        for element in stream_obj.getElementsByClass(m21note.GeneralNote):
            element_offset = float(stream_obj.elementOffset(element))
            if isinstance(element, m21note.Rest):
                if (
                    hasattr(element.style, "hideObjectOnPrint")
                    and element.style.hideObjectOnPrint
                ):
                    continue
                events.append(VoiceEvent(
                    kind="rest",
                    beat=element_offset + 1.0,
                    duration=float(element.duration.quarterLength),
                    part_index=part_idx,
                    measure_number=measure_number,
                    voice=voice_id,
                    tie_status=None,
                    is_grace=bool(getattr(element.duration, "isGrace", False)),
                    dots=element.duration.dots,
                    grace_slash=self._extract_grace_slash(element),
                    grace_duration=self._extract_grace_duration(element),
                    offset=element_offset,
                    tuplets=self._extract_tuplets(element),
                    _source=element,
                ))
                continue

            if isinstance(element, m21chord.Chord):
                pitches = [
                    self._format_pitch_with_octave(pitch_obj.nameWithOctave)
                    for pitch_obj in element.pitches
                ]
                pitch_classes = self._unique_preserving_order([
                    self._format_pitch_class(pitch_obj)
                    for pitch_obj in element.pitches
                ])
                events.append(VoiceEvent(
                    kind="chord",
                    beat=element_offset + 1.0,
                    duration=float(element.duration.quarterLength),
                    part_index=part_idx,
                    measure_number=measure_number,
                    voice=voice_id,
                    tie_status=self._extract_tie_status(element),
                    is_grace=bool(getattr(element.duration, "isGrace", False)),
                    dots=element.duration.dots,
                    grace_slash=self._extract_grace_slash(element),
                    grace_duration=self._extract_grace_duration(element),
                    offset=element_offset,
                    pitches=pitches,
                    pitch_classes=pitch_classes,
                    tuplets=self._extract_tuplets(element),
                    _source=element,
                ))
                continue

            if isinstance(element, m21note.Note):
                pitch = self._format_pitch_with_octave(element.pitch.nameWithOctave)
                events.append(VoiceEvent(
                    kind="note",
                    beat=element_offset + 1.0,
                    duration=float(element.duration.quarterLength),
                    part_index=part_idx,
                    measure_number=measure_number,
                    voice=voice_id,
                    tie_status=self._extract_tie_status(element),
                    is_grace=bool(getattr(element.duration, "isGrace", False)),
                    dots=element.duration.dots,
                    grace_slash=self._extract_grace_slash(element),
                    grace_duration=self._extract_grace_duration(element),
                    offset=element_offset,
                    pitch=pitch,
                    tuplets=self._extract_tuplets(element),
                    _source=element,
                ))

        return events


    def _voice_event_to_row(self, event: VoiceEvent) -> BarEventRow:
        """Project one internal voice event into a compact public row."""
        pitch_data: object
        if event.kind == "note":
            pitch_data = event.pitch
        elif event.kind == "chord":
            pitch_data = list(event.pitches or [])
        else:
            pitch_data = None

        return [
            event.kind,
            event.beat,
            pitch_data,
            event.duration,
            event.tie_status,
            event.is_grace,
            event.dots,
            event.grace_slash,
            event.grace_duration,
        ]


    def _build_tuplet_rows(
        self,
        events: list[VoiceEvent],
    ) -> list[TupletSpanRow]:
        """Compress per-event tuplet info into voice-level span rows."""
        if not events:
            return []

        ratio_runs: dict[tuple[int, int], list[int]] = {}
        all_rows: list[TupletSpanRow] = []

        def flush_ratio(ratio: tuple[int, int]) -> None:
            run = ratio_runs.pop(ratio, [])
            if not run:
                return
            actual_notes, normal_notes = ratio
            beat_range = self._event_beat_range(events, run[0], run[-1])
            if actual_notes <= 0:
                all_rows.append([
                    [actual_notes, normal_notes],
                    beat_range,
                ])
                return
            if len(run) % actual_notes != 0:
                all_rows.append([
                    [actual_notes, normal_notes],
                    beat_range,
                ])
                return
            for start in range(0, len(run), actual_notes):
                chunk = run[start:start + actual_notes]
                all_rows.append([
                    [actual_notes, normal_notes],
                    self._event_beat_range(events, chunk[0], chunk[-1]),
                ])

        active_ratios: set[tuple[int, int]] = set()
        for index, event in enumerate(events, start=1):
            event_ratios = {
                (tuplet.actual_notes, tuplet.normal_notes)
                for tuplet in (event.tuplets or [])
            }
            for ratio in list(active_ratios - event_ratios):
                flush_ratio(ratio)
                active_ratios.discard(ratio)
            for ratio in event_ratios:
                ratio_runs.setdefault(ratio, []).append(index)
                active_ratios.add(ratio)

        for ratio in list(active_ratios):
            flush_ratio(ratio)

        all_rows.sort(
            key=lambda row: (row[1][0], row[1][1], row[0][0], row[0][1])
        )
        return all_rows


    @staticmethod
    def _event_beat_range(
        events: list[VoiceEvent],
        start_index: int,
        end_index: int,
    ) -> list[float]:
        """Return beat values for a 1-based inclusive event-index range."""
        if not events:
            return []

        safe_start = max(1, min(start_index, len(events)))
        safe_end = max(1, min(end_index, len(events)))
        return [
            float(events[safe_start - 1].beat),
            float(events[safe_end - 1].beat),
        ]


    def _find_qualifying_measures(
        self,
        voice_sequences: dict[tuple[int, int], list[VoiceEvent]],
        sequence: list[ParsedQueryEvent],
        chord_mode: str,
    ) -> set[int]:
        """Run the parsed sequence matcher and collect touched measures."""
        qualifying_measures: set[int] = set()
        for events in voice_sequences.values():
            if not events or len(events) < len(sequence):
                continue
            for start_index in range(len(events) - len(sequence) + 1):
                window = events[start_index:start_index + len(sequence)]
                if self._window_matches(window, sequence, chord_mode):
                    qualifying_measures.update(
                        event.measure_number for event in window
                    )
        return qualifying_measures


    def _window_matches(
        self,
        window: list[VoiceEvent],
        sequence: list[ParsedQueryEvent],
        chord_mode: str,
    ) -> bool:
        """Return True when one event window matches the parsed sequence."""
        for event, expected in zip(window, sequence):
            if not self._event_matches(event, expected, chord_mode):
                return False
        return True


    def _event_matches(
        self,
        event: VoiceEvent,
        expected: ParsedQueryEvent,
        chord_mode: str,
    ) -> bool:
        """Return True when one internal event matches one query element."""
        if expected.kind != "any" and event.kind != expected.kind:
            return False

        if expected.duration is not None:
            if abs(event.duration - expected.duration) > 1e-9:
                return False

        if expected.kind == "note":
            return event.pitch == expected.pitch

        if expected.kind == "rest":
            return event.kind == "rest"

        if expected.kind == "chord":
            event_pitch_classes = set(event.pitch_classes or [])
            expected_pitch_classes = set(expected.pitch_classes or [])
            if chord_mode == "exact":
                return event_pitch_classes == expected_pitch_classes
            return expected_pitch_classes.issubset(event_pitch_classes)

        return True


    @staticmethod
    def _resolve_voice_id(voice_stream: m21stream.Stream) -> int:
        """Resolve a voice stream to the public 1-based voice number."""
        voice_id = getattr(voice_stream, "id", None)
        if str(voice_id).isdigit():
            return int(voice_id)
        return 1


    @staticmethod
    def _extract_tuplets(general_note: m21note.GeneralNote) -> list[TupletInfo]:
        """Extract compact tuplet metadata from a general note."""
        tuplets: list[TupletInfo] = []
        for tuplet in general_note.duration.tuplets:
            tuplets.append(TupletInfo(
                actual_notes=int(tuplet.numberNotesActual),
                normal_notes=int(tuplet.numberNotesNormal),
            ))
        return tuplets


    @staticmethod
    def _extract_grace_slash(
        general_note: m21note.GeneralNote,
    ) -> Optional[bool]:
        """Return grace-note slash state, or None for non-grace events."""
        if not bool(getattr(general_note.duration, "isGrace", False)):
            return None
        return bool(getattr(general_note.duration, "slash", False))


    @staticmethod
    def _extract_grace_duration(
        general_note: m21note.GeneralNote,
    ) -> Optional[str]:
        """Return the written grace-note duration type, if applicable."""
        if not bool(getattr(general_note.duration, "isGrace", False)):
            return None
        return str(getattr(general_note.duration, "type", None) or "eighth")


    @staticmethod
    def _extract_tie_status(general_note: m21note.GeneralNote) -> Optional[str]:
        """Return ``"start"``, ``"continue"``, ``"stop"``, or ``None``.

        Maps music21's ``Tie.type`` onto the minimal three-state alphabet
        used in the public bar payload. Unusual tie types (``let-ring``,
        ``continue-let-ring``) are reported as ``"continue"`` so the
        agent has enough information to know the note is mid-tie without
        us inventing new vocabulary.
        """
        tie = getattr(general_note, "tie", None)
        if tie is None:
            return None
        tie_type = getattr(tie, "type", None)
        if tie_type is None:
            return None
        if tie_type == "start" or tie_type == "stop" or tie_type == "continue":
            return tie_type
        return "continue"


    @staticmethod
    def _unique_preserving_order(values: list[str]) -> list[str]:
        """Return values with duplicates removed while preserving order."""
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result


    def _normalize_pitch_class(
        self,
        raw_pitch: PitchInput,
        field_name: str,
    ) -> str:
        """Normalize one query pitch class to public spelling."""
        try:
            pitch_obj = normalize_pitch(raw_pitch)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} contains an invalid pitch class: {raw_pitch!r}."
            ) from exc
        return self._format_pitch_class(pitch_obj)


    def _format_pitch_class(
        self,
        pitch_obj: m21pitch.Pitch,
    ) -> str:
        """Format a music21 Pitch as a public pitch class string."""
        return pitch_obj.name.replace("-", "b")


    def _format_pitch_with_octave(
        self,
        raw_pitch: PitchInput,
    ) -> str:
        """Format a pitch input as a public pitch-with-octave string."""
        pitch_obj = normalize_pitch(raw_pitch)
        name_with_octave = pitch_obj.nameWithOctave or pitch_obj.name
        return name_with_octave.replace("-", "b")
