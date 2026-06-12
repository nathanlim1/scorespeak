"""Deterministic symbolic fact extraction for long-task benchmarks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from music21 import chord as m21chord
from music21 import dynamics as m21dynamics
from music21 import note as m21note
from music21 import spanner as m21spanner
from music21 import stream as m21stream

from scorespeak import ScoreSpeak
from scorespeak.agent.context_renderers import render_exact_context


SCORE_ROOT_FACT_ID = "score_root"
SUPPORTED_POINT_MARKINGS = frozenset(
    {
        "dynamic",
        "articulation",
        "lyric",
        "text_expression",
        "chord_symbol",
        "ornament",
    }
)
SUPPORTED_SPANS = frozenset({"hairpin", "slur"})
LAYOUT_NOTATION_KEYS = frozenset({"system_break", "page_break"})


@dataclass(frozen=True)
class DependencyEdge:
    """One typed dependency from a target fact to its required anchor."""

    kind: str
    fact_id: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DependencyEdge":
        """Build a dependency edge from a JSON-like dictionary."""
        return cls(kind=str(data["kind"]), fact_id=str(data["fact_id"]))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class SymbolicFact:
    """One deterministic fact describing a target MusicXML score."""

    fact_id: str
    channel: str
    location: dict[str, Any]
    payload: dict[str, Any]
    anchor_type: str
    depends_on: list[DependencyEdge] = field(default_factory=list)
    canonical_text: str = ""
    supported: bool = True
    source_span: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SymbolicFact":
        """Build a symbolic fact from a JSON-like dictionary."""
        return cls(
            fact_id=str(data["fact_id"]),
            channel=str(data["channel"]),
            location=dict(data.get("location") or {}),
            payload=dict(data.get("payload") or {}),
            anchor_type=str(data.get("anchor_type") or ""),
            depends_on=[
                DependencyEdge.from_dict(item)
                for item in data.get("depends_on") or []
            ],
            canonical_text=str(data.get("canonical_text") or ""),
            supported=bool(data.get("supported", True)),
            source_span=dict(data.get("source_span") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        data = asdict(self)
        data["depends_on"] = [edge.to_dict() for edge in self.depends_on]
        return data


@dataclass(frozen=True)
class _PreparedEvent:
    """One validated event row ready to become a symbolic fact."""

    event_index: int
    row: list[Any]
    payload: dict[str, Any]
    is_grace: bool
    beat: float


@dataclass(frozen=True)
class _TieEvent:
    """One non-grace rhythmic event used for deriving pairwise ties."""

    fact_id: str
    part_index: int
    measure: int
    voice: int
    beat: float
    kind: str
    pitch: Any
    duration: float
    dots: int
    tie: str | None


@dataclass(frozen=True)
class _RowSpanFragment:
    """One per-measure row span fragment from exact context."""

    kind: str
    value: Any
    part_index: int
    measure: int
    voice: int
    flags: str
    start_beat: float
    end_beat: float


@dataclass
class FactExtractionResult:
    """Extracted facts plus excluded target-content summaries."""

    facts: list[SymbolicFact]
    unsupported_symbolic_facts: list[dict[str, Any]] = field(default_factory=list)
    ignored_layout_items: list[dict[str, Any]] = field(default_factory=list)
    ignored_boundary_items: list[dict[str, Any]] = field(default_factory=list)

    @property
    def target_fact_count_supported(self) -> int:
        """Return the number of supported target facts."""
        return len(self.facts)

    @property
    def target_fact_count_symbolic_all(self) -> int:
        """Return supported plus unsupported symbolic target fact count."""
        return len(self.facts) + len(self.unsupported_symbolic_facts)

    @property
    def target_fact_count_all(self) -> int:
        """Return the legacy symbolic target fact count alias."""
        return self.target_fact_count_symbolic_all

    @property
    def unsupported_symbolic_count(self) -> int:
        """Return the number of unsupported symbolic target facts."""
        return len(self.unsupported_symbolic_facts)

    @property
    def ignored_layout_count(self) -> int:
        """Return the number of ignored layout-only items."""
        return len(self.ignored_layout_items)

    @property
    def ignored_boundary_count(self) -> int:
        """Return the number of ignored boundary-crossing notation items."""
        return len(self.ignored_boundary_items)

    @property
    def unsupported_facts(self) -> list[dict[str, Any]]:
        """Return the legacy unsupported symbolic fact list alias."""
        return self.unsupported_symbolic_facts

    def facts_by_id(self) -> dict[str, SymbolicFact]:
        """Return supported facts keyed by fact id."""
        return {fact.fact_id: fact for fact in self.facts}


def extract_long_task_facts(
    score_or_path: ScoreSpeak | str | Path,
) -> FactExtractionResult:
    """Extract supported long-task facts from a score or MusicXML path."""
    score_state = (
        score_or_path
        if isinstance(score_or_path, ScoreSpeak)
        else ScoreSpeak.from_musicxml(score_or_path)
    )
    exact = render_exact_context(
        score_state._build_bar_result_set({"scope": {}}),
        include=["all_current_channels"],
    )
    hidden_rests = _hidden_rest_keys(score_state)
    builder = _FactBuilder(score_state, exact, hidden_rests)
    return builder.extract()


def stable_fact_id(channel: str, payload: dict[str, Any]) -> str:
    """Return a stable id for one symbolic fact payload."""
    encoded = json.dumps(
        {"channel": channel, "payload": _stable_fact_id_payload(channel, payload)},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]
    return f"{channel}:{digest}"


def _stable_fact_id_payload(channel: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return the payload fields that define a stable fact identity."""
    if channel != "part":
        return _normalize_json(payload)
    part_index = int(payload.get("part_index", -1))
    label = _clean_part_label(
        payload.get("part_name")
        or payload.get("display_name")
        or f"part {part_index}"
    )
    return {"part_index": part_index, "part_name": label}


def fact_identity_payload(fact: SymbolicFact) -> dict[str, Any]:
    """Return the canonical identity payload used for comparing repeated facts."""
    return {
        "channel": fact.channel,
        "location": _normalize_json(fact.location),
        "payload": _normalize_json(fact.payload),
        "anchor_type": fact.anchor_type,
    }


def duration_label(quarter_length: float) -> str:
    """Return a stable English duration label for a quarter-length."""
    labels = {
        4.0: "whole",
        3.0: "dotted half",
        2.0: "half",
        1.5: "dotted quarter",
        1.0: "quarter",
        0.75: "dotted eighth",
        0.5: "eighth",
        0.25: "sixteenth",
        0.125: "thirty-second",
    }
    return labels.get(float(quarter_length), f"{quarter_length:g}-quarter-length")


class _FactBuilder:
    """Stateful helper for converting exact context into facts."""

    def __init__(
        self,
        score_state: ScoreSpeak,
        exact_context: dict[str, Any],
        hidden_rests: set[tuple[int, int, int, float, float, int]],
    ) -> None:
        """Store extraction state."""
        self.score_state = score_state
        self.exact_context = exact_context
        self.hidden_rests = hidden_rests
        self.facts: list[SymbolicFact] = []
        self.unsupported_symbolic: list[dict[str, Any]] = []
        self.ignored_layout: list[dict[str, Any]] = []
        self.ignored_boundary: list[dict[str, Any]] = []
        self.part_fact_ids: dict[int, str] = {}
        self.measure_fact_ids: dict[tuple[int, int], str] = {}
        self.voice_fact_ids: dict[tuple[int, int, int], str] = {}
        self.event_fact_ids: dict[tuple[int, int, int, int], str] = {}
        self.event_rows: dict[tuple[int, int, int, int], list[Any]] = {}
        self.regular_event_ids_by_position: dict[
            tuple[int, int, int, float],
            str,
        ] = {}
        self.event_fact_ids_by_position: dict[
            tuple[int, int, int, float, bool, str],
            list[str],
        ] = {}
        self.canonical_span_ranges: list[dict[str, Any]] = []
        self.row_span_fragments: list[_RowSpanFragment] = []
        self.tie_events: list[_TieEvent] = []
        self.fact_ids_seen: set[str] = set()

    def extract(self) -> FactExtractionResult:
        """Extract all supported facts in dependency-friendly order."""
        self._add_part_facts()
        self._add_measure_and_attribute_facts()
        self._add_voice_and_event_facts()
        self._add_tie_span_facts()
        self._add_canonical_span_facts()
        self._add_marking_and_span_facts()
        self._add_row_span_facts()
        return FactExtractionResult(
            facts=self.facts,
            unsupported_symbolic_facts=self.unsupported_symbolic,
            ignored_layout_items=self.ignored_layout,
            ignored_boundary_items=self.ignored_boundary,
        )

    def _add_part_facts(self) -> None:
        """Extract part anchor facts."""
        info_by_index = {
            int(info.index): info
            for info in self.score_state.list_parts()
        }
        for part_index, raw_part_name in self._iter_parts():
            part_name = _clean_part_label(raw_part_name)
            info = info_by_index.get(part_index)
            display_name = (
                _clean_part_label(info.display_name)
                if info is not None and info.display_name
                else part_name
            )
            payload = {
                "part_index": part_index,
                "part_name": part_name,
                "instrument": _part_instrument_label(info, part_name),
                "display_name": display_name,
                "hand": (
                    _clean_part_label(info.hand)
                    if info is not None and info.hand
                    else None
                ),
            }
            fact = self._fact(
                "part",
                {"part_index": part_index},
                payload,
                "part_anchor",
                [DependencyEdge("score_root", SCORE_ROOT_FACT_ID)],
                f"Create part {part_index}: {part_name}.",
            )
            self.part_fact_ids[part_index] = fact.fact_id

    def _add_measure_and_attribute_facts(self) -> None:
        """Extract measure anchors and global/part attributes."""
        seen_attribute_keys: set[tuple[Any, ...]] = set()
        for bar in self._bars():
            measure = int(bar.get("measure_number"))
            for part in bar.get("parts", []):
                part_index = int(part.get("part_index"))
                self._add_measure_fact(part_index, measure)
                self._add_clef_fact(part, measure, seen_attribute_keys)
            self._add_bar_attribute_facts(bar, measure, seen_attribute_keys)
            self._add_unsupported_structure(bar)

    def _add_voice_and_event_facts(self) -> None:
        """Extract voice anchors and event facts."""
        for bar in self._bars():
            measure = int(bar.get("measure_number"))
            for part in bar.get("parts", []):
                part_index = int(part.get("part_index"))
                for voice in part.get("voices", []):
                    voice_number = int(voice.get("voice", 1))
                    self._add_voice_fact(part_index, measure, voice_number)
                    self._add_event_facts(
                        part_index,
                        measure,
                        voice_number,
                        voice,
                    )
                    self._add_tuplet_facts(
                        part_index,
                        measure,
                        voice_number,
                        voice,
                    )

    def _add_marking_and_span_facts(self) -> None:
        """Extract supported point markings and spans."""
        for bar in self._bars():
            measure = int(bar.get("measure_number"))
            for part in bar.get("parts", []):
                part_index = int(part.get("part_index"))
                for voice in part.get("voices", []):
                    voice_number = int(voice.get("voice", 1))
                    for row in voice.get("markings", []):
                        self._add_marking_fact(
                            part_index,
                            measure,
                            voice_number,
                            row,
                        )
                    for row in voice.get("spans", []):
                        self._add_span_fact(
                            part_index,
                            measure,
                            voice_number,
                            row,
                        )

    def _add_measure_fact(self, part_index: int, measure: int) -> None:
        """Add one part-specific measure anchor fact."""
        key = (part_index, measure)
        if key in self.measure_fact_ids:
            return
        payload = {"part_index": part_index, "measure": measure}
        fact = self._fact(
            "measure",
            payload,
            payload,
            "measure_anchor",
            [
                DependencyEdge(
                    "part_anchor",
                    self.part_fact_ids.get(part_index, ""),
                )
            ],
            f"Part {part_index} has measure {measure}.",
        )
        self.measure_fact_ids[key] = fact.fact_id

    def _add_voice_fact(
        self,
        part_index: int,
        measure: int,
        voice_number: int,
    ) -> None:
        """Add one voice anchor fact."""
        key = (part_index, measure, voice_number)
        if key in self.voice_fact_ids:
            return
        payload = {
            "part_index": part_index,
            "measure": measure,
            "voice": voice_number,
        }
        fact = self._fact(
            "voice",
            payload,
            payload,
            "voice_anchor",
            [
                DependencyEdge(
                    "measure_anchor",
                    self.measure_fact_ids.get((part_index, measure), ""),
                )
            ],
            (
                f"Use voice {voice_number} in part {part_index} "
                f"measure {measure}."
            ),
        )
        self.voice_fact_ids[key] = fact.fact_id

    def _add_clef_fact(
        self,
        part: dict[str, Any],
        measure: int,
        seen_attribute_keys: set[tuple[Any, ...]],
    ) -> None:
        """Add a clef fact when the exact context reports one."""
        notation = part.get("notation")
        if not isinstance(notation, dict) or not notation.get("clef"):
            return
        part_index = int(part.get("part_index"))
        key = ("clef", part_index, measure, notation["clef"])
        if key in seen_attribute_keys:
            return
        seen_attribute_keys.add(key)
        payload = {
            "kind": "clef",
            "value": str(notation["clef"]),
            "part_index": part_index,
            "measure": measure,
        }
        self._fact(
            "attribute",
            {"part_index": part_index, "measure": measure, "kind": "clef"},
            payload,
            "measure_position_anchor",
            [
                DependencyEdge(
                    "measure_anchor",
                    self.measure_fact_ids.get((part_index, measure), ""),
                )
            ],
            (
                f"Set part {part_index} measure {measure} clef to "
                f"{payload['value']}."
            ),
        )

    def _add_bar_attribute_facts(
        self,
        bar: dict[str, Any],
        measure: int,
        seen_attribute_keys: set[tuple[Any, ...]],
    ) -> None:
        """Add global time/key/tempo facts reported on this bar."""
        notation = bar.get("notation")
        if not isinstance(notation, dict):
            return
        active = notation.get("active")
        if not isinstance(active, dict):
            return
        for key, label in (
            ("time", "time_signature"),
            ("key", "key_signature"),
            ("tempo", "tempo"),
        ):
            if key not in active or active[key] in (None, ""):
                continue
            attribute_key = (label, measure, active[key])
            if attribute_key in seen_attribute_keys:
                continue
            seen_attribute_keys.add(attribute_key)
            payload = {
                "kind": label,
                "value": active[key],
                "measure": measure,
            }
            self._fact(
                "attribute",
                {"measure": measure, "kind": label},
                payload,
                "measure_position_anchor",
                self._global_measure_edges(measure),
                (
                    f"Set measure {measure} {label.replace('_', ' ')} "
                    f"to {active[key]}."
                ),
            )

    def _add_event_facts(
        self,
        part_index: int,
        measure: int,
        voice_number: int,
        voice: dict[str, Any],
    ) -> None:
        """Add note/rest/chord event facts for one voice."""
        prepared_events = self._prepare_event_rows(
            part_index,
            measure,
            voice_number,
            voice,
        )
        for is_grace_pass in (False, True):
            for prepared in prepared_events:
                if prepared.is_grace != is_grace_pass:
                    continue
                self._add_prepared_event_fact(
                    part_index,
                    measure,
                    voice_number,
                    prepared,
                )

    def _prepare_event_rows(
        self,
        part_index: int,
        measure: int,
        voice_number: int,
        voice: dict[str, Any],
    ) -> list[_PreparedEvent]:
        """Validate and normalize event rows for one voice."""
        prepared_events: list[_PreparedEvent] = []
        for event_index, row in enumerate(voice.get("events", []), start=1):
            if not isinstance(row, list) or len(row) < 7:
                self._unsupported(
                    "event",
                    "malformed_event_row",
                    {
                        "part_index": part_index,
                        "measure": measure,
                        "voice": voice_number,
                        "row": row,
                    },
                )
                continue
            if self._is_hidden_rest(part_index, measure, voice_number, row):
                continue
            kind = str(row[0])
            if kind not in {"note", "rest", "chord"}:
                self._unsupported(
                    "event",
                    "unsupported_event_kind",
                    {
                        "part_index": part_index,
                        "measure": measure,
                        "voice": voice_number,
                        "kind": kind,
                    },
                )
                continue
            is_grace = bool(row[5])
            payload = {
                "kind": kind,
                "part_index": part_index,
                "measure": measure,
                "voice": voice_number,
                "beat": _rounded_float(row[1]),
                "pitch": _normalize_json(row[2]),
                "duration": _rounded_float(row[3]),
                "is_grace": is_grace,
                "dots": int(row[6] or 0),
                "grace_slash": _grace_slash(row),
            }
            if is_grace:
                payload["grace_duration"] = _grace_duration(row)
            prepared_events.append(
                _PreparedEvent(
                    event_index=event_index,
                    row=row,
                    payload=payload,
                    is_grace=is_grace,
                    beat=float(payload["beat"]),
                )
            )
        return prepared_events

    def _add_prepared_event_fact(
        self,
        part_index: int,
        measure: int,
        voice_number: int,
        prepared: _PreparedEvent,
    ) -> None:
        """Create one event fact from a validated event row."""
        voice_id = self.voice_fact_ids.get((part_index, measure, voice_number), "")
        depends_on = [DependencyEdge("voice_anchor", voice_id)]
        if prepared.is_grace:
            principal_id = self.regular_event_ids_by_position.get(
                (part_index, measure, voice_number, prepared.beat)
            )
            if principal_id:
                depends_on = [DependencyEdge("event_anchor", principal_id)]

        fact = self._fact(
            "event",
            {
                "part_index": part_index,
                "measure": measure,
                "voice": voice_number,
                "beat": prepared.payload["beat"],
                "is_grace": prepared.is_grace,
            },
            prepared.payload,
            "event_anchor",
            depends_on,
            _event_text(prepared.payload),
        )
        event_key = (part_index, measure, voice_number, prepared.event_index)
        self.event_fact_ids[event_key] = fact.fact_id
        self.event_rows[event_key] = prepared.row
        position_key = (
            part_index,
            measure,
            voice_number,
            prepared.beat,
            prepared.is_grace,
            str(prepared.payload["kind"]),
        )
        self.event_fact_ids_by_position.setdefault(position_key, []).append(
            fact.fact_id
        )
        if not prepared.is_grace:
            principal_key = (part_index, measure, voice_number, prepared.beat)
            self.regular_event_ids_by_position.setdefault(
                principal_key,
                fact.fact_id,
            )
            self._remember_tie_event(
                fact.fact_id,
                part_index,
                measure,
                voice_number,
                prepared,
            )

    def _remember_tie_event(
        self,
        fact_id: str,
        part_index: int,
        measure: int,
        voice_number: int,
        prepared: _PreparedEvent,
    ) -> None:
        """Store one non-grace rhythmic event for adjacent tie derivation."""
        self.tie_events.append(
            _TieEvent(
                fact_id=fact_id,
                part_index=part_index,
                measure=measure,
                voice=voice_number,
                beat=prepared.beat,
                kind=str(prepared.payload["kind"]),
                pitch=prepared.payload.get("pitch"),
                duration=float(prepared.payload["duration"]),
                dots=int(prepared.payload.get("dots") or 0),
                tie=_normalized_tie_type(prepared.row[4]),
            )
        )

    def _add_tie_span_facts(self) -> None:
        """Derive supported pairwise tie facts from adjacent event tie states."""
        grouped_events: dict[tuple[int, int], list[_TieEvent]] = {}
        for event in self.tie_events:
            grouped_events.setdefault((event.part_index, event.voice), []).append(
                event
            )

        for events in grouped_events.values():
            ordered_events = sorted(
                events,
                key=lambda event: (event.measure, event.beat, event.fact_id),
            )
            for index, event in enumerate(ordered_events):
                previous_event = ordered_events[index - 1] if index > 0 else None
                next_event = (
                    ordered_events[index + 1]
                    if index + 1 < len(ordered_events)
                    else None
                )
                self._record_tie_incoming_boundary(event, previous_event)
                self._add_tie_fact_to_next_event(event, next_event)

    def _record_tie_incoming_boundary(
        self,
        event: _TieEvent,
        previous_event: _TieEvent | None,
    ) -> None:
        """Record unpaired incoming tie endpoints at the excerpt boundary."""
        if event.tie not in {"stop", "continue"}:
            return
        if _tie_events_form_pair(previous_event, event):
            return
        if previous_event is None:
            self._ignored_boundary_item(
                "span",
                "boundary_tie_missing_previous_endpoint",
                _tie_boundary_location(event, "previous"),
            )
            return
        self._ignored_boundary_item(
            "span",
            "tie_previous_endpoint_not_tool_representable",
            {
                "part_index": event.part_index,
                "measure": event.measure,
                "voice": event.voice,
                "beat": event.beat,
                "kind": "tie",
                "previous_measure": previous_event.measure,
                "previous_beat": previous_event.beat,
            },
        )

    def _add_tie_fact_to_next_event(
        self,
        event: _TieEvent,
        next_event: _TieEvent | None,
    ) -> None:
        """Emit a tie fact from one event to the adjacent matching next event."""
        if event.tie not in {"start", "continue"}:
            return
        if next_event is None:
            self._ignored_boundary_item(
                "span",
                "boundary_tie_missing_next_endpoint",
                _tie_boundary_location(event, "next"),
            )
            return
        if not _tie_events_form_pair(event, next_event):
            self._ignored_boundary_item(
                "span",
                "tie_next_endpoint_not_tool_representable",
                {
                    "part_index": event.part_index,
                    "measure": event.measure,
                    "voice": event.voice,
                    "beat": event.beat,
                    "kind": "tie",
                    "next_measure": next_event.measure,
                    "next_beat": next_event.beat,
                    "next_kind": next_event.kind,
                },
            )
            return

        payload = {
            "kind": "tie",
            "value": "",
            "part_index": event.part_index,
            "voice": event.voice,
            "start_measure": event.measure,
            "start_beat": event.beat,
            "end_measure": next_event.measure,
            "end_beat": next_event.beat,
            "event_kind": event.kind,
            "pitch": _normalize_json(event.pitch),
            "end_pitch": _normalize_json(next_event.pitch),
            "start_duration": _rounded_float(event.duration),
            "end_duration": _rounded_float(next_event.duration),
            "start_dots": event.dots,
            "end_dots": next_event.dots,
        }
        self._fact(
            "span",
            {
                "part_index": event.part_index,
                "measure": next_event.measure,
                "voice": event.voice,
                "beat": next_event.beat,
                "kind": "tie",
                "start_measure": event.measure,
                "start_beat": event.beat,
                "end_measure": next_event.measure,
                "end_beat": next_event.beat,
            },
            payload,
            "span_endpoint_anchor",
            [
                DependencyEdge("span_start_anchor", event.fact_id),
                DependencyEdge("span_end_anchor", next_event.fact_id),
            ],
            _tie_text(payload),
        )

    def _unique_event_fact_id_at_beat(
        self,
        part_index: int,
        measure: int,
        voice_number: int,
        beat: float,
    ) -> str | None:
        """Return the only non-hidden event fact at a beat, if unambiguous."""
        matches: list[str] = []
        target_beat = _rounded_float(beat)
        for key, row in self.event_rows.items():
            row_part, row_measure, row_voice, event_index = key
            if (
                row_part != part_index
                or row_measure != measure
                or row_voice != voice_number
            ):
                continue
            if _rounded_float(row[1]) != target_beat:
                continue
            event_id = self.event_fact_ids.get(
                (row_part, row_measure, row_voice, event_index)
            )
            if event_id is not None:
                matches.append(event_id)

        if len(matches) == 1:
            return matches[0]
        return None

    def _add_marking_fact(
        self,
        part_index: int,
        measure: int,
        voice_number: int,
        row: Any,
    ) -> None:
        """Add one supported marking fact or unsupported summary."""
        if not isinstance(row, list) or len(row) < 3:
            self._unsupported(
                "marking",
                "malformed_marking_row",
                {"part_index": part_index, "measure": measure, "row": row},
            )
            return
        kind = str(row[0])
        if kind not in SUPPORTED_POINT_MARKINGS:
            self._unsupported(
                "marking",
                "unsupported_marking_kind",
                {
                    "part_index": part_index,
                    "measure": measure,
                    "voice": voice_number,
                    "kind": kind,
                },
            )
            return
        beat = _rounded_float(row[2])
        value = (
            _normalize_ornament_payload(row[1])
            if kind == "ornament"
            else _normalize_json(row[1])
        )
        payload = {
            "kind": kind,
            "value": value,
            "part_index": part_index,
            "measure": measure,
            "voice": voice_number,
            "beat": beat,
        }
        if kind in {"dynamic", "chord_symbol", "text_expression"}:
            depends_on = [
                DependencyEdge(
                    "measure_position_anchor",
                    self.measure_fact_ids.get((part_index, measure), ""),
                )
            ]
            anchor_type = "measure_position_anchor"
        else:
            event_id = self._unique_event_fact_id_at_beat(
                part_index,
                measure,
                voice_number,
                beat,
            )
            if event_id is None:
                depends_on = [
                    DependencyEdge(
                        "measure_position_anchor",
                        self.measure_fact_ids.get((part_index, measure), ""),
                    )
                ]
                anchor_type = "measure_position_anchor"
            else:
                depends_on = [DependencyEdge("event_anchor", event_id)]
                anchor_type = "event_anchor"
        self._fact(
            "marking",
            {
                "part_index": part_index,
                "measure": measure,
                "voice": voice_number,
                "beat": beat,
                "kind": kind,
            },
            payload,
            anchor_type,
            depends_on,
            _marking_text(payload),
        )

    def _add_span_fact(
        self,
        part_index: int,
        measure: int,
        voice_number: int,
        row: Any,
    ) -> None:
        """Add one supported span fact or unsupported summary."""
        if not isinstance(row, list) or len(row) < 4:
            self._unsupported(
                "span",
                "malformed_span_row",
                {"part_index": part_index, "measure": measure, "row": row},
            )
            return
        kind = str(row[0])
        flags = str(row[2] or "")
        beat_range = row[3]
        if kind in {"slur", "hairpin"}:
            if not isinstance(beat_range, list) or len(beat_range) != 2:
                self._unsupported(
                    "span",
                    "malformed_span_range",
                    {
                        "part_index": part_index,
                        "measure": measure,
                        "voice": voice_number,
                        "kind": kind,
                    },
                )
                return
            if self._row_span_has_canonical_overlap(
                part_index,
                measure,
                voice_number,
                kind,
                row[1],
            ):
                return
            self.row_span_fragments.append(
                _RowSpanFragment(
                    kind=kind,
                    value=_normalize_json(row[1]),
                    part_index=part_index,
                    measure=measure,
                    voice=voice_number,
                    flags=flags,
                    start_beat=_rounded_float(beat_range[0]),
                    end_beat=_rounded_float(beat_range[1]),
                )
            )
            return
        if kind not in SUPPORTED_SPANS:
            self._unsupported(
                "span",
                "unsupported_span_kind",
                {
                    "part_index": part_index,
                    "measure": measure,
                    "voice": voice_number,
                    "kind": kind,
                },
            )
            return
        if not isinstance(beat_range, list) or len(beat_range) != 2:
            self._unsupported(
                "span",
                "malformed_span_range",
                {
                    "part_index": part_index,
                    "measure": measure,
                    "voice": voice_number,
                    "kind": kind,
                },
            )
            return
        start_beat = _rounded_float(beat_range[0])
        end_beat = _rounded_float(beat_range[1])

        payload = {
            "kind": kind,
            "value": _normalize_json(row[1]),
            "part_index": part_index,
            "measure": measure,
            "voice": voice_number,
            "start_beat": start_beat,
            "end_beat": end_beat,
        }
        start_id = self._unique_event_fact_id_at_beat(
            part_index,
            measure,
            voice_number,
            start_beat,
        )
        end_id = self._unique_event_fact_id_at_beat(
            part_index,
            measure,
            voice_number,
            end_beat,
        )
        if start_id is None or end_id is None:
            depends_on = [
                DependencyEdge(
                    "measure_position_anchor",
                    self.measure_fact_ids.get((part_index, measure), ""),
                )
            ]
            anchor_type = "measure_position_anchor"
        else:
            depends_on = [
                DependencyEdge("span_start_anchor", start_id),
                DependencyEdge("span_end_anchor", end_id),
            ]
            anchor_type = "span_endpoint_anchor"
        self._fact(
            "span",
            {
                "part_index": part_index,
                "measure": measure,
                "voice": voice_number,
                "kind": kind,
                "start_beat": payload["start_beat"],
                "end_beat": payload["end_beat"],
            },
            payload,
            anchor_type,
            depends_on,
            _span_text(payload),
        )

    def _add_row_span_facts(self) -> None:
        """Merge row-level slur/hairpin fragments into final-measure facts."""
        for fragments in self._row_span_groups():
            if not fragments:
                continue
            first = fragments[0]
            last = fragments[-1]
            if "L" in first.flags or "R" in last.flags:
                self._ignored_boundary_item(
                    "span",
                    "boundary_span_fragment",
                    {
                        "part_index": first.part_index,
                        "measure": last.measure,
                        "voice": first.voice,
                        "kind": first.kind,
                        "flags": first.flags + last.flags,
                        "beat_range": [
                            first.start_beat,
                            last.end_beat,
                        ],
                    },
                )
                continue
            self._add_row_span_group_fact(fragments)

    def _row_span_groups(self) -> list[list[_RowSpanFragment]]:
        """Return row span fragments grouped into contiguous chains."""
        groups: list[list[_RowSpanFragment]] = []
        active_groups: dict[tuple[Any, ...], list[_RowSpanFragment]] = {}
        fragments = sorted(
            self.row_span_fragments,
            key=lambda fragment: (
                fragment.part_index,
                fragment.voice,
                fragment.kind,
                json.dumps(fragment.value, sort_keys=True),
                fragment.measure,
                fragment.start_beat,
                fragment.end_beat,
            ),
        )
        for fragment in fragments:
            key = self._row_span_group_key(fragment)
            group = active_groups.get(key) if "L" in fragment.flags else None
            if group is None:
                group = []
            group.append(fragment)
            if "R" in fragment.flags:
                active_groups[key] = group
                continue
            if active_groups.get(key) is group:
                del active_groups[key]
            groups.append(group)

        for group in active_groups.values():
            if group not in groups:
                groups.append(group)
        return groups

    @staticmethod
    def _row_span_group_key(fragment: _RowSpanFragment) -> tuple[Any, ...]:
        """Return a stable grouping key for one row span fragment."""
        return (
            fragment.part_index,
            fragment.voice,
            fragment.kind,
            json.dumps(fragment.value, sort_keys=True, separators=(",", ":")),
        )

    def _add_row_span_group_fact(
        self,
        fragments: list[_RowSpanFragment],
    ) -> None:
        """Add one supported fact for a merged row span group."""
        first = fragments[0]
        last = fragments[-1]
        payload = {
            "kind": first.kind,
            "value": first.value,
            "part_index": first.part_index,
            "voice": first.voice,
            "start_measure": first.measure,
            "start_beat": first.start_beat,
            "end_measure": last.measure,
            "end_beat": last.end_beat,
        }
        if first.kind == "slur":
            start_id = self._unique_event_fact_id_at_beat(
                first.part_index,
                first.measure,
                first.voice,
                first.start_beat,
            )
            end_id = self._unique_event_fact_id_at_beat(
                last.part_index,
                last.measure,
                last.voice,
                last.end_beat,
            )
            if start_id is not None and end_id is not None:
                depends_on = [
                    DependencyEdge("span_start_anchor", start_id),
                    DependencyEdge("span_end_anchor", end_id),
                ]
                anchor_type = "span_endpoint_anchor"
            else:
                depends_on = [
                    DependencyEdge(
                        "measure_position_anchor",
                        self.measure_fact_ids.get((last.part_index, last.measure), ""),
                    )
                ]
                anchor_type = "measure_position_anchor"
        else:
            depends_on = [
                DependencyEdge(
                    "span_start_anchor",
                    self.measure_fact_ids.get((first.part_index, first.measure), ""),
                ),
                DependencyEdge(
                    "span_end_anchor",
                    self.measure_fact_ids.get((last.part_index, last.measure), ""),
                ),
            ]
            anchor_type = "measure_position_span_anchor"

        self._fact(
            "span",
            {
                "part_index": first.part_index,
                "measure": last.measure,
                "voice": first.voice,
                "kind": first.kind,
                "start_measure": first.measure,
                "start_beat": first.start_beat,
                "end_measure": last.measure,
                "end_beat": last.end_beat,
            },
            payload,
            anchor_type,
            depends_on,
            _span_text(payload),
        )

    def _add_canonical_span_facts(self) -> None:
        """Extract one canonical fact for each supported music21 spanner."""
        for part_index, part_obj in enumerate(self.score_state.score.parts):
            self._add_canonical_slur_facts(part_index, part_obj)
            self._add_canonical_hairpin_facts(part_index, part_obj)

    def _add_canonical_slur_facts(
        self,
        part_index: int,
        part_obj: m21stream.Part,
    ) -> None:
        """Extract canonical slur facts from one part."""
        for slur in part_obj.getElementsByClass(m21spanner.Slur):
            spanned = list(slur.getSpannedElements())
            if len(spanned) < 2:
                self._ignored_boundary_item(
                    "span",
                    "boundary_span_missing_endpoint",
                    {"part_index": part_index, "kind": "slur"},
                )
                continue
            start = self._event_endpoint(part_index, spanned[0])
            end = self._event_endpoint(part_index, spanned[-1])
            if start is None or end is None:
                self._record_missing_canonical_endpoint(
                    part_index,
                    "slur",
                    start,
                    end,
                )
                continue
            start_event_id = self._event_fact_id_for_endpoint(start)
            end_event_id = self._event_fact_id_for_endpoint(end)
            if start_event_id is None or end_event_id is None:
                self._unsupported(
                    "span",
                    "missing_supported_span_endpoint",
                    {
                        "part_index": part_index,
                        "kind": "slur",
                        "start": _normalize_json(start),
                        "end": _normalize_json(end),
                    },
                )
                continue
            if _same_span_endpoint(start, end):
                self._ignored_boundary_item(
                    "span",
                    "degenerate_span",
                    {
                        "part_index": part_index,
                        "kind": "slur",
                        "start": _normalize_json(start),
                        "end": _normalize_json(end),
                    },
                )
                continue
            payload = {
                "kind": "slur",
                "value": "",
                "part_index": part_index,
                "voice": int(start["voice"]),
                "start_measure": int(start["measure"]),
                "start_beat": float(start["beat"]),
                "end_measure": int(end["measure"]),
                "end_beat": float(end["beat"]),
            }
            if bool(start.get("is_grace")) or bool(end.get("is_grace")):
                payload["start_is_grace"] = bool(start.get("is_grace"))
                payload["end_is_grace"] = bool(end.get("is_grace"))
            self._fact(
                "span",
                {
                    "part_index": part_index,
                    "measure": payload["end_measure"],
                    "voice": payload["voice"],
                    "kind": "slur",
                    "start_measure": payload["start_measure"],
                    "start_beat": payload["start_beat"],
                    "end_measure": payload["end_measure"],
                    "end_beat": payload["end_beat"],
                },
                payload,
                "span_endpoint_anchor",
                [
                    DependencyEdge("span_start_anchor", start_event_id),
                    DependencyEdge("span_end_anchor", end_event_id),
                ],
                _span_text(payload),
            )
            self._record_canonical_span_range(payload)

    def _add_canonical_hairpin_facts(
        self,
        part_index: int,
        part_obj: m21stream.Part,
    ) -> None:
        """Extract canonical hairpin facts from one part."""
        for hairpin in part_obj.getElementsByClass(m21dynamics.DynamicWedge):
            spanned = list(hairpin.getSpannedElements())
            if len(spanned) < 2:
                self._ignored_boundary_item(
                    "span",
                    "boundary_span_missing_endpoint",
                    {"part_index": part_index, "kind": "hairpin"},
                )
                continue
            start = self._logical_position(part_index, hairpin, spanned[0], "start")
            end = self._logical_position(part_index, hairpin, spanned[-1], "end")
            if start is None or end is None:
                self._record_missing_canonical_endpoint(
                    part_index,
                    "hairpin",
                    start,
                    end,
                )
                continue
            start_measure_id = self.measure_fact_ids.get(
                (part_index, int(start["measure"]))
            )
            end_measure_id = self.measure_fact_ids.get(
                (part_index, int(end["measure"]))
            )
            if start_measure_id is None or end_measure_id is None:
                self._ignored_boundary_item(
                    "span",
                    "boundary_span_missing_measure",
                    {
                        "part_index": part_index,
                        "kind": "hairpin",
                        "start": _normalize_json(start),
                        "end": _normalize_json(end),
                    },
                )
                continue
            if _same_span_endpoint(start, end):
                self._ignored_boundary_item(
                    "span",
                    "degenerate_span",
                    {
                        "part_index": part_index,
                        "kind": "hairpin",
                        "start": _normalize_json(start),
                        "end": _normalize_json(end),
                    },
                )
                continue
            payload = {
                "kind": "hairpin",
                "value": _hairpin_value(hairpin),
                "part_index": part_index,
                "voice": int(start["voice"]),
                "start_measure": int(start["measure"]),
                "start_beat": float(start["beat"]),
                "end_measure": int(end["measure"]),
                "end_beat": float(end["beat"]),
            }
            self._fact(
                "span",
                {
                    "part_index": part_index,
                    "measure": payload["end_measure"],
                    "voice": payload["voice"],
                    "kind": "hairpin",
                    "start_measure": payload["start_measure"],
                    "start_beat": payload["start_beat"],
                    "end_measure": payload["end_measure"],
                    "end_beat": payload["end_beat"],
                },
                payload,
                "measure_position_span_anchor",
                [
                    DependencyEdge("span_start_anchor", start_measure_id),
                    DependencyEdge("span_end_anchor", end_measure_id),
                ],
                _span_text(payload),
            )
            self._record_canonical_span_range(payload)

    def _event_endpoint(
        self,
        part_index: int,
        element: Any,
    ) -> dict[str, Any] | None:
        """Return the symbolic event endpoint represented by a music21 element."""
        if not isinstance(element, (m21note.Note, m21chord.Chord)):
            return None
        position = self._element_position(element)
        if position is None:
            return None
        measure, beat = position
        return {
            "part_index": part_index,
            "measure": measure,
            "voice": _span_voice_id(element),
            "beat": beat,
            "is_grace": bool(getattr(element.duration, "isGrace", False)),
            "kind": "chord" if isinstance(element, m21chord.Chord) else "note",
        }

    def _logical_position(
        self,
        part_index: int,
        spanner: m21spanner.Spanner,
        fallback_element: Any,
        side: str,
    ) -> dict[str, Any] | None:
        """Return a logical span endpoint using ScoreSpeak metadata if present."""
        measure_attr = f"scorespeak_{side}_measure"
        beat_attr = f"scorespeak_{side}_beat"
        measure_value = getattr(spanner, measure_attr, None)
        beat_value = getattr(spanner, beat_attr, None)
        if measure_value is not None and beat_value is not None:
            try:
                measure = int(measure_value)
                beat = _rounded_float(beat_value)
            except (TypeError, ValueError):
                return None
            return {
                "part_index": part_index,
                "measure": measure,
                "voice": _span_voice_id(fallback_element),
                "beat": beat,
            }

        position = self._element_position(fallback_element)
        if position is None:
            return None
        measure, beat = position
        return {
            "part_index": part_index,
            "measure": measure,
            "voice": _span_voice_id(fallback_element),
            "beat": beat,
        }

    def _element_position(self, element: Any) -> tuple[int, float] | None:
        """Return measure number and 1-based beat for a music21 element."""
        measure_obj = element.getContextByClass(m21stream.Measure)
        if measure_obj is None:
            return None
        try:
            offset = float(measure_obj.elementOffset(element))
        except Exception:
            voice_obj = getattr(element, "activeSite", None)
            if not isinstance(voice_obj, m21stream.Voice):
                return None
            try:
                voice_offset = float(measure_obj.elementOffset(voice_obj))
                element_offset = float(voice_obj.elementOffset(element))
            except Exception:
                return None
            offset = voice_offset + element_offset
        return int(measure_obj.number), _rounded_float(offset + 1.0)

    def _event_fact_id_for_endpoint(
        self,
        endpoint: dict[str, Any],
    ) -> str | None:
        """Return the fact id for a direct music21 event endpoint."""
        key = (
            int(endpoint["part_index"]),
            int(endpoint["measure"]),
            int(endpoint["voice"]),
            float(endpoint["beat"]),
            bool(endpoint["is_grace"]),
            str(endpoint["kind"]),
        )
        candidates = self.event_fact_ids_by_position.get(key, [])
        if not candidates:
            return None
        return candidates[0]

    def _record_missing_canonical_endpoint(
        self,
        part_index: int,
        kind: str,
        start: dict[str, Any] | None,
        end: dict[str, Any] | None,
    ) -> None:
        """Record a canonical span whose endpoints cannot both be represented."""
        if start is None or end is None:
            self._ignored_boundary_item(
                "span",
                "boundary_span_missing_endpoint",
                {
                    "part_index": part_index,
                    "kind": kind,
                    "start": _normalize_json(start),
                    "end": _normalize_json(end),
                },
            )
            return
        self._unsupported(
            "span",
            "unsupported_canonical_span_endpoint",
            {
                "part_index": part_index,
                "kind": kind,
                "start": _normalize_json(start),
                "end": _normalize_json(end),
            },
        )

    def _record_canonical_span_range(self, payload: dict[str, Any]) -> None:
        """Remember a canonical span range for suppressing row fragments."""
        self.canonical_span_ranges.append(
            {
                "part_index": int(payload["part_index"]),
                "kind": str(payload["kind"]),
                "voice": int(payload["voice"]),
                "value": _normalize_json(payload.get("value")),
                "start_measure": int(payload["start_measure"]),
                "start_beat": _rounded_float(payload["start_beat"]),
                "end_measure": int(payload["end_measure"]),
                "end_beat": _rounded_float(payload["end_beat"]),
            }
        )

    def _row_span_has_canonical_overlap(
        self,
        part_index: int,
        measure: int,
        voice_number: int,
        kind: str,
        value: Any,
    ) -> bool:
        """Return whether a row-fragment span is covered by a canonical fact."""
        normalized_value = _normalize_json(value)
        for span_range in self.canonical_span_ranges:
            if span_range["part_index"] != part_index:
                continue
            if span_range["kind"] != kind:
                continue
            if span_range["voice"] != voice_number:
                continue
            if span_range["value"] != normalized_value:
                continue
            if span_range["start_measure"] <= measure <= span_range["end_measure"]:
                return True
        return False

    def _add_tuplet_facts(
        self,
        part_index: int,
        measure: int,
        voice_number: int,
        voice: dict[str, Any],
    ) -> None:
        """Add supported tuplet facts for one voice."""
        for row in voice.get("tuplets", []):
            payload = self._tuplet_payload(part_index, measure, voice_number, row)
            if payload is None:
                continue
            event_edges = [
                DependencyEdge("event_anchor", fact_id)
                for fact_id in self._event_fact_ids_in_beat_range(
                    part_index,
                    measure,
                    voice_number,
                    float(payload["start_beat"]),
                    float(payload["end_beat"]),
                )
            ]
            anchor_type = "event_span_anchor"
            if not event_edges:
                anchor_type = "voice_anchor"
                event_edges = [
                    DependencyEdge(
                        "voice_anchor",
                        self.voice_fact_ids.get((part_index, measure, voice_number), ""),
                    )
                ]
            self._fact(
                "tuplet",
                {
                    "part_index": part_index,
                    "measure": measure,
                    "voice": voice_number,
                    "beat": payload["start_beat"],
                },
                payload,
                anchor_type,
                event_edges,
                _tuplet_text(payload),
            )

    def _tuplet_payload(
        self,
        part_index: int,
        measure: int,
        voice_number: int,
        row: Any,
    ) -> dict[str, Any] | None:
        """Return a normalized tuplet payload, or record an unsupported row."""
        if not isinstance(row, list) or len(row) < 2:
            self._unsupported(
                "tuplet",
                "malformed_tuplet_row",
                {
                    "part_index": part_index,
                    "measure": measure,
                    "voice": voice_number,
                    "row": row,
                },
            )
            return None
        ratio = row[0]
        beat_range = row[1]
        if (
            not isinstance(ratio, list)
            or len(ratio) != 2
            or not isinstance(beat_range, list)
            or len(beat_range) != 2
        ):
            self._unsupported(
                "tuplet",
                "malformed_tuplet_row",
                {
                    "part_index": part_index,
                    "measure": measure,
                    "voice": voice_number,
                    "row": row,
                },
            )
            return None
        try:
            actual_notes = int(ratio[0])
            normal_notes = int(ratio[1])
            start_beat = _rounded_float(beat_range[0])
            end_beat = _rounded_float(beat_range[1])
        except (TypeError, ValueError):
            self._unsupported(
                "tuplet",
                "malformed_tuplet_row",
                {
                    "part_index": part_index,
                    "measure": measure,
                    "voice": voice_number,
                    "row": row,
                },
            )
            return None
        return {
            "kind": "tuplet",
            "part_index": part_index,
            "measure": measure,
            "voice": voice_number,
            "actual_notes": actual_notes,
            "normal_notes": normal_notes,
            "start_beat": start_beat,
            "end_beat": end_beat,
        }

    def _event_fact_ids_in_beat_range(
        self,
        part_index: int,
        measure: int,
        voice_number: int,
        start_beat: float,
        end_beat: float,
    ) -> list[str]:
        """Return event fact ids whose beats fall inside a tuplet row."""
        matches = []
        start = _rounded_float(start_beat)
        end = _rounded_float(end_beat)
        for key, row in self.event_rows.items():
            row_part, row_measure, row_voice, event_index = key
            if (
                row_part != part_index
                or row_measure != measure
                or row_voice != voice_number
            ):
                continue
            beat = _rounded_float(row[1])
            if start <= beat <= end:
                event_id = self.event_fact_ids.get(
                    (row_part, row_measure, row_voice, event_index)
                )
                if event_id is not None:
                    matches.append(event_id)
        return matches

    def _add_unsupported_structure(self, bar: dict[str, Any]) -> None:
        """Record structural notation as ignored layout or unsupported symbolic."""
        notation = bar.get("notation")
        if not isinstance(notation, dict):
            return
        for key, value in notation.items():
            if key in {"active", "changed_here"} or value in (None, False, "", []):
                continue
            location = {
                "measure": int(bar.get("measure_number")),
                "kind": key,
                "value": _normalize_json(value),
            }
            if key in LAYOUT_NOTATION_KEYS:
                self._ignored_layout_item(
                    "structure",
                    "layout_not_scored",
                    location,
                )
            else:
                self._unsupported(
                    "structure",
                    "structure_not_supported_in_v1",
                    location,
                )

    def _global_measure_edges(self, measure: int) -> list[DependencyEdge]:
        """Return measure-anchor edges for score-wide attributes."""
        edges = []
        for part_index in sorted(self.part_fact_ids):
            fact_id = self.measure_fact_ids.get((part_index, measure))
            if fact_id:
                edges.append(DependencyEdge("measure_anchor", fact_id))
        if not edges:
            edges.append(DependencyEdge("score_root", SCORE_ROOT_FACT_ID))
        return edges

    def _fact(
        self,
        channel: str,
        location: dict[str, Any],
        payload: dict[str, Any],
        anchor_type: str,
        depends_on: list[DependencyEdge],
        canonical_text: str,
    ) -> SymbolicFact:
        """Create, store, and return one fact."""
        clean_payload = _normalize_json(payload)
        fact_id = stable_fact_id(channel, clean_payload)
        fact = SymbolicFact(
            fact_id=fact_id,
            channel=channel,
            location=_normalize_json(location),
            payload=clean_payload,
            anchor_type=anchor_type,
            depends_on=[edge for edge in depends_on if edge.fact_id],
            canonical_text=canonical_text,
            supported=True,
        )
        if fact_id in self.fact_ids_seen:
            return fact
        self.fact_ids_seen.add(fact_id)
        self.facts.append(fact)
        return fact

    def _unsupported(
        self,
        channel: str,
        reason: str,
        location: dict[str, Any],
    ) -> None:
        """Record one unsupported symbolic target item."""
        self.unsupported_symbolic.append(
            {
                "channel": channel,
                "reason": reason,
                "location": _normalize_json(location),
            }
        )

    def _ignored_layout_item(
        self,
        channel: str,
        reason: str,
        location: dict[str, Any],
    ) -> None:
        """Record one ignored layout-only target item."""
        self.ignored_layout.append(
            {
                "channel": channel,
                "reason": reason,
                "location": _normalize_json(location),
            }
        )

    def _ignored_boundary_item(
        self,
        channel: str,
        reason: str,
        location: dict[str, Any],
    ) -> None:
        """Record one ignored boundary-crossing notation item."""
        self.ignored_boundary.append(
            {
                "channel": channel,
                "reason": reason,
                "location": _normalize_json(location),
            }
        )

    def _iter_parts(self) -> list[tuple[int, str]]:
        """Return unique part labels in score order."""
        parts = {}
        for bar in self._bars():
            for part in bar.get("parts", []):
                part_index = int(part.get("part_index"))
                parts.setdefault(
                    part_index,
                    _clean_part_label(part.get("part_name") or f"Part {part_index}"),
                )
        return sorted(parts.items(), key=lambda item: item[0])

    def _bars(self) -> list[dict[str, Any]]:
        """Return exact-context bars."""
        return [
            bar
            for bar in self.exact_context.get("bars", [])
            if isinstance(bar, dict)
        ]

    def _is_hidden_rest(
        self,
        part_index: int,
        measure: int,
        voice_number: int,
        row: list[Any],
    ) -> bool:
        """Return whether an event row is a hidden rest."""
        if str(row[0]) != "rest":
            return False
        key = (
            part_index,
            measure,
            voice_number,
            _rounded_float(row[1]),
            _rounded_float(row[3]),
            int(row[6] or 0),
        )
        return key in self.hidden_rests


def _hidden_rest_keys(
    score_state: ScoreSpeak,
) -> set[tuple[int, int, int, float, float, int]]:
    """Return keys for hidden rests in a ScoreSpeak score."""
    keys: set[tuple[int, int, int, float, float, int]] = set()
    for part_index, part_obj in enumerate(score_state.score.parts):
        for measure_obj in part_obj.getElementsByClass(m21stream.Measure):
            measure_number = int(measure_obj.number)
            for voice_obj in measure_obj.voices:
                voice_number = _voice_number(voice_obj)
                keys.update(
                    _hidden_rest_keys_from_stream(
                        voice_obj,
                        part_index,
                        measure_number,
                        voice_number,
                    )
                )
            keys.update(
                _hidden_rest_keys_from_stream(
                    measure_obj,
                    part_index,
                    measure_number,
                    1,
                )
            )
    return keys


def _part_instrument_label(info: Any, part_name: str) -> str:
    """Return a stable instrument label for part identity."""
    if info is None or not getattr(info, "instrument", None):
        return _clean_part_label(part_name)
    instrument = _clean_part_label(info.instrument)
    if instrument == "Unknown":
        return _clean_part_label(part_name)
    return instrument


def _clean_part_label(value: Any) -> str:
    """Return a compact part label without MusicXML indentation artifacts."""
    return " ".join(str(value or "").split())


def _same_span_endpoint(
    start: dict[str, Any],
    end: dict[str, Any],
) -> bool:
    """Return whether a span starts and ends at the same logical endpoint."""
    same_position = (
        int(start.get("part_index", -1)) == int(end.get("part_index", -2))
        and int(start.get("measure", -1)) == int(end.get("measure", -2))
        and int(start.get("voice", -1)) == int(end.get("voice", -2))
        and _rounded_float(start.get("beat", 0.0))
        == _rounded_float(end.get("beat", 1.0))
    )
    if not same_position:
        return False
    if "is_grace" in start or "is_grace" in end:
        return bool(start.get("is_grace")) == bool(end.get("is_grace")) and str(
            start.get("kind", "")
        ) == str(end.get("kind", ""))
    return True


def _normalized_tie_type(value: Any) -> str | None:
    """Return a supported tie type string from an event row value."""
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"start", "stop", "continue"}:
        return text
    return None


def _tie_events_form_pair(
    start: _TieEvent | None,
    end: _TieEvent,
) -> bool:
    """Return whether two adjacent events can become one add_tie operation."""
    if start is None:
        return False
    if start.tie not in {"start", "continue"}:
        return False
    if end.tie not in {"stop", "continue"}:
        return False
    if start.kind not in {"note", "chord"} or end.kind not in {"note", "chord"}:
        return False
    if start.kind != end.kind:
        return False
    return _same_tie_pitch(start.pitch, end.pitch)


def _same_tie_pitch(start_pitch: Any, end_pitch: Any) -> bool:
    """Return whether two note/chord payloads have matching pitch content."""
    return _normalize_json(start_pitch) == _normalize_json(end_pitch)


def _tie_boundary_location(event: _TieEvent, missing_side: str) -> dict[str, Any]:
    """Return an ignored-boundary location for an unpaired tie endpoint."""
    return {
        "part_index": event.part_index,
        "measure": event.measure,
        "voice": event.voice,
        "beat": event.beat,
        "kind": "tie",
        "event_kind": event.kind,
        "pitch": _normalize_json(event.pitch),
        "tie": event.tie,
        "missing_side": missing_side,
    }


def _hidden_rest_keys_from_stream(
    stream_obj: m21stream.Stream,
    part_index: int,
    measure_number: int,
    voice_number: int,
) -> set[tuple[int, int, int, float, float, int]]:
    """Return hidden rest keys for direct rest children in one stream."""
    keys: set[tuple[int, int, int, float, float, int]] = set()
    for element in stream_obj:
        if not isinstance(element, m21note.Rest):
            continue
        if not bool(getattr(element.style, "hideObjectOnPrint", False)):
            continue
        beat = _rounded_float(stream_obj.elementOffset(element) + 1.0)
        duration = _rounded_float(element.quarterLength)
        dots = int(getattr(element.duration, "dots", 0) or 0)
        keys.add(
            (
                part_index,
                measure_number,
                voice_number,
                beat,
                duration,
                dots,
            )
        )
    return keys


def _voice_number(voice_obj: m21stream.Voice) -> int:
    """Return an integer voice id, defaulting to one."""
    try:
        return int(voice_obj.id)
    except (TypeError, ValueError):
        return 1


def _span_voice_id(element: Any) -> int:
    """Return the voice id containing ``element``."""
    parent = getattr(element, "activeSite", None)
    while parent is not None and not isinstance(
        parent,
        (m21stream.Voice, m21stream.Measure),
    ):
        parent = getattr(parent, "activeSite", None)
    if isinstance(parent, m21stream.Voice):
        return _voice_number(parent)
    return 1


def _grace_slash(row: list[Any]) -> bool | None:
    """Return the grace slash field from an event row."""
    if len(row) < 8:
        return None
    value = row[7]
    if value is None:
        return None
    return bool(value)


def _grace_duration(row: list[Any]) -> str:
    """Return the written grace duration from an event row."""
    if len(row) < 9 or row[8] in (None, ""):
        return "eighth"
    return str(row[8])


def _normalize_ornament_payload(value: Any) -> Any:
    """Return a stable ornament payload."""
    if isinstance(value, dict):
        ornament_type = str(value.get("type") or "").strip().lower()
        if ornament_type == "tremolo":
            payload: dict[str, Any] = {"type": "tremolo"}
            marks = value.get("marks")
            if marks is not None:
                payload["marks"] = int(marks)
            return payload
        return _normalize_json(value)
    text = " ".join(str(value or "").strip().lower().split())
    if text in {"trill", "mordent", "turn", "tremolo"}:
        return text
    return text.replace("-", " ")


def _hairpin_value(hairpin: m21spanner.Spanner) -> str:
    """Return the stable value for a canonical hairpin fact."""
    if isinstance(hairpin, m21dynamics.Crescendo):
        return "crescendo"
    if isinstance(hairpin, m21dynamics.Diminuendo):
        return "diminuendo"
    return "hairpin"


def _event_text(payload: dict[str, Any]) -> str:
    """Return a canonical human-readable event description."""
    kind = str(payload["kind"])
    duration = duration_label(float(payload["duration"]))
    beat = f"{float(payload['beat']):g}"
    if kind == "rest":
        body = f"{duration} rest"
    elif kind == "chord":
        pitches = ", ".join(str(item) for item in payload["pitch"])
        body = f"{duration} chord ({pitches})"
    else:
        body = f"{duration} {payload['pitch']} note"
    if bool(payload.get("is_grace")):
        slash = "slashed " if payload.get("grace_slash") else "unslashed "
        grace_duration = str(payload.get("grace_duration") or "").strip()
        if grace_duration:
            body = f"{slash}{grace_duration} grace {kind}"
            if kind == "note":
                body = f"{slash}{grace_duration} grace {payload['pitch']} note"
            elif kind == "chord":
                pitches = ", ".join(str(item) for item in payload["pitch"])
                body = f"{slash}{grace_duration} grace chord ({pitches})"
            elif kind == "rest":
                body = f"{slash}{grace_duration} grace rest"
            return (
                f"Part {payload['part_index']} measure {payload['measure']} "
                f"voice {payload['voice']} beat {beat}: {body}."
            )
        body = f"{slash}grace {body}"
    return (
        f"Part {payload['part_index']} measure {payload['measure']} "
        f"voice {payload['voice']} beat {beat}: {body}."
    )


def _marking_text(payload: dict[str, Any]) -> str:
    """Return a canonical human-readable marking description."""
    kind = str(payload["kind"])
    beat = f"{float(payload['beat']):g}"
    value = payload["value"]
    if kind == "ornament" and isinstance(value, dict):
        if value.get("type") == "tremolo" and value.get("marks") is not None:
            value = f"tremolo with {value['marks']} marks"
    return (
        f"Part {payload['part_index']} measure {payload['measure']} "
        f"voice {payload['voice']} beat {beat}: "
        f"{kind} {value}."
    )


def _tuplet_text(payload: dict[str, Any]) -> str:
    """Return a canonical human-readable tuplet description."""
    start = f"{float(payload['start_beat']):g}"
    end = f"{float(payload['end_beat']):g}"
    return (
        f"Part {payload['part_index']} measure {payload['measure']} "
        f"voice {payload['voice']}: {payload['actual_notes']}:"
        f"{payload['normal_notes']} tuplet from beat {start} through beat {end}."
    )


def _span_text(payload: dict[str, Any]) -> str:
    """Return a canonical human-readable span description."""
    if str(payload.get("kind")) == "tie":
        return _tie_text(payload)
    start = f"{float(payload['start_beat']):g}"
    end = f"{float(payload['end_beat']):g}"
    start_label = "grace beat" if payload.get("start_is_grace") else "beat"
    end_label = "grace beat" if payload.get("end_is_grace") else "beat"
    if "start_measure" in payload and "end_measure" in payload:
        return (
            f"Part {payload['part_index']} voice {payload['voice']}: "
            f"{payload['kind']} {payload['value']} from measure "
            f"{payload['start_measure']} {start_label} {start} to measure "
            f"{payload['end_measure']} {end_label} {end}."
        )
    return (
        f"Part {payload['part_index']} measure {payload['measure']} "
        f"voice {payload['voice']}: {payload['kind']} {payload['value']} "
        f"from {start_label} {start} to {end_label} {end}."
    )


def _tie_text(payload: dict[str, Any]) -> str:
    """Return a canonical human-readable tie endpoint description."""
    start = f"{float(payload['start_beat']):g}"
    end = f"{float(payload['end_beat']):g}"
    start_event = _tie_event_label(
        str(payload.get("event_kind") or "note"),
        payload.get("pitch"),
        float(payload["start_duration"]),
    )
    end_event = _tie_event_label(
        str(payload.get("event_kind") or "note"),
        payload.get("end_pitch", payload.get("pitch")),
        float(payload["end_duration"]),
    )
    return (
        f"Part {payload['part_index']} voice {payload['voice']}: tie the "
        f"{start_event} at measure {payload['start_measure']} beat {start} "
        f"to the {end_event} at measure {payload['end_measure']} beat {end}."
    )


def _tie_event_label(kind: str, pitch: Any, duration: float) -> str:
    """Return a compact tied note or chord endpoint label."""
    label = duration_label(float(duration))
    if kind == "chord":
        pitches = ", ".join(str(item) for item in _normalize_json(pitch))
        return f"{label} chord ({pitches})"
    return f"{label} {pitch} note"


def _normalize_json(value: Any) -> Any:
    """Normalize nested JSON-like values for stable fact identity."""
    if isinstance(value, dict):
        return {
            str(key): _normalize_json(inner)
            for key, inner in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json(item) for item in value]
    if isinstance(value, float):
        return _rounded_float(value)
    return value


def _rounded_float(value: Any) -> float:
    """Return a stable rounded float."""
    return round(float(value), 9)


def slug(value: Any) -> str:
    """Return an ASCII slug for identifiers."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "item"
