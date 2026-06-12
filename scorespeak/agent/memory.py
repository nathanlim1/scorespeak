"""Session-scoped memory for ScoreSpeak agent turns."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Optional, Sequence

from langchain_core.tools import StructuredTool


MemoryPool = Literal["instruction", "execution"]
MemoryPromptDetail = Literal["brief", "full_previous_turn"]

_VALID_POOLS: tuple[MemoryPool, ...] = ("instruction", "execution")
_MAX_TOOL_RESULT_CHARS = 360
_MAX_TEXT_CHARS = 4000
_MAX_TOOL_ARGUMENT_CHARS = 500
_MAX_TOOL_ARGUMENT_ITEMS = 20
_MAX_SEARCH_LIMIT = 20
_MAX_PROMPT_INSTRUCTION_CHARS = 260
_MAX_PROMPT_RESULT_CHARS = 360
_MAX_PROMPT_ERROR_CHARS = 220
_MAX_PROMPT_TOOL_NAMES = 5
_RESULT_DETAILS_SEPARATOR = " | details="
_OMIT_MEMORY_RESULT_TOOL_NAMES = frozenset({
    "inspect_score_attributes",
    "inspect_score_region",
    "memory_search",
    "search_score",
    "tool_search",
})

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_#b]+")
_MEASURE_PATTERN = re.compile(
    r"\b(?:m(?:easure)?|measures|bar|bars)\s+"
    r"(\d+)(?:\s*(?:-|to|through|thru)\s*(\d+))?",
    re.IGNORECASE,
)
_STANDALONE_RANGE_PATTERN = re.compile(r"\b(\d+)\s*[-]\s*(\d+)\b")
_PART_REFERENCE_PATTERN = re.compile(
    r"\b(?:part|staff)\s+(?:named\s+|called\s+)?([a-zA-Z0-9_#-]+)",
    re.IGNORECASE,
)
_MEASURE_KEYS = frozenset({
    "bar",
    "bars",
    "bar_range",
    "end_bar",
    "end_measure",
    "measure",
    "measure_number",
    "measure_numbers",
    "measure_range",
    "measures",
    "start_bar",
    "start_measure",
})
_PART_KEYS = frozenset({
    "part",
    "part_id",
    "part_ids",
    "part_index",
    "part_indices",
    "part_name",
    "part_names",
    "parts",
    "staff",
    "staff_id",
    "staff_name",
})
_KNOWN_PART_WORDS = frozenset({
    "bass",
    "cello",
    "clarinet",
    "contrabass",
    "flute",
    "guitar",
    "horn",
    "oboe",
    "organ",
    "percussion",
    "piano",
    "soprano",
    "tenor",
    "trombone",
    "trumpet",
    "viola",
    "violin",
    "voice",
})


@dataclass(frozen=True)
class InstructionMemoryEntry:
    """Prior user prompt and lightweight structured hints."""

    turn_id: str
    timestamp: datetime
    user_prompt: str
    measure_hints: tuple[int, ...]
    measure_ranges: tuple[tuple[int, int], ...]
    part_hints: tuple[str, ...]


@dataclass(frozen=True)
class MemoryToolCallTrace:
    """Compact record of one agent tool call and its result."""

    name: str
    arguments: dict[str, Any]
    result: str | None = None
    ok: bool | None = None


@dataclass(frozen=True)
class ExecutionMemoryEntry:
    """Completed agent turn summary and compact tool-call trace."""

    turn_id: str
    timestamp: datetime
    tool_trace: tuple[MemoryToolCallTrace, ...]
    final_response: str
    status: str
    error: str | None = None


@dataclass(frozen=True)
class _QueryFeatures:
    """Structured search features extracted from a query."""

    raw_text: str
    tokens: frozenset[str]
    measure_numbers: frozenset[int]
    measure_ranges: tuple[tuple[int, int], ...]
    part_hints: frozenset[str]


@dataclass(frozen=True)
class _IndexedMemoryEntry:
    """Searchable projection of one memory entry."""

    pool: MemoryPool
    entry: InstructionMemoryEntry | ExecutionMemoryEntry
    text: str
    tokens: frozenset[str]
    measure_numbers: frozenset[int]
    measure_ranges: tuple[tuple[int, int], ...]
    part_hints: frozenset[str]
    tool_names: frozenset[str]


class LexicalMemorySearch:
    """Lexical and structured memory search isolated from the store."""

    def rank(
        self,
        entries: Iterable[tuple[MemoryPool, InstructionMemoryEntry | ExecutionMemoryEntry]],
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return ranked memory payloads for a non-empty query."""
        features = _query_features(query)
        scored: list[tuple[float, datetime, dict[str, Any]]] = []

        for pool, entry in entries:
            indexed = _index_entry(pool, entry)
            score = _score_indexed_entry(indexed, features)
            if score <= 0:
                continue
            payload = _entry_payload(pool, entry, score=score)
            scored.append((score, entry.timestamp, payload))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [payload for _, _, payload in scored[:limit]]


class AgentMemoryStore:
    """Append-only bounded in-memory store for one agent session."""

    def __init__(self, max_entries_per_pool: int = 200) -> None:
        """Create an empty memory store with per-pool bounds."""
        self._max_entries_per_pool = max(1, int(max_entries_per_pool))
        self._instruction_entries: list[InstructionMemoryEntry] = []
        self._execution_entries: list[ExecutionMemoryEntry] = []
        self._turn_counter = 0
        self._search = LexicalMemorySearch()

    @property
    def instruction_entries(self) -> tuple[InstructionMemoryEntry, ...]:
        """Return instruction entries in append order."""
        return tuple(self._instruction_entries)

    @property
    def execution_entries(self) -> tuple[ExecutionMemoryEntry, ...]:
        """Return execution entries in append order."""
        return tuple(self._execution_entries)

    def has_entries(self) -> bool:
        """Return whether any memory entry exists."""
        return bool(self._instruction_entries or self._execution_entries)

    def next_turn_id(self) -> str:
        """Allocate a stable session-scoped turn id."""
        self._turn_counter += 1
        return f"turn-{self._turn_counter}"

    def append_instruction(
        self,
        turn_id: str,
        user_prompt: str,
        *,
        timestamp: datetime | None = None,
    ) -> InstructionMemoryEntry:
        """Append a user prompt to instruction memory."""
        self._observe_turn_id(turn_id)
        prompt = _compact_text(user_prompt, _MAX_TEXT_CHARS)
        measures, ranges = _extract_measure_hints(prompt)
        entry = InstructionMemoryEntry(
            turn_id=str(turn_id),
            timestamp=_timestamp(timestamp),
            user_prompt=prompt,
            measure_hints=tuple(sorted(measures)),
            measure_ranges=tuple(ranges),
            part_hints=tuple(sorted(_extract_part_hints_from_text(prompt))),
        )
        self._instruction_entries.append(entry)
        self._trim_pool(self._instruction_entries)
        return entry

    def append_execution(
        self,
        turn_id: str,
        tool_trace: Sequence[MemoryToolCallTrace],
        final_response: str,
        status: str,
        *,
        error: str | None = None,
        timestamp: datetime | None = None,
    ) -> ExecutionMemoryEntry:
        """Append completed turn execution details to execution memory."""
        self._observe_turn_id(turn_id)
        entry = ExecutionMemoryEntry(
            turn_id=str(turn_id),
            timestamp=_timestamp(timestamp),
            tool_trace=tuple(_compact_trace_item(item) for item in tool_trace),
            final_response=_compact_text(final_response, _MAX_TEXT_CHARS),
            status=str(status),
            error=_compact_text(error, _MAX_TEXT_CHARS) if error else None,
        )
        self._execution_entries.append(entry)
        self._trim_pool(self._execution_entries)
        return entry

    def record_turn(
        self,
        turn_id: str,
        user_prompt: str,
        tool_trace: Sequence[MemoryToolCallTrace],
        final_response: str,
        status: str,
        *,
        error: str | None = None,
    ) -> tuple[InstructionMemoryEntry, ExecutionMemoryEntry]:
        """Append instruction and execution memory for one completed turn."""
        instruction = self.append_instruction(turn_id, user_prompt)
        execution = self.append_execution(
            turn_id,
            tool_trace,
            final_response,
            status,
            error=error,
        )
        return instruction, execution

    def search(
        self,
        query: str = "",
        pools: Sequence[str] | None = None,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        """Retrieve recent entries for blank queries or ranked search matches."""
        safe_limit = _normalize_limit(limit)
        normalized_pools = _normalize_pools(pools)
        entries = self._entries_for_pools(normalized_pools)
        if not query.strip():
            return self._recent_payloads(entries, safe_limit)
        return self._search.rank(entries, query, safe_limit)

    def _entries_for_pools(
        self,
        pools: Sequence[MemoryPool],
    ) -> list[tuple[MemoryPool, InstructionMemoryEntry | ExecutionMemoryEntry]]:
        """Return stored entries for the requested pools."""
        entries: list[tuple[MemoryPool, InstructionMemoryEntry | ExecutionMemoryEntry]] = []
        if "instruction" in pools:
            entries.extend(("instruction", entry) for entry in self._instruction_entries)
        if "execution" in pools:
            entries.extend(("execution", entry) for entry in self._execution_entries)
        return entries

    def _recent_payloads(
        self,
        entries: Sequence[tuple[MemoryPool, InstructionMemoryEntry | ExecutionMemoryEntry]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return newest entries across the requested pools."""
        newest = sorted(entries, key=lambda item: item[1].timestamp, reverse=True)
        return [
            _entry_payload(pool, entry, score=None)
            for pool, entry in newest[:limit]
        ]

    def _trim_pool(
        self,
        entries: list[InstructionMemoryEntry] | list[ExecutionMemoryEntry],
    ) -> None:
        """Trim a pool to the configured bound."""
        overflow = len(entries) - self._max_entries_per_pool
        if overflow > 0:
            del entries[:overflow]

    def _observe_turn_id(self, turn_id: str) -> None:
        """Advance the internal counter when callers append explicit turn ids."""
        match = re.fullmatch(r"turn-(\d+)", str(turn_id))
        if match is not None:
            self._turn_counter = max(self._turn_counter, int(match.group(1)))


def format_memory_context_for_prompt(
    memory_store: AgentMemoryStore,
    detail: MemoryPromptDetail = "brief",
) -> str:
    """Render direct previous-turn memory context for the agent prompt."""
    if detail not in {"brief", "full_previous_turn"}:
        raise ValueError(
            "detail must be either 'brief' or 'full_previous_turn'."
        )

    turn_id = _latest_memory_turn_id(memory_store)
    if turn_id is None:
        return ""

    instruction = _latest_instruction_for_turn(memory_store, turn_id)
    execution = _latest_execution_for_turn(memory_store, turn_id)
    lines = _memory_context_header(detail)
    if instruction is not None:
        lines.append(_format_prompt_instruction(instruction, detail))
    if execution is not None:
        if detail == "full_previous_turn":
            lines.extend(_format_full_prompt_execution(execution))
        else:
            summary = _execution_prompt_summary(execution)
            if summary:
                lines.append(f"- Previous result ({execution.turn_id}): {summary}")
    return "\n".join(lines)


def _memory_context_header(detail: MemoryPromptDetail) -> list[str]:
    """Return common prompt guidance for previous-turn memory."""
    if detail == "full_previous_turn":
        detail_text = (
            "Full compact previous-turn detail is shown because current part "
            "or bar scope is missing."
        )
    else:
        detail_text = (
            "Brief previous-turn summary is shown because current part and "
            "bar scope are resolved."
        )
    return [
        "DIRECT PREVIOUS TURN MEMORY (SESSION MEMORY):",
        (
            "- This is the immediately previous completed user request, agent "
            "response, and actions. The current user message is not stored "
            "here yet."
        ),
        (
            "- Use this previous turn to resolve references like same, that, "
            "there, again, previous, and missing part/bar scope unless the "
            "current request gives a conflicting explicit target."
        ),
        (
            "- Use memory_search only for older turns or exact details not "
            "shown here; blank query gets newest memory, query text searches "
            "prior instructions/results."
        ),
        f"- Detail mode: {detail}. {detail_text}",
    ]


def _format_prompt_instruction(
    entry: InstructionMemoryEntry,
    detail: MemoryPromptDetail,
) -> str:
    """Render the previous user request for prompt memory."""
    limit = (
        _MAX_TEXT_CHARS
        if detail == "full_previous_turn"
        else _MAX_PROMPT_INSTRUCTION_CHARS
    )
    prompt = _compact_text(entry.user_prompt, limit)
    return f"- Previous user request ({entry.turn_id}): {prompt}"


def _format_full_prompt_execution(entry: ExecutionMemoryEntry) -> list[str]:
    """Render full compact execution details for prompt memory."""
    lines = []
    if entry.final_response:
        response = _compact_text(entry.final_response, _MAX_TEXT_CHARS)
        lines.append(f"- Previous agent response ({entry.turn_id}): {response}")
    if entry.status != "ok":
        lines.append(f"- Previous status ({entry.turn_id}): {entry.status}")
    if entry.error:
        error = _compact_text(entry.error, _MAX_TEXT_CHARS)
        lines.append(f"- Previous error ({entry.turn_id}): {error}")
    if entry.tool_trace:
        lines.append("- Previous tool calls/actions:")
        for index, item in enumerate(entry.tool_trace, start=1):
            lines.append(_format_full_prompt_tool_call(index, item))
    return lines


def _format_full_prompt_tool_call(
    index: int,
    item: MemoryToolCallTrace,
) -> str:
    """Render one stored tool call with compact arguments and result."""
    name = item.name or "(unknown_tool)"
    fragments = [
        f"{index}. {name}",
        f"ok={item.ok}",
        f"args={_json_dumps(item.arguments)}",
    ]
    if item.result:
        result = _compact_tool_result_for_memory(item.name, item.result)
        if result:
            fragments.append(f"result={result}")
    return "  - " + "; ".join(fragments)


def _compact_tool_result_for_memory(tool_name: str, result: str) -> str | None:
    """Return a prompt-safe tool result, or omit noisy retrieval results."""
    if tool_name in _OMIT_MEMORY_RESULT_TOOL_NAMES:
        return None
    text = str(result)
    if _RESULT_DETAILS_SEPARATOR in text:
        text = text.split(_RESULT_DETAILS_SEPARATOR, 1)[0]
    return _compact_text(text, _MAX_TOOL_RESULT_CHARS)


def _latest_memory_turn_id(memory_store: AgentMemoryStore) -> str | None:
    """Return the turn id for the newest memory entry across both pools."""
    newest: tuple[datetime, str] | None = None
    for entry in (*memory_store.instruction_entries, *memory_store.execution_entries):
        candidate = (entry.timestamp, entry.turn_id)
        if newest is None or candidate > newest:
            newest = candidate
    if newest is None:
        return None
    return newest[1]


def _latest_instruction_for_turn(
    memory_store: AgentMemoryStore,
    turn_id: str,
) -> InstructionMemoryEntry | None:
    """Return the latest instruction entry for a turn if present."""
    for entry in reversed(memory_store.instruction_entries):
        if entry.turn_id == turn_id:
            return entry
    return None


def _latest_execution_for_turn(
    memory_store: AgentMemoryStore,
    turn_id: str,
) -> ExecutionMemoryEntry | None:
    """Return the latest execution entry for a turn if present."""
    for entry in reversed(memory_store.execution_entries):
        if entry.turn_id == turn_id:
            return entry
    return None


def _execution_prompt_summary(entry: ExecutionMemoryEntry) -> str:
    """Render one execution entry as a compact prompt-safe summary."""
    parts = []
    if entry.final_response:
        parts.append(
            "reply="
            f"{_compact_text(entry.final_response, _MAX_PROMPT_RESULT_CHARS)}"
        )
    tool_names = _tool_trace_prompt_summary(entry.tool_trace)
    if tool_names:
        parts.append(f"tools={tool_names}")
    if entry.status != "ok":
        parts.append(f"status={entry.status}")
    if entry.error:
        parts.append(f"error={_compact_text(entry.error, _MAX_PROMPT_ERROR_CHARS)}")
    return "; ".join(parts)


def _tool_trace_prompt_summary(tool_trace: Sequence[MemoryToolCallTrace]) -> str:
    """Return a bounded tool-name status summary for prompt memory."""
    summaries = []
    for item in tool_trace[:_MAX_PROMPT_TOOL_NAMES]:
        if not item.name:
            continue
        if item.ok is True:
            suffix = "ok"
        elif item.ok is False:
            suffix = "error"
        else:
            suffix = "unknown"
        summaries.append(f"{item.name}({suffix})")
    if len(tool_trace) > _MAX_PROMPT_TOOL_NAMES:
        summaries.append("...")
    return ", ".join(summaries)


def make_memory_search_tool(memory_store: AgentMemoryStore) -> StructuredTool:
    """Build the model-facing memory retrieval tool."""
    def memory_search(
        query: str = "",
        pools: Optional[list[str]] = None,
        limit: int = 6,
    ) -> str:
        """Search prior session memory or retrieve recent memory entries."""
        try:
            matches = memory_store.search(query=query, pools=pools, limit=limit)
        except ValueError as exc:
            return f"ERROR (ValueError): {exc}"
        payload = {
            "query": query,
            "pools": list(_normalize_pools(pools)),
            "matches": matches,
        }
        return _json_dumps(payload)

    return StructuredTool.from_function(
        func=memory_search,
        name="memory_search",
        description=(
            "memory_search(query='', pools=None, limit=6)\n\n"
            "Search prior completed turns in this session. Use it for older "
            "turns or exact details not present in DIRECT PREVIOUS TURN "
            "MEMORY. Blank query returns newest entries; non-empty query "
            "searches instruction prompts and execution summaries. pools may "
            "contain 'instruction' and/or 'execution'."
        ),
    )


def _timestamp(value: datetime | None) -> datetime:
    """Return an aware UTC timestamp."""
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalize_limit(limit: int) -> int:
    """Clamp a user-supplied memory search limit."""
    return max(1, min(int(limit), _MAX_SEARCH_LIMIT))


def _normalize_pools(pools: Sequence[str] | None) -> tuple[MemoryPool, ...]:
    """Validate and normalize memory pool names."""
    if pools is None:
        return _VALID_POOLS
    normalized: list[MemoryPool] = []
    invalid: list[str] = []
    for pool in pools:
        text = str(pool).strip().lower()
        if text in _VALID_POOLS:
            normalized.append(text)  # type: ignore[arg-type]
        elif text:
            invalid.append(text)
    if invalid:
        raise ValueError(f"Unknown memory pool(s): {', '.join(invalid)}")
    if not normalized:
        return _VALID_POOLS
    return tuple(dict.fromkeys(normalized))


def _entry_payload(
    pool: MemoryPool,
    entry: InstructionMemoryEntry | ExecutionMemoryEntry,
    *,
    score: float | None,
) -> dict[str, Any]:
    """Serialize one memory entry for tool output."""
    payload: dict[str, Any] = {
        "pool": pool,
        "turn_id": entry.turn_id,
        "timestamp": entry.timestamp.isoformat(),
    }
    if score is not None:
        payload["score"] = round(score, 3)
    if isinstance(entry, InstructionMemoryEntry):
        payload.update({
            "user_prompt": entry.user_prompt,
            "measure_hints": list(entry.measure_hints),
            "measure_ranges": [list(item) for item in entry.measure_ranges],
            "part_hints": list(entry.part_hints),
        })
    else:
        payload.update({
            "tool_trace": [asdict(item) for item in entry.tool_trace],
            "final_response": entry.final_response,
            "status": entry.status,
            "error": entry.error,
        })
    return payload


def _index_entry(
    pool: MemoryPool,
    entry: InstructionMemoryEntry | ExecutionMemoryEntry,
) -> _IndexedMemoryEntry:
    """Build a searchable projection for one entry."""
    if isinstance(entry, InstructionMemoryEntry):
        text = " ".join(
            [
                entry.user_prompt,
                " ".join(str(item) for item in entry.measure_hints),
                " ".join(f"{start}-{end}" for start, end in entry.measure_ranges),
                " ".join(entry.part_hints),
            ]
        )
        return _IndexedMemoryEntry(
            pool=pool,
            entry=entry,
            text=text,
            tokens=_tokenize(text),
            measure_numbers=frozenset(entry.measure_hints),
            measure_ranges=entry.measure_ranges,
            part_hints=frozenset(entry.part_hints),
            tool_names=frozenset(),
        )

    tool_names = frozenset(item.name for item in entry.tool_trace if item.name)
    trace_text = " ".join(_trace_text(item) for item in entry.tool_trace)
    text = " ".join(
        part
        for part in [
            trace_text,
            entry.final_response,
            entry.status,
            entry.error or "",
        ]
        if part
    )
    measures, ranges = _extract_measure_hints(text)
    parts = _extract_part_hints_from_text(text)
    for trace_item in entry.tool_trace:
        measures.update(_extract_measure_hints_from_value(trace_item.arguments))
        argument_ranges = _extract_measure_ranges_from_value(trace_item.arguments)
        ranges.extend(argument_ranges)
        for start, end in argument_ranges:
            measures.update(_bounded_range_numbers(start, end))
        parts.update(_extract_part_hints_from_value(trace_item.arguments))
    return _IndexedMemoryEntry(
        pool=pool,
        entry=entry,
        text=text,
        tokens=_tokenize(text).union(_tool_name_tokens(tool_names)),
        measure_numbers=frozenset(measures),
        measure_ranges=tuple(ranges),
        part_hints=frozenset(parts),
        tool_names=tool_names,
    )


def _query_features(query: str) -> _QueryFeatures:
    """Extract lexical and structured query features."""
    measures, ranges = _extract_measure_hints(query)
    return _QueryFeatures(
        raw_text=query.lower(),
        tokens=_tokenize(query),
        measure_numbers=frozenset(measures),
        measure_ranges=tuple(ranges),
        part_hints=frozenset(_extract_part_hints_from_text(query)),
    )


def _score_indexed_entry(indexed: _IndexedMemoryEntry, query: _QueryFeatures) -> float:
    """Score one indexed entry against query features."""
    token_overlap = query.tokens.intersection(indexed.tokens)
    score = float(len(token_overlap))
    if query.raw_text and query.raw_text in indexed.text.lower():
        score += 2.0

    measure_matches = query.measure_numbers.intersection(indexed.measure_numbers)
    if measure_matches:
        score += 3.0 * len(measure_matches)
    contained_measures = [
        measure
        for measure in query.measure_numbers
        if any(start <= measure <= end for start, end in indexed.measure_ranges)
    ]
    if contained_measures:
        score += 2.5 * len(contained_measures)
    score += 3.0 * _range_overlap_count(query.measure_ranges, indexed.measure_ranges)

    part_matches = query.part_hints.intersection(indexed.part_hints)
    if part_matches:
        score += 2.5 * len(part_matches)

    for tool_name in indexed.tool_names:
        lowered = tool_name.lower()
        if lowered in query.raw_text:
            score += 4.0
            continue
        pieces = set(lowered.split("_"))
        if pieces and pieces.issubset(query.tokens):
            score += 2.0

    return score


def _range_overlap_count(
    query_ranges: Sequence[tuple[int, int]],
    entry_ranges: Sequence[tuple[int, int]],
) -> int:
    """Count overlapping structured measure ranges."""
    count = 0
    for query_start, query_end in query_ranges:
        for entry_start, entry_end in entry_ranges:
            if query_start <= entry_end and entry_start <= query_end:
                count += 1
                break
    return count


def _tokenize(text: str) -> frozenset[str]:
    """Return normalized lexical tokens."""
    tokens = []
    for match in _TOKEN_PATTERN.finditer(text.lower()):
        token = match.group(0).strip("_")
        if not token:
            continue
        tokens.append(token)
        if "_" in token:
            tokens.extend(part for part in token.split("_") if part)
    return frozenset(tokens)


def _extract_measure_hints(text: str) -> tuple[set[int], list[tuple[int, int]]]:
    """Extract measure numbers and ranges from free text."""
    measures: set[int] = set()
    ranges: list[tuple[int, int]] = []
    for match in _MEASURE_PATTERN.finditer(text):
        start = int(match.group(1))
        end_text = match.group(2)
        if end_text is None:
            measures.add(start)
            continue
        end = int(end_text)
        low, high = sorted((start, end))
        ranges.append((low, high))
        measures.update(range(low, high + 1))
    for match in _STANDALONE_RANGE_PATTERN.finditer(text):
        start = int(match.group(1))
        end = int(match.group(2))
        low, high = sorted((start, end))
        ranges.append((low, high))
    return measures, _dedupe_ranges(ranges)


def _dedupe_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return ranges without duplicates while preserving order."""
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    for item in ranges:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _extract_measure_hints_from_value(value: Any) -> set[int]:
    """Extract measure numbers from structured tool arguments/results."""
    return _extract_measure_hints_from_value_impl(value, structured_key=False)


def _extract_measure_hints_from_value_impl(value: Any, *, structured_key: bool) -> set[int]:
    """Extract measure numbers from nested values with key context."""
    measures: set[int] = set()
    if isinstance(value, bool) or value is None:
        return measures
    if isinstance(value, int):
        if structured_key and value > 0:
            measures.add(value)
        return measures
    if isinstance(value, float):
        if structured_key and value.is_integer() and value > 0:
            measures.add(int(value))
        return measures
    if isinstance(value, str):
        text_measures, _ = _extract_measure_hints(value)
        return text_measures
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key).lower()
            if key in _MEASURE_KEYS or key.endswith("_measure"):
                measures.update(_extract_measure_hints_from_value_impl(item, structured_key=True))
            elif isinstance(item, (dict, list, tuple, set, frozenset)):
                measures.update(_extract_measure_hints_from_value_impl(item, structured_key=False))
        return measures
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            measures.update(_extract_measure_hints_from_value_impl(item, structured_key=structured_key))
    return measures


def _extract_measure_ranges_from_value(value: Any) -> list[tuple[int, int]]:
    """Extract measure ranges from structured tool arguments/results."""
    ranges: list[tuple[int, int]] = []
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key).lower()
            if key in {"bar_range", "measure_range"}:
                parsed = _range_from_value(item)
                if parsed is not None:
                    ranges.append(parsed)
            elif isinstance(item, (dict, list, tuple, set, frozenset)):
                ranges.extend(_extract_measure_ranges_from_value(item))
        start = _first_int(
            value.get("start_measure", value.get("start_bar"))
        )
        end = _first_int(value.get("end_measure", value.get("end_bar")))
        if start is not None and end is not None:
            ranges.append(tuple(sorted((start, end))))
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            ranges.extend(_extract_measure_ranges_from_value(item))
    return _dedupe_ranges(ranges)


def _range_from_value(value: Any) -> tuple[int, int] | None:
    """Parse a two-item value into a normalized range."""
    if isinstance(value, str):
        match = _STANDALONE_RANGE_PATTERN.search(value)
        if match is not None:
            return tuple(sorted((int(match.group(1)), int(match.group(2)))))
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start = _first_int(value[0])
    end = _first_int(value[1])
    if start is None or end is None:
        return None
    return tuple(sorted((start, end)))


def _first_int(value: Any) -> int | None:
    """Return a positive integer from a scalar value when possible."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def _bounded_range_numbers(start: int, end: int) -> set[int]:
    """Return measure numbers in a range, capped to avoid large expansions."""
    low, high = sorted((start, end))
    if high - low > 500:
        return {low, high}
    return set(range(low, high + 1))


def _extract_part_hints_from_text(text: str) -> set[str]:
    """Extract likely part names or ids from free text."""
    hints = {
        _normalize_part_hint(match.group(1))
        for match in _PART_REFERENCE_PATTERN.finditer(text)
    }
    tokens = _tokenize(text)
    hints.update(token for token in tokens if token in _KNOWN_PART_WORDS)
    return {hint for hint in hints if hint}


def _extract_part_hints_from_value(value: Any) -> set[str]:
    """Extract likely part hints from structured tool arguments/results."""
    return _extract_part_hints_from_value_impl(value, structured_key=False)


def _extract_part_hints_from_value_impl(value: Any, *, structured_key: bool) -> set[str]:
    """Extract likely part hints from nested values with key context."""
    hints: set[str] = set()
    if value is None or isinstance(value, bool):
        return hints
    if isinstance(value, (int, float)):
        if structured_key:
            hints.add(_normalize_part_hint(str(int(value))))
        return hints
    if isinstance(value, str):
        if structured_key:
            hints.add(_normalize_part_hint(value))
        hints.update(_extract_part_hints_from_text(value))
        return {hint for hint in hints if hint}
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key).lower()
            if key in _PART_KEYS or key.endswith("_part"):
                hints.update(_extract_part_hints_from_value_impl(item, structured_key=True))
            elif isinstance(item, (dict, list, tuple, set, frozenset)):
                hints.update(_extract_part_hints_from_value_impl(item, structured_key=False))
        return hints
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            hints.update(_extract_part_hints_from_value_impl(item, structured_key=structured_key))
    return {hint for hint in hints if hint}


def _normalize_part_hint(value: str) -> str:
    """Normalize a part hint for structured matching."""
    text = re.sub(r"[^a-zA-Z0-9#b]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", text)


def _trace_text(item: MemoryToolCallTrace) -> str:
    """Render a compact trace item as searchable text."""
    return " ".join(
        part
        for part in [
            item.name,
            _json_dumps(item.arguments),
            item.result or "",
            "" if item.ok is None else str(item.ok),
        ]
        if part
    )


def _tool_name_tokens(tool_names: Iterable[str]) -> frozenset[str]:
    """Return searchable tokens from tool names."""
    tokens: set[str] = set()
    for name in tool_names:
        lowered = name.lower()
        tokens.add(lowered)
        tokens.update(part for part in lowered.split("_") if part)
    return frozenset(tokens)


def _compact_trace_item(item: MemoryToolCallTrace) -> MemoryToolCallTrace:
    """Return a bounded copy of a tool trace item."""
    return MemoryToolCallTrace(
        name=_compact_text(item.name, 120),
        arguments=_compact_arguments(item.arguments),
        result=(
            _compact_tool_result_for_memory(item.name, item.result)
            if item.result
            else None
        ),
        ok=item.ok,
    )


def _compact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return compact JSON-like tool arguments."""
    compacted: dict[str, Any] = {}
    for key, value in arguments.items():
        compacted[str(key)] = _compact_argument_value(value)
    return compacted


def _compact_argument_value(value: Any) -> Any:
    """Bound one tool argument value while preserving simple structure."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            return _compact_text(value, _MAX_TOOL_ARGUMENT_CHARS)
        return value
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_TOOL_ARGUMENT_ITEMS:
                compacted["__truncated__"] = True
                break
            compacted[str(key)] = _compact_argument_value(item)
        return compacted
    if isinstance(value, (list, tuple, set, frozenset)):
        compacted_list = [
            _compact_argument_value(item)
            for item in list(value)[:_MAX_TOOL_ARGUMENT_ITEMS]
        ]
        if len(value) > _MAX_TOOL_ARGUMENT_ITEMS:
            compacted_list.append("...")
        return compacted_list
    return _compact_text(str(value), _MAX_TOOL_ARGUMENT_CHARS)


def _compact_text(value: object, limit: int) -> str:
    """Collapse whitespace and truncate text."""
    text = "" if value is None else str(value)
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _json_dumps(value: Any) -> str:
    """Serialize a value with stable compact JSON formatting."""
    return json.dumps(value, default=str, separators=(",", ":"), sort_keys=True)
