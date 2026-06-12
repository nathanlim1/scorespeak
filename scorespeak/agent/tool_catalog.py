"""Searchable catalog of ScoreSpeak agent tool capabilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ..retrieval import (
    MethodRecord,
    SYNONYM_MAP,
    _is_add_rest_visibility_query,
    _is_fill_measure_gaps_query,
    _is_hide_rest_query,
    _is_rest_spelling_query,
    _phrase_synonyms,
)


@dataclass(frozen=True)
class ToolCatalogEntry:
    """One searchable tool/capability entry returned to the agent."""

    name: str
    signature: str
    mixin: str
    description: str
    tags: list[str]


class ToolCatalog:
    """Search over all ScoreSpeak method records available to the agent."""

    def __init__(self, records: Iterable[MethodRecord]) -> None:
        self._records = list(records)
        self._records_by_name = {record.name: record for record in self._records}
        self._record_positions = {
            record.name: index
            for index, record in enumerate(self._records)
        }

    @property
    def records(self) -> list[MethodRecord]:
        """All catalog records in stable index order."""
        return self._records

    @property
    def names(self) -> set[str]:
        """Names of all cataloged ScoreSpeak methods."""
        return set(self._records_by_name)

    def get(self, name: str) -> MethodRecord | None:
        """Return the record for ``name`` if it exists."""
        return self._records_by_name.get(name)

    def resolve_names(self, names: Iterable[str] | None) -> tuple[list[str], list[str]]:
        """Split requested tool names into valid and invalid names."""
        if names is None:
            return [], []

        valid: list[str] = []
        invalid: list[str] = []
        seen_valid: set[str] = set()
        seen_invalid: set[str] = set()
        for raw_name in names:
            name = str(raw_name).strip()
            if not name:
                continue
            if name in self._records_by_name:
                if name not in seen_valid:
                    valid.append(name)
                    seen_valid.add(name)
                continue
            if name not in seen_invalid:
                invalid.append(name)
                seen_invalid.add(name)
        return valid, invalid

    def search(
        self,
        query: str,
        *,
        tool_names: Iterable[str] | None = None,
        limit: int = 8,
    ) -> list[ToolCatalogEntry]:
        """Search capabilities by natural-language query and/or exact names."""
        selected: list[MethodRecord] = []
        seen: set[str] = set()

        valid_names, _ = self.resolve_names(tool_names)
        for name in valid_names:
            record = self._records_by_name[name]
            selected.append(record)
            seen.add(record.name)

        if query.strip():
            for record in self._ranked_query_matches(query):
                if record.name in seen:
                    continue
                selected.append(record)
                seen.add(record.name)
                if len(selected) >= limit:
                    break

        return [self._entry_for(record) for record in selected[:limit]]

    def _ranked_query_matches(self, query: str) -> list[MethodRecord]:
        """Return query matches sorted by intent-aware lexical score."""
        tokens = _tokenize(query)
        if not tokens:
            return []

        expanded_tokens = _expand_tokens(tokens)
        query_verbs = expanded_tokens & _VERBS
        normalized_query = " ".join(tokens)
        scored: list[tuple[float, int, int, MethodRecord]] = []
        for record in self._records:
            score = _score_record(
                record,
                tokens,
                expanded_tokens,
                query_verbs,
                normalized_query,
            )
            if score > 0:
                scored.append((
                    score,
                    len(record.tags),
                    self._record_positions.get(record.name, len(self._records)),
                    record,
                ))

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [record for _score, _tag_count, _position, record in scored]

    @staticmethod
    def _entry_for(record: MethodRecord) -> ToolCatalogEntry:
        """Convert a method record to a compact catalog entry."""
        return ToolCatalogEntry(
            name=record.name,
            signature=record.signature,
            mixin=record.mixin,
            description=_short_description(record.docstring),
            tags=sorted(record.tags),
        )


def _tokenize(text: str) -> list[str]:
    """Tokenize free text into lowercase search tokens."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


_VERBS = frozenset({
    "add",
    "insert",
    "remove",
    "clear",
    "set",
    "get",
    "list",
    "replace",
    "copy",
    "reshape",
    "fill",
})


def _expand_tokens(tokens: list[str]) -> set[str]:
    """Expand query tokens through the shared music synonym map."""
    expanded = set(tokens)
    for token in tokens:
        expanded.update(SYNONYM_MAP.get(token, []))
    expanded.update(_phrase_synonyms(" ".join(tokens)))
    return expanded


def _score_record(
    record: MethodRecord,
    tokens: list[str],
    expanded_tokens: set[str],
    query_verbs: set[str],
    normalized_query: str,
) -> float:
    """Return an intent-aware search score for a tool record."""
    name_text = record.name.replace("_", " ")
    searchable_text = " ".join(
        [
            name_text,
            record.signature,
            record.docstring,
            " ".join(record.tags),
        ]
    ).lower()

    tag_overlap = expanded_tokens & set(record.tags)
    text_overlap = sum(1 for token in tokens if token in searchable_text)
    score = float(len(tag_overlap) * 10 + text_overlap)

    if record.name in normalized_query or name_text in normalized_query:
        score += 20.0

    method_verb = record.name.split("_", 1)[0]
    if query_verbs and method_verb in _VERBS:
        if method_verb in query_verbs:
            score += 6.0
        else:
            score -= 4.0

    if record.name == "add_rest" and _is_add_rest_visibility_query(
        normalized_query
    ):
        score += 30.0
    elif record.name == "fill_measure_gaps" and _is_fill_measure_gaps_query(
        normalized_query
    ):
        score += 30.0
    elif record.name == "reshape_rests" and _is_rest_spelling_query(normalized_query):
        score += 30.0
    elif record.name == "remove_rests" and _is_hide_rest_query(normalized_query):
        score += 30.0

    return max(0.0, score)


def _short_description(docstring: str, limit: int = 240) -> str:
    """Return the first docstring paragraph capped to ``limit`` characters."""
    if not docstring:
        return ""
    first = docstring.strip().split("\n\n", 1)[0].strip()
    if len(first) <= limit:
        return first
    return first[: limit - 1].rstrip() + "…"
