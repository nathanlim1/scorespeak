"""
ScoreSpeak retrieval helpers for tools and lexical score context.

Maps natural language music-editing queries to ScoreSpeak API methods using
token overlap scoring with synonym expansion. Also provides a lightweight
context retriever that extracts explicit part/bar references from raw user
queries and retrieves the corresponding score bars without pitch or rhythm
matching.
"""

from __future__ import annotations

import inspect
import re
import warnings
from dataclasses import dataclass, field

from typing import Literal, Optional

from .core import ScoreSpeak
from .score.staff_groups import build_part_display_labels
from .types import BarQuery, BarResultSet


DEFAULT_AUTO_CONTEXT_MEASURE_LIMIT = 8
DEFAULT_AUTO_CONTEXT_PART_BAR_LIMIT = 16
PUBLIC_TOOL_EXCLUDED_NAMES = frozenset()


@dataclass
class MethodRecord:
    """Structured representation of a single ScoreSpeak public method."""

    name: str                  # e.g. "add_dynamic"
    mixin: str                 # e.g. "ExpressionsMixin"
    signature: str             # e.g. "(level, measure_number, beat=1.0, part=None)"
    docstring: str             # first paragraph of the docstring, or ""
    tags: frozenset[str]       # tokens from method name, e.g. frozenset({"add", "dynamic"})


SYNONYM_MAP: dict[str, list[str]] = {
    # Dynamics
    "forte":        ["dynamic"],
    "piano":        ["dynamic"],
    "loud":         ["dynamic"],
    "louder":       ["dynamic"],
    "softer":       ["dynamic"],
    "quiet":        ["dynamic"],
    "volume":       ["dynamic"],
    "pp":           ["dynamic"],
    "ff":           ["dynamic"],
    "mf":           ["dynamic"],
    "mp":           ["dynamic"],
    # Hairpins
    "crescendo":    ["hairpin"],
    "decrescendo":  ["hairpin"],
    "diminuendo":   ["hairpin"],
    "swell":        ["hairpin"],
    "fade":         ["hairpin"],
    # Structural
    "bar":          ["measure", "measures"],
    "bars":         ["measure", "measures"],
    "measure":      ["measure", "measures"],
    "measures":     ["measure", "measures"],
    "empty":        ["clear", "measure", "measures"],
    "blank":        ["clear", "measure", "measures"],
    "contents":     ["contents"],
    # Articulation
    "staccato":     ["articulation"],
    "staccatissimo": ["articulation"],
    "accent":       ["articulation"],
    "marcato":      ["articulation"],
    "tenuto":       ["articulation"],
    "fermata":      ["articulation"],
    "breath":       ["articulation"],
    "caesura":      ["articulation"],
    "bow":          ["articulation"],
    "upbow":        ["articulation"],
    "downbow":      ["articulation"],
    "harmonic":     ["articulation"],
    "stopped":      ["articulation"],
    # Slur / legato
    "legato":       ["slur"],
    "smooth":       ["slur"],
    "connect":      ["slur"],
    "tie":          ["tie"],
    # Pitch / notes
    "note":         ["notes"],
    "pitch":        ["note", "notes"],
    "notes":        ["note"],
    "tone":         ["note", "notes"],
    "sound":        ["note", "notes"],
    "rest":         ["rests"],
    "rests":        ["rest"],
    "silence":      ["rest", "rests"],
    "pause":        ["rest", "rests"],
    "gap":          ["gaps"],
    "gaps":         ["gap"],
    "missing":      ["gap", "gaps"],
    "hide":         ["remove"],
    "hidden":       ["remove"],
    "split":        ["reshape"],
    "merge":        ["reshape"],
    "reformat":     ["reshape"],
    "format":       ["reshape"],
    "spelling":     ["reshape"],
    "spell":        ["reshape"],
    # Chords
    "chord":        ["chord"],
    "harmony":      ["chord", "symbol"],
    # Tempo
    "speed":        ["tempo"],
    "bpm":          ["tempo"],
    "metronome":    ["tempo"],
    "faster":       ["tempo"],
    "slower":       ["tempo"],
    # Key / time
    "key":          ["key", "signature", "transpose"],
    "time":         ["time", "signature"],
    "meter":        ["time", "signature"],
    "signature":    ["signature"],
    # Ornaments
    "trill":        ["ornament"],
    "mordent":      ["ornament"],
    "turn":         ["ornament"],
    "grace":        ["grace"],
    # Lyrics
    "lyric":        ["lyric"],
    "lyrics":       ["lyric"],
    "text":         ["text", "lyric"],
    "word":         ["lyric"],
    "words":        ["lyric"],
    # Layout
    "title":        ["title"],
    "composer":     ["composer"],
    "transpose":    ["transpose"],
    "octave":       ["transpose", "ottava"],
    "8va":          ["ottava"],
    # Parts
    "instrument":   ["part", "parts", "score"],
    "instruments":  ["part", "parts", "score"],
    "instrumentation": ["part", "parts", "score"],
    "ensemble":     ["part", "parts", "score"],
    "roster":       ["part", "parts", "score"],
    "lineup":       ["part", "parts", "score"],
    "part":         ["parts", "score"],
    "parts":        ["part", "score"],
    "voice":        [
        "note", "notes", "rest", "rests", "chord", "measure", "measures",
    ],
    "voices":       [
        "note", "notes", "rest", "rests", "chord", "measure", "measures",
    ],
    "staff":        ["part", "parts", "score"],
    "staves":       ["part", "parts", "score"],
    # Pedal
    "pedal":        ["pedal"],
    "sustain":      ["pedal"],
    # Repeat
    "repeat":       ["repeat"],
    "volta":        ["ending"],
    "voltas":       ["ending"],
    # Rehearsal
    "rehearsal":    ["rehearsal"],
    "marker":       ["rehearsal"],
    # Glissando
    "gliss":        ["glissando"],
    "slide":        ["glissando"],
    # Arpeggio
    "arpeggio":     ["arpeggio"],
    "arpeggiate":   ["arpeggio"],
    "roll":         ["arpeggio"],
    # Tuplet
    "triplet":      ["tuplet"],
    "tuplet":       ["tuplet"],
    # Clef
    "clef":         ["clef"],
    "treble":       ["clef"],
    "bass":         ["clef"],
    # Barline
    "barline":      ["barline"],
    "double":       ["barline"],
    "final":        ["barline"],
    # Navigation marks
    "coda":         ["coda"],
    "segno":        ["segno"],
    "tocoda":       ["to", "coda"],
    "to_coda":      ["to", "coda"],
    "dc":           ["da", "capo"],
    "ds":           ["dal", "segno"],
    "navigation":   ["navigation", "mark"],
    "jump":         ["navigation", "mark"],
    "fine":         ["fine", "da", "capo", "dal", "segno"],
    # Chord symbols (note: "harmony" already maps to ["chord", "symbol"] above)
    # Tremolo ornament
    "tremolo":      ["ornament"],
    "trem":         ["ornament"],
    # Verbs — action words users say instead of the API verb tokens
    # "add" synonyms
    "make":         ["add"],
    "create":       ["add"],
    "put":          ["add"],
    "place":        ["add", "insert"],
    "write":        ["add"],
    "attach":       ["add"],
    "apply":        ["add"],
    "append":       ["add"],
    "unhide":       ["add"],
    "reveal":       ["add"],
    # "remove" synonyms
    "delete":       ["remove"],
    "erase":        ["clear", "remove"],
    "clear":        ["clear", "remove"],
    "take":         ["remove"],
    "strip":        ["remove"],
    "drop":         ["remove"],
    "eliminate":    ["remove"],
    # "set" synonyms
    "change":       ["set", "replace"],
    "update":       ["set", "replace"],
    "configure":    ["set"],
    "assign":       ["set"],
    "define":       ["set"],
    "specify":      ["set"],
    "adjust":       ["set"],
    "initialize":   ["set", "score", "part", "parts"],
    "initialise":   ["set", "score", "part", "parts"],
    "setup":        ["set", "score", "part", "parts"],
    "reset":        ["set", "replace"],
    # "get" synonyms
    "show":         ["get", "list"],
    "fetch":        ["get"],
    "find":         ["get"],
    "retrieve":     ["get"],
    "display":      ["get", "list"],
    "read":         ["get"],
    # "insert" synonyms
    "inject":       ["insert"],
    # "copy" synonyms
    "copy":         ["copy"],
    "paste":        ["copy"],
    "duplicate":    ["copy"],
    "clone":        ["copy"],
    "same":         ["copy"],
    "identical":    ["copy"],
    "mirror":       ["copy"],
    # "replace" synonyms
    "swap":         ["replace"],
    "substitute":   ["replace"],
    "overwrite":    ["replace"],
    # "list" synonyms
    "enumerate":    ["list"],
}


# Recognized verbs and their sort precedence (lower = earlier).
_VERB_ORDER: dict[str, int] = {
    "add": 0,
    "insert": 1,
    "remove": 2,
    "clear": 3,
    "set": 4,
    "get": 5,
    "list": 6,
    "replace": 7,
    "copy": 8,
    "reshape": 9,
    "fill": 10,
}

_PART_TOKEN_STOPWORDS = {"part", "parts", "staff", "staves", "voice", "voices"}
_BAR_PREFIX_PATTERN = re.compile(
    r"(?:\b(?:bars?|measures?)\b|mm\.?|m\.)",
    re.IGNORECASE,
)
_BAR_RANGE_AT_POSITION_PATTERN = re.compile(
    r"\s*(\d+)\s*(?:-|to|through)\s*(\d+)",
    re.IGNORECASE,
)
_BAR_NUMBER_AT_POSITION_PATTERN = re.compile(
    r"\s*(\d+)",
    re.IGNORECASE,
)
_BAR_SEPARATOR_AT_POSITION_PATTERN = re.compile(
    r"\s*(?:,|and)\s*",
    re.IGNORECASE,
)
_END_ORIENTED_CONTEXT_PATTERNS = (
    re.compile(r"\bat\s+the\s+end\b", re.IGNORECASE),
    re.compile(r"\bfinal\s+(?:bars?|measures?)\b", re.IGNORECASE),
    re.compile(r"\bcontinue\b", re.IGNORECASE),
    re.compile(r"\bappend\b", re.IGNORECASE),
    re.compile(r"\bafter\s+the\s+existing\s+music\b", re.IGNORECASE),
)

BarContextStatus = Literal["explicit", "end_fallback", "missing"]


def _domain_token(name: str) -> str:
    """Return the non-verb portion of a method name as a sort key.

    If the leading underscore-split token is a recognized verb, the domain
    is everything after it (joined by ``_``).  Otherwise the entire name is
    the domain.
    """
    parts = name.split("_")
    if parts[0] in _VERB_ORDER:
        return "_".join(parts[1:])
    return name


def _verb_sort_key(name: str) -> int:
    """Return the verb precedence for sorting within a domain cluster.

    Methods whose leading token is not a recognized verb sort last.
    """
    leading = name.split("_")[0]
    return _VERB_ORDER.get(leading, len(_VERB_ORDER))


class MethodIndex:
    """Index of all public methods on a ``ScoreSpeak`` instance.

    Built once at initialization by introspecting the live object.
    Records are sorted by mixin MRO order → domain token → verb precedence.
    """

    def __init__(self, score_state: object) -> None:
        members = inspect.getmembers(score_state, predicate=inspect.ismethod)
        public = [
            (n, m)
            for n, m in members
            if not n.startswith("_") and n not in PUBLIC_TOOL_EXCLUDED_NAMES
        ]

        if not public:
            raise ValueError(
                "MethodIndex: no public methods found on ScoreSpeak instance."
            )

        # Build a mapping: mixin class → MRO position (lower = earlier).
        mro = type(score_state).__mro__
        mro_rank: dict[str, int] = {cls.__name__: idx for idx, cls in enumerate(mro)}

        records: list[MethodRecord] = []
        for name, method in public:
            # Determine originating mixin via MRO walk.
            mixin_name = type(score_state).__name__  # fallback
            for cls in mro:
                if name in vars(cls):
                    mixin_name = cls.__name__
                    break

            # Signature
            try:
                sig = str(inspect.signature(method))
            except (ValueError, TypeError):
                warnings.warn(
                    f"MethodIndex: could not extract signature for '{name}'",
                    stacklevel=2,
                )
                sig = "(unknown)"

            # Docstring
            doc = inspect.getdoc(method) or ""
            if not doc:
                warnings.warn(
                    f"MethodIndex: method '{name}' has no docstring",
                    stacklevel=2,
                )

            tags = frozenset(name.split("_"))

            records.append(
                MethodRecord(
                    name=name,
                    mixin=mixin_name,
                    signature=sig,
                    docstring=doc,
                    tags=tags,
                )
            )

        # Sort: mixin MRO order → domain token → verb precedence.
        records.sort(
            key=lambda r: (
                mro_rank.get(r.mixin, len(mro)),
                _domain_token(r.name),
                _verb_sort_key(r.name),
            )
        )

        self._records: list[MethodRecord] = records

    @property
    def records(self) -> list[MethodRecord]:
        """All indexed method records, sorted by mixin/domain/verb order."""
        return self._records


class LexicalRetriever:
    """Lexical query engine over a :class:`MethodIndex`.

    Tokenizes a natural-language query, expands tokens through a synonym map,
    and scores each indexed method by the fraction of its tags covered by the
    expanded query tokens.  Returns all methods at or above the configured
    threshold, sorted descending by score.
    """

    def __init__(
        self,
        index: MethodIndex,
        synonym_map: dict[str, list[str]] | None = None,
        threshold: float = 0.5,
    ) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"threshold must be in [0.0, 1.0], got {threshold}"
            )
        self._index = index
        self._synonym_map: dict[str, list[str]] = (
            synonym_map if synonym_map is not None else SYNONYM_MAP
        )
        self._threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, text: str) -> list[tuple[MethodRecord, float]]:
        """Return ``(record, score)`` pairs for all records at or above
        the configured threshold, sorted descending by score.

        Scoring: ``score = |Q ∩ T| / |T|`` where *Q* is the expanded query
        token set and *T* is the record's tag set.

        A method is only included if at least one *domain* token (non-verb)
        overlaps with the expanded query.  This prevents verb-only matches
        like "add" pulling in every ``add_*`` method.
        """
        text_lower = text.lower()
        tokens = re.sub(r"[^\w\s]", "", text_lower).split()
        if not tokens:
            return []

        # Expand each token through the synonym map.
        expanded: set[str] = set()
        for tok in tokens:
            expanded.add(tok)
            if tok in self._synonym_map:
                expanded.update(self._synonym_map[tok])
        expanded.update(_phrase_synonyms(text_lower))

        _verbs = set(_VERB_ORDER)

        results: list[tuple[MethodRecord, float]] = []
        for record in self._index.records:
            tags = record.tags
            overlap = expanded & tags
            # Require at least one non-verb tag to match.
            if not (overlap - _verbs):
                continue
            score = len(overlap) / len(tags)
            if record.name == "add_rest" and _is_add_rest_visibility_query(
                text_lower
            ):
                score += 1.0
            elif record.name == "fill_measure_gaps" and _is_fill_measure_gaps_query(
                text_lower
            ):
                score += 1.0
            elif record.name == "reshape_rests" and _is_rest_spelling_query(text_lower):
                score += 1.0
            elif record.name == "remove_rests" and _is_hide_rest_query(text_lower):
                score += 1.0
            if score >= self._threshold:
                results.append((record, score))

        results.sort(key=lambda pair: pair[1], reverse=True)
        return results


def _phrase_synonyms(text: str) -> set[str]:
    """Return extra query tokens for multi-word music notation phrases."""
    expanded: set[str] = set()

    if re.search(
        r"\b("
        r"initialize\s+(?:the\s+)?score\s+with|"
        r"set\s+up\s+(?:a\s+)?score\s+for|"
        r"score\s+for|"
        r"change\s+instrumentation\s+to|"
        r"use\s+these\s+instruments|"
        r"make\s+this\s+for"
        r")\b",
        text,
    ):
        expanded.update({"set", "score", "parts"})
    if re.search(
        r"\b("
        r"string\s+quartet|"
        r"piano\s+trio|"
        r"brass\s+quintet|"
        r"woodwind\s+quintet|"
        r"satb"
        r")\b",
        text,
    ):
        expanded.update({"set", "score", "parts"})
    if re.search(r"\bend\s+brackets?\b", text):
        expanded.add("ending")
    if re.search(r"\bvoltas?\b", text):
        expanded.update({"ending", "bracket"})
    if re.search(r"\bshow\s+hidden\s+rests?\b", text):
        expanded.add("add")
    if re.search(r"\bmake\s+rests?\s+visible\b", text):
        expanded.add("add")
    if re.search(r"\breveal\s+rests?\b", text):
        expanded.add("add")
    if re.search(r"\bunhide\s+rests?\b", text):
        expanded.add("add")
    if re.search(r"\bfill\s+(?:measure\s+)?gaps?\b", text):
        expanded.update({"fill", "measure", "gaps"})
    if re.search(r"\bfill\s+(?:missing\s+)?rests?\b", text):
        expanded.update({"fill", "measure", "gaps"})
    if re.search(r"\bmissing\s+rests?\b", text):
        expanded.update({"fill", "gaps"})
    if re.search(
        r"\b("
        r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|"
        r"\d+(?:st|nd|rd|th)?"
        r")\s+endings?\b",
        text,
    ):
        expanded.add("bracket")

    return expanded


def _is_add_rest_visibility_query(text: str) -> bool:
    """Return whether a query asks to add or reveal one visible rest."""
    return bool(
        re.search(r"\b(?:add|insert|put|place|write)\s+(?:a\s+)?rests?\b", text)
        or re.search(r"\b(?:unhide|reveal)\s+(?:a\s+)?rests?\b", text)
        or re.search(r"\bshow\s+hidden\s+rests?\b", text)
        or re.search(r"\bmake\s+(?:a\s+)?rests?\s+visible\b", text)
    )


def _is_fill_measure_gaps_query(text: str) -> bool:
    """Return whether a query asks to fill actual uncovered rhythmic gaps."""
    return bool(
        re.search(r"\bfill\s+(?:measure\s+)?gaps?\b", text)
        or re.search(r"\bfill\s+(?:missing\s+)?rests?\b", text)
        or re.search(r"\bmissing\s+rests?\b", text)
        or re.search(r"\bcomplete\s+(?:the\s+)?measure\s+with\s+rests?\b", text)
    )


def _is_rest_spelling_query(text: str) -> bool:
    """Return whether a query asks to reshape existing rest spelling."""
    if not re.search(r"\brests?\b", text):
        return False
    return bool(
        re.search(r"\b(?:split|merge|reformat|format)\b", text)
        or re.search(r"\b(?:change|update|adjust)\s+rests?\s+spelling\b", text)
        or re.search(r"\brests?\s+spelling\b", text)
        or re.search(r"\bspell\s+(?:the\s+)?rests?\b", text)
    )


def _is_hide_rest_query(text: str) -> bool:
    """Return whether a query asks to hide visible rest notation."""
    return bool(
        re.search(r"\bhide\s+(?:a\s+)?rests?\b", text)
        or re.search(r"\bremove\s+(?:visible\s+)?rests?\b", text)
    )


@dataclass(frozen=True)
class PartMatchCandidate:
    """Normalized searchable representation of one score part.

    ``part_name`` is the display name (e.g. ``"Piano RH"``) that should
    appear in scope messages and ambiguity lists. ``raw_name`` retains
    the original ``music21`` ``partName`` for base-name tokenization so
    ``"piano"`` still matches both staves of a grand staff. ``hand`` is
    ``"RH" | "LH" | "Pedal" | None`` and enables hand-synonym matching
    (``"right hand"`` → RH parts).
    """

    part_index: int
    part_name: str
    normalized_name: str
    token_candidates: tuple[str, ...]
    raw_name: str = ""
    normalized_raw_name: str = ""
    hand: Optional[str] = None
    group_index_within_name: Optional[int] = None


_HAND_SYNONYMS: dict[str, str] = {
    "right hand": "RH",
    "rh": "RH",
    "treble staff": "RH",
    "upper staff": "RH",
    "left hand": "LH",
    "lh": "LH",
    "bass staff": "LH",
    "lower staff": "LH",
    "pedal": "Pedal",
    "pedals": "Pedal",
    "pedal staff": "Pedal",
}


@dataclass
class ExtractedContextScope:
    """Lexically extracted bar and part scope for context retrieval."""

    part_indices: list[int] | None = None
    measure_numbers: list[int] | None = None
    bar_range: tuple[int, int] | None = None
    matched_part_names: list[str] = field(default_factory=list)
    ambiguity_messages: list[str] = field(default_factory=list)
    context_truncation_messages: list[str] = field(default_factory=list)
    explicit_part_mention: bool = False
    explicit_bar_mention: bool = False
    used_fallback_bar: bool = False
    bar_context_status: BarContextStatus = "missing"


@dataclass
class QueryWithContextResult:
    """Combined method and context-bar retrieval result for one user query."""

    methods: list[tuple[MethodRecord, float]]
    scope: ExtractedContextScope
    context_query: BarQuery | None
    context_bars: BarResultSet


def _normalize_lexical_text(text: str) -> str:
    """Lowercase text and collapse punctuation into single spaces."""
    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    collapsed = re.sub(r"\s+", " ", normalized)
    return collapsed.strip()


def _query_contains_phrase(query_text: str, phrase: str) -> bool:
    """Return True when a normalized query contains a whole phrase."""
    if not phrase:
        return False
    pattern = rf"\b{re.escape(phrase)}\b"
    return re.search(pattern, query_text) is not None


def _is_end_oriented_context_request(text: str) -> bool:
    """Return whether missing bar scope should mean the score's final bar."""
    return any(
        pattern.search(text) is not None
        for pattern in _END_ORIENTED_CONTEXT_PATTERNS
    )


def _build_part_match_candidates(score_state: ScoreSpeak) -> list[PartMatchCandidate]:
    """Build normalized lexical match candidates from the score's parts.

    Each candidate's ``part_name`` is the agent-facing display name
    (``"Piano RH"``) while ``raw_name`` retains the original
    ``partName`` so token-based matching on ``"piano"`` still selects
    both staves. ``hand`` enables hand-synonym queries such as
    ``"right hand"``.
    """
    candidates = []
    labels = build_part_display_labels(score_state.score)
    for part_index, part in enumerate(score_state.score.parts):
        raw_name = part.partName or f"Part {part_index + 1}"
        label = labels.get(part_index)
        display_name = label.display_name if label is not None else raw_name
        hand = label.hand if label is not None else None
        group_index = label.group_index if label is not None else None

        normalized_name = _normalize_lexical_text(display_name)
        normalized_raw_name = _normalize_lexical_text(raw_name)

        tokens = []
        for token in normalized_raw_name.split():
            if token in _PART_TOKEN_STOPWORDS:
                continue
            if token.isdigit():
                continue
            if len(token) < 2:
                continue
            tokens.append(token)

        candidates.append(
            PartMatchCandidate(
                part_index=part_index,
                part_name=display_name,
                normalized_name=normalized_name,
                token_candidates=tuple(tokens),
                raw_name=raw_name,
                normalized_raw_name=normalized_raw_name,
                hand=hand,
                group_index_within_name=group_index,
            )
        )
    return candidates


def _detect_hand_mentions(normalized_query: str) -> set[str]:
    """Return the set of hand labels (``RH``, ``LH``, ``Pedal``) mentioned.

    Uses whole-phrase matches so bogus substrings like ``"archive"``
    (contains ``"rh"``) do not trigger a false RH match. Multi-word
    synonyms like ``"right hand"`` take precedence over their shorter
    aliases.
    """
    hands: set[str] = set()
    for phrase, hand in _HAND_SYNONYMS.items():
        if _query_contains_phrase(normalized_query, phrase):
            hands.add(hand)
    return hands


def _detect_group_index_for_name(
    normalized_query: str,
    normalized_raw_name: str,
) -> Optional[int]:
    """Return an explicit ``1`` / ``2`` following a base-name mention.

    Example: for raw name ``"piano"`` the query ``"piano 2 rh"`` yields
    ``2``. Requires the digit to immediately follow the base name so
    bar numbers like ``"piano bar 3"`` do not collide.
    """
    if not normalized_raw_name:
        return None
    pattern = rf"\b{re.escape(normalized_raw_name)}\s+(\d{{1,2}})\b"
    match = re.search(pattern, normalized_query)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _parse_measure_spans_after_prefix(
    text: str,
    start_index: int,
) -> list[tuple[int, int]]:
    """Parse one measure-reference clause after a bar/measure prefix."""
    spans: list[tuple[int, int]] = []
    cursor = start_index

    while True:
        if spans:
            separator_match = _BAR_SEPARATOR_AT_POSITION_PATTERN.match(text, cursor)
            if separator_match is not None and separator_match.end() > cursor:
                cursor = separator_match.end()

        range_match = _BAR_RANGE_AT_POSITION_PATTERN.match(text, cursor)
        if range_match is not None:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            spans.append((min(start, end), max(start, end)))
            cursor = range_match.end()
            continue

        number_match = _BAR_NUMBER_AT_POSITION_PATTERN.match(text, cursor)
        if number_match is not None:
            value = int(number_match.group(1))
            spans.append((value, value))
            cursor = number_match.end()
            continue

        break

    return spans


def _expand_measure_spans(spans: list[tuple[int, int]]) -> list[int]:
    """Expand inclusive measure spans into a sorted unique measure list."""
    measure_numbers: set[int] = set()
    for start, end in spans:
        for measure_number in range(start, end + 1):
            measure_numbers.add(measure_number)
    return sorted(measure_numbers)


def _contiguous_bar_range(
    measure_numbers: list[int],
) -> tuple[int, int] | None:
    """Return a contiguous bar range when the measures form one span."""
    if not measure_numbers:
        return None

    expected = list(range(measure_numbers[0], measure_numbers[-1] + 1))
    if measure_numbers == expected:
        return measure_numbers[0], measure_numbers[-1]
    return None


def _extract_measure_numbers_from_text(text: str) -> list[int] | None:
    """Extract one or more explicit measure references from raw text."""
    spans: list[tuple[int, int]] = []
    for prefix_match in _BAR_PREFIX_PATTERN.finditer(text):
        spans.extend(_parse_measure_spans_after_prefix(text, prefix_match.end()))

    if not spans:
        return None
    return _expand_measure_spans(spans)


def _extract_part_indices_from_text(
    score_state: ScoreSpeak,
    text: str,
) -> tuple[list[int] | None, list[str], list[str], bool]:
    """Extract part scope from a raw query using conservative lexical matching.

    Matching proceeds in layered passes so the most specific mention
    wins:

    1. **Exact display-name match** (``"piano rh"`` → one staff).
    2. **Base-name + hand intersection** (``"piano right hand"``).
    3. **Base-name + group-index intersection** (``"piano 2"``).
    4. **Hand-only** (``"right hand"`` across all groups).
    5. **Exact raw-name match** (``"piano"`` selects both staves).
    6. **Token fallback** (loose, partial matches).
    """
    normalized_query = _normalize_lexical_text(text)
    candidates = _build_part_match_candidates(score_state)
    hand_mentions = _detect_hand_mentions(normalized_query)

    # ---------- Pass 1: exact display-name match ---------------------
    display_matches = [
        candidate
        for candidate in candidates
        if candidate.normalized_name
        and candidate.normalized_name != candidate.normalized_raw_name
        and _query_contains_phrase(normalized_query, candidate.normalized_name)
    ]
    if display_matches:
        return _finalize_matches(display_matches)

    # ---------- Pass 2 and 3: base-name + hand/group intersection ----
    raw_name_matches: dict[str, list[PartMatchCandidate]] = {}
    for candidate in candidates:
        if not candidate.normalized_raw_name:
            continue
        if _query_contains_phrase(normalized_query, candidate.normalized_raw_name):
            raw_name_matches.setdefault(
                candidate.normalized_raw_name, []
            ).append(candidate)

    if raw_name_matches and hand_mentions:
        intersected: list[PartMatchCandidate] = []
        for normalized_raw_name, group in raw_name_matches.items():
            group_index = _detect_group_index_for_name(
                normalized_query, normalized_raw_name
            )
            for candidate in group:
                if candidate.hand not in hand_mentions:
                    continue
                if (
                    group_index is not None
                    and candidate.group_index_within_name != group_index
                ):
                    continue
                intersected.append(candidate)
        if intersected:
            return _finalize_matches(intersected)

    if raw_name_matches:
        intersected_by_group: list[PartMatchCandidate] = []
        any_group_index_seen = False
        for normalized_raw_name, group in raw_name_matches.items():
            group_index = _detect_group_index_for_name(
                normalized_query, normalized_raw_name
            )
            if group_index is None:
                intersected_by_group.extend(group)
                continue
            any_group_index_seen = True
            intersected_by_group.extend(
                candidate
                for candidate in group
                if candidate.group_index_within_name == group_index
            )
        if any_group_index_seen and intersected_by_group:
            return _finalize_matches(intersected_by_group)
        if intersected_by_group and not hand_mentions:
            return _finalize_matches(intersected_by_group)

    # ---------- Pass 4: hand-only match ------------------------------
    if hand_mentions:
        hand_matches = [
            candidate
            for candidate in candidates
            if candidate.hand in hand_mentions
        ]
        if hand_matches:
            return _finalize_matches(hand_matches)

    # ---------- Pass 5: loose token match (existing fallback) --------
    token_matches: dict[str, list[PartMatchCandidate]] = {}
    for candidate in candidates:
        for token in candidate.token_candidates:
            if not _query_contains_phrase(normalized_query, token):
                continue
            token_matches.setdefault(token, []).append(candidate)

    if not token_matches:
        return None, [], [], False

    matched_part_names: list[str] = []
    ambiguity_messages: list[str] = []
    matched_part_indices: list[int] = []
    seen_part_indices: set[int] = set()

    for token in sorted(token_matches):
        token_candidates = token_matches[token]
        part_indices_for_token = sorted(
            {candidate.part_index for candidate in token_candidates}
        )

        if len(part_indices_for_token) == 1:
            part_index = part_indices_for_token[0]
            if part_index in seen_part_indices:
                continue

            candidate = next(
                item for item in token_candidates if item.part_index == part_index
            )
            matched_part_indices.append(part_index)
            matched_part_names.append(candidate.part_name)
            seen_part_indices.add(part_index)
            continue

        candidate_names: list[str] = []
        seen_names: set[str] = set()
        for candidate in token_candidates:
            if candidate.part_name in seen_names:
                continue
            candidate_names.append(candidate.part_name)
            seen_names.add(candidate.part_name)

        ambiguity_messages.append(
            f"Part mention '{token}' is ambiguous across parts: "
            + ", ".join(candidate_names)
        )

    if matched_part_indices:
        matched_part_indices.sort()
        return matched_part_indices, matched_part_names, ambiguity_messages, True

    return None, [], ambiguity_messages, True


def _finalize_matches(
    matches: list[PartMatchCandidate],
) -> tuple[list[int], list[str], list[str], bool]:
    """Deduplicate matches and return the standard 4-tuple result shape."""
    unique_matches: list[PartMatchCandidate] = []
    seen_part_indices: set[int] = set()
    for candidate in matches:
        if candidate.part_index in seen_part_indices:
            continue
        unique_matches.append(candidate)
        seen_part_indices.add(candidate.part_index)

    unique_matches.sort(key=lambda candidate: candidate.part_index)
    part_indices = [candidate.part_index for candidate in unique_matches]
    matched_part_names = [candidate.part_name for candidate in unique_matches]
    return part_indices, matched_part_names, [], True


def extract_lexical_context_scope(
    score_state: ScoreSpeak,
    text: str,
) -> ExtractedContextScope:
    """Extract explicit part/bar scope cues from a raw user query."""
    part_indices, matched_part_names, ambiguity_messages, explicit_part_mention = (
        _extract_part_indices_from_text(score_state, text)
    )
    measure_numbers = _extract_measure_numbers_from_text(text)
    bar_range = None
    if measure_numbers is not None:
        bar_range = _contiguous_bar_range(measure_numbers)
    if measure_numbers is not None:
        bar_context_status: BarContextStatus = "explicit"
    elif _is_end_oriented_context_request(text):
        bar_context_status = "end_fallback"
    else:
        bar_context_status = "missing"

    return ExtractedContextScope(
        part_indices=part_indices,
        measure_numbers=measure_numbers,
        bar_range=bar_range,
        matched_part_names=matched_part_names,
        ambiguity_messages=ambiguity_messages,
        explicit_part_mention=explicit_part_mention,
        explicit_bar_mention=measure_numbers is not None,
        bar_context_status=bar_context_status,
    )


def _get_last_measure_number(
    score_state: ScoreSpeak,
    part_indices: list[int] | None,
) -> int:
    """Return the final measure number for the scoped parts."""
    scoped_parts = list(score_state.score.parts)
    if part_indices is not None:
        scoped_parts = [scoped_parts[index] for index in part_indices]

    measure_counts = []
    for part in scoped_parts:
        measure_counts.append(score_state._get_measure_count(part))

    if not measure_counts:
        return 1
    return max(1, max(measure_counts))


def _all_part_indices(score_state: ScoreSpeak) -> list[int]:
    """Return all score part indices in stable order."""
    return list(range(len(list(score_state.score.parts))))


def _context_part_selection(
    score_state: ScoreSpeak,
    scope: ExtractedContextScope,
    part_bar_limit: int,
) -> tuple[list[int] | None, int]:
    """Return scoped parts for automatic context and the selected part count."""
    all_indices = _all_part_indices(score_state)
    scoped_indices = (
        list(scope.part_indices)
        if scope.part_indices is not None
        else list(all_indices)
    )
    if not scoped_indices:
        return scope.part_indices, 1

    safe_part_bar_limit = max(1, int(part_bar_limit))
    if len(scoped_indices) <= safe_part_bar_limit:
        return scope.part_indices, max(1, len(scoped_indices))

    capped_indices = scoped_indices[:safe_part_bar_limit]
    scope.context_truncation_messages.append(
        "automatic context limited to parts "
        f"{capped_indices}; use inspect_score_region for omitted parts."
    )
    return capped_indices, len(capped_indices)


def _context_measure_budget(
    part_count: int,
    measure_limit: int,
    part_bar_limit: int,
) -> int:
    """Return the automatic context measure budget for the selected parts."""
    safe_part_count = max(1, int(part_count))
    safe_measure_limit = max(1, int(measure_limit))
    safe_part_bar_limit = max(1, int(part_bar_limit))
    part_bar_budget = max(1, safe_part_bar_limit // safe_part_count)
    return max(1, min(safe_measure_limit, part_bar_budget))


def _measure_list_label(measure_numbers: list[int]) -> str:
    """Return a compact label for a sorted measure list."""
    if not measure_numbers:
        return "(none)"
    if len(measure_numbers) <= 8:
        return ", ".join(str(number) for number in measure_numbers)
    head = ", ".join(str(number) for number in measure_numbers[:8])
    return f"{head}, ..."


def _cap_bar_range(
    scope: ExtractedContextScope,
    bar_range: tuple[int, int],
    measure_budget: int,
) -> tuple[int, int]:
    """Cap an automatic bar range to the measure budget."""
    start, end = bar_range
    capped_end = min(end, start + measure_budget - 1)
    if capped_end < end:
        scope.context_truncation_messages.append(
            "automatic context limited to bars "
            f"{start}-{capped_end} from requested bars {start}-{end}; "
            "use inspect_score_region for omitted bars."
        )
    return start, capped_end


def _cap_measure_numbers(
    scope: ExtractedContextScope,
    measure_numbers: list[int],
    measure_budget: int,
) -> list[int]:
    """Cap automatic disconnected measure context to the measure budget."""
    capped = list(measure_numbers[:measure_budget])
    if len(capped) < len(measure_numbers):
        scope.context_truncation_messages.append(
            "automatic context limited to measures "
            f"{_measure_list_label(capped)} from requested measures "
            f"{_measure_list_label(measure_numbers)}; use inspect_score_region "
            "for omitted bars."
        )
    return capped


def build_context_bar_query(
    score_state: ScoreSpeak,
    scope: ExtractedContextScope,
    *,
    measure_limit: int = DEFAULT_AUTO_CONTEXT_MEASURE_LIMIT,
    part_bar_limit: int = DEFAULT_AUTO_CONTEXT_PART_BAR_LIMIT,
) -> BarQuery | None:
    """Build a scope-only bar-result query from extracted lexical scope."""
    scope_payload = {}
    has_bar_scope = scope.bar_range is not None or scope.measure_numbers is not None
    if not has_bar_scope and scope.bar_context_status == "missing":
        return None

    selected_parts, part_count = _context_part_selection(
        score_state,
        scope,
        part_bar_limit,
    )
    measure_budget = _context_measure_budget(
        part_count,
        measure_limit,
        part_bar_limit,
    )
    if selected_parts is not None:
        scope_payload["parts"] = list(selected_parts)

    if scope.bar_range is not None:
        scope_payload["bar_range"] = _cap_bar_range(
            scope,
            scope.bar_range,
            measure_budget,
        )
    elif scope.measure_numbers is not None:
        scope_payload["measure_numbers"] = _cap_measure_numbers(
            scope,
            list(scope.measure_numbers),
            measure_budget,
        )
    elif scope.bar_context_status == "end_fallback":
        last_measure = _get_last_measure_number(score_state, scope.part_indices)
        scope_payload["bar_range"] = (last_measure, last_measure)
        scope.used_fallback_bar = True
    else:
        return None

    return {"scope": scope_payload}


class LexicalContextRetriever:
    """Combined retriever for tool suggestions and lexical bar context."""

    def __init__(
        self,
        score_state: ScoreSpeak,
        synonym_map: dict[str, list[str]] | None = None,
        threshold: float = 0.5,
    ) -> None:
        """Build the combined retriever for one live `ScoreSpeak` instance."""
        self._score_state = score_state
        self._method_retriever = LexicalRetriever(
            MethodIndex(score_state),
            synonym_map=synonym_map,
            threshold=threshold,
        )

    def query(self, text: str) -> QueryWithContextResult:
        """Return lexical method hits plus scope-derived context bars."""
        methods = self._method_retriever.query(text)
        scope = extract_lexical_context_scope(self._score_state, text)
        context_query = build_context_bar_query(self._score_state, scope)
        if context_query is None:
            context_bars = self._score_state._empty_bar_result_set()
        else:
            try:
                context_bars = self._score_state._build_bar_result_set(context_query)
            except ValueError as exc:
                if "Score is empty" not in str(exc):
                    raise
                scope.context_truncation_messages.append(
                    "automatic score context omitted because the score is empty"
                )
                context_query = None
                context_bars = self._score_state._empty_bar_result_set()

        return QueryWithContextResult(
            methods=methods,
            scope=scope,
            context_query=context_query,
            context_bars=context_bars,
        )

    @property
    def method_records(self) -> list[MethodRecord]:
        """Return all public ScoreSpeak method records in index order."""
        return self._method_retriever._index.records


class ResultFormatter:
    """Formats scored retrieval results into plain-text terminal output."""

    DISPLAY_LIMIT: int = 10

    @staticmethod
    def _truncate_docstring(docstring: str) -> str:
        """Return the first sentence or up to 120 chars, whichever is shorter.

        A "first sentence" ends at the first period followed by a space or
        at the first period at end-of-string.  If the docstring is empty,
        returns ``"(no description)"``.
        """
        if not docstring:
            return "(no description)"

        # Find first sentence boundary: period followed by space or end of string.
        match = re.search(r"\.\s|\.\Z", docstring)
        if match:
            first_sentence = docstring[: match.start() + 1]  # include the period
        else:
            first_sentence = docstring

        # Apply 120-char cap.
        if len(first_sentence) > 120:
            return first_sentence[:120]
        return first_sentence

    def format(self, results: list[tuple[MethodRecord, float]]) -> str:
        """Return a plain-text string for terminal display."""
        if not results:
            return "No matching methods found. Try rephrasing your query."

        lines: list[str] = []
        visible = results[: self.DISPLAY_LIMIT]

        for i, (record, score) in enumerate(visible, start=1):
            lines.append(
                f"#{i}  {record.name}  [{record.mixin}]  score={score:.2f}"
            )
            lines.append(f"    Signature: {record.signature}")
            lines.append(f"    {self._truncate_docstring(record.docstring)}")

        if len(results) > self.DISPLAY_LIMIT:
            omitted = len(results) - self.DISPLAY_LIMIT
            lines.append(f"... and {omitted} more results omitted")

        return "\n".join(lines)
