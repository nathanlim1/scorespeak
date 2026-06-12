"""
Render retrieved score regions for agent prompts and inspection tools.

The summary renderer is intentionally human-readable and lossy.  It keeps
automatic prompt context small by showing rhythm-count buckets, register
ranges, present-only markings, and structural cues per bar/part.

The exact renderer filters the internal compact bar-result payload so
inspection tools can return mandatory event detail plus explicitly requested
optional channels.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fractions import Fraction
from typing import Any, Optional

from music21 import pitch as m21pitch

from ..types import BarResultSet


DEFAULT_SUMMARY_PART_BAR_LIMIT = 16

_EVENT_KIND_INDEX = 0
_EVENT_PITCH_INDEX = 2
_EVENT_DURATION_INDEX = 3
_EVENT_DOTS_INDEX = 6

_COMMON_DURATION_NAMES = (
    (8.0, "breve"),
    (4.0, "whole"),
    (3.0, "dotted half"),
    (2.0, "half"),
    (1.5, "dotted quarter"),
    (1.0, "quarter"),
    (0.75, "dotted eighth"),
    (0.5, "eighth"),
    (0.375, "dotted 16th"),
    (0.25, "16th"),
    (0.125, "32nd"),
    (0.0625, "64th"),
)

_STRUCTURE_KEYS = (
    "barline_start",
    "barline_end",
    "repeat_start",
    "repeat_end",
    "ending_number",
    "rehearsal_mark",
    "navigation",
    "system_break",
    "page_break",
)

_ALL_OPTIONAL_CATEGORIES = {
    "attributes",
    "structure",
    "dynamics",
    "hairpins",
    "articulations",
    "slurs",
    "lyrics",
    "text",
    "chord_symbols",
    "ornaments",
    "technique",
    "markings",
    "spans",
}


def render_summary_context(
    context_bars: Optional[BarResultSet | dict],
    scope: Any = None,
    *,
    max_part_bars: int = DEFAULT_SUMMARY_PART_BAR_LIMIT,
    empty_message: str = "(no scoped bars)",
) -> str:
    """Render retrieved bars as compact default prompt context.

    Args:
        context_bars: The exact ``BarResultSet`` produced by retrieval.
        scope: Reserved for future scope-sensitive formatting.  The value is
            accepted so callers can pass the retrieval scope without branching.
        max_part_bars: Maximum number of bar/part summaries to emit.
        empty_message: Message to return when no bars are available.

    Returns:
        A deterministic, line-oriented summary suitable for the base prompt.
    """
    del scope

    bars = _get_bars(context_bars)
    if not bars:
        return empty_message

    lines = []
    emitted_part_bars = 0
    first_measure_number = bars[0].get("measure_number")
    metadata = _get_search_metadata(context_bars)
    if metadata.get("truncated") is True:
        returned = metadata.get("returned_matches", len(bars))
        total = metadata.get("total_matches", "?")
        limit = metadata.get("limit", returned)
        lines.append(
            f"search_score limit reached: returned first {returned} of "
            f"{total} matching bars (limit={limit}); narrow the scope or "
            "raise limit for more."
        )

    for bar in bars:
        measure_number = bar.get("measure_number", "?")
        header_fields = [f"m{measure_number}"]
        attributes = _format_bar_attributes(
            bar,
            is_first_bar=measure_number == first_measure_number,
        )
        if attributes:
            header_fields.append(f"attrs={attributes}")
        structure = _format_structure(bar.get("notation", {}))
        if structure:
            header_fields.append(f"structure={structure}")
        matches = _format_search_matches(bar.get("matches", []))
        if matches:
            header_fields.append(f"matches={matches}")
        lines.append(" ".join(header_fields))

        for part in bar.get("parts", []):
            if emitted_part_bars >= max_part_bars:
                lines.append(
                    f"... truncated after {max_part_bars} part-bar summaries; "
                    "use inspect_score_region for exact detail."
                )
                return "\n".join(lines)
            lines.append(f"- {_format_part_label(part)}: {_format_part_summary(part, bar)}")
            emitted_part_bars += 1

    return "\n".join(lines)


def render_exact_context(
    context_bars: Optional[BarResultSet | dict],
    *,
    include: Optional[list[str] | str] = None,
) -> dict[str, Any]:
    """Return filtered exact symbolic context for explicit inspection.

    Mandatory core data is always retained: compact schemas, measures,
    parts/staves, voices, events, tie/dot/grace fields carried in event rows,
    tuplets, and first-bar-plus-changes attributes.  Optional channels are
    included only when requested through ``include``.
    """
    bars = _get_bars(context_bars)
    include_set, warnings = _normalize_include(include)

    result = {
        "retrieval": "explicit inspection",
        "include": sorted(include_set),
        "event_schema": _copy_key(context_bars, "event_schema"),
        "tuplet_schema": _copy_key(context_bars, "tuplet_schema"),
        "bars": [],
    }

    if _needs_marking_schema(include_set):
        result["marking_schema"] = _copy_key(context_bars, "marking_schema")
    if _needs_span_schema(include_set):
        result["span_schema"] = _copy_key(context_bars, "span_schema")
    if warnings:
        result["warnings"] = warnings

    first_measure_number = bars[0].get("measure_number") if bars else None
    for bar in bars:
        exact_bar = {
            "measure_number": bar.get("measure_number"),
            "parts": [],
        }

        notation = _filter_bar_notation(
            bar.get("notation", {}),
            include_set,
            is_first_bar=bar.get("measure_number") == first_measure_number,
        )
        if notation:
            exact_bar["notation"] = notation

        for part in bar.get("parts", []):
            exact_bar["parts"].append(_filter_part(part, include_set))

        result["bars"].append(exact_bar)

    return result


def _get_bars(context_bars: Optional[BarResultSet | dict]) -> list[dict[str, Any]]:
    """Return the bar list from a possible ``BarResultSet``."""
    if not isinstance(context_bars, dict):
        return []
    bars = context_bars.get("bars")
    if not isinstance(bars, list):
        return []
    return [bar for bar in bars if isinstance(bar, dict)]


def _get_search_metadata(
    context_bars: Optional[BarResultSet | dict],
) -> dict[str, Any]:
    """Return search metadata from a possible ``BarResultSet``."""
    if not isinstance(context_bars, dict):
        return {}
    metadata = context_bars.get("search_metadata")
    if not isinstance(metadata, dict):
        return {}
    return metadata


def _copy_key(context_bars: Optional[BarResultSet | dict], key: str) -> Any:
    """Return a defensive copy of a top-level payload key."""
    if not isinstance(context_bars, dict):
        return []
    return deepcopy(context_bars.get(key, []))


def _format_bar_attributes(bar: dict[str, Any], *, is_first_bar: bool) -> str:
    """Render first-bar or changed active attributes for a summary header."""
    notation = bar.get("notation", {})
    if not isinstance(notation, dict):
        return ""

    changed = notation.get("changed_here") or []
    if not is_first_bar and not changed:
        return ""

    active = notation.get("active", {})
    if not isinstance(active, dict):
        return ""

    fragments = []
    if active.get("time") not in ("", None):
        fragments.append(f"time={_format_scalar(active['time'])}")
    key_value = active.get("concert_key", active.get("key"))
    if key_value not in ("", None):
        fragments.append(f"concert_key={_format_scalar(key_value)}")
    if active.get("tempo") not in ("", None):
        fragments.append(f"tempo={_format_scalar(active['tempo'])}")
    if changed:
        fragments.append("changed=" + ",".join(str(item) for item in changed))
    return ",".join(fragments)


def _format_structure(notation: Any) -> str:
    """Render present structural notation fields for a summary header."""
    if not isinstance(notation, dict):
        return ""

    fragments = []
    for key in _STRUCTURE_KEYS:
        value = notation.get(key)
        if value in (None, False, "", []):
            continue
        if value is True:
            fragments.append(key)
        elif isinstance(value, list):
            fragments.append(f"{key}={','.join(str(item) for item in value)}")
        else:
            fragments.append(f"{key}={_format_scalar(value)}")
    return ",".join(fragments)


def _format_search_matches(rows: Any) -> str:
    """Render semantic search match reasons for a summary header."""
    if not isinstance(rows, list):
        return ""
    fragments = []
    seen = set()
    for row in rows:
        if not isinstance(row, list) or len(row) < 2:
            continue
        detail = str(row[1])
        if not detail or detail in seen:
            continue
        fragments.append(detail)
        seen.add(detail)
    return ",".join(fragments[:4])


def _format_part_label(part: dict[str, Any]) -> str:
    """Return a compact part/staff label."""
    name = part.get("part_name") or "Part"
    index = part.get("part_index")
    if index is None:
        return str(name)
    return f"{name} [{index}]"


def _format_part_summary(part: dict[str, Any], bar: dict[str, Any]) -> str:
    """Render one part/staff summary within a bar."""
    events = _collect_events(part)
    fragments = []

    if not events:
        fragments.append("empty")
    elif _is_full_measure_rest(events, bar):
        fragments.append("full-measure rest")
    else:
        rhythm = _format_rhythm_counts(part)
        if rhythm:
            fragments.append(f"rhythm={rhythm}")

    pitch_range = _format_pitch_range(events)
    if pitch_range:
        fragments.append(f"range={pitch_range}")

    clef = _part_clef(part)
    if clef:
        fragments.append(f"clef={clef}")

    key_fragment = _part_key_fragment(part)
    if key_fragment:
        fragments.append(key_fragment)

    markings = _format_present_markings(part)
    if markings:
        fragments.append(f"markings={markings}")

    return "; ".join(fragments) if fragments else "empty"


def _collect_events(part: dict[str, Any]) -> list[list[Any]]:
    """Collect all event rows in a part/staff payload."""
    events = []
    for voice in part.get("voices", []):
        if not isinstance(voice, dict):
            continue
        for event in voice.get("events", []):
            if isinstance(event, list):
                events.append(event)
    return events


def _format_rhythm_counts(part: dict[str, Any]) -> str:
    """Return rhythmic duration buckets for one part/staff."""
    counts = Counter()
    for voice in part.get("voices", []):
        if not isinstance(voice, dict):
            continue
        tuplet_ratios = _event_tuplet_ratio_map(voice)
        for index, event in enumerate(voice.get("events", []), start=1):
            if not isinstance(event, list):
                continue
            label = _event_bucket_label(event, tuplet_ratios.get(index))
            counts[label] += 1

    return ", ".join(
        _format_count(label, count)
        for label, count in sorted(counts.items(), key=lambda item: item[0])
    )


def _event_tuplet_ratio_map(voice: dict[str, Any]) -> dict[int, tuple[int, int]]:
    """Map event indices to tuplet ratios for one voice payload."""
    ratios = {}
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
        actual, normal = ratio
        if not all(isinstance(value, int) for value in (actual, normal)):
            continue
        try:
            start_beat = float(beat_range[0])
            end_beat = float(beat_range[1])
        except (TypeError, ValueError):
            continue
        for index, event in enumerate(voice.get("events", []), start=1):
            if not isinstance(event, list) or len(event) <= 1:
                continue
            try:
                event_beat = float(event[1])
            except (TypeError, ValueError):
                continue
            if start_beat - 1e-9 <= event_beat <= end_beat + 1e-9:
                ratios[index] = (actual, normal)
    return ratios


def _event_bucket_label(event: list[Any], tuplet_ratio: Optional[tuple[int, int]]) -> str:
    """Return the summary rhythm bucket for one compact event row."""
    kind = str(event[_EVENT_KIND_INDEX]) if len(event) > _EVENT_KIND_INDEX else "note"
    duration = event[_EVENT_DURATION_INDEX] if len(event) > _EVENT_DURATION_INDEX else 0.0
    dots = event[_EVENT_DOTS_INDEX] if len(event) > _EVENT_DOTS_INDEX else 0
    duration_name = _duration_name(duration, dots=dots, tuplet_ratio=tuplet_ratio)

    if tuplet_ratio is not None:
        actual, normal = tuplet_ratio
        return f"{duration_name}-note tuplet ({actual}:{normal})"
    if kind == "rest":
        return f"{duration_name} rest"
    if kind == "chord":
        return f"{duration_name} chord"
    return f"{duration_name} note"


def _duration_name(
    quarter_length: Any,
    *,
    dots: Any = 0,
    tuplet_ratio: Optional[tuple[int, int]] = None,
) -> str:
    """Return a compact duration name from a quarter length value."""
    try:
        value = float(quarter_length)
    except (TypeError, ValueError):
        return str(quarter_length)

    if tuplet_ratio is not None:
        actual, normal = tuplet_ratio
        if normal:
            value = value * actual / normal

    for size, name in _COMMON_DURATION_NAMES:
        if abs(value - size) < 1e-6:
            return name

    if not tuplet_ratio and dots == 1:
        undotted = value / 1.5
        for size, name in _COMMON_DURATION_NAMES:
            if abs(undotted - size) < 1e-6:
                return f"dotted {name}"

    return f"{Fraction(value).limit_denominator(64)}ql"


def _format_count(label: str, count: int) -> str:
    """Render a count plus a rhythm bucket label."""
    if count == 1:
        return f"1 {label}"
    if "tuplet (" in label:
        return f"{count} {label.replace('tuplet (', 'tuplets (')}"
    if label.endswith("rest"):
        return f"{count} {label}s"
    if label.endswith("chord"):
        return f"{count} {label}s"
    return f"{count} {label}s"


def _format_pitch_range(events: list[list[Any]]) -> str:
    """Return the lowest-to-highest pitch range for event rows."""
    pitches = []
    for event in events:
        if len(event) <= _EVENT_PITCH_INDEX:
            continue
        pitch_payload = event[_EVENT_PITCH_INDEX]
        if isinstance(pitch_payload, list):
            pitches.extend(str(item) for item in pitch_payload if item)
        elif pitch_payload:
            pitches.append(str(pitch_payload))

    parsed = []
    for pitch_name in pitches:
        try:
            parsed.append(m21pitch.Pitch(pitch_name))
        except Exception:
            continue
    if not parsed:
        return ""

    parsed.sort(key=lambda pitch_obj: pitch_obj.ps)
    return f"{_format_pitch(parsed[0])}-{_format_pitch(parsed[-1])}"


def _format_pitch(pitch_obj: m21pitch.Pitch) -> str:
    """Format a music21 pitch in the public ASCII spelling."""
    name = pitch_obj.nameWithOctave or pitch_obj.name
    return name.replace("-", "b")


def _part_clef(part: dict[str, Any]) -> str:
    """Return the active/change clef fragment for a part payload."""
    notation = part.get("notation")
    if not isinstance(notation, dict):
        return ""
    clef = notation.get("clef")
    return str(clef) if clef else ""


def _part_key_fragment(part: dict[str, Any]) -> str:
    """Return a compact part-key fragment with concert-key context."""
    notation = part.get("notation")
    if not isinstance(notation, dict):
        return ""

    key_value = notation.get("key")
    if not key_value:
        return ""

    concert_key = notation.get("concert_key")
    key_role = notation.get("key_role")
    if key_role == "transposed_written_key":
        if concert_key:
            return f"written_key={key_value} (concert_key={concert_key})"
        return f"written_key={key_value}"
    if key_role == "local_staff_key":
        if concert_key:
            return f"local_key={key_value} (concert_key={concert_key})"
        return f"local_key={key_value}"
    if concert_key and concert_key != key_value:
        return f"part_key={key_value} (concert_key={concert_key})"
    return f"part_key={key_value}"


def _format_present_markings(part: dict[str, Any]) -> str:
    """Render only marking/span categories present in a part/staff."""
    fragments = []
    seen = set()

    for voice in part.get("voices", []):
        if not isinstance(voice, dict):
            continue
        for row in voice.get("markings", []):
            fragment = _summary_marking_fragment(row)
            if fragment and fragment not in seen:
                fragments.append(fragment)
                seen.add(fragment)
        for row in voice.get("spans", []):
            fragment = _summary_span_fragment(row)
            if fragment and fragment not in seen:
                fragments.append(fragment)
                seen.add(fragment)

    return ",".join(fragments)


def _summary_marking_fragment(row: Any) -> str:
    """Return a compact summary fragment for one marking row."""
    if not isinstance(row, list) or len(row) < 3:
        return ""
    kind = str(row[0])
    payload = row[1]
    if kind == "dynamic" and payload:
        return f"dynamic({_format_scalar(payload)})"
    if kind == "chord_symbol" and payload:
        return f"chord_symbol({_format_scalar(payload)})"
    if kind == "lyric":
        return "lyric"
    if kind == "articulation":
        return "articulation"
    if kind in {"ornament", "fingering", "arpeggio"}:
        return kind
    return kind


def _summary_span_fragment(row: Any) -> str:
    """Return a compact summary fragment for one span row."""
    if not isinstance(row, list) or len(row) < 3:
        return ""
    kind = str(row[0])
    payload = row[1]
    if kind == "hairpin" and payload:
        return f"hairpin({_format_scalar(payload)})"
    if kind == "text_expression" and payload:
        return f"text({_format_scalar(payload)})"
    return kind


def _format_scalar(value: Any) -> str:
    """Return a short scalar string for prompt fragments."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace(" ", "_")


def _is_full_measure_rest(events: list[list[Any]], bar: dict[str, Any]) -> bool:
    """Return True when events represent one visible full-measure rest."""
    if len(events) != 1:
        return False
    event = events[0]
    if len(event) <= _EVENT_DURATION_INDEX or event[_EVENT_KIND_INDEX] != "rest":
        return False
    bar_length = _bar_quarter_length(bar)
    if bar_length is None:
        return False
    try:
        return abs(float(event[_EVENT_DURATION_INDEX]) - bar_length) < 1e-6
    except (TypeError, ValueError):
        return False


def _bar_quarter_length(bar: dict[str, Any]) -> Optional[float]:
    """Return the active bar duration in quarter lengths when available."""
    notation = bar.get("notation", {})
    if not isinstance(notation, dict):
        return None
    active = notation.get("active", {})
    if not isinstance(active, dict):
        return None
    time_signature = active.get("time")
    if not isinstance(time_signature, str) or "/" not in time_signature:
        return None
    try:
        numerator_text, denominator_text = time_signature.split("/", 1)
        numerator = int(numerator_text)
        denominator = int(denominator_text)
    except ValueError:
        return None
    if denominator <= 0:
        return None
    return numerator * (4.0 / denominator)


def _normalize_include(include: Optional[list[str] | str]) -> tuple[set[str], list[str]]:
    """Normalize exact inspection include categories."""
    warnings = []
    if include is None:
        return set(), warnings
    if isinstance(include, str):
        raw_items = [item.strip() for item in include.split(",") if item.strip()]
    elif isinstance(include, list):
        raw_items = [str(item).strip() for item in include if str(item).strip()]
    else:
        return set(), [f"Unsupported include value {include!r}; using core fields only."]

    include_set = set()
    for item in raw_items:
        normalized = item.lower().replace("-", "_").replace(" ", "_")
        if normalized in {"all", "full", "all_current_channels"}:
            include_set.update(_ALL_OPTIONAL_CATEGORIES)
            continue
        if normalized == "dynamics_hairpins":
            include_set.update({"dynamics", "hairpins"})
            continue
        if normalized == "lyrics_text":
            include_set.update({"lyrics", "text"})
            continue
        if normalized in {"text_expression", "text_expressions"}:
            include_set.add("text")
            continue
        if normalized == "articulations_fermata":
            include_set.add("articulations")
            continue
        if normalized not in _ALL_OPTIONAL_CATEGORIES:
            warnings.append(f"Unsupported include category {item!r}; ignored.")
            continue
        include_set.add(normalized)
    return include_set, warnings


def _needs_marking_schema(include_set: set[str]) -> bool:
    """Return True when exact output can include marking rows."""
    return bool(
        include_set
        & {
            "dynamics",
            "articulations",
            "lyrics",
            "text",
            "chord_symbols",
            "ornaments",
            "technique",
            "markings",
        }
    )


def _needs_span_schema(include_set: set[str]) -> bool:
    """Return True when exact output can include span rows."""
    return bool(include_set & {"hairpins", "slurs", "technique", "spans"})


def _filter_bar_notation(
    notation: Any,
    include_set: set[str],
    *,
    is_first_bar: bool,
) -> dict[str, Any]:
    """Filter bar notation to mandatory attributes plus optional structure."""
    if not isinstance(notation, dict):
        return {}

    filtered = {}
    changed = notation.get("changed_here")
    if is_first_bar or changed:
        active = notation.get("active")
        if isinstance(active, dict):
            filtered["active"] = deepcopy(active)
        if changed:
            filtered["changed_here"] = deepcopy(changed)

    if "structure" in include_set:
        for key in _STRUCTURE_KEYS:
            if key in notation:
                filtered[key] = deepcopy(notation[key])

    return filtered


def _filter_part(part: dict[str, Any], include_set: set[str]) -> dict[str, Any]:
    """Filter one part/staff payload for exact inspection."""
    filtered = {
        "part_index": part.get("part_index"),
        "part_name": part.get("part_name"),
        "voices": [],
    }
    if "hand" in part:
        filtered["hand"] = part["hand"]
    if "notation" in part:
        filtered["notation"] = deepcopy(part["notation"])

    for voice in part.get("voices", []):
        if isinstance(voice, dict):
            filtered["voices"].append(_filter_voice(voice, include_set))

    return filtered


def _filter_voice(voice: dict[str, Any], include_set: set[str]) -> dict[str, Any]:
    """Filter one voice payload for exact inspection."""
    filtered = {
        "voice": voice.get("voice"),
        "events": deepcopy(voice.get("events", [])),
        "tuplets": deepcopy(voice.get("tuplets", [])),
    }

    markings = [
        deepcopy(row)
        for row in voice.get("markings", [])
        if _include_marking_row(row, include_set)
    ]
    if markings:
        filtered["markings"] = markings

    spans = [
        deepcopy(row)
        for row in voice.get("spans", [])
        if _include_span_row(row, include_set)
    ]
    if spans:
        filtered["spans"] = spans

    return filtered


def _include_marking_row(row: Any, include_set: set[str]) -> bool:
    """Return True when a marking row belongs to requested categories."""
    if "markings" in include_set:
        return True
    if not isinstance(row, list) or len(row) < 2:
        return False
    kind = str(row[0])
    categories = {
        "dynamic": {"dynamics"},
        "articulation": {"articulations"},
        "lyric": {"lyrics"},
        "text_expression": {"text"},
        "chord_symbol": {"chord_symbols"},
        "ornament": {"ornaments"},
        "fingering": {"technique"},
        "arpeggio": {"technique"},
    }.get(kind, {kind})
    return bool(categories & include_set)


def _include_span_row(row: Any, include_set: set[str]) -> bool:
    """Return True when a span row belongs to requested categories."""
    if "spans" in include_set:
        return True
    if not isinstance(row, list) or len(row) < 2:
        return False
    kind = str(row[0])
    categories = {
        "hairpin": {"hairpins"},
        "slur": {"slurs"},
        "glissando": {"technique"},
        "ottava": {"technique"},
        "pedal": {"technique"},
    }.get(kind, {kind})
    return bool(categories & include_set)
