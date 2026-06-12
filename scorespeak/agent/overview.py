"""
Compact global overview of a ``ScoreSpeak`` for agent prompts.

Provides a small snapshot — parts, total bar count, active signatures and
tempo at bar 1, pickup flag, metadata — that is included in every agent
turn regardless of what retrieval returns.  The agent must always be aware
of the full shape of the score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from music21 import stream as m21stream

from ..core import ScoreSpeak
from ..music.pitch_space import (
    part_transposition_interval,
    stored_pitch_space_label,
)
from ..score.staff_groups import build_part_display_labels, detect_staff_groups


@dataclass
class PartSnapshot:
    """Minimal per-part info included in the global overview.

    ``voice_count`` reports the largest number of voices seen across
    this part's measures (1 when the part is single-voice). ``transposition``
    is a short label such as ``"Bb"`` or ``"sounding"`` describing how
    written pitch maps to sounding pitch; ``None`` when the part uses no
    transposition and plays at written pitch.

    ``display_name`` is the agent-facing label (``"Piano RH"``) and
    defaults to ``name``. ``hand`` is ``"RH" | "LH" | "Pedal" | None``
    and is only non-null for parts inside a detected brace group.
    """

    index: int
    name: str
    instrument: str
    measure_count: int
    voice_count: int = 1
    transposition: Optional[str] = None
    stored_pitch: Optional[str] = None
    display_name: Optional[str] = None
    hand: Optional[str] = None


@dataclass
class StaffGroupSnapshot:
    """One detected grand-staff grouping surfaced in the overview.

    Parallels :class:`scorespeak.score.staff_groups.StaffGroupInfo` but uses
    plain Python lists so it serializes cleanly alongside the rest of
    the overview.
    """

    name: str
    part_indices: list[int] = field(default_factory=list)
    hand_labels: list[str] = field(default_factory=list)
    group_index_within_name: int = 1
    total_groups_with_name: int = 1


@dataclass
class SignatureTimeline:
    """Global time / key / tempo change points across the score.

    Each list contains ``(measure_number, value)`` tuples in ascending
    measure order; bar 1 is always included if the score has measures.
    ``tempo`` carries numeric BPM values, ``time`` the ratio string
    (``"4/4"``), and ``key`` the formatted key label.
    """

    time: list[tuple[int, str]] = field(default_factory=list)
    key: list[tuple[int, str]] = field(default_factory=list)
    tempo: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class ScoreOverview:
    """Compact, serializable snapshot of the whole score."""

    title: str
    composer: str
    subtitle: str
    total_bars: int
    pickup: bool
    pickup_quarter_length: Optional[float]
    time_signature_at_bar_1: Optional[str]
    key_signature_at_bar_1: Optional[str]
    tempo_at_bar_1: Optional[float]
    parts: list[PartSnapshot] = field(default_factory=list)
    signature_timeline: SignatureTimeline = field(
        default_factory=SignatureTimeline
    )
    staff_groups: list[StaffGroupSnapshot] = field(default_factory=list)


def _first_part(score_state: ScoreSpeak) -> Optional[m21stream.Part]:
    """Return the first part of the score, or ``None`` if there are no parts."""
    parts = list(score_state.score.parts)
    if not parts:
        return None
    return parts[0]


def _detect_pickup(part: m21stream.Part, nominal_ql: Optional[float]) -> tuple[bool, Optional[float]]:
    """Return whether the first measure looks like a pickup and its quarter length.

    A pickup is detected when the first measure's ``paddingLeft`` is positive
    or when its total duration is shorter than the time signature's bar
    length at that measure.
    """
    measures = list(part.getElementsByClass(m21stream.Measure))
    if not measures:
        return False, None

    first = measures[0]
    padding_left = float(getattr(first, "paddingLeft", 0.0) or 0.0)
    if padding_left > 0.0:
        effective = first.duration.quarterLength
        return True, float(effective)

    if nominal_ql is not None:
        effective = float(first.duration.quarterLength)
        if 0.0 < effective < nominal_ql - 1e-6:
            return True, effective

    return False, None


def _nominal_bar_quarter_length(time_signature: Optional[str]) -> Optional[float]:
    """Parse a ``time_signature`` string like ``"4/4"`` into its bar length."""
    if not time_signature or "/" not in time_signature:
        return None
    try:
        numerator_str, denominator_str = time_signature.split("/", 1)
        numerator = int(numerator_str.strip())
        denominator = int(denominator_str.strip())
    except ValueError:
        return None
    if denominator <= 0:
        return None
    return numerator * (4.0 / denominator)


def _safe_call(fn, *args, **kwargs):
    """Invoke ``fn`` and swallow exceptions, returning ``None`` on failure.

    The overview is cosmetic for the agent; it must never crash the turn
    if one helper raises on a partially populated score (e.g. empty).
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _max_voice_count(part: m21stream.Part) -> int:
    """Return the largest voice count seen in any measure of ``part``.

    Measures with no explicit voices report a count of 1. If a measure
    has explicit Voice streams that do not include voice "1" but also
    carry top-level GeneralNotes, those top-level notes live in an
    implicit voice 1 and are counted alongside the explicit voices.
    """
    from music21 import note as m21note  # local import; avoids cycle at top

    max_count = 1
    for measure in part.getElementsByClass(m21stream.Measure):
        voices = list(measure.voices)
        if not voices:
            continue
        voice_ids = {str(getattr(v, "id", "")) for v in voices}
        total = len(voices)
        if "1" not in voice_ids:
            top_level_notes = [
                el
                for el in measure.getElementsByClass(m21note.GeneralNote)
            ]
            if top_level_notes:
                total += 1
        if total > max_count:
            max_count = total
    return max_count


def _part_transposition_label(part: m21stream.Part) -> Optional[str]:
    """Return a short transposition label for ``part`` or ``None``.

    Transposing instruments (clarinet in Bb, horn in F, etc.) report
    their transposition interval; concert-pitch instruments return
    ``None`` so the overview can omit the field entirely.
    """
    interval = part_transposition_interval(part)
    if interval is None:
        return None

    semitones = getattr(interval, "semitones", None)
    if semitones is None or int(semitones) == 0:
        return None

    label = getattr(interval, "directedName", None) or getattr(interval, "name", None)
    if not label:
        label = f"{int(semitones):+d}st"

    return str(label)


def _collect_signature_timeline(
    score_state: ScoreSpeak,
    total_bars: int,
) -> SignatureTimeline:
    """Build the global time/key/tempo change timeline from part 0.

    Scans bar-by-bar and records every point where the active value
    differs from the previous bar's. Bar 1's value is always included
    (when present) so the agent sees an initial anchor even for scores
    with no mid-score changes.
    """
    timeline = SignatureTimeline()
    if total_bars <= 0:
        return timeline

    prev_time: Optional[str] = None
    prev_key: Optional[str] = None
    prev_tempo: Optional[float] = None

    for bar in range(1, total_bars + 1):
        time_value = _safe_call(score_state.get_active_time_signature, bar)
        key_value = _safe_call(score_state.get_active_key_signature, bar)
        tempo_value = _safe_call(score_state.get_active_tempo, bar)

        if time_value is not None and time_value != prev_time:
            timeline.time.append((bar, str(time_value)))
            prev_time = time_value
        if key_value is not None and key_value != prev_key:
            timeline.key.append((bar, str(key_value)))
            prev_key = key_value
        if tempo_value is not None and tempo_value != prev_tempo:
            timeline.tempo.append((bar, float(tempo_value)))
            prev_tempo = tempo_value

    return timeline


def build_score_overview(score_state: ScoreSpeak) -> ScoreOverview:
    """Assemble a :class:`ScoreOverview` from a live ``ScoreSpeak``.

    Args:
        score_state: The score to summarize.  May be empty (no parts / no
            measures); the overview still returns sensible defaults.

    Returns:
        A :class:`ScoreOverview` with metadata, parts, bar count, the
        active time/key/tempo at bar 1, per-part voice counts and
        transpositions, and a mid-score signature-change timeline.
    """
    metadata = _safe_call(score_state.get_metadata) or {}
    part_infos = _safe_call(score_state.list_parts) or []
    raw_parts = list(score_state.score.parts)
    labels = _safe_call(build_part_display_labels, score_state.score) or {}

    parts: list[PartSnapshot] = []
    for info in part_infos:
        raw_part = raw_parts[info.index] if 0 <= info.index < len(raw_parts) else None
        voice_count = _max_voice_count(raw_part) if raw_part is not None else 1
        transposition = _part_transposition_label(raw_part) if raw_part is not None else None
        stored_pitch = (
            stored_pitch_space_label(raw_part)
            if raw_part is not None and transposition is not None
            else None
        )
        label = labels.get(info.index)
        display_name = (
            label.display_name if label is not None else info.display_name or info.name
        )
        hand = label.hand if label is not None else info.hand
        parts.append(
            PartSnapshot(
                index=info.index,
                name=info.name,
                instrument=info.instrument,
                measure_count=info.measure_count,
                voice_count=voice_count,
                transposition=transposition,
                stored_pitch=stored_pitch,
                display_name=display_name,
                hand=hand,
            )
        )

    total_bars = max((info.measure_count for info in part_infos), default=0)

    time_signature = None
    key_signature = None
    tempo = None
    pickup = False
    pickup_ql = None

    if parts and total_bars > 0:
        time_signature = _safe_call(score_state.get_active_time_signature, 1)
        key_signature = _safe_call(score_state.get_active_key_signature, 1)
        tempo = _safe_call(score_state.get_active_tempo, 1)

        first_part = _first_part(score_state)
        if first_part is not None:
            nominal_ql = _nominal_bar_quarter_length(time_signature)
            pickup, pickup_ql = _detect_pickup(first_part, nominal_ql)

    signature_timeline = _collect_signature_timeline(score_state, total_bars)
    staff_groups = _collect_staff_group_snapshots(score_state)

    return ScoreOverview(
        title=metadata.get("title", "") or "",
        composer=metadata.get("composer", "") or "",
        subtitle=metadata.get("subtitle", "") or "",
        total_bars=total_bars,
        pickup=pickup,
        pickup_quarter_length=pickup_ql,
        time_signature_at_bar_1=time_signature,
        key_signature_at_bar_1=key_signature,
        tempo_at_bar_1=tempo,
        parts=parts,
        signature_timeline=signature_timeline,
        staff_groups=staff_groups,
    )


def _collect_staff_group_snapshots(
    score_state: ScoreSpeak,
) -> list[StaffGroupSnapshot]:
    """Convert detected :class:`StaffGroupInfo` objects into snapshots."""
    groups = _safe_call(detect_staff_groups, score_state.score) or []
    snapshots: list[StaffGroupSnapshot] = []
    for group in groups:
        snapshots.append(
            StaffGroupSnapshot(
                name=group.name,
                part_indices=list(group.part_indices),
                hand_labels=list(group.hand_labels),
                group_index_within_name=group.group_index_within_name,
                total_groups_with_name=group.total_groups_with_name,
            )
        )
    return snapshots


def _format_part_snapshot(snapshot: PartSnapshot) -> str:
    """Render one :class:`PartSnapshot` as a compact fragment."""
    extras: list[str] = [f"{snapshot.measure_count} bars"]
    if snapshot.voice_count > 1:
        extras.append(f"voices={snapshot.voice_count}")
    if snapshot.transposition:
        extras.append(f"transp={snapshot.transposition}")
    if snapshot.stored_pitch:
        extras.append(f"stored={snapshot.stored_pitch}")

    shown_name = snapshot.display_name or snapshot.name
    return (
        f"[{snapshot.index}] {shown_name} "
        f"({snapshot.instrument}, {', '.join(extras)})"
    )


def _format_staff_group(group: StaffGroupSnapshot) -> str:
    """Render one :class:`StaffGroupSnapshot` as ``Piano [1, 2] (RH, LH)``.

    When multiple groups share the same base name, the group's 1-based
    index is included so the fragment reads ``Piano 1 [1, 2] (RH, LH)``.
    """
    if group.total_groups_with_name > 1:
        label = f"{group.name} {group.group_index_within_name}"
    else:
        label = group.name
    indices = ", ".join(str(idx) for idx in group.part_indices)
    hands = ", ".join(group.hand_labels)
    return f"{label} [{indices}] ({hands})"


def _format_timeline_points(points: list[tuple[int, object]]) -> str:
    """Render a list of ``(bar, value)`` points as ``"1:4/4, 33:3/4"``."""
    fragments: list[str] = []
    for bar, value in points:
        if isinstance(value, float):
            formatted = f"{value:g}"
        else:
            formatted = str(value)
        fragments.append(f"{bar}:{formatted}")
    return ", ".join(fragments)


def format_overview_for_prompt(overview: ScoreOverview) -> str:
    """Render a :class:`ScoreOverview` as a compact text block for prompts.

    The output is intentionally terse and line-oriented so the agent can
    parse it without spending tokens on filler.
    """
    lines: list[str] = []

    title = overview.title or "(untitled)"
    composer = overview.composer or "(no composer)"
    lines.append(f"Title: {title}    Composer: {composer}")
    if overview.subtitle:
        lines.append(f"Subtitle: {overview.subtitle}")

    if overview.parts:
        part_fragments = [_format_part_snapshot(p) for p in overview.parts]
        lines.append("Parts: " + ", ".join(part_fragments))
    else:
        lines.append("Parts: (none)")

    if overview.staff_groups:
        group_fragments = [_format_staff_group(g) for g in overview.staff_groups]
        lines.append("Groups: " + " | ".join(group_fragments))

    pickup_label = "yes" if overview.pickup else "no"
    if overview.pickup and overview.pickup_quarter_length is not None:
        pickup_label = f"yes ({overview.pickup_quarter_length:g} ql)"
    lines.append(
        f"Total bars: {overview.total_bars}    Pickup: {pickup_label}"
    )

    if overview.total_bars > 0:
        fields: list[str] = []
        if overview.time_signature_at_bar_1:
            fields.append(f"time={overview.time_signature_at_bar_1}")
        if overview.key_signature_at_bar_1:
            fields.append(f"concert_key={overview.key_signature_at_bar_1}")
        if overview.tempo_at_bar_1 is not None:
            fields.append(f"tempo={overview.tempo_at_bar_1:g}")
        if fields:
            lines.append("At bar 1: " + " ".join(fields))

    timeline = overview.signature_timeline
    has_time_changes = len(timeline.time) > 1
    has_key_changes = len(timeline.key) > 1
    has_tempo_changes = len(timeline.tempo) > 1

    if has_time_changes or has_key_changes or has_tempo_changes:
        change_fragments: list[str] = []
        if has_time_changes:
            change_fragments.append(
                f"time=[{_format_timeline_points(list(timeline.time))}]"
            )
        if has_key_changes:
            change_fragments.append(
                f"concert_key=[{_format_timeline_points(list(timeline.key))}]"
            )
        if has_tempo_changes:
            change_fragments.append(
                f"tempo=[{_format_timeline_points(list(timeline.tempo))}]"
            )
        lines.append("Changes: " + " ".join(change_fragments))

    return "\n".join(lines)


def overview_as_dict(overview: ScoreOverview) -> dict:
    """Return the overview as a plain dict (useful for tests and logging)."""
    return asdict(overview)
