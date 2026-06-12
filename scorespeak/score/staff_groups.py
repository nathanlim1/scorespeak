"""
Detect grand-staff groupings inside a ``ScoreSpeak`` and assign RH/LH labels.

Piano and organ parts are represented in MusicXML (and therefore in
``music21``) as multiple ``Part`` or ``PartStaff`` streams joined by a
``music21.layout.StaffGroup`` with ``symbol="brace"``. The agent-facing
surfaces need to know which staff is the right hand and which is the
left, and must disambiguate multiple instruments of the same name (two
pianos in a four-hand piece, for example).

This module is the single source of truth for those display labels. It
is read-only with respect to the underlying score: no ``music21``
objects are mutated. Callers receive a mapping from part index to a
``PartLabel`` that carries the display name, hand, staff position, and
group index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from music21 import layout as m21layout
from music21 import stream as m21stream


_BRACE_SYMBOLS = frozenset({"brace"})

_HAND_LABELS_BY_SIZE: dict[int, tuple[str, ...]] = {
    2: ("RH", "LH"),
    3: ("RH", "LH", "Pedal"),
}


@dataclass(frozen=True)
class StaffGroupInfo:
    """One detected multi-staff grouping in document order.

    ``name`` is the base instrument name (e.g. ``"Piano"``) without any
    numeric disambiguator. ``part_indices`` lists the 0-based score part
    indices belonging to this group in staff order from top (RH) to
    bottom. ``hand_labels`` is a tuple of the same length with values
    ``"RH" | "LH" | "Pedal" | "Staff N"`` depending on group size.
    """

    name: str
    part_indices: tuple[int, ...]
    hand_labels: tuple[str, ...]
    group_index_within_name: int
    total_groups_with_name: int


@dataclass(frozen=True)
class PartLabel:
    """Display metadata derived for one part.

    ``display_name`` is the label that should replace the raw
    ``part.partName`` when rendering the overview or bar context.
    ``hand`` is ``None`` when the part is not inside any detected group.
    """

    part_index: int
    base_name: str
    display_name: str
    hand: Optional[str] = None
    group_index: Optional[int] = None
    total_groups_with_name: Optional[int] = None
    staff_position: Optional[int] = None
    staff_count: Optional[int] = None


@dataclass
class _RawGroup:
    """Intermediate mutable record used while collecting groups."""

    name: str
    part_indices: list[int] = field(default_factory=list)


def _score_parts(score: m21stream.Score) -> list[m21stream.Part]:
    """Return the list of parts in document order."""
    return list(score.parts)


def _group_qualifies(group: m21layout.StaffGroup) -> bool:
    """Return True when ``group`` is a keyboard-style brace grouping."""
    symbol = getattr(group, "symbol", None)
    if symbol not in _BRACE_SYMBOLS:
        return False

    spanned = list(group.getSpannedElements())
    if len(spanned) < 2:
        return False

    return all(isinstance(element, m21stream.Part) for element in spanned)


def _resolve_part_indices(
    spanned: list[m21stream.Part],
    parts: list[m21stream.Part],
) -> Optional[list[int]]:
    """Return 0-based indices of ``spanned`` within ``parts`` or ``None``.

    Returns ``None`` when any spanned part cannot be located in the
    score's top-level part list (indicating a malformed score layout
    that we should ignore rather than crash on).
    """
    indices: list[int] = []
    for element in spanned:
        try:
            indices.append(parts.index(element))
        except ValueError:
            return None
    return indices


def _group_display_name(group: m21layout.StaffGroup, parts_in_group: list[m21stream.Part]) -> str:
    """Return the base instrument name for a detected staff group.

    Prefers the ``StaffGroup.name`` attribute when set; otherwise falls
    back to the first part's ``partName`` so lone ``PartStaff`` pairs
    without an explicit group name (rare) still get a sensible label.
    """
    name = getattr(group, "name", None)
    if name:
        return str(name)
    first = parts_in_group[0]
    return str(first.partName) if first.partName else "Keyboard"


def _hand_labels_for_size(size: int) -> tuple[str, ...]:
    """Return hand labels for a group of ``size`` staves.

    Groups of size 2 are labeled ``("RH", "LH")``; size 3 adds
    ``"Pedal"`` for organs. Larger groups fall back to generic
    ``"Staff 1" .. "Staff N"`` to avoid inventing musical semantics.
    """
    if size in _HAND_LABELS_BY_SIZE:
        return _HAND_LABELS_BY_SIZE[size]
    return tuple(f"Staff {position}" for position in range(1, size + 1))


def detect_staff_groups(score: m21stream.Score) -> list[StaffGroupInfo]:
    """Return all brace-style keyboard groups in document order.

    Each group is emitted once, with its parts ordered from top staff
    (RH) to bottom (pedal or lowest LH). Groups whose base name collides
    with another group are assigned 1-based ``group_index_within_name``
    values so callers can render ``"Piano 1 RH"`` / ``"Piano 2 RH"``.

    Args:
        score: The underlying ``music21`` score (usually ``ScoreSpeak.score``).
    """
    parts = _score_parts(score)
    if not parts:
        return []

    raw_groups: list[_RawGroup] = []

    for group in score.recurse().getElementsByClass(m21layout.StaffGroup):
        if not _group_qualifies(group):
            continue

        spanned = list(group.getSpannedElements())
        indices = _resolve_part_indices(spanned, parts)
        if indices is None or len(indices) < 2:
            continue

        base_name = _group_display_name(group, spanned)
        raw_groups.append(_RawGroup(name=base_name, part_indices=indices))

    counts: dict[str, int] = {}
    for raw in raw_groups:
        counts[raw.name] = counts.get(raw.name, 0) + 1

    seen: dict[str, int] = {}
    infos: list[StaffGroupInfo] = []
    for raw in raw_groups:
        total = counts[raw.name]
        seen[raw.name] = seen.get(raw.name, 0) + 1
        index_within = seen[raw.name]

        hand_labels = _hand_labels_for_size(len(raw.part_indices))
        infos.append(
            StaffGroupInfo(
                name=raw.name,
                part_indices=tuple(raw.part_indices),
                hand_labels=hand_labels,
                group_index_within_name=index_within,
                total_groups_with_name=total,
            )
        )

    return infos


def _build_display_name(
    base_name: str,
    group_index: int,
    total_groups: int,
    hand_label: str,
) -> str:
    """Assemble the final display name for one staff in a group."""
    if total_groups > 1:
        return f"{base_name} {group_index} {hand_label}"
    return f"{base_name} {hand_label}"


def build_part_display_labels(score: m21stream.Score) -> dict[int, PartLabel]:
    """Return per-part display metadata for every part in ``score``.

    Parts that belong to a detected brace group receive an RH / LH /
    Pedal label (with a numeric group prefix when multiple groups share
    the same name). Parts outside any group receive a ``PartLabel`` with
    ``hand=None`` and ``display_name`` equal to their raw ``partName``.

    Args:
        score: The underlying ``music21`` score (usually ``ScoreSpeak.score``).
    """
    parts = _score_parts(score)
    labels: dict[int, PartLabel] = {}

    for index, part in enumerate(parts):
        raw_name = part.partName or f"Part {index}"
        labels[index] = PartLabel(
            part_index=index,
            base_name=raw_name,
            display_name=raw_name,
        )

    for group in detect_staff_groups(score):
        for position, (part_index, hand) in enumerate(
            zip(group.part_indices, group.hand_labels), start=1
        ):
            display_name = _build_display_name(
                base_name=group.name,
                group_index=group.group_index_within_name,
                total_groups=group.total_groups_with_name,
                hand_label=hand,
            )
            labels[part_index] = PartLabel(
                part_index=part_index,
                base_name=group.name,
                display_name=display_name,
                hand=hand,
                group_index=group.group_index_within_name,
                total_groups_with_name=group.total_groups_with_name,
                staff_position=position,
                staff_count=len(group.part_indices),
            )

    return labels
