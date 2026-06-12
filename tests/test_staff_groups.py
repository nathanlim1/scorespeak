"""
Unit tests for ScoreSpeak staff-group helpers.

Covers brace-group detection, position-based hand assignment, the
multi-piano numeric-prefix disambiguator, and the ``PartLabel`` shape
used downstream by the overview, bar context, and lexical retrieval.
"""

from __future__ import annotations

from music21 import clef, layout, note, stream

from scorespeak import ScoreSpeak
from scorespeak.score.staff_groups import (
    PartLabel,
    build_part_display_labels,
    detect_staff_groups,
)


def _make_piano_pair(
    first_clef: clef.Clef,
    second_clef: clef.Clef,
    part_name: str = "Piano",
) -> tuple[stream.PartStaff, stream.PartStaff]:
    """Return an (upper, lower) ``PartStaff`` pair with one measure each."""
    upper = stream.PartStaff()
    upper.partName = part_name
    upper.append(first_clef)
    measure_upper = stream.Measure(number=1)
    measure_upper.append(note.Note("C4"))
    upper.append(measure_upper)

    lower = stream.PartStaff()
    lower.partName = part_name
    lower.append(second_clef)
    measure_lower = stream.Measure(number=1)
    measure_lower.append(note.Note("C3"))
    lower.append(measure_lower)

    return upper, lower


def _make_named_part(part_name: str) -> stream.Part:
    """Return one named single-staff part with a one-note measure."""
    part = stream.Part()
    part.partName = part_name
    part.append(clef.TrebleClef())
    measure = stream.Measure(number=1)
    measure.append(note.Note("C4"))
    part.append(measure)
    return part


def _single_piano_score() -> ScoreSpeak:
    """Return a ``ScoreSpeak`` with one soprano part and one piano grand staff."""
    soprano = stream.Part()
    soprano.partName = "Soprano"
    soprano.append(clef.TrebleClef())
    measure = stream.Measure(number=1)
    measure.append(note.Note("E5"))
    soprano.append(measure)

    rh, lh = _make_piano_pair(clef.TrebleClef(), clef.BassClef())

    score = stream.Score()
    score.insert(0, soprano)
    score.insert(0, rh)
    score.insert(0, lh)
    score.insert(0, layout.StaffGroup([rh, lh], name="Piano", symbol="brace"))
    return ScoreSpeak(score)


def _two_piano_score() -> ScoreSpeak:
    """Return a ``ScoreSpeak`` with two piano grand staves back-to-back."""
    rh1, lh1 = _make_piano_pair(clef.TrebleClef(), clef.BassClef())
    rh2, lh2 = _make_piano_pair(clef.TrebleClef(), clef.BassClef())

    score = stream.Score()
    score.insert(0, rh1)
    score.insert(0, lh1)
    score.insert(0, rh2)
    score.insert(0, lh2)
    score.insert(0, layout.StaffGroup([rh1, lh1], name="Piano", symbol="brace"))
    score.insert(0, layout.StaffGroup([rh2, lh2], name="Piano", symbol="brace"))
    return ScoreSpeak(score)


def test_detect_staff_groups_single_piano():
    """One brace group wrapping two PartStaff siblings is detected as RH/LH."""
    ss = _single_piano_score()
    groups = detect_staff_groups(ss.score)

    assert len(groups) == 1
    group = groups[0]
    assert group.name == "Piano"
    assert group.part_indices == (1, 2)
    assert group.hand_labels == ("RH", "LH")
    assert group.group_index_within_name == 1
    assert group.total_groups_with_name == 1


def test_detect_staff_groups_skips_non_braced():
    """Regular ``Part`` instances without a brace group are not labeled."""
    ss = _single_piano_score()
    labels = build_part_display_labels(ss.score)

    soprano_label = labels[0]
    assert isinstance(soprano_label, PartLabel)
    assert soprano_label.hand is None
    assert soprano_label.display_name == "Soprano"


def test_build_part_display_labels_single_piano():
    """Single piano uses bare ``Piano RH`` / ``Piano LH`` labels."""
    ss = _single_piano_score()
    labels = build_part_display_labels(ss.score)

    assert labels[1].display_name == "Piano RH"
    assert labels[1].hand == "RH"
    assert labels[1].group_index == 1
    assert labels[1].total_groups_with_name == 1

    assert labels[2].display_name == "Piano LH"
    assert labels[2].hand == "LH"


def test_build_part_display_labels_two_pianos_add_numeric_prefix():
    """Multiple piano groups get a 1-based numeric prefix."""
    ss = _two_piano_score()
    labels = build_part_display_labels(ss.score)

    assert labels[0].display_name == "Piano 1 RH"
    assert labels[0].hand == "RH"
    assert labels[0].group_index == 1

    assert labels[1].display_name == "Piano 1 LH"
    assert labels[2].display_name == "Piano 2 RH"
    assert labels[2].group_index == 2
    assert labels[3].display_name == "Piano 2 LH"


def test_detect_staff_groups_ignores_two_treble_cross_staff():
    """Hand assignment is position-based even when both staves use treble clef.

    The user explicitly requested position-based labeling so cross-staff
    pieces (two treble clefs) still get RH/LH by order, not by clef.
    """
    rh, lh = _make_piano_pair(clef.TrebleClef(), clef.TrebleClef())
    score = stream.Score()
    score.insert(0, rh)
    score.insert(0, lh)
    score.insert(0, layout.StaffGroup([rh, lh], name="Piano", symbol="brace"))
    ss = ScoreSpeak(score)

    labels = build_part_display_labels(ss.score)
    assert labels[0].hand == "RH"
    assert labels[1].hand == "LH"


def test_detect_staff_groups_three_staff_organ():
    """Three-staff brace groups label the third staff as ``Pedal``."""
    rh = stream.PartStaff(); rh.partName = "Organ"
    rh.append(clef.TrebleClef())
    m = stream.Measure(number=1); m.append(note.Note("C5")); rh.append(m)

    lh = stream.PartStaff(); lh.partName = "Organ"
    lh.append(clef.BassClef())
    m = stream.Measure(number=1); m.append(note.Note("C3")); lh.append(m)

    pedal = stream.PartStaff(); pedal.partName = "Organ"
    pedal.append(clef.BassClef())
    m = stream.Measure(number=1); m.append(note.Note("C2")); pedal.append(m)

    score = stream.Score()
    score.insert(0, rh)
    score.insert(0, lh)
    score.insert(0, pedal)
    score.insert(0, layout.StaffGroup([rh, lh, pedal], name="Organ", symbol="brace"))
    ss = ScoreSpeak(score)

    labels = build_part_display_labels(ss.score)
    assert labels[0].hand == "RH"
    assert labels[1].hand == "LH"
    assert labels[2].hand == "Pedal"
    assert labels[2].display_name == "Organ Pedal"


def test_detect_staff_groups_no_brace_symbol():
    """Orchestral ``bracketSquare`` groups are not treated as grand staves."""
    v1 = stream.Part(); v1.partName = "Violin 1"
    v1.append(clef.TrebleClef())
    m = stream.Measure(number=1); m.append(note.Note("C4")); v1.append(m)

    v2 = stream.Part(); v2.partName = "Violin 2"
    v2.append(clef.TrebleClef())
    m = stream.Measure(number=1); m.append(note.Note("C4")); v2.append(m)

    score = stream.Score()
    score.insert(0, v1)
    score.insert(0, v2)
    score.insert(0, layout.StaffGroup([v1, v2], name="Violins", symbol="square"))
    ss = ScoreSpeak(score)

    groups = detect_staff_groups(ss.score)
    assert groups == []

    labels = build_part_display_labels(ss.score)
    assert labels[0].hand is None
    assert labels[0].display_name == "Violin 1"


def test_detect_staff_groups_ignores_bracketed_ensemble_parts() -> None:
    """Bracketed ensemble groups should keep raw musical part names."""
    parts = [
        _make_named_part("Violin 1"),
        _make_named_part("Violin 2"),
        _make_named_part("Viola"),
        _make_named_part("Violoncello"),
    ]
    score = stream.Score()
    for part in parts:
        score.insert(0, part)
    score.insert(0, layout.StaffGroup(parts, name="Strings", symbol="bracket"))
    score_state = ScoreSpeak(score)

    groups = detect_staff_groups(score_state.score)
    labels = build_part_display_labels(score_state.score)

    assert groups == []
    assert [labels[index].display_name for index in range(4)] == [
        "Violin 1",
        "Violin 2",
        "Viola",
        "Violoncello",
    ]
    assert [labels[index].hand for index in range(4)] == [None, None, None, None]


def test_build_part_display_labels_empty_score():
    """Empty scores return an empty label map without raising."""
    ss = ScoreSpeak(stream.Score())
    assert build_part_display_labels(ss.score) == {}
    assert detect_staff_groups(ss.score) == []
