"""Tests for measure management operations."""

from xml.etree import ElementTree as ET

import pytest
from music21 import bar as m21bar
from music21 import dynamics as m21dynamics
from music21 import expressions as m21expressions
from music21 import harmony as m21harmony
from music21 import spanner as m21spanner
from music21 import stream as m21stream

from scorespeak import ScoreSpeak


def _right_barline(ss: ScoreSpeak, measure_number: int, part=None):
    """Return the right barline object of a given measure."""
    part_obj, _ = ss._resolve_part(part)
    measures = sorted(
        part_obj.getElementsByClass(m21stream.Measure),
        key=lambda m: m.number,
    )
    for m in measures:
        if m.number == measure_number:
            return m.rightBarline
    raise AssertionError(f"Measure {measure_number} not found.")


def _hidden_rests_in_measure(ss: ScoreSpeak, measure_number: int) -> list[ET.Element]:
    """Return exported hidden rest notes for one measure."""
    root = ET.fromstring(ss.to_musicxml_string())
    measure = root.find(f".//{{*}}measure[@number='{measure_number}']")
    assert measure is not None
    return [
        note
        for note in measure.findall("{*}note")
        if note.attrib.get("print-object") == "no"
        and note.find("{*}rest") is not None
    ]


def _spanner_measure_numbers(spanner_obj: m21spanner.Spanner) -> list[int]:
    """Return measure numbers for a spanner's endpoints."""
    numbers = []
    for element in spanner_obj.getSpannedElements():
        measure = element.getContextByClass(m21stream.Measure)
        if measure is not None:
            numbers.append(int(measure.number))
    return numbers


class TestAddMeasures:
    """Tests for adding measures."""

    def test_add_one_measure(self):
        ss = ScoreSpeak.create(measures=2)
        result = ss.add_measures(1)
        assert result.success
        assert ss.measure_count == 3

    def test_add_multiple_measures(self):
        ss = ScoreSpeak.create(measures=2)
        result = ss.add_measures(5)
        assert result.success
        assert ss.measure_count == 7

    def test_add_measures_to_empty_score(self):
        ss = ScoreSpeak.create(measures=0)
        result = ss.add_measures(3)
        assert result.success
        assert ss.measure_count == 3

    def test_add_measures_updates_all_parts(self):
        ss = ScoreSpeak.create(parts=["violin", "cello"], measures=2)
        result = ss.add_measures(2)
        assert result.success
        for part in ss.score.parts:
            assert len(list(part.getElementsByClass(m21stream.Measure))) == 4

    def test_add_measures_inherits_time_signature(self):
        ss = ScoreSpeak.create(time_signature="3/4", measures=2)
        ss.add_measures(2)
        for m in range(1, 5):
            assert ss.get_active_time_signature(m) == "3/4"

    def test_add_zero_measures_fails(self):
        ss = ScoreSpeak.create(measures=2)
        with pytest.raises(ValueError, match="at least 1"):
            ss.add_measures(0)

    def test_add_measures_moves_final_barline_to_new_end(self):
        ss = ScoreSpeak.create(measures=4)
        ss.set_barline("final", 4)

        result = ss.add_measures(2)

        assert result.success
        old_last = _right_barline(ss, 4)
        new_last = _right_barline(ss, 6)
        assert old_last is None or old_last.type in ("regular", "normal")
        assert new_last is not None
        assert new_last.type == "final"
        assert not isinstance(new_last, m21bar.Repeat)

    def test_add_measures_moves_double_barline_to_new_end(self):
        ss = ScoreSpeak.create(measures=3)
        ss.set_barline("double", 3)

        ss.add_measures(1)

        old_last = _right_barline(ss, 3)
        new_last = _right_barline(ss, 4)
        assert old_last is None or old_last.type in ("regular", "normal")
        assert new_last is not None
        assert new_last.type == "double"

    def test_add_measures_reports_moved_barline(self):
        ss = ScoreSpeak.create(measures=2)
        ss.set_barline("final", 2)

        result = ss.add_measures(3)

        moved = result.details.get("barlines_moved")
        assert moved is not None
        assert len(moved) == 1
        assert moved[0]["barline_type"] == "final"
        assert moved[0]["from_measure"] == 2
        assert moved[0]["to_measure"] == 5

    def test_add_measures_leaves_regular_barline_alone(self):
        ss = ScoreSpeak.create(measures=3)

        result = ss.add_measures(2)

        assert "barlines_moved" not in result.details
        new_last = _right_barline(ss, 5)
        assert new_last is None or new_last.type in ("regular", "normal")

    def test_add_measures_does_not_move_repeat_end(self):
        ss = ScoreSpeak.create(measures=3)
        ss.add_repeat(1, 3)

        result = ss.add_measures(2)

        assert "barlines_moved" not in result.details
        old_last = _right_barline(ss, 3)
        assert isinstance(old_last, m21bar.Repeat)
        new_last = _right_barline(ss, 5)
        assert new_last is None or new_last.type in ("regular", "normal")

    def test_add_measures_moves_final_barline_per_part(self):
        ss = ScoreSpeak.create(parts=["violin", "cello"], measures=4)
        ss.set_barline("final", 4)

        ss.add_measures(2)

        for part_idx in (0, 1):
            old_last = _right_barline(ss, 4, part=part_idx)
            new_last = _right_barline(ss, 6, part=part_idx)
            assert old_last is None or old_last.type in ("regular", "normal")
            assert new_last is not None
            assert new_last.type == "final"


class TestInsertMeasure:
    """Tests for inserting measures."""

    def test_insert_at_beginning(self):
        ss = ScoreSpeak.create(measures=3)
        result = ss.insert_measure(before=1)
        assert result.success
        assert ss.measure_count == 4

    def test_insert_in_middle(self):
        ss = ScoreSpeak.create(measures=4)
        result = ss.insert_measure(before=3)
        assert result.success
        assert ss.measure_count == 5

    def test_insert_at_end(self):
        ss = ScoreSpeak.create(measures=3)
        result = ss.insert_measure(before=4)
        assert result.success
        assert ss.measure_count == 4

    def test_insert_multiple(self):
        ss = ScoreSpeak.create(measures=3)
        result = ss.insert_measure(before=2, count=3)
        assert result.success
        assert ss.measure_count == 6

    def test_insert_preserves_continuity(self):
        ss = ScoreSpeak.create(time_signature="6/8", measures=3)
        ss.insert_measure(before=2)
        for m in range(1, 5):
            assert ss.get_active_time_signature(m) == "6/8"

    def test_insert_beyond_range_fails(self):
        ss = ScoreSpeak.create(measures=3)
        with pytest.raises(ValueError, match="Cannot insert"):
            ss.insert_measure(before=10)

    def test_insert_at_end_moves_final_barline(self):
        ss = ScoreSpeak.create(measures=3)
        ss.set_barline("final", 3)

        ss.insert_measure(before=4, count=2)

        old_last = _right_barline(ss, 3)
        new_last = _right_barline(ss, 5)
        assert old_last is None or old_last.type in ("regular", "normal")
        assert new_last is not None
        assert new_last.type == "final"

    def test_insert_in_middle_preserves_final_barline_on_last_measure(self):
        ss = ScoreSpeak.create(measures=4)
        ss.set_barline("final", 4)

        ss.insert_measure(before=2, count=1)

        last = _right_barline(ss, 5)
        assert last is not None
        assert last.type == "final"
        middle = _right_barline(ss, 2)
        assert middle is None or middle.type in ("regular", "normal")


class TestDeleteMeasures:
    """Tests for deleting measures."""

    def test_delete_single(self):
        ss = ScoreSpeak.create(measures=4)
        result = ss.delete_measure(2)
        assert result.success
        assert ss.measure_count == 3

    def test_delete_range(self):
        ss = ScoreSpeak.create(measures=6)
        result = ss.delete_measures(2, 4)
        assert result.success
        assert ss.measure_count == 3

    def test_delete_first_measure(self):
        ss = ScoreSpeak.create(measures=4)
        ss.delete_measure(1)
        assert ss.measure_count == 3

    def test_delete_last_measure(self):
        ss = ScoreSpeak.create(measures=4)
        ss.delete_measure(4)
        assert ss.measure_count == 3

    def test_delete_nonexistent_fails(self):
        ss = ScoreSpeak.create(measures=3)
        with pytest.raises(ValueError, match="Cannot delete"):
            ss.delete_measure(5)

    def test_delete_invalid_range_fails(self):
        ss = ScoreSpeak.create(measures=4)
        with pytest.raises(ValueError, match="must be >="):
            ss.delete_measures(4, 2)

    def test_delete_renumbers_correctly(self):
        ss = ScoreSpeak.create(measures=5)
        ss.delete_measure(3)
        part_obj, _ = ss._resolve_part(None)
        from music21 import stream as m21stream
        measures = sorted(
            part_obj.getElementsByClass(m21stream.Measure),
            key=lambda m: m.number,
        )
        numbers = [m.number for m in measures]
        assert numbers == [1, 2, 3, 4]

    def test_delete_removes_spanners_touching_deleted_measures(self) -> None:
        """Deleting a measure removes note-anchored spans that touch it."""
        ss = ScoreSpeak.create(measures=3)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=2, beat=1)
        ss.add_slur(1, 1.0, 2, 1.0)

        result = ss.delete_measure(2)

        part_obj, _ = ss._resolve_part(None)
        assert list(part_obj.getElementsByClass(m21spanner.Slur)) == []
        assert result.details["removed_spanners"][0]["type"] == "Slur"

    def test_delete_measure_clears_ties_touching_deleted_notes(self) -> None:
        """Deleting a measure clears tied markers on surviving tied notes."""
        ss = ScoreSpeak.create(measures=2)
        ss._add_note_one("C4", "whole", measure=1, beat=1)
        ss._add_note_one("C4", "whole", measure=2, beat=1)
        ss.add_tie(measure=1, beat=1)

        result = ss.delete_measure(1)

        sounding = [note for note in ss.get_notes(measure=1) if not note.is_rest]
        assert len(sounding) == 1
        assert not sounding[0].is_tied
        assert len(result.details["removed_ties"]) == 2

    def test_delete_removes_anchor_backed_spanners_touching_deleted_measures(
        self,
    ) -> None:
        """Deleting a measure removes spans backed by explicit anchors."""
        ss = ScoreSpeak.create(measures=3)
        ss._add_note_one("C4", "whole", measure=1, beat=1)
        ss._add_note_one("D4", "whole", measure=2, beat=1)
        ss.add_hairpin("crescendo", 1, 1.0, 2, 1.0, part=0)

        result = ss.delete_measure(2)

        part_obj, _ = ss._resolve_part(None)
        assert list(part_obj.getElementsByClass(m21dynamics.Crescendo)) == []
        assert result.details["removed_spanners"][0]["type"] == "Crescendo"
        assert result.details["removed_spanner_anchors"] == 1

    def test_delete_shifts_surviving_spanner_measure_attrs(self) -> None:
        """Deleting earlier measures shifts logical span measure metadata."""
        ss = ScoreSpeak.create(measures=4)
        ss._add_note_one("C4", "quarter", measure=3, beat=1)
        ss._add_note_one("D4", "quarter", measure=4, beat=2)
        ss.add_hairpin("crescendo", 3, 1.0, 4, 2.0, part=0)

        result = ss.delete_measure(1)

        part_obj, _ = ss._resolve_part(None)
        hairpins = list(part_obj.getElementsByClass(m21dynamics.Crescendo))
        assert len(hairpins) == 1
        assert _spanner_measure_numbers(hairpins[0]) == [2, 3]
        assert hairpins[0].scorespeak_start_measure == 2
        assert hairpins[0].scorespeak_end_measure == 3
        assert result.details["shifted_spanner_measure_attrs"] == 2


class TestClearMeasures:
    """Tests for clearing measure contents while preserving bars."""

    def test_clear_single_measure_restores_visible_full_measure_rest(self):
        ss = ScoreSpeak.create(measures=2)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss.reshape_rests(
            measure=1,
            part=0,
            voice=1,
            start_beat=2,
            total_duration="quarter",
            rests=[{"duration": "quarter"}],
        )

        result = ss.clear_measures(1)

        assert result.success
        assert ss.measure_count == 2
        rests = [note for note in ss.get_notes(measure=1) if note.is_rest]
        assert [(rest.beat, rest.duration_type) for rest in rests] == [
            (1.0, "whole")
        ]
        part_payload = ss._build_bar_result_set({
            "scope": {"bar_range": (1, 1)},
        })["bars"][0]["parts"][0]
        assert part_payload["voices"][0]["events"][0][0] == "rest"
        assert len(_hidden_rests_in_measure(ss, 1)) == 0
        assert result.details["measures_cleared"][0]["events_removed"] == 3
        assert result.details["measures_cleared"][0][
            "rest_quarter_length"
        ] == 4.0

        second_result = ss.clear_measures(1)
        assert second_result.details["measures_cleared"][0][
            "events_removed"
        ] == 1

    def test_clear_range_in_all_parts(self):
        ss = ScoreSpeak.create(parts=["violin", "cello"], measures=3)
        for part in (0, 1):
            ss._add_note_one("C4", "quarter", measure=1, beat=1, part=part)
            ss._add_note_one("D4", "quarter", measure=2, beat=1, part=part)

        result = ss.clear_measures(1, 2)

        assert result.details["parts"] == [0, 1]
        assert len(result.details["measures_cleared"]) == 4
        for part in (0, 1):
            assert [
                (note.beat, note.duration_type)
                for note in ss.get_notes(measure=1, part=part)
                if note.is_rest
            ] == [(1.0, "whole")]
            assert [
                (note.beat, note.duration_type)
                for note in ss.get_notes(measure=2, part=part)
                if note.is_rest
            ] == [(1.0, "whole")]

    def test_clear_can_target_one_part(self):
        ss = ScoreSpeak.create(parts=["violin", "cello"], measures=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
        ss._add_note_one("C3", "quarter", measure=1, beat=1, part=1)

        result = ss.clear_measures(1, part=1)

        assert result.details["parts"] == [1]
        assert [
            note.pitch
            for note in ss.get_notes(measure=1, part=0)
            if not note.is_rest
        ] == ["C4"]
        assert [
            (note.beat, note.duration_type)
            for note in ss.get_notes(measure=1, part=1)
            if note.is_rest
        ] == [(1.0, "whole")]

    def test_clear_multi_voice_measure_requires_explicit_scope(self) -> None:
        """Broad clears reject multi-voice measures unless scope is explicit."""
        ss = ScoreSpeak.create(measures=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=1, voice=1)
        ss._add_note_one("G3", "half", measure=1, beat=1, voice=2)

        with pytest.raises(ValueError, match="multiple rhythmic voices"):
            ss.clear_measures(1)

    def test_clear_can_target_one_voice(self) -> None:
        """A voice-scoped clear removes only the selected rhythmic voice."""
        ss = ScoreSpeak.create(measures=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=1, voice=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=2, voice=1)
        ss._add_note_one("G3", "half", measure=1, beat=1, voice=2)

        result = ss.clear_measures(1, voice=1)

        assert result.success
        assert result.details["voice"] == 1
        assert [
            note.pitch
            for note in ss.get_notes(measure=1, voice=2)
            if not note.is_rest
        ] == ["G3"]
        assert [
            (note.beat, note.duration_type)
            for note in ss.get_notes(measure=1, voice=1)
            if note.is_rest
        ] == [(1.0, "whole")]

    def test_clear_multi_voice_measure_all_voices_is_explicit(self) -> None:
        """all_voices=True intentionally clears every voice in the measure."""
        ss = ScoreSpeak.create(measures=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=1, voice=1)
        ss._add_note_one("G3", "half", measure=1, beat=1, voice=2)

        result = ss.clear_measures(1, all_voices=True)

        assert result.success
        assert result.details["all_voices"] is True
        assert [
            (note.beat, note.duration_type)
            for note in ss.get_notes(measure=1)
            if note.is_rest
        ] == [(1.0, "whole")]

    def test_clear_measure_clears_ties_touching_removed_notes(self) -> None:
        """Clearing a measure clears tie markers on notes outside the measure."""
        ss = ScoreSpeak.create(measures=2)
        ss._add_note_one("C4", "whole", measure=1, beat=1)
        ss._add_note_one("C4", "whole", measure=2, beat=1)
        ss.add_tie(measure=1, beat=1)

        result = ss.clear_measures(1)

        sounding = [note for note in ss.get_notes(measure=2) if not note.is_rest]
        assert len(sounding) == 1
        assert not sounding[0].is_tied
        assert result.details["measures_cleared"][0]["ties_removed"] == 2

    def test_clear_voice_clears_ties_touching_removed_voice_notes(self) -> None:
        """Voice-scoped clearing clears tied markers in that voice only."""
        ss = ScoreSpeak.create(measures=2)
        ss._add_note_one("C4", "whole", measure=1, beat=1, voice=1)
        ss._add_note_one("C4", "whole", measure=2, beat=1, voice=1)
        ss._add_note_one("G3", "half", measure=1, beat=1, voice=2)
        ss.add_tie(measure=1, beat=1, voice=1)

        result = ss.clear_measures(1, voice=1)

        voice_one = [
            note for note in ss.get_notes(measure=2, voice=1) if not note.is_rest
        ]
        voice_two = [
            note for note in ss.get_notes(measure=1, voice=2) if not note.is_rest
        ]
        assert len(voice_one) == 1
        assert not voice_one[0].is_tied
        assert [note.pitch for note in voice_two] == ["G3"]
        assert result.details["measures_cleared"][0]["ties_removed"] == 2

    def test_clear_preserves_structure_and_removes_note_spanners(self):
        ss = ScoreSpeak.create(measures=2)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=2, beat=1)
        ss.add_slur(1, 1.0, 2, 1.0)
        ss.set_barline("double", 1)
        ss.set_time_signature("3/4", 2)

        result = ss.clear_measures(1)

        part_obj, _ = ss._resolve_part(None)
        assert result.details["measures_cleared"][0]["spanners_removed"] == 1
        assert list(part_obj.getElementsByClass(m21spanner.Slur)) == []
        assert _right_barline(ss, 1).type == "double"
        assert ss.get_active_time_signature(2) == "3/4"
        assert [
            note.pitch
            for note in ss.get_notes(measure=2)
            if not note.is_rest
        ] == ["D4"]

    def test_clear_invalid_range_does_not_mutate(self):
        ss = ScoreSpeak.create(measures=2)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)

        with pytest.raises(ValueError, match="Cannot clear through measure 3"):
            ss.clear_measures(1, 3)

        assert [
            note.pitch
            for note in ss.get_notes(measure=1)
            if not note.is_rest
        ] == ["C4"]

    def test_clear_pickup_measure_uses_effective_duration(self):
        ss = ScoreSpeak.create(time_signature="4/4", measures=1)
        ss.set_pickup_measure(1.0)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)

        result = ss.clear_measures(1)

        assert result.details["measures_cleared"][0][
            "rest_quarter_length"
        ] == 1.0
        rests = [note for note in ss.get_notes(measure=1) if note.is_rest]
        assert [(rest.beat, rest.quarter_length) for rest in rests] == [
            (1.0, 1.0)
        ]


class TestCopyMeasureContents:
    """Tests for copying measure contents while replacing target music."""

    def test_copy_single_measure_replaces_target_contents_and_markings(self):
        """Copying a bar replaces notes and local musical markings."""
        ss = ScoreSpeak.create(measures=2)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss.add_dynamic("mf", 1, 1)
        ss.add_text_expression("dolce", 1, 1)
        ss.add_chord_symbol("C", 1, 1)
        ss._add_note_one("D4", "quarter", measure=2, beat=1)
        ss.add_dynamic("ff", 2, 1)

        result = ss.copy_measure_contents(1, 2)

        assert result.success
        part_obj, _ = ss._resolve_part(None)
        measure = part_obj.measure(2)
        real_notes = [
            element
            for element in measure.recurse().notes
            if not isinstance(element, m21harmony.ChordSymbol)
        ]
        assert [element.pitch.nameWithOctave for element in real_notes] == ["C4"]
        dynamics = list(measure.getElementsByClass(m21dynamics.Dynamic))
        text = list(measure.getElementsByClass(m21expressions.TextExpression))
        chords = list(measure.getElementsByClass(m21harmony.ChordSymbol))
        assert [dynamic.value for dynamic in dynamics] == ["mf"]
        assert [expression.content for expression in text] == ["dolce"]
        assert len(chords) == 1
        assert result.details["measures_copied"][0]["removed_events"] == 2
        assert result.details["measures_copied"][0][
            "removed_local_markings"
        ] == 1

    def test_copy_range_defaults_to_corresponding_all_parts(self):
        """Omitting parts copies the range independently in every part."""
        ss = ScoreSpeak.create(parts=["violin", "cello"], measures=4)
        for part, pitches in enumerate((("C4", "D4"), ("C3", "D3"))):
            ss._add_note_one(pitches[0], "quarter", measure=1, beat=1, part=part)
            ss._add_note_one(pitches[1], "quarter", measure=2, beat=1, part=part)

        result = ss.copy_measure_contents(1, 3, count=2)

        assert result.details["parts"] == [
            {"source_part": 0, "target_part": 0},
            {"source_part": 1, "target_part": 1},
        ]
        assert [note.pitch for note in ss.get_notes(measure=3, part=0) if not note.is_rest] == ["C4"]
        assert [note.pitch for note in ss.get_notes(measure=4, part=0) if not note.is_rest] == ["D4"]
        assert [note.pitch for note in ss.get_notes(measure=3, part=1) if not note.is_rest] == ["C3"]
        assert [note.pitch for note in ss.get_notes(measure=4, part=1) if not note.is_rest] == ["D3"]

    def test_copy_cross_part_preserves_written_pitch(self):
        """Cross-part copy duplicates written material without transposition."""
        ss = ScoreSpeak.create(parts=["violin", "cello"], measures=2)
        ss._add_note_one("F#4", "quarter", measure=1, beat=1, part=0)

        ss.copy_measure_contents(1, 2, source_part=0, target_part=1)

        assert [
            note.pitch
            for note in ss.get_notes(measure=2, part=1)
            if not note.is_rest
        ] == ["F#4"]

    def test_copy_preserves_target_structure_and_rehearsal_mark(self):
        """Target-owned structural notation is not overwritten by copy."""
        ss = ScoreSpeak.create(measures=2)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss.set_barline("double", 2)
        ss.set_key_signature("G", 2)
        ss.add_rehearsal_mark("B", 2)
        ss.add_ending_bracket(1, start_measure=2, end_measure=2)

        ss.copy_measure_contents(1, 2)

        part_obj, _ = ss._resolve_part(None)
        measure = part_obj.measure(2)
        rehearsal_marks = list(
            measure.getElementsByClass(m21expressions.RehearsalMark)
        )
        endings = list(part_obj.getElementsByClass(m21spanner.RepeatBracket))
        assert _right_barline(ss, 2).type == "double"
        assert "G" in ss.get_active_key_signature(2)
        assert [mark.content for mark in rehearsal_marks] == ["B"]
        assert len(endings) == 1
        assert _spanner_measure_numbers(endings[0]) == [2]

    def test_copy_rejects_duration_mismatch_without_mutation(self):
        """Mismatched target duration fails before changing the live score."""
        ss = ScoreSpeak.create(measures=2)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=2, beat=1)
        ss.set_time_signature("3/4", 2)

        with pytest.raises(ValueError, match="mismatched effective durations"):
            ss.copy_measure_contents(1, 2)

        assert [
            note.pitch
            for note in ss.get_notes(measure=2)
            if not note.is_rest
        ] == ["D4"]

    def test_copy_rejects_target_part_without_source_part(self):
        """A target-only part request is ambiguous and rejected."""
        ss = ScoreSpeak.create(parts=["violin", "cello"], measures=2)

        with pytest.raises(ValueError, match="target_part cannot be provided"):
            ss.copy_measure_contents(1, 2, target_part=1)

    def test_copy_empty_source_restores_visible_full_measure_rest(self):
        """Copying an empty source measure leaves a valid empty target bar."""
        ss = ScoreSpeak.create(measures=2)
        ss._add_note_one("D4", "quarter", measure=2, beat=1)

        result = ss.copy_measure_contents(1, 2)

        rests = [note for note in ss.get_notes(measure=2) if note.is_rest]
        assert [(rest.beat, rest.duration_type) for rest in rests] == [
            (1.0, "whole")
        ]
        assert len(_hidden_rests_in_measure(ss, 2)) == 0
        assert not result.details["measures_copied"][0]["rest_added"]

    def test_copy_fully_contained_spanners(self) -> None:
        """Contained source slurs and hairpins are copied to target endpoints."""
        ss = ScoreSpeak.create(measures=4)
        for measure, pitch in enumerate(("C4", "D4", "E4", "F4"), start=1):
            ss._add_note_one(pitch, "quarter", measure=measure, beat=1)
        ss.add_slur(1, 1.0, 2, 1.0)
        ss.add_hairpin("crescendo", 1, 1.0, 2, 1.0)

        result = ss.copy_measure_contents(1, 3, count=2)

        part_obj, _ = ss._resolve_part(None)
        slur_spans = [
            _spanner_measure_numbers(spanner_obj)
            for spanner_obj in part_obj.getElementsByClass(m21spanner.Slur)
        ]
        hairpins = list(part_obj.getElementsByClass(m21dynamics.DynamicWedge))
        hairpin_spans = [
            _spanner_measure_numbers(spanner_obj)
            for spanner_obj in hairpins
        ]
        copied_hairpins = [
            spanner_obj
            for spanner_obj in hairpins
            if _spanner_measure_numbers(spanner_obj) == [3, 3]
        ]
        crescendo_details = [
            item
            for item in result.details["copied_spanners"]
            if item["type"] == "Crescendo"
        ]
        assert [3, 4] in slur_spans
        assert [3, 3] in hairpin_spans
        assert len(copied_hairpins) == 1
        assert copied_hairpins[0].scorespeak_start_measure == 3
        assert copied_hairpins[0].scorespeak_end_measure == 4
        assert crescendo_details[0]["measures"] == [3, 4]
        assert {item["type"] for item in result.details["copied_spanners"]} == {
            "Slur",
            "Crescendo",
        }

    def test_copy_skips_partial_logical_hairpin(self) -> None:
        """A hairpin with a logical endpoint outside the source is skipped."""
        ss = ScoreSpeak.create(measures=4)
        for measure, pitch in enumerate(("C4", "D4", "E4", "F4"), start=1):
            ss._add_note_one(pitch, "quarter", measure=measure, beat=1)
        ss.add_hairpin("crescendo", 1, 1.0, 2, 1.0)

        result = ss.copy_measure_contents(1, 3)

        part_obj, _ = ss._resolve_part(None)
        hairpin_spans = [
            _spanner_measure_numbers(spanner_obj)
            for spanner_obj in part_obj.getElementsByClass(
                m21dynamics.DynamicWedge
            )
        ]
        assert [3, 3] not in hairpin_spans
        assert result.details["copied_spanners"] == []
        assert result.details["skipped_spanners"][0]["type"] == "Crescendo"
        assert result.details["skipped_spanners"][0]["measures"] == [1, 2]

    def test_copy_removes_target_logical_hairpin_endpoint(self) -> None:
        """Replacing a logical endpoint bar removes its existing hairpin."""
        ss = ScoreSpeak.create(measures=4)
        for measure, pitch in enumerate(("C4", "D4", "E4", "F4"), start=1):
            ss._add_note_one(pitch, "quarter", measure=measure, beat=1)
        ss.add_hairpin("crescendo", 3, 1.0, 4, 1.0)

        result = ss.copy_measure_contents(1, 4)

        part_obj, _ = ss._resolve_part(None)
        hairpins = list(part_obj.getElementsByClass(m21dynamics.DynamicWedge))
        assert hairpins == []
        assert result.details["removed_target_spanners"][0]["type"] == "Crescendo"
        assert result.details["removed_target_spanners"][0]["measures"] == [3, 4]

    @pytest.mark.parametrize(
        ("span_type", "span_class", "detail_type"),
        [
            ("slur", m21spanner.Slur, "Slur"),
            ("glissando", m21spanner.Glissando, "Glissando"),
        ],
    )
    def test_copy_physical_spans_without_logical_metadata(
        self,
        span_type: str,
        span_class: type[m21spanner.Spanner],
        detail_type: str,
    ) -> None:
        """Spanners without logical metadata still copy by physical anchors."""
        ss = ScoreSpeak.create(measures=4)
        for measure, pitch in enumerate(("C4", "D4", "E4", "F4"), start=1):
            ss._add_note_one(pitch, "quarter", measure=measure, beat=1)
        if span_type == "slur":
            ss.add_slur(1, 1.0, 2, 1.0)
        else:
            ss.add_glissando(1, 1.0, 2, 1.0)

        result = ss.copy_measure_contents(1, 3, count=2)

        part_obj, _ = ss._resolve_part(None)
        spans = [
            _spanner_measure_numbers(spanner_obj)
            for spanner_obj in part_obj.getElementsByClass(span_class)
        ]
        assert [3, 4] in spans
        assert detail_type in {
            item["type"]
            for item in result.details["copied_spanners"]
        }

    @pytest.mark.parametrize(
        ("span_type", "span_class", "detail_type"),
        [
            ("slur", m21spanner.Slur, "Slur"),
            ("glissando", m21spanner.Glissando, "Glissando"),
        ],
    )
    def test_copy_skips_partial_physical_spans_without_logical_metadata(
        self,
        span_type: str,
        span_class: type[m21spanner.Spanner],
        detail_type: str,
    ) -> None:
        """Spanners without logical metadata skip partial physical copies."""
        ss = ScoreSpeak.create(measures=4)
        for measure, pitch in enumerate(("C4", "D4", "E4", "F4"), start=1):
            ss._add_note_one(pitch, "quarter", measure=measure, beat=1)
        if span_type == "slur":
            ss.add_slur(1, 1.0, 2, 1.0)
        else:
            ss.add_glissando(1, 1.0, 2, 1.0)

        result = ss.copy_measure_contents(1, 3)

        part_obj, _ = ss._resolve_part(None)
        spans = [
            _spanner_measure_numbers(spanner_obj)
            for spanner_obj in part_obj.getElementsByClass(span_class)
        ]
        assert [3, 4] not in spans
        assert detail_type in {
            item["type"]
            for item in result.details["skipped_spanners"]
        }

    def test_copy_skips_partial_source_spans_and_removes_target_spans(self):
        """Boundary-crossing source spans are skipped; target spans are removed."""
        ss = ScoreSpeak.create(measures=4)
        for measure, pitch in enumerate(("C4", "D4", "E4", "F4"), start=1):
            ss._add_note_one(pitch, "quarter", measure=measure, beat=1)
        ss.add_slur(1, 1.0, 2, 1.0)
        ss.add_slur(3, 1.0, 4, 1.0)

        result = ss.copy_measure_contents(1, 3)

        part_obj, _ = ss._resolve_part(None)
        slur_spans = [
            _spanner_measure_numbers(spanner_obj)
            for spanner_obj in part_obj.getElementsByClass(m21spanner.Slur)
        ]
        assert [1, 2] in slur_spans
        assert [3, 4] not in slur_spans
        assert result.details["skipped_spanners"][0]["type"] == "Slur"
        assert result.details["removed_target_spanners"][0]["type"] == "Slur"

    def test_copy_strips_boundary_ties_but_keeps_contained_ties(self):
        """Dangling tie fragments are stripped while complete copied ties remain."""
        ss = ScoreSpeak.create(measures=4)
        for measure in range(1, 5):
            ss._add_note_one("C4", "whole", measure=measure, beat=1)
        ss.add_tie(measure=1, beat=1.0)

        single_result = ss.copy_measure_contents(2, 4)

        assert not ss.get_notes(measure=4)[0].is_tied
        assert single_result.details["measures_copied"][0]["stripped_ties"] == 1

        range_result = ss.copy_measure_contents(1, 3, count=2)

        assert ss.get_notes(measure=3)[0].is_tied
        assert ss.get_notes(measure=4)[0].is_tied
        assert sum(
            item["stripped_ties"]
            for item in range_result.details["measures_copied"]
        ) == 0

    def test_copy_preserves_voices_chords_and_lyrics(self):
        """Cloned measure contents preserve voices, chords, and attached lyrics."""
        ss = ScoreSpeak.create(measures=2)
        ss.add_chord(["C4", "E4"], "quarter", measure=1, beat=1, voice=1)
        ss.add_lyric("la", 1, 1, voice=1)
        ss._add_note_one("G4", "quarter", measure=1, beat=1, voice=2)

        ss.copy_measure_contents(1, 2)

        notes = ss.get_notes(measure=2, voice=1)
        second_voice = ss.get_notes(measure=2, voice=2)
        chord_notes = [note_info for note_info in notes if note_info.is_chord]
        assert {note_info.pitch for note_info in chord_notes} == {"C4", "E4"}
        assert second_voice[0].pitch == "G4"
        part_obj, _ = ss._resolve_part(None)
        copied_chord = list(part_obj.measure(2).recurse().notes)[0]
        assert copied_chord.lyric == "la"


class TestGetMeasureInfo:
    """Tests for measure info queries."""

    def test_basic_info(self):
        ss = ScoreSpeak.create(
            time_signature="4/4",
            key_signature="G",
            measures=4,
        )
        info = ss.get_measure_info(1)
        assert info.number == 1
        assert info.time_signature == "4/4"
        assert "G" in info.key_signature
        assert info.beat_count == 4.0

    def test_info_with_tempo(self):
        ss = ScoreSpeak.create(tempo=120.0, measures=2)
        info = ss.get_measure_info(1)
        assert info.tempo == 120.0
