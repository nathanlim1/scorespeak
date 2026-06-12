"""Tests for part management operations."""

from pathlib import Path

from music21 import bar as m21bar
from music21 import clef as m21clef
from music21 import expressions as m21expressions
from music21 import layout as m21layout
from music21 import note as m21note
from music21 import repeat as m21repeat
from music21 import stream as m21stream
from music21 import tempo as m21tempo
import pytest

from scorespeak import ScoreSpeak
from scorespeak.music.pitch_space import part_transposition_interval
from scorespeak.score.staff_groups import detect_staff_groups


def _part_names(score_state: ScoreSpeak) -> list[str]:
    """Return part names in the score's visible part order."""
    return [part.name for part in score_state.list_parts()]


def _make_grand_staff_score() -> ScoreSpeak:
    """Build a minimal score with one piano grand staff."""
    rh = m21stream.PartStaff()
    rh.partName = "Piano"
    rh.append(m21clef.TrebleClef())
    rh_measure = m21stream.Measure(number=1)
    rh_measure.append(m21note.Note("C4"))
    rh.append(rh_measure)

    lh = m21stream.PartStaff()
    lh.partName = "Piano"
    lh.append(m21clef.BassClef())
    lh_measure = m21stream.Measure(number=1)
    lh_measure.append(m21note.Note("C3"))
    lh.append(lh_measure)

    score = m21stream.Score()
    score.insert(0, rh)
    score.insert(0, lh)
    score.insert(0, m21layout.StaffGroup([rh, lh], name="Piano", symbol="brace"))
    return ScoreSpeak(score)


class TestAddPart:
    """Tests for adding parts."""

    def test_add_part(self):
        ss = ScoreSpeak.create(parts=["piano"], measures=4)
        result = ss.add_part(instrument="violin")
        assert result.success
        assert ss.part_count == 2

    @pytest.mark.parametrize(
        ("instrument_name", "clef_class"),
        [
            ("violin", m21clef.TrebleClef),
            ("viola", m21clef.AltoClef),
            ("cello", m21clef.BassClef),
            ("bassoon", m21clef.BassClef),
            ("guitar", m21clef.Treble8vbClef),
            ("snare drum", m21clef.PercussionClef),
        ],
    )
    def test_add_part_uses_default_clef_conventions(
        self,
        instrument_name: str,
        clef_class: type[m21clef.Clef],
    ) -> None:
        """``add_part`` chooses default clefs from instrument conventions."""
        ss = ScoreSpeak.create(parts=["piano"], measures=1)
        ss.add_part(instrument=instrument_name)

        added_part = list(ss.score.parts)[1]
        active_clef = ss._get_active_clef_obj(added_part, 1)

        assert isinstance(active_clef, clef_class)

    def test_add_part_unknown_instrument_defaults_to_treble(self) -> None:
        """Unknown instruments fall back to treble clef."""
        ss = ScoreSpeak.create(parts=["piano"], measures=1)
        ss.add_part(instrument="theremin")

        added_part = list(ss.score.parts)[1]
        active_clef = ss._get_active_clef_obj(added_part, 1)

        assert isinstance(active_clef, m21clef.TrebleClef)

    def test_add_part_resolves_dashed_instrument_labels(self) -> None:
        """Instrument lookup treats dashed labels as word separators."""
        ss = ScoreSpeak.create(parts=["piano"], measures=1)
        ss.add_part(name="English Horn", instrument="english-horn")

        added_part = list(ss.score.parts)[1]
        instrument = added_part.getInstrument(returnDefault=False)

        assert instrument is not None
        assert instrument.instrumentName == "English Horn"

    @pytest.mark.parametrize(
        "instrument_name",
        [
            "Horn in E♭",
            "Horn in E-flat",
            "Horn in E flat",
            "E-flat horn",
        ],
    )
    def test_create_normalizes_eflat_horn_labels(
        self,
        instrument_name: str,
    ) -> None:
        """E-flat horn spellings resolve to the correct transposition."""
        ss = ScoreSpeak.create(
            parts=[instrument_name],
            measures=1,
            key_signature="Eb",
        )
        part = list(ss.score.parts)[0]
        interval = part_transposition_interval(part)

        assert interval is not None
        assert interval.semitones == -9
        assert part.atSoundingPitch is False
        assert ss.get_active_key_signature(1, part=0) == "C major"

    def test_add_part_with_name(self):
        ss = ScoreSpeak.create(parts=["piano"], measures=4)
        result = ss.add_part(name="Solo Violin", instrument="violin")
        assert result.success
        parts = ss.list_parts()
        assert any(p.name == "Solo Violin" for p in parts)

    def test_add_part_rejects_duplicate_display_name(self) -> None:
        """Adding a part with an existing display name should fail clearly."""
        ss = ScoreSpeak.create(parts=[], measures=0)
        ss.set_score_parts(
            parts=[
                {"name": "1st Violin Staff 1", "instrument": "violin"},
                {"name": "1st Violin Staff 2", "instrument": "violin"},
            ]
        )

        with pytest.raises(ValueError, match="already exists"):
            ss.add_part(name="1st Violin Staff 1", instrument="violin")

    def test_added_part_has_matching_measures(self):
        ss = ScoreSpeak.create(parts=["piano"], measures=4)
        ss.add_part(instrument="violin")
        parts = ss.list_parts()
        for p in parts:
            assert p.measure_count == 4

    def test_add_part_to_empty_score(self):
        ss = ScoreSpeak.create(parts=["piano"], measures=0)
        result = ss.add_part(instrument="violin")
        assert result.success

    def test_add_part_appends_without_index(self) -> None:
        """``index=None`` appends the new part after existing parts."""
        ss = ScoreSpeak.create(parts=["piano", "violin"], measures=1)
        result = ss.add_part(name="Horn", instrument="horn")

        assert _part_names(ss) == ["Piano", "Violin", "Horn"]
        assert result.details["part_index"] == 2

    def test_add_part_insert_at_index_zero(self) -> None:
        """``index=0`` inserts the new part before the current first part."""
        ss = ScoreSpeak.create(parts=["piano", "violin"], measures=1)
        result = ss.add_part(name="Horn", instrument="horn", index=0)

        assert _part_names(ss) == ["Horn", "Piano", "Violin"]
        assert ss.part_count == 3
        assert result.details["part_index"] == 0

    def test_add_part_insert_at_middle_index(self) -> None:
        """A middle index inserts between existing parts."""
        ss = ScoreSpeak.create(parts=["piano", "violin"], measures=1)
        result = ss.add_part(name="Horn", instrument="horn", index=1)

        assert _part_names(ss) == ["Piano", "Horn", "Violin"]
        assert result.details["part_index"] == 1

    def test_add_part_insert_at_len_appends(self) -> None:
        """``index=len(parts)`` appends just like the default behavior."""
        ss = ScoreSpeak.create(parts=["piano", "violin"], measures=1)
        result = ss.add_part(name="Horn", instrument="horn", index=2)

        assert _part_names(ss) == ["Piano", "Violin", "Horn"]
        assert result.details["part_index"] == 2

    def test_add_part_index_out_of_range_raises(self) -> None:
        """``index=len(parts)+1`` is outside the valid insertion range."""
        ss = ScoreSpeak.create(parts=["piano"], measures=1)
        with pytest.raises(ValueError, match="out of range"):
            ss.add_part(instrument="flute", index=2)

    def test_add_part_negative_index_raises(self) -> None:
        """Negative indices are invalid and do not use Python indexing."""
        ss = ScoreSpeak.create(parts=["piano"], measures=1)
        with pytest.raises(ValueError, match="out of range"):
            ss.add_part(instrument="flute", index=-1)

    def test_add_part_inserted_index_controls_resolution(self) -> None:
        """Part resolution follows the visible order after insertion."""
        ss = ScoreSpeak.create(parts=["piano", "violin"], measures=1)
        ss.add_part(name="Horn", instrument="horn", index=1)

        result = ss.remove_part(1)

        assert result.details["removed_part_name"] == "Horn"
        assert _part_names(ss) == ["Piano", "Violin"]

    def test_add_part_before_grand_staff_preserves_group_labels(self) -> None:
        """Grand-staff labels survive when a new part shifts their indices."""
        ss = _make_grand_staff_score()
        result = ss.add_part(name="Flute", instrument="flute", index=0)

        parts = ss.list_parts()

        assert [part.name for part in parts] == ["Flute", "Piano", "Piano"]
        assert result.details["part_index"] == 0
        assert parts[1].display_name == "Piano RH"
        assert parts[1].hand == "RH"
        assert parts[2].display_name == "Piano LH"
        assert parts[2].hand == "LH"

    def test_add_part_grand_staff_creates_grouped_piano_staves(self) -> None:
        """``grand_staff=True`` adds adjacent RH/LH piano staves."""
        ss = ScoreSpeak.create(parts=["violin"], measures=2)

        result = ss.add_part(name="Piano", instrument="piano", grand_staff=True)

        parts = ss.list_parts()
        groups = detect_staff_groups(ss.score)
        score_parts = list(ss.score.parts)

        assert result.success
        assert result.details["part_indices"] == [1, 2]
        assert result.details["display_names"] == ["Piano RH", "Piano LH"]
        assert result.details["grand_staff"] is True
        assert [part.display_name for part in parts] == [
            "Violin",
            "Piano RH",
            "Piano LH",
        ]
        assert parts[1].measure_count == 2
        assert parts[2].measure_count == 2
        assert isinstance(ss._get_active_clef_obj(score_parts[1], 1), m21clef.TrebleClef)
        assert isinstance(ss._get_active_clef_obj(score_parts[2], 1), m21clef.BassClef)
        assert groups[-1].part_indices == (1, 2)
        assert groups[-1].hand_labels == ("RH", "LH")

    def test_add_part_grand_staff_to_empty_score_creates_one_measure(self) -> None:
        """Adding a grand staff to a partless score mirrors add_part defaults."""
        ss = ScoreSpeak.create(parts=[], measures=0)

        result = ss.add_part(name="Piano", instrument="piano", grand_staff=True)

        assert result.success
        assert ss.part_count == 2
        assert [part.measure_count for part in ss.list_parts()] == [1, 1]

    def test_add_part_grand_staff_insert_at_index_zero(self) -> None:
        """A grand staff can be inserted before existing parts."""
        ss = ScoreSpeak.create(parts=["violin"], measures=1)

        result = ss.add_part(
            name="Piano",
            instrument="piano",
            index=0,
            grand_staff=True,
        )

        assert result.details["part_indices"] == [0, 1]
        assert [part.display_name for part in ss.list_parts()] == [
            "Piano RH",
            "Piano LH",
            "Violin",
        ]

    def test_add_part_grand_staff_insert_at_middle_index(self) -> None:
        """The insertion index points to the RH staff of the new group."""
        ss = ScoreSpeak.create(parts=["flute", "violin"], measures=1)

        result = ss.add_part(
            name="Piano",
            instrument="piano",
            index=1,
            grand_staff=True,
        )

        assert result.details["part_indices"] == [1, 2]
        assert [part.display_name for part in ss.list_parts()] == [
            "Flute",
            "Piano RH",
            "Piano LH",
            "Violin",
        ]

    def test_add_part_multiple_grand_staves_get_numbered_labels(self) -> None:
        """Duplicate piano groups are disambiguated by group number."""
        ss = ScoreSpeak.create(parts=[], measures=0)
        ss.add_part(name="Piano", instrument="piano", grand_staff=True)

        result = ss.add_part(name="Piano", instrument="piano", grand_staff=True)

        assert result.details["part_indices"] == [2, 3]
        assert [part.display_name for part in ss.list_parts()] == [
            "Piano 1 RH",
            "Piano 1 LH",
            "Piano 2 RH",
            "Piano 2 LH",
        ]

    def test_create_part_spec_can_request_grand_staff(self) -> None:
        """``ScoreSpeak.create`` accepts an opt-in grand-staff part spec."""
        ss = ScoreSpeak.create(
            parts=[{"instrument": "piano", "name": "Piano", "grand_staff": True}],
            measures=2,
        )

        parts = ss.list_parts()

        assert ss.part_count == 2
        assert [part.display_name for part in parts] == ["Piano RH", "Piano LH"]
        assert [part.hand for part in parts] == ["RH", "LH"]
        assert [part.measure_count for part in parts] == [2, 2]

    def test_add_notes_accepts_grand_staff_display_label(self) -> None:
        """Note tools can target labels such as ``Piano LH`` directly."""
        ss = ScoreSpeak.create(
            parts=[{"instrument": "piano", "name": "Piano", "grand_staff": True}],
            measures=1,
        )

        result = ss.add_notes(
            measure=1,
            part="Piano LH",
            voice=1,
            notes=[
                {"pitch": "C3", "beat": 1.0, "duration": "quarter", "dots": 0}
            ],
        )

        lh_notes = [note for note in ss.get_notes(measure=1, part=1) if not note.is_rest]
        rh_notes = [note for note in ss.get_notes(measure=1, part=0) if not note.is_rest]

        assert result.success
        assert result.details["part"] == 1
        assert [note.pitch for note in lh_notes] == ["C3"]
        assert rh_notes == []

    def test_created_grand_staff_round_trips_through_musicxml(
        self,
        tmp_path: Path,
    ) -> None:
        """Export/import preserves the brace group needed for RH/LH labels."""
        ss = ScoreSpeak.create(
            parts=[{"instrument": "piano", "name": "Piano", "grand_staff": True}],
            measures=1,
        )
        path = tmp_path / "grand_staff.musicxml"

        ss.to_musicxml(path)
        loaded = ScoreSpeak.from_musicxml(path)
        exported_xml = path.read_text(encoding="utf-8")

        assert "<staves>2</staves>" in exported_xml
        assert "<part-group" not in exported_xml
        assert [part.display_name for part in loaded.list_parts()] == [
            "Piano RH",
            "Piano LH",
        ]


class TestSetScoreParts:
    """Tests for replacement-style part setup."""

    def test_set_score_parts_replaces_and_orders_requested_parts(self) -> None:
        """Requested parts are kept in argument order and omitted parts vanish."""
        ss = ScoreSpeak.create(
            parts=[
                {"instrument": "violin", "name": "Violin"},
                {"instrument": "cello", "name": "Cello"},
                {"instrument": "flute", "name": "Flute"},
            ],
            measures=2,
        )

        result = ss.set_score_parts([
            {"instrument": "flute", "name": "Flute"},
            {"instrument": "violin", "name": "Violin"},
        ])

        assert result.success
        assert [part.display_name for part in ss.list_parts()] == [
            "Flute",
            "Violin",
        ]
        assert result.details["removed_parts"] == 1
        assert result.details["created_parts"] == 0

    def test_set_score_parts_moves_matching_part_with_content(self) -> None:
        """A matching existing part is moved rather than recreated."""
        ss = ScoreSpeak.create(
            parts=[
                {"instrument": "violin", "name": "Violin"},
                {"instrument": "cello", "name": "Cello"},
            ],
            measures=1,
        )
        ss.add_notes(
            measure=1,
            part="Cello",
            voice=1,
            notes=[{"pitch": "C3", "beat": 1.0, "duration": "quarter", "dots": 0}],
        )

        ss.set_score_parts([
            {"instrument": "cello", "name": "Cello"},
            {"instrument": "violin", "name": "Violin"},
        ])

        cello_notes = [
            note for note in ss.get_notes(measure=1, part=0) if not note.is_rest
        ]
        assert [part.display_name for part in ss.list_parts()] == [
            "Cello",
            "Violin",
        ]
        assert [note.pitch for note in cello_notes] == ["C3"]

    def test_set_score_parts_preserves_duplicate_matches_in_old_order(self) -> None:
        """Duplicate requested names reuse duplicate existing parts by occurrence."""
        ss = ScoreSpeak.create(
            parts=[
                {"instrument": "violin", "name": "Violin"},
                {"instrument": "violin", "name": "Violin"},
            ],
            measures=1,
        )
        ss.add_notes(
            measure=1,
            part=0,
            voice=1,
            notes=[{"pitch": "C4", "beat": 1.0, "duration": "quarter", "dots": 0}],
        )
        ss.add_notes(
            measure=1,
            part=1,
            voice=1,
            notes=[{"pitch": "D4", "beat": 1.0, "duration": "quarter", "dots": 0}],
        )

        ss.set_score_parts([
            {"instrument": "cello", "name": "Cello"},
            {"instrument": "violin", "name": "Violin"},
            {"instrument": "violin", "name": "Violin"},
        ])

        first_violin_notes = [
            note for note in ss.get_notes(measure=1, part=1) if not note.is_rest
        ]
        second_violin_notes = [
            note for note in ss.get_notes(measure=1, part=2) if not note.is_rest
        ]
        assert [note.pitch for note in first_violin_notes] == ["C4"]
        assert [note.pitch for note in second_violin_notes] == ["D4"]

    def test_set_score_parts_infers_keyed_transposition_from_display_name(
        self,
    ) -> None:
        """Generic transposing families use keyed names as a fallback."""
        ss = ScoreSpeak.create(parts=["piano"], measures=1, key_signature="Eb")

        result = ss.set_score_parts([
            {"instrument": "clarinet", "name": "B♭ Clarinet 1"},
            {"instrument": "horn", "name": "E♭ Horn 1"},
            {"instrument": "trumpet", "name": "C Trumpet 1"},
        ])
        parts = list(ss.score.parts)
        clarinet_interval = part_transposition_interval(parts[0])
        horn_interval = part_transposition_interval(parts[1])

        assert result.success
        assert clarinet_interval is not None
        assert clarinet_interval.semitones == -2
        assert horn_interval is not None
        assert horn_interval.semitones == -9
        assert part_transposition_interval(parts[2]) is None
        assert parts[2].atSoundingPitch is True
        assert ss.get_active_key_signature(1, part=0) == "F major"
        assert ss.get_active_key_signature(1, part=1) == "C major"
        assert ss.get_active_key_signature(1, part=2) == "E- major"

    def test_set_score_parts_rejects_malformed_specs(self) -> None:
        """The replacement tool validates its list and required instrument."""
        ss = ScoreSpeak.create(parts=["violin"], measures=1)

        with pytest.raises(ValueError, match="non-empty list"):
            ss.set_score_parts("violin")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="at least one"):
            ss.set_score_parts([])
        with pytest.raises(ValueError, match="instrument"):
            ss.set_score_parts([{"name": "Solo"}])  # type: ignore[list-item]
        with pytest.raises(ValueError, match="name"):
            ss.set_score_parts([{"instrument": "flute", "name": 7}])  # type: ignore[list-item]
        with pytest.raises(ValueError, match="Unsupported"):
            ss.set_score_parts([{"instrument": "piano", "grand_staff": True}])  # type: ignore[list-item]

    def test_set_score_parts_auto_creates_piano_grand_staff(self) -> None:
        """Piano requests automatically create RH/LH staves and default clefs."""
        ss = ScoreSpeak.create(parts=["violin"], measures=1)

        result = ss.set_score_parts([{"instrument": "piano", "name": "Piano"}])

        score_parts = list(ss.score.parts)
        assert result.success
        assert [part.display_name for part in ss.list_parts()] == [
            "Piano RH",
            "Piano LH",
        ]
        assert isinstance(ss._get_active_clef_obj(score_parts[0], 1), m21clef.TrebleClef)
        assert isinstance(ss._get_active_clef_obj(score_parts[1], 1), m21clef.BassClef)

    def test_set_score_parts_auto_creates_organ_pedal_staff(self) -> None:
        """Organ requests automatically create RH/LH/Pedal staves."""
        ss = ScoreSpeak.create(parts=["violin"], measures=1)

        ss.set_score_parts([{"instrument": "organ", "name": "Organ"}])

        assert [(part.display_name, part.hand) for part in ss.list_parts()] == [
            ("Organ RH", "RH"),
            ("Organ LH", "LH"),
            ("Organ Pedal", "Pedal"),
        ]

    def test_set_score_parts_preserves_existing_grand_staff_content(self) -> None:
        """Matching existing grand-staff groups are reused as logical parts."""
        ss = ScoreSpeak.create(
            parts=[{"instrument": "piano", "name": "Piano", "grand_staff": True}],
            measures=1,
        )
        ss.add_notes(
            measure=1,
            part="Piano LH",
            voice=1,
            notes=[{"pitch": "C3", "beat": 1.0, "duration": "quarter", "dots": 0}],
        )

        result = ss.set_score_parts([
            {"instrument": "piano", "name": "Piano"},
        ])

        lh_notes = [
            note for note in ss.get_notes(measure=1, part="Piano LH")
            if not note.is_rest
        ]
        assert result.details["created_parts"] == 0
        assert [note.pitch for note in lh_notes] == ["C3"]

    def test_set_score_parts_expands_existing_single_piano(self) -> None:
        """A matching single-staff piano seeds the RH staff of a grand staff."""
        ss = ScoreSpeak.create(parts=["piano"], measures=1)
        ss.add_notes(
            measure=1,
            part=0,
            voice=1,
            notes=[{"pitch": "C4", "beat": 1.0, "duration": "quarter", "dots": 0}],
        )

        result = ss.set_score_parts([
            {"instrument": "piano", "name": "Piano"},
        ])

        rh_notes = [
            note for note in ss.get_notes(measure=1, part="Piano RH")
            if not note.is_rest
        ]
        assert result.details["created_parts"] == 1
        assert [part.display_name for part in ss.list_parts()] == [
            "Piano RH",
            "Piano LH",
        ]
        assert [note.pitch for note in rh_notes] == ["C4"]

    def test_set_score_parts_preserves_score_level_structure_on_new_first_part(
        self,
    ) -> None:
        """Score-level marks survive when their original carrier part is removed."""
        ss = ScoreSpeak.create(parts=["piano"], measures=4, tempo=120)
        ss.set_time_signature("3/4", measure_number=2)
        ss.set_key_signature("G", measure_number=2)
        ss.set_barline("double", measure_number=1)
        ss.add_repeat(start_measure=2, end_measure=4)
        ss.add_system_break(measure_number=3)
        ss.add_page_break(measure_number=4)
        ss.set_tempo(90, measure_number=3)
        ss.add_rehearsal_mark("A", measure_number=2)
        ss.add_coda(measure_number=3)

        ss.set_score_parts([{"instrument": "flute", "name": "Flute"}])

        part_obj = list(ss.score.parts)[0]
        measure_1 = ss._resolve_measure(part_obj, 1)
        measure_2 = ss._resolve_measure(part_obj, 2)
        measure_3 = ss._resolve_measure(part_obj, 3)
        measure_4 = ss._resolve_measure(part_obj, 4)

        assert ss.get_active_time_signature(2) == "3/4"
        assert ss.get_active_key_signature(2) == "G major"
        assert ss.get_active_tempo(3) == 90.0
        assert isinstance(measure_1.rightBarline, m21bar.Barline)
        assert measure_1.rightBarline.type == "double"
        assert isinstance(measure_2.leftBarline, m21bar.Repeat)
        assert isinstance(measure_4.rightBarline, m21bar.Repeat)
        assert any(
            layout.isNew
            for layout in measure_3.getElementsByClass(m21layout.SystemLayout)
        )
        assert any(
            layout.isNew
            for layout in measure_4.getElementsByClass(m21layout.PageLayout)
        )
        assert len(list(measure_3.getElementsByClass(m21tempo.MetronomeMark))) >= 1
        assert len(list(measure_2.getElementsByClass(m21expressions.RehearsalMark))) == 1
        assert len(list(measure_3.getElementsByClass(m21repeat.Coda))) == 1

    def test_set_score_parts_round_trips_auto_grand_staff(
        self,
        tmp_path: Path,
    ) -> None:
        """Automatically created grand staves survive MusicXML export/import."""
        ss = ScoreSpeak.create(parts=["violin"], measures=1)
        ss.set_score_parts([{"instrument": "piano", "name": "Piano"}])
        path = tmp_path / "set_parts_grand_staff.musicxml"

        ss.to_musicxml(path)
        loaded = ScoreSpeak.from_musicxml(path)

        assert [part.display_name for part in loaded.list_parts()] == [
            "Piano RH",
            "Piano LH",
        ]


class TestRemovePart:
    """Tests for removing parts."""

    def test_remove_part_by_index(self):
        ss = ScoreSpeak.create(parts=["piano", "violin"], measures=2)
        result = ss.remove_part(1)
        assert result.success
        assert ss.part_count == 1

    def test_remove_part_by_name(self):
        ss = ScoreSpeak.create(parts=["piano", "violin"], measures=2)
        result = ss.remove_part("Violin")
        assert result.success
        assert ss.part_count == 1

    def test_remove_last_part_fails(self):
        ss = ScoreSpeak.create(parts=["piano"], measures=2)
        with pytest.raises(ValueError, match="Cannot remove the last part"):
            ss.remove_part(0)

    def test_remove_invalid_index_fails(self):
        ss = ScoreSpeak.create(parts=["piano", "violin"], measures=2)
        with pytest.raises(ValueError, match="out of range"):
            ss.remove_part(5)


class TestListParts:
    """Tests for listing parts."""

    def test_list_parts(self):
        ss = ScoreSpeak.create(
            parts=["violin", "viola", "cello"],
            measures=2,
        )
        parts = ss.list_parts()
        assert len(parts) == 3
        assert all(p.measure_count == 2 for p in parts)

    def test_get_part_info(self):
        ss = ScoreSpeak.create(parts=["flute"], measures=4)
        info = ss.get_part_info(0)
        assert info.index == 0
        assert info.measure_count == 4
