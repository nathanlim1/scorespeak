"""Tests for LayoutMixin — metadata, layout breaks, and transposition."""

import os
import tempfile

import pytest
from music21 import chord as m21chord
from music21 import instrument as m21instrument
from music21 import interval as m21interval
from music21 import layout as m21layout
from music21 import note as m21note
from music21 import stream as m21stream

from scorespeak import ScoreSpeak
from scorespeak.music.pitch_space import part_transposition_interval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_score(
    title: str = "Test",
    composer: str = "",
    time_signature: str = "4/4",
    key_signature: str = "C",
    measures: int = 4,
    parts: list | None = None,
) -> ScoreSpeak:
    """Create a basic ScoreSpeak for testing."""
    return ScoreSpeak.create(
        title=title,
        composer=composer,
        time_signature=time_signature,
        key_signature=key_signature,
        measures=measures,
        parts=parts or ["piano"],
    )


def _add_note(ss: ScoreSpeak, pitch: str, measure: int, part: int = 0) -> None:
    """Helper to add a note at beat 1 of a measure."""
    ss._add_note_one(pitch, "quarter", measure, beat=1, part=part)


# ===================================================================
# Metadata Tests
# ===================================================================


class TestSetTitle:
    """Tests for set_title."""

    def test_set_title(self):
        ss = _make_score(title="Original")
        result = ss.set_title("New Title")
        assert result.success
        assert result.details["old_title"] == "Original"
        assert result.details["new_title"] == "New Title"
        assert ss.title == "New Title"

    def test_overwrite_title(self):
        ss = _make_score(title="First")
        ss.set_title("Second")
        result = ss.set_title("Third")
        assert result.success
        assert result.details["old_title"] == "Second"
        assert ss.title == "Third"

    def test_set_empty_title(self):
        ss = _make_score(title="Something")
        result = ss.set_title("")
        assert result.success
        assert ss.title == ""

    def test_set_title_on_bare_score(self):
        score = m21stream.Score()
        ss = ScoreSpeak(score)
        result = ss.set_title("From Scratch")
        assert result.success
        assert ss.title == "From Scratch"


class TestSetSubtitle:
    """Tests for set_subtitle."""

    def test_set_subtitle(self):
        ss = _make_score()
        result = ss.set_subtitle("Op. 1")
        assert result.success
        assert result.details["new_subtitle"] == "Op. 1"
        metadata = ss.get_metadata()
        assert metadata["subtitle"] == "Op. 1"

    def test_overwrite_subtitle(self):
        ss = _make_score()
        ss.set_subtitle("First Movement")
        result = ss.set_subtitle("Allegro con brio")
        assert result.success
        assert result.details["old_subtitle"] == "First Movement"
        assert ss.get_metadata()["subtitle"] == "Allegro con brio"

    def test_clear_subtitle(self):
        ss = _make_score()
        ss.set_subtitle("Tempo di valse")
        ss.set_subtitle("")
        assert ss.get_metadata()["subtitle"] == ""


class TestSetComposer:
    """Tests for set_composer."""

    def test_set_composer(self):
        ss = _make_score(composer="")
        result = ss.set_composer("J.S. Bach")
        assert result.success
        assert result.details["new_composer"] == "J.S. Bach"
        assert ss.composer == "J.S. Bach"

    def test_overwrite_composer(self):
        ss = _make_score(composer="Mozart")
        result = ss.set_composer("Beethoven")
        assert result.success
        assert result.details["old_composer"] == "Mozart"
        assert ss.composer == "Beethoven"


class TestGetMetadata:
    """Tests for get_metadata."""

    def test_full_metadata(self):
        ss = _make_score(title="Symphony No. 5", composer="Beethoven")
        ss.set_subtitle("Allegro con brio")
        md = ss.get_metadata()
        assert md["title"] == "Symphony No. 5"
        assert md["subtitle"] == "Allegro con brio"
        assert md["composer"] == "Beethoven"

    def test_default_metadata(self):
        ss = _make_score(title="Untitled", composer="")
        md = ss.get_metadata()
        assert md["title"] == "Untitled"
        assert md["subtitle"] == ""
        assert md["composer"] == ""

    def test_metadata_on_bare_score(self):
        score = m21stream.Score()
        ss = ScoreSpeak(score)
        md = ss.get_metadata()
        assert md["title"] == ""
        assert md["subtitle"] == ""
        assert md["composer"] == ""


# ===================================================================
# System Break Tests
# ===================================================================


class TestAddSystemBreak:
    """Tests for add_system_break."""

    def test_add_system_break(self):
        ss = _make_score(measures=4)
        result = ss.add_system_break(2)
        assert result.success
        assert result.description == (
            "System break added: measure 2 now starts a new system."
        )
        assert result.details["measure_number"] == 2

        part = list(ss.score.parts)[0]
        measure = part.measure(2)
        sys_layouts = list(measure.getElementsByClass(m21layout.SystemLayout))
        assert any(sl.isNew for sl in sys_layouts)

    def test_add_system_break_idempotent(self):
        ss = _make_score(measures=4)
        ss.add_system_break(2)
        ss.add_system_break(2)
        part = list(ss.score.parts)[0]
        measure = part.measure(2)
        sys_layouts = [
            sl for sl in measure.getElementsByClass(m21layout.SystemLayout)
            if sl.isNew
        ]
        assert len(sys_layouts) == 1

    def test_add_system_break_all_parts(self):
        ss = _make_score(measures=4, parts=["violin", "cello"])
        result = ss.add_system_break(3)
        assert result.success
        assert len(result.details["parts"]) == 2

        for part in ss.score.parts:
            m = part.measure(3)
            sys_layouts = list(m.getElementsByClass(m21layout.SystemLayout))
            assert any(sl.isNew for sl in sys_layouts)

    def test_add_system_break_part_argument_removed(self):
        ss = _make_score(measures=4, parts=["violin", "cello"])
        with pytest.raises(TypeError):
            ss.add_system_break(2, part=0)


class TestRemoveSystemBreak:
    """Tests for remove_system_break."""

    def test_remove_existing_system_break(self):
        ss = _make_score(measures=4)
        ss.add_system_break(2)
        result = ss.remove_system_break(2)
        assert result.success
        assert result.description == (
            "Removed system break(s) that made measure 2 start a new system."
        )
        assert result.details["removed_count"] >= 1

        part = list(ss.score.parts)[0]
        measure = part.measure(2)
        sys_layouts = [
            sl for sl in measure.getElementsByClass(m21layout.SystemLayout)
            if sl.isNew
        ]
        assert len(sys_layouts) == 0

    def test_remove_nonexistent_system_break(self):
        ss = _make_score(measures=4)
        result = ss.remove_system_break(2)
        assert not result.success
        assert "nothing to remove" in result.description.lower()


# ===================================================================
# Page Break Tests
# ===================================================================


class TestAddPageBreak:
    """Tests for add_page_break."""

    def test_add_page_break(self):
        ss = _make_score(measures=4)
        result = ss.add_page_break(3)
        assert result.success
        assert result.description == (
            "Page break added: measure 3 now starts a new page."
        )

        part = list(ss.score.parts)[0]
        measure = part.measure(3)
        page_layouts = list(measure.getElementsByClass(m21layout.PageLayout))
        assert any(pl.isNew for pl in page_layouts)

    def test_add_page_break_idempotent(self):
        ss = _make_score(measures=4)
        ss.add_page_break(3)
        ss.add_page_break(3)
        part = list(ss.score.parts)[0]
        measure = part.measure(3)
        page_layouts = [
            pl for pl in measure.getElementsByClass(m21layout.PageLayout)
            if pl.isNew
        ]
        assert len(page_layouts) == 1

    def test_add_page_break_all_parts(self):
        ss = _make_score(measures=4, parts=["violin", "cello"])
        result = ss.add_page_break(2)
        assert result.success
        assert len(result.details["parts"]) == 2


class TestRemovePageBreak:
    """Tests for remove_page_break."""

    def test_remove_existing_page_break(self):
        ss = _make_score(measures=4)
        ss.add_page_break(3)
        result = ss.remove_page_break(3)
        assert result.success
        assert result.description == (
            "Removed page break(s) that made measure 3 start a new page."
        )

        part = list(ss.score.parts)[0]
        measure = part.measure(3)
        page_layouts = [
            pl for pl in measure.getElementsByClass(m21layout.PageLayout)
            if pl.isNew
        ]
        assert len(page_layouts) == 0

    def test_remove_nonexistent_page_break(self):
        ss = _make_score(measures=4)
        result = ss.remove_page_break(2)
        assert not result.success


# ===================================================================
# Transposition Tests
# ===================================================================


class TestTransposeByInterval:
    """Tests for transpose with interval strings."""

    def test_transpose_up_perfect_fifth(self):
        ss = _make_score(measures=2)
        _add_note(ss, "C4", 1)
        result = ss.transpose("P5")
        assert result.success
        assert result.details["notes_transposed"] >= 1

        part = list(ss.score.parts)[0]
        m = part.measure(1)
        notes = [n for n in m.notes if isinstance(n, m21note.Note)]
        pitches = [n.nameWithOctave for n in notes]
        assert "G4" in pitches

    def test_transpose_down_minor_third(self):
        ss = _make_score(measures=2)
        _add_note(ss, "E4", 1)
        result = ss.transpose("-m3")
        assert result.success

        part = list(ss.score.parts)[0]
        m = part.measure(1)
        notes = [n for n in m.notes if isinstance(n, m21note.Note)]
        found_c_sharp = any(n.pitch.midi == 61 for n in notes)
        assert found_c_sharp

    def test_transpose_up_major_second(self):
        ss = _make_score(measures=2)
        _add_note(ss, "C4", 1)
        ss.transpose("M2")

        part = list(ss.score.parts)[0]
        m = part.measure(1)
        notes = [n for n in m.notes if isinstance(n, m21note.Note)]
        pitches = [n.nameWithOctave for n in notes]
        assert "D4" in pitches


class TestTransposeBySemitones:
    """Tests for transpose with integer semitone counts."""

    def test_transpose_up_by_semitones(self):
        ss = _make_score(measures=2)
        _add_note(ss, "C4", 1)
        result = ss.transpose(7)
        assert result.success

        part = list(ss.score.parts)[0]
        m = part.measure(1)
        notes = [n for n in m.notes if isinstance(n, m21note.Note)]
        assert any(n.pitch.midi == 67 for n in notes)

    def test_transpose_down_by_semitones(self):
        ss = _make_score(measures=2)
        _add_note(ss, "C4", 1)
        ss.transpose(-3)

        part = list(ss.score.parts)[0]
        m = part.measure(1)
        notes = [n for n in m.notes if isinstance(n, m21note.Note)]
        assert any(n.pitch.midi == 57 for n in notes)

    def test_transpose_zero_semitones(self):
        ss = _make_score(measures=2)
        _add_note(ss, "C4", 1)
        result = ss.transpose(0)
        assert result.success

        part = list(ss.score.parts)[0]
        m = part.measure(1)
        notes = [n for n in m.notes if isinstance(n, m21note.Note)]
        assert any(n.pitch.midi == 60 for n in notes)


class TestTransposeMeasureRange:
    """Tests for transpose with start/end measure range."""

    def test_transpose_specific_range(self):
        ss = _make_score(measures=4)
        _add_note(ss, "C4", 1)
        _add_note(ss, "C4", 2)
        _add_note(ss, "C4", 3)
        _add_note(ss, "C4", 4)

        ss.transpose("P5", start_measure=2, end_measure=3)

        part = list(ss.score.parts)[0]

        m1_notes = [
            n for n in part.measure(1).notes if isinstance(n, m21note.Note)
        ]
        assert any(n.pitch.midi == 60 for n in m1_notes)

        m2_notes = [
            n for n in part.measure(2).notes if isinstance(n, m21note.Note)
        ]
        assert any(n.pitch.midi == 67 for n in m2_notes)

        m3_notes = [
            n for n in part.measure(3).notes if isinstance(n, m21note.Note)
        ]
        assert any(n.pitch.midi == 67 for n in m3_notes)

        m4_notes = [
            n for n in part.measure(4).notes if isinstance(n, m21note.Note)
        ]
        assert any(n.pitch.midi == 60 for n in m4_notes)

    def test_transpose_single_measure(self):
        ss = _make_score(measures=4)
        _add_note(ss, "C4", 1)
        _add_note(ss, "C4", 2)

        ss.transpose("M2", start_measure=2, end_measure=2)

        part = list(ss.score.parts)[0]
        m1_notes = [
            n for n in part.measure(1).notes if isinstance(n, m21note.Note)
        ]
        assert any(n.nameWithOctave == "C4" for n in m1_notes)

        m2_notes = [
            n for n in part.measure(2).notes if isinstance(n, m21note.Note)
        ]
        assert any(n.nameWithOctave == "D4" for n in m2_notes)

    def test_transpose_refreshes_only_requested_range(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Range transposition should refresh accidentals only in that range."""
        ss = _make_score(measures=4)
        _add_note(ss, "C4", 1)
        _add_note(ss, "C4", 2)
        _add_note(ss, "C4", 3)
        _add_note(ss, "C4", 4)
        calls: list[int] = []
        original_refresh = type(ss)._refresh_measure_accidentals

        def record_refresh(
            score_state: ScoreSpeak,
            part_obj: object,
            measure_number: int,
        ) -> None:
            """Record refreshed measures while preserving real behavior."""
            calls.append(measure_number)
            original_refresh(score_state, part_obj, measure_number)

        monkeypatch.setattr(type(ss), "_refresh_measure_accidentals", record_refresh)

        ss.transpose("M2", start_measure=2, end_measure=3)

        assert calls == [2, 3]


class TestTransposeSpecificPart:
    """Tests for transpose on a specific part."""

    def test_transpose_single_part(self):
        ss = _make_score(measures=2, parts=["violin", "cello"])
        _add_note(ss, "C4", 1, part=0)
        _add_note(ss, "C3", 1, part=1)

        ss.transpose("P5", part=0)

        parts = list(ss.score.parts)

        violin_notes = [
            n for n in parts[0].measure(1).notes
            if isinstance(n, m21note.Note)
        ]
        assert any(n.pitch.midi == 67 for n in violin_notes)

        cello_notes = [
            n for n in parts[1].measure(1).notes
            if isinstance(n, m21note.Note)
        ]
        assert any(n.pitch.midi == 48 for n in cello_notes)

    def test_transpose_all_parts(self):
        ss = _make_score(measures=2, parts=["violin", "cello"])
        _add_note(ss, "C4", 1, part=0)
        _add_note(ss, "C3", 1, part=1)

        ss.transpose("P5")

        parts = list(ss.score.parts)
        violin_notes = [
            n for n in parts[0].measure(1).notes
            if isinstance(n, m21note.Note)
        ]
        assert any(n.pitch.midi == 67 for n in violin_notes)

        cello_notes = [
            n for n in parts[1].measure(1).notes
            if isinstance(n, m21note.Note)
        ]
        assert any(n.pitch.midi == 55 for n in cello_notes)


class TestTransposeChords:
    """Tests for transpose with chords."""

    def test_transpose_chord(self):
        ss = _make_score(measures=2)
        ss.add_chord(["C4", "E4", "G4"], "quarter", 1, beat=1)
        ss.transpose("M2")

        part = list(ss.score.parts)[0]
        m = part.measure(1)
        chords = list(m.getElementsByClass(m21chord.Chord))
        assert len(chords) >= 1
        midi_values = sorted([p.midi for p in chords[0].pitches])
        assert midi_values == [62, 66, 69]


class TestTransposeErrors:
    """Tests for transpose error handling."""

    def test_invalid_interval_string(self):
        ss = _make_score(measures=2)
        with pytest.raises(ValueError, match="[Cc]annot parse interval"):
            ss.transpose("XYZ")

    def test_empty_interval_string(self):
        ss = _make_score(measures=2)
        with pytest.raises(ValueError, match="empty"):
            ss.transpose("")

    def test_invalid_measure_range_start(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="out of range"):
            ss.transpose("P5", start_measure=10)

    def test_invalid_measure_range_end(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="out of range"):
            ss.transpose("P5", start_measure=1, end_measure=10)

    def test_end_before_start(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="out of range"):
            ss.transpose("P5", start_measure=3, end_measure=1)


# ===================================================================
# Concert / Written Pitch Tests
# ===================================================================


class TestTransposeToConcertPitch:
    """Tests for transpose_to_concert_pitch."""

    def test_no_transposing_instruments(self):
        ss = _make_score(measures=2, parts=["piano"])
        _add_note(ss, "C4", 1)
        result = ss.transpose_to_concert_pitch()
        assert result.success
        assert "already at concert pitch" in result.description.lower()

    def test_transposing_instrument(self):
        score = m21stream.Score()
        part = m21stream.Part(id="P1")
        part.partName = "Bb Clarinet"
        clarinet = m21instrument.Clarinet()
        part.insert(0, clarinet)

        m = m21stream.Measure(number=1)
        m.append(m21note.Note("C4", quarterLength=4.0))
        part.append(m)
        score.insert(0, part)

        ss = ScoreSpeak(score)
        result = ss.transpose_to_concert_pitch()
        assert result.success
        assert len(result.details["transposed_parts"]) == 1

        notes = list(part.measure(1).getElementsByClass(m21note.Note))
        transposition = clarinet.transposition
        expected_midi = 60 + transposition.semitones
        assert notes[0].pitch.midi == expected_midi

    def test_converts_written_key_and_marks_part_sounding(self) -> None:
        ss = _make_score(measures=1, parts=["clarinet"], key_signature="F")
        part = list(ss.score.parts)[0]
        assert part.atSoundingPitch is False
        assert ss.get_active_key_signature(1, part=0) == "G major"

        result = ss.transpose_to_concert_pitch()

        assert result.success
        assert part.atSoundingPitch is True
        assert ss.get_active_key_signature(1, part=0) == "F major"
        assert result.details["transposed_parts"][0]["key_signatures_changed"] == 1

    def test_transpose_to_concert_pitch_is_idempotent(self) -> None:
        ss = _make_score(measures=1, parts=["clarinet"], key_signature="F")
        ss.transpose_to_concert_pitch()

        result = ss.transpose_to_concert_pitch()

        assert result.success
        assert result.details["transposed_parts"] == []
        assert "already at concert pitch" in result.description.lower()

    def test_import_preserves_written_key_and_transpose_export(self) -> None:
        """Imported transposing MusicXML keeps part key and transpose metadata."""
        musicxml = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name>Bb Clarinet</part-name>
      <score-instrument id="P1-I1">
        <instrument-name>Clarinet</instrument-name>
      </score-instrument>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>2</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
        <transpose><diatonic>-1</diatonic><chromatic>-2</chromatic></transpose>
      </attributes>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>4</duration><type>whole</type>
      </note>
    </measure>
  </part>
</score-partwise>
"""
        ss = ScoreSpeak.from_musicxml(musicxml)
        part = list(ss.score.parts)[0]

        assert part.getInstrument().transposition is not None
        assert part.atSoundingPitch is False
        assert ss.get_active_key_signature(1) != ss.get_active_key_signature(
            1,
            part=0,
        )
        assert "<transpose>" in ss.to_musicxml_string(make_notation=False)

    def test_import_prefers_musicxml_transpose_over_eflat_horn_label(self) -> None:
        """Imported E-flat horns use explicit MusicXML transpose metadata."""
        musicxml = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name>Eb Horn 1</part-name>
      <score-instrument id="P1-I1">
        <instrument-name>Horn in E♭</instrument-name>
      </score-instrument>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
        <transpose><diatonic>-5</diatonic><chromatic>-9</chromatic></transpose>
      </attributes>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>4</duration><type>whole</type>
      </note>
    </measure>
  </part>
</score-partwise>
"""
        ss = ScoreSpeak.from_musicxml(musicxml)
        part = list(ss.score.parts)[0]
        interval = part_transposition_interval(part)

        assert interval is not None
        assert interval.semitones == -9
        assert part.atSoundingPitch is False
        assert ss.get_active_key_signature(1) == "3 flats"

        ss.transpose_to_concert_pitch()
        note = next(part.recurse().notes)

        assert note.pitch.midi == 51
        assert ss.get_active_key_signature(1, part=0) == "3 flats"
        assert "<transpose>" not in ss.to_musicxml_string(make_notation=False)

    def test_concert_pitch_conversion_survives_export_reimport(self) -> None:
        """Converted sounding-pitch parts do not revert after MusicXML export."""
        ss = _make_score(measures=1, parts=["clarinet"], key_signature="F")
        ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
        ss.transpose_to_concert_pitch()

        round_tripped = ScoreSpeak.from_musicxml(
            ss.to_musicxml_string(make_notation=False)
        )
        part = list(round_tripped.score.parts)[0]
        result = round_tripped.transpose_to_concert_pitch()

        assert part.atSoundingPitch is True
        assert round_tripped.get_active_key_signature(1, part=0) == "F major"
        assert result.details["transposed_parts"] == []


class TestTransposeToWrittenPitch:
    """Tests for transpose_to_written_pitch."""

    def test_no_transposing_instruments(self):
        ss = _make_score(measures=2, parts=["piano"])
        result = ss.transpose_to_written_pitch()
        assert result.success
        assert "already at written pitch" in result.description.lower()

    def test_round_trip_concert_then_written(self):
        score = m21stream.Score()
        part = m21stream.Part(id="P1")
        part.partName = "Bb Clarinet"
        clarinet = m21instrument.Clarinet()
        part.insert(0, clarinet)

        m = m21stream.Measure(number=1)
        m.append(m21note.Note("C4", quarterLength=4.0))
        part.append(m)
        score.insert(0, part)

        ss = ScoreSpeak(score)
        ss.transpose_to_concert_pitch()
        ss.transpose_to_written_pitch()

        notes = list(part.measure(1).getElementsByClass(m21note.Note))
        assert notes[0].pitch.midi == 60

    def test_converts_concert_key_and_marks_part_written(self) -> None:
        ss = _make_score(measures=1, parts=["clarinet"], key_signature="F")
        part = list(ss.score.parts)[0]
        ss.transpose_to_concert_pitch()

        result = ss.transpose_to_written_pitch()

        assert result.success
        assert part.atSoundingPitch is False
        assert ss.get_active_key_signature(1, part=0) == "G major"
        assert result.details["transposed_parts"][0]["key_signatures_changed"] == 1

    def test_transpose_to_written_pitch_is_idempotent(self) -> None:
        ss = _make_score(measures=1, parts=["clarinet"], key_signature="F")

        result = ss.transpose_to_written_pitch()

        assert result.success
        assert result.details["transposed_parts"] == []
        assert "already at written pitch" in result.description.lower()


# ===================================================================
# Round-trip Test (export → re-import → verify)
# ===================================================================


class TestRoundTrip:
    """End-to-end round-trip: create, edit, export, re-import, verify."""

    def test_metadata_round_trip(self):
        ss = _make_score(title="Original", composer="Bach", measures=4)
        ss.set_subtitle("BWV 999")
        ss.add_system_break(2)
        ss.add_page_break(3)
        _add_note(ss, "C4", 1)
        _add_note(ss, "E4", 2)

        with tempfile.NamedTemporaryFile(
            suffix=".musicxml", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            ss.to_musicxml(tmp_path)
            ss2 = ScoreSpeak.from_musicxml(tmp_path)

            assert ss2.title == "Original"
            assert ss2.composer == "Bach"

            md = ss2.get_metadata()
            assert md["subtitle"] == "BWV 999"
        finally:
            os.unlink(tmp_path)

    def test_transpose_round_trip(self):
        ss = _make_score(measures=2)
        _add_note(ss, "C4", 1)
        _add_note(ss, "E4", 2)

        ss.transpose("P5")
        ss.transpose("-P5")

        part = list(ss.score.parts)[0]
        m1_notes = [
            n for n in part.measure(1).notes if isinstance(n, m21note.Note)
        ]
        assert any(n.pitch.midi == 60 for n in m1_notes)

        m2_notes = [
            n for n in part.measure(2).notes if isinstance(n, m21note.Note)
        ]
        assert any(n.pitch.midi == 64 for n in m2_notes)


# ===================================================================
# Edge Cases
# ===================================================================


class TestEdgeCases:
    """Additional edge-case coverage."""

    def test_system_and_page_break_same_measure(self):
        ss = _make_score(measures=4)
        ss.add_system_break(2)
        ss.add_page_break(2)

        part = list(ss.score.parts)[0]
        m = part.measure(2)
        sys_layouts = list(m.getElementsByClass(m21layout.SystemLayout))
        page_layouts = list(m.getElementsByClass(m21layout.PageLayout))
        assert any(sl.isNew for sl in sys_layouts)
        assert any(pl.isNew for pl in page_layouts)

    def test_remove_only_system_break_leaves_page_break(self):
        ss = _make_score(measures=4)
        ss.add_system_break(2)
        ss.add_page_break(2)
        ss.remove_system_break(2)

        part = list(ss.score.parts)[0]
        m = part.measure(2)
        sys_layouts = [
            sl for sl in m.getElementsByClass(m21layout.SystemLayout)
            if sl.isNew
        ]
        page_layouts = [
            pl for pl in m.getElementsByClass(m21layout.PageLayout)
            if pl.isNew
        ]
        assert len(sys_layouts) == 0
        assert len(page_layouts) == 1

    def test_transpose_empty_measures(self):
        ss = _make_score(measures=4)
        result = ss.transpose("P5")
        assert result.success
        assert result.details["notes_transposed"] == 0

    def test_metadata_persists_through_operations(self):
        ss = _make_score(title="My Score", composer="Me")
        ss.set_subtitle("Draft")
        ss.transpose("M2")
        ss.add_system_break(2)

        md = ss.get_metadata()
        assert md["title"] == "My Score"
        assert md["subtitle"] == "Draft"
        assert md["composer"] == "Me"
