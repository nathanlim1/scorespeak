"""
Integration tests for the ScoreSpeak framework.

Tests cross-tier interactions, round-trip MusicXML export/import,
and real-world file operations.
"""

import os
import tempfile

import pytest

from scorespeak import ScoreSpeak, OperationResult


TEST_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "MU114 Midterm Final modified.musicxml",
)


class TestRoundTrip:
    """Test export → re-import fidelity."""

    def test_round_trip_empty_score(self):
        ss = ScoreSpeak.create(measures=4, time_signature="4/4", key_signature="C")

        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
            path = f.name
        try:
            ss.to_musicxml(path)
            ss2 = ScoreSpeak.from_musicxml(path)
            assert ss2.part_count == 1
            assert ss2.measure_count >= 4
        finally:
            os.unlink(path)

    def test_round_trip_with_notes(self):
        ss = ScoreSpeak.create(
            title="Round Trip Test",
            composer="Test Composer",
            time_signature="3/4",
            key_signature="G",
            tempo=120.0,
            parts=["violin"],
            measures=4,
        )

        ss._add_note_one("G4", "quarter", measure=1, beat=1)
        ss._add_note_one("A4", "quarter", measure=1, beat=2)
        ss._add_note_one("B4", "quarter", measure=1, beat=3)
        ss._add_note_one("C5", "half", measure=2, beat=1)
        ss._add_note_one("D5", "quarter", measure=2, beat=3)

        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
            path = f.name
        try:
            ss.to_musicxml(path)
            ss2 = ScoreSpeak.from_musicxml(path)
            assert ss2.part_count == 1
            assert ss2.measure_count >= 4
            notes = ss2.get_notes(measure=1)
            pitched = [n for n in notes if not n.is_rest]
            assert len(pitched) >= 3
        finally:
            os.unlink(path)

    def test_round_trip_with_dynamics_and_articulations(self):
        ss = ScoreSpeak.create(measures=4, parts=["piano"])
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("E4", "quarter", measure=1, beat=2)
        ss._add_note_one("G4", "quarter", measure=1, beat=3)
        ss._add_note_one("C5", "quarter", measure=1, beat=4)
        ss.add_dynamic("mf", measure_number=1, beat=1)
        ss.add_articulation("staccato", measure_number=1, beat=1)

        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
            path = f.name
        try:
            ss.to_musicxml(path)
            ss2 = ScoreSpeak.from_musicxml(path)
            assert ss2.part_count == 1
            notes = ss2.get_notes(measure=1)
            assert len([n for n in notes if not n.is_rest]) >= 4
        finally:
            os.unlink(path)

    def test_round_trip_preserves_key_signature(self):
        ss = ScoreSpeak.create(key_signature="Bb", measures=4)
        ss._add_note_one("Bb4", "whole", measure=1, beat=1)

        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
            path = f.name
        try:
            ss.to_musicxml(path)
            ss2 = ScoreSpeak.from_musicxml(path)
            ks = ss2.get_active_key_signature(1)
            assert "B" in ks
        finally:
            os.unlink(path)


class TestRealWorldFile:
    """Tests using the real MusicXML test file."""

    @pytest.fixture
    def real_score(self):
        if not os.path.exists(TEST_FILE):
            pytest.skip("Real-world test file not found")
        return ScoreSpeak.from_musicxml(TEST_FILE)

    def test_part_count(self, real_score):
        # Piano grand staff may be split into 2 parts by music21
        assert real_score.part_count >= 2

    def test_measure_count(self, real_score):
        assert real_score.measure_count == 112

    def test_part_names(self, real_score):
        parts = real_score.list_parts()
        names = [p.name for p in parts]
        assert any("Soprano" in n or "S." in n for n in names)

    def test_piano_grand_staff_gets_rh_lh_display_names(self, real_score):
        """MU114's piano parts surface as ``Piano RH`` / ``Piano LH``."""
        parts = real_score.list_parts()
        piano_staves = [p for p in parts if "Piano" in p.name]
        assert len(piano_staves) == 2

        display_names = {p.display_name for p in piano_staves}
        hands = {p.hand for p in piano_staves}
        assert display_names == {"Piano RH", "Piano LH"}
        assert hands == {"RH", "LH"}

    def test_time_signature(self, real_score):
        ts = real_score.get_active_time_signature(1)
        assert ts == "3/4"

    def test_key_signature(self, real_score):
        ks = real_score.get_active_key_signature(1)
        assert "flat" in ks.lower() or "b" in ks.lower() or "E" in ks

    def test_query_notes(self, real_score):
        notes = real_score.get_notes(measure=50, part=0)
        assert isinstance(notes, list)

    def test_query_notes_second_part(self, real_score):
        notes = real_score.get_notes(measure=10, part=1)
        assert isinstance(notes, list)

    def test_measure_info(self, real_score):
        info = real_score.get_measure_info(1)
        assert info.time_signature == "3/4"
        assert info.beat_count == 3.0

    def test_export_reimport(self, real_score):
        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
            path = f.name
        try:
            real_score.to_musicxml(path)
            ss2 = ScoreSpeak.from_musicxml(path)
            assert ss2.part_count >= 2
            assert ss2.measure_count >= 100
        finally:
            os.unlink(path)

    def test_bar_result_set_surfaces_rich_notation(self, real_score):
        """bar result projection should surface dynamics, spans, clefs, and bar notation."""
        result = real_score._build_bar_result_set({"scope": {"bar_range": (1, 3)}})

        assert "marking_schema" in result
        assert "span_schema" in result
        assert "bar_notation_keys" in result
        assert "part_notation_keys" in result

        bars = result["bars"]
        assert len(bars) == 3

        bar1 = bars[0]
        assert "active" in bar1["notation"]
        assert bar1["notation"]["active"]["time"] == "3/4"

        has_clef_on_bar1 = any(
            part.get("notation", {}).get("clef")
            for part in bar1["parts"]
        )
        assert has_clef_on_bar1, "clef should appear on scope bar 1"

        has_clef_on_bar2 = any(
            part.get("notation", {}).get("clef")
            for part in bars[1]["parts"]
        )
        assert not has_clef_on_bar2, "clef should be omitted on unchanged bars"

        has_markings = False
        has_spans = False
        for bar in bars:
            for part in bar["parts"]:
                for voice in part["voices"]:
                    if voice.get("markings"):
                        has_markings = True
                    if voice.get("spans"):
                        has_spans = True
        assert has_markings, "expected at least one marking in bars 1-3"
        assert has_spans, "expected at least one span in bars 1-3"


class TestEndToEndWorkflow:
    """End-to-end tests simulating realistic editing sessions."""

    def test_compose_simple_melody(self):
        """Create a simple 8-bar melody from scratch."""
        ss = ScoreSpeak.create(
            title="Simple Melody",
            composer="AI Composer",
            time_signature="4/4",
            key_signature="C",
            tempo=120.0,
            parts=["flute"],
            measures=8,
        )

        melody = [
            (1, "C5", "quarter", 1), (1, "D5", "quarter", 2),
            (1, "E5", "quarter", 3), (1, "F5", "quarter", 4),
            (2, "G5", "half", 1), (2, "E5", "half", 3),
            (3, "F5", "quarter", 1), (3, "E5", "quarter", 2),
            (3, "D5", "quarter", 3), (3, "C5", "quarter", 4),
            (4, "D5", "whole", 1), (5, "D5", "whole", 1),
        ]
        for m, pitch, dur, beat in melody:
            result = ss._add_note_one(pitch, dur, measure=m, beat=beat)
            assert result.success

        ss.add_dynamic("mf", measure_number=1, beat=1)
        ss.add_hairpin("crescendo", 1, 1.0, 2, 3.0)
        ss.add_dynamic("f", measure_number=2, beat=3)
        ss.add_hairpin("diminuendo", 3, 1.0, 4, 1.0)
        ss.add_dynamic("p", measure_number=4, beat=1)

        ss.add_articulation("staccato", 1, beat=3)
        ss.add_slur(1, 1.0, 1, 2.0)
        ss.add_tie(measure=4, beat=1)
        ss.add_text_expression("dolce", 1, beat=1)
        ss.set_barline("double", measure_number=4)
        ss.set_barline("final", measure_number=8)

        assert ss.measure_count == 8
        notes = ss.get_notes()
        pitched = [n for n in notes if not n.is_rest]
        assert len(pitched) >= 10

        xml_str = ss.to_musicxml_string()
        assert "Simple Melody" in xml_str
        assert "score-partwise" in xml_str

    def test_multi_part_arrangement(self):
        """Create a multi-part arrangement with different instruments."""
        ss = ScoreSpeak.create(
            title="Chamber Piece",
            parts=["violin", "viola", "cello"],
            time_signature="3/4",
            key_signature="D",
            tempo=100.0,
            measures=4,
        )

        ss._add_note_one("F#5", "half", measure=1, beat=1, part=0)
        ss._add_note_one("D5", "quarter", measure=1, beat=3, part=0)

        ss._add_note_one("A4", "half", measure=1, beat=1, part=1)
        ss._add_note_one("F#4", "quarter", measure=1, beat=3, part=1)

        ss._add_note_one("D3", "half", measure=1, beat=1, part=2)
        ss._add_note_one("A2", "quarter", measure=1, beat=3, part=2)

        ss.add_dynamic("p", measure_number=1, beat=1, part=0)
        ss.add_dynamic("p", measure_number=1, beat=1, part=1)
        ss.add_dynamic("p", measure_number=1, beat=1, part=2)

        assert ss.part_count == 3
        for i in range(3):
            notes = ss.get_notes(measure=1, part=i)
            pitched = [n for n in notes if not n.is_rest]
            assert len(pitched) == 2

    def test_key_change_midpiece(self):
        """Change key signature mid-piece and verify continuity."""
        ss = ScoreSpeak.create(
            key_signature="C",
            time_signature="4/4",
            measures=8,
        )

        ss._add_note_one("C4", "whole", measure=1, beat=1)
        ss._add_note_one("D4", "whole", measure=2, beat=1)
        ss._add_note_one("E4", "whole", measure=3, beat=1)
        ss._add_note_one("F4", "whole", measure=4, beat=1)

        ss.set_key_signature("G", measure_number=5)

        assert "C" in ss.get_active_key_signature(1)
        assert "C" in ss.get_active_key_signature(4)
        assert "G" in ss.get_active_key_signature(5)
        assert "G" in ss.get_active_key_signature(8)

    def test_time_signature_change(self):
        """Change time signature and verify measures adapt."""
        ss = ScoreSpeak.create(
            time_signature="4/4",
            measures=4,
        )

        ss.set_time_signature("6/8", measure_number=3)

        assert ss.get_active_time_signature(1) == "4/4"
        assert ss.get_active_time_signature(2) == "4/4"
        assert ss.get_active_time_signature(3) == "6/8"
        assert ss.get_active_time_signature(4) == "6/8"

    def test_chord_progression(self):
        """Build a chord progression and verify structure."""
        ss = ScoreSpeak.create(
            time_signature="4/4",
            key_signature="C",
            measures=4,
        )

        chords = [
            (1, ["C4", "E4", "G4"]),
            (2, ["F4", "A4", "C5"]),
            (3, ["G4", "B4", "D5"]),
            (4, ["C4", "E4", "G4"]),
        ]
        for m, pitches in chords:
            result = ss.add_chord(pitches, "whole", measure=m, beat=1)
            assert result.success

        for m in range(1, 5):
            notes = ss.get_notes(measure=m)
            chord_notes = [n for n in notes if n.is_chord]
            assert len(chord_notes) == 3

    def test_tuplets_in_context(self):
        """Add tuplets and verify they fit within the measure."""
        ss = ScoreSpeak.create(time_signature="4/4", measures=2)

        result = ss.add_tuplet(
            [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")],
            actual_notes=3,
            normal_notes=2,
            measure=1,
            beat=1,
        )
        assert result.success

        notes = ss.get_notes(measure=1)
        assert len(notes) >= 3

    def test_grace_notes_before_melody(self):
        """Add grace notes before melody notes."""
        ss = ScoreSpeak.create(time_signature="4/4", measures=2)
        ss._add_note_one("C5", "quarter", measure=1, beat=1)
        ss.add_grace_note("B4", measure=1, beat=1)

        notes = ss.get_notes(measure=1)
        grace_notes = [n for n in notes if n.is_grace]
        assert len(grace_notes) == 1
        assert grace_notes[0].pitch == "B4"

    def test_layout_operations(self):
        """Test metadata and layout operations together."""
        ss = ScoreSpeak.create(
            title="Old Title",
            measures=8,
        )

        ss.set_title("New Title")
        ss.set_subtitle("Op. 1")
        ss.set_composer("J. Doe")

        md = ss.get_metadata()
        assert md["title"] == "New Title"
        assert md["subtitle"] == "Op. 1"
        assert md["composer"] == "J. Doe"

        ss.add_system_break(4)
        ss.add_page_break(4)
        ss.add_rehearsal_mark("A", 1)
        ss.add_rehearsal_mark("B", 5)

    def test_transpose_section(self):
        """Transpose a section of the score."""
        ss = ScoreSpeak.create(
            time_signature="4/4",
            key_signature="C",
            measures=4,
        )

        ss._add_note_one("C4", "whole", measure=1, beat=1)
        ss._add_note_one("D4", "whole", measure=2, beat=1)
        ss._add_note_one("E4", "whole", measure=3, beat=1)
        ss._add_note_one("F4", "whole", measure=4, beat=1)

        result = ss.transpose("M2", start_measure=3, end_measure=4)
        assert result.success

        notes_m3 = ss.get_notes(measure=3)
        pitched_m3 = [n for n in notes_m3 if not n.is_rest]
        assert len(pitched_m3) >= 1
        assert pitched_m3[0].pitch in ("F#4", "F♯4", "G-4")

    def test_multiple_voices(self):
        """Add notes in multiple voices."""
        ss = ScoreSpeak.create(time_signature="4/4", measures=2)

        ss._add_note_one("C5", "whole", measure=1, beat=1, voice=1)
        ss._add_note_one("E4", "whole", measure=1, beat=1, voice=2)

        all_notes = ss.get_notes(measure=1)
        pitched = [n for n in all_notes if not n.is_rest]
        assert len(pitched) == 2

        v1_notes = ss.get_notes(measure=1, voice=1)
        v2_notes = ss.get_notes(measure=1, voice=2)
        assert len([n for n in v1_notes if not n.is_rest]) == 1
        assert len([n for n in v2_notes if not n.is_rest]) == 1


class TestImpliedContinuityAcrossOperations:
    """Verify implied continuity persists across various operations."""

    def test_continuity_after_insert(self):
        ss = ScoreSpeak.create(
            time_signature="6/8",
            key_signature="Bb",
            measures=4,
        )
        ss.insert_measure(before=3)

        for m in range(1, 6):
            assert ss.get_active_time_signature(m) == "6/8"
            assert "B" in ss.get_active_key_signature(m)

    def test_continuity_after_delete(self):
        ss = ScoreSpeak.create(
            time_signature="3/4",
            key_signature="D",
            tempo=144.0,
            measures=6,
        )
        ss.delete_measure(3)

        for m in range(1, 6):
            assert ss.get_active_time_signature(m) == "3/4"
            assert "D" in ss.get_active_key_signature(m)

    def test_continuity_after_add_part(self):
        ss = ScoreSpeak.create(
            time_signature="4/4",
            key_signature="G",
            parts=["piano"],
            measures=4,
        )
        ss.add_part(instrument="violin")

        assert ss.part_count == 2
        for m in range(1, 5):
            assert ss.get_active_time_signature(m, part=0) == "4/4"

    def test_continuity_with_time_sig_change_and_insert(self):
        ss = ScoreSpeak.create(
            time_signature="4/4",
            measures=6,
        )
        ss.set_time_signature("3/4", measure_number=4)

        assert ss.get_active_time_signature(3) == "4/4"
        assert ss.get_active_time_signature(4) == "3/4"
        assert ss.get_active_time_signature(6) == "3/4"

    def test_added_measures_inherit_latest_state(self):
        ss = ScoreSpeak.create(
            time_signature="4/4",
            key_signature="C",
            measures=4,
        )
        ss.set_time_signature("7/8", measure_number=3)
        ss.add_measures(2)

        assert ss.get_active_time_signature(5) == "7/8"
        assert ss.get_active_time_signature(6) == "7/8"


class TestErrorMessages:
    """Verify that error messages are musical and descriptive."""

    def test_beat_overflow_message(self):
        ss = ScoreSpeak.create(time_signature="4/4", measures=2)
        ss._add_note_one("C4", "half", measure=1, beat=1)
        ss._add_note_one("D4", "half", measure=1, beat=3)
        with pytest.raises(ValueError, match="beats"):
            ss._add_note_one("E4", "quarter", measure=1, beat=4.5)

    def test_measure_not_found_message(self):
        ss = ScoreSpeak.create(measures=4)
        with pytest.raises(ValueError, match="does not exist"):
            ss.get_measure_info(10)

    def test_part_not_found_message(self):
        ss = ScoreSpeak.create(parts=["piano"])
        with pytest.raises(ValueError, match="No part named"):
            ss._add_note_one("C4", part="violin")

    def test_invalid_pitch_message(self):
        from scorespeak.music.validation import normalize_pitch
        with pytest.raises(ValueError, match="Cannot parse"):
            normalize_pitch("XYZ")

    def test_invalid_duration_message(self):
        from scorespeak.music.validation import normalize_duration
        with pytest.raises(ValueError, match="Cannot parse"):
            normalize_duration("bogus")

    def test_invalid_dynamic_message(self):
        ss = ScoreSpeak.create(measures=2)
        with pytest.raises(ValueError, match="not a valid dynamic"):
            ss.add_dynamic("xxx", measure_number=1)

    def test_remove_last_part_message(self):
        ss = ScoreSpeak.create(parts=["piano"])
        with pytest.raises(ValueError, match="Cannot remove the last part"):
            ss.remove_part(0)
