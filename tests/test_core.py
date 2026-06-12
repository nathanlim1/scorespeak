"""Tests for ScoreSpeak core functionality — creation, import, export."""

import os
import tempfile
from pathlib import Path

import pytest

from scorespeak import ScoreSpeak


class TestCreate:
    """Tests for ScoreSpeak.create()."""

    def test_create_default(self):
        ss = ScoreSpeak.create()
        assert ss.title == "Untitled"
        assert ss.part_count == 1
        assert ss.measure_count == 0

    def test_create_with_title_and_composer(self):
        ss = ScoreSpeak.create(title="My Song", composer="J. S. Bach")
        assert ss.title == "My Song"
        assert ss.composer == "J. S. Bach"

    def test_create_with_measures(self):
        ss = ScoreSpeak.create(
            time_signature="3/4",
            key_signature="G",
            tempo=100.0,
            measures=4,
        )
        assert ss.measure_count == 4
        assert ss.get_active_time_signature(1) == "3/4"
        assert "G" in ss.get_active_key_signature(1)
        assert ss.get_active_tempo(1) == 100.0

    def test_create_multiple_parts(self):
        ss = ScoreSpeak.create(
            parts=["violin", "viola", "cello"],
            measures=2,
        )
        assert ss.part_count == 3
        parts = ss.list_parts()
        assert any("Violin" in p.name for p in parts)
        assert any("Viola" in p.name for p in parts)

    def test_create_with_dict_parts(self):
        ss = ScoreSpeak.create(
            parts=[
                {"name": "Melody", "instrument": "flute"},
                {"name": "Bass", "instrument": "cello", "clef": "bass"},
            ],
            measures=2,
        )
        assert ss.part_count == 2
        parts = ss.list_parts()
        assert parts[0].name == "Melody"
        assert parts[1].name == "Bass"

    def test_create_with_flat_key(self):
        ss = ScoreSpeak.create(key_signature="Bb", measures=1)
        ks = ss.get_active_key_signature(1)
        assert "B" in ks

    def test_create_with_minor_key(self):
        ss = ScoreSpeak.create(key_signature="A minor", measures=1)
        ks = ss.get_active_key_signature(1)
        assert "minor" in ks.lower()


class TestImportExport:
    """Tests for MusicXML import and export."""

    def test_export_and_reimport(self):
        ss = ScoreSpeak.create(
            title="Round Trip",
            parts=["piano"],
            time_signature="4/4",
            key_signature="C",
            measures=4,
        )

        with tempfile.NamedTemporaryFile(
            suffix=".musicxml", delete=False
        ) as f:
            tmp_path = f.name

        try:
            result = ss.to_musicxml(tmp_path)
            assert result.success
            assert os.path.exists(tmp_path)

            ss2 = ScoreSpeak.from_musicxml(Path(tmp_path))
            assert ss2.part_count == 1
            assert ss2.measure_count >= 4
        finally:
            os.unlink(tmp_path)

    def test_to_musicxml_string(self):
        ss = ScoreSpeak.create(measures=2)
        xml_str = ss.to_musicxml_string()
        assert "<?xml" in xml_str
        assert "score-partwise" in xml_str

    def test_large_ensemble_export_suppresses_midi_channel_warning(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Large ensemble exports should not leak music21 MIDI-channel warnings."""
        ss = ScoreSpeak.create(
            parts=[
                "flute",
                "oboe",
                "clarinet",
                "bassoon",
                "horn",
                "trumpet",
                "trombone",
                "tuba",
                "violin",
                "viola",
                "cello",
                "contrabass",
                "soprano",
                "alto",
                "tenor",
                "baritone",
                "guitar",
                "harp",
            ],
            measures=1,
        )

        result = ss.to_musicxml(tmp_path / "large_ensemble.musicxml")
        captured = capsys.readouterr()

        assert result.success
        assert "we are out of midi channels" not in captured.err
        assert "we are out of midi channels" not in captured.out

    def test_import_real_file(self):
        """Test importing the real-world MusicXML test file."""
        test_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "MU114 Midterm Final modified.musicxml",
        )
        if not os.path.exists(test_file):
            pytest.skip("Real-world test file not found")

        ss = ScoreSpeak.from_musicxml(test_file)
        assert ss.part_count >= 1
        assert ss.measure_count > 0
        parts = ss.list_parts()
        assert len(parts) >= 1

    def test_import_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            ScoreSpeak.from_musicxml("/nonexistent/path.musicxml")


class TestImpliedContinuity:
    """Tests for implied continuity of musical state."""

    def test_time_signature_inherits(self):
        ss = ScoreSpeak.create(time_signature="6/8", measures=4)
        for m in range(1, 5):
            assert ss.get_active_time_signature(m) == "6/8"

    def test_key_signature_inherits(self):
        ss = ScoreSpeak.create(key_signature="D", measures=4)
        for m in range(1, 5):
            ks = ss.get_active_key_signature(m)
            assert "D" in ks

    def test_tempo_inherits(self):
        ss = ScoreSpeak.create(tempo=144.0, measures=4)
        for m in range(1, 5):
            assert ss.get_active_tempo(m) == 144.0

    def test_tempo_inherits_without_music21_context_lookup(self) -> None:
        """Tempo lookup handles empty-start scores and explicit later tempos."""
        ss = ScoreSpeak.create(parts=[], measures=0)
        ss.add_part(name="Violin", instrument="violin")
        ss.add_measures(8)

        assert ss.get_active_tempo(8) is None

        ss.set_tempo(96, measure_number=3, part=0)

        assert ss.get_active_tempo(1) is None
        assert ss.get_active_tempo(3) == 96.0
        assert ss.get_active_tempo(8) == 96.0


class TestRepr:
    """Tests for string representation."""

    def test_repr(self):
        ss = ScoreSpeak.create(title="Test", parts=["piano"], measures=4)
        r = repr(ss)
        assert "Test" in r
        assert "4" in r
