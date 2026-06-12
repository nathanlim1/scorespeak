"""Tests for extended markings (lyrics, ornaments, spanners, voltas, arpeggio)."""

import inspect
import os
import tempfile

from lxml import etree
from music21 import spanner as m21spanner
import pytest

from scorespeak import ScoreSpeak


def _musicxml_endings(musicxml: str) -> list:
    """Return all MusicXML ending elements from an exported score."""
    root = etree.fromstring(musicxml.encode("utf-8"))
    return root.xpath(".//*[local-name()='ending']")


def _musicxml_note_pitches(musicxml: str) -> list[str]:
    """Return MusicXML note pitches as compact step/octave strings."""
    root = etree.fromstring(musicxml.encode("utf-8"))
    pitches = []
    for pitch in root.xpath(".//*[local-name()='note']/*[local-name()='pitch']"):
        step = pitch.xpath("./*[local-name()='step']/text()")
        octave = pitch.xpath("./*[local-name()='octave']/text()")
        if step and octave:
            pitches.append(f"{step[0]}{octave[0]}")
    return pitches


def _musicxml_tremolo_values(musicxml: str) -> list[str]:
    """Return tremolo mark counts from exported MusicXML."""
    root = etree.fromstring(musicxml.encode("utf-8"))
    return [
        str(value)
        for value in root.xpath(".//*[local-name()='tremolo']/text()")
    ]


@pytest.fixture
def simple_score():
    ss = ScoreSpeak.create(measures=4, time_signature="4/4", key_signature="C")
    ss._add_note_one("C4", measure=1)
    ss._add_note_one("D4", measure=1, beat=2)
    ss._add_note_one("E4", measure=2)
    ss._add_note_one("F4", measure=2, beat=2)
    ss.add_chord(["C4", "E4", "G4"], measure=3, beat=1)
    ss._add_note_one("B4", measure=4)
    return ss


class TestLyrics:
    def test_add_and_get_lyric(self, simple_score):
        ss = simple_score
        r = ss.add_lyric("Do", measure_number=1, beat=1.0, lyric_number=1)
        assert r.success
        lyrics = ss.get_lyrics()
        assert len(lyrics) == 1
        assert lyrics[0].text == "Do"
        assert lyrics[0].measure_number == 1

    def test_second_verse(self, simple_score):
        ss = simple_score
        ss.add_lyric("Do", 1, 1.0, lyric_number=1)
        ss.add_lyric("Un", 1, 1.0, lyric_number=2)
        lyrics = ss.get_lyrics(measure_number=1, voice=1)
        assert len(lyrics) == 2

    def test_add_lyric_rejects_unsupported_voice(self, simple_score) -> None:
        """Lyric tools reject voice numbers outside the public range."""
        with pytest.raises(ValueError, match="between 1 and 4"):
            simple_score.add_lyric("Do", 1, 1.0, voice=5)

    def test_get_lyrics_rejects_unsupported_voice(self, simple_score) -> None:
        """Lyric queries reject voice numbers outside the public range."""
        with pytest.raises(ValueError, match="between 1 and 4"):
            simple_score.get_lyrics(voice=5)

    def test_remove_lyric(self, simple_score):
        ss = simple_score
        ss.add_lyric("x", 1, 1.0)
        ss.remove_lyric(1, 1.0)
        assert ss.get_lyrics() == []

    def test_remove_missing_lyric_raises(self, simple_score):
        ss = simple_score
        ss.add_lyric("a", 1, 1.0)
        with pytest.raises(ValueError, match="No lyric number 2"):
            ss.remove_lyric(1, 1.0, lyric_number=2)

    def test_lyric_on_rest_raises(self):
        ss = ScoreSpeak.create(measures=1)
        with pytest.raises(ValueError, match="No note or chord"):
            ss.add_lyric("la", 1, 1.0)


class TestOrnaments:
    def test_trill(self, simple_score):
        ss = simple_score
        ss.add_ornament("trill", 1, 1.0)
        ss.remove_ornament("trill", 1, 1.0)

    def test_tremolo_slashes(self, simple_score: ScoreSpeak) -> None:
        """Tremolo ornaments can be added and removed."""
        ss = simple_score
        ss.add_ornament("tremolo", 1, 2.0, tremolo_marks=3)
        ss.remove_ornament("tremolo", 1, 2.0)

    @pytest.mark.parametrize("marks", [1, 2, 3, 4])
    def test_tremolo_marks_export_to_musicxml(
        self,
        simple_score: ScoreSpeak,
        marks: int,
    ) -> None:
        """Tremolo slash count should survive MusicXML export."""
        simple_score.add_ornament(
            "tremolo",
            1,
            2.0,
            tremolo_marks=marks,
        )

        assert _musicxml_tremolo_values(simple_score.to_musicxml_string()) == [
            str(marks)
        ]

    def test_unknown_ornament(self, simple_score):
        with pytest.raises(ValueError, match="Unknown ornament"):
            simple_score.add_ornament("not_real", 1, 1.0)


class TestOttavaGlissandoPedal:
    def test_ottava(self, simple_score):
        ss = simple_score
        ss.add_ottava("8va", 1, 1.0, 2, 2.0)
        ss.remove_ottava(1, 1.0)

    def test_ottava_spans_notes_between_inclusive_beats(
        self,
        simple_score: ScoreSpeak,
    ) -> None:
        """Ottava spanners include every note from start beat through end beat."""
        ss = simple_score
        ss.add_ottava("8va", 1, 1.0, 2, 2.0)

        ottavas = list(ss.score.parts[0].getElementsByClass(m21spanner.Ottava))
        spanned_pitches = [
            element.pitch.nameWithOctave
            for element in ottavas[0].getSpannedElements()
        ]

        assert spanned_pitches == ["C4", "D4", "E4", "F4"]

    @pytest.mark.parametrize(
        ("ottava_type", "internal_pitch", "exported_pitch"),
        [
            ("8va", "C4", ["C5"]),
            ("8vb", "C4", ["C3"]),
            ("15ma", "C4", ["C6"]),
            ("15mb", "C4", ["C2"]),
        ],
    )
    def test_ottava_default_keeps_internal_pitch_and_exports_shifted_musicxml(
        self,
        ottava_type: str,
        internal_pitch: str,
        exported_pitch: list[str],
    ) -> None:
        """Default ottavas keep stored pitches and export octave-shifted XML."""
        ss = ScoreSpeak.create(measures=1)
        ss._add_note_one(internal_pitch, "quarter", measure=1, beat=1)
        result = ss.add_ottava(ottava_type, 1, 1.0, 1, 1.0)

        note = ss.get_notes(measure=1)[0]
        ottavas = list(ss.score.parts[0].getElementsByClass(m21spanner.Ottava))

        assert note.pitch == internal_pitch
        assert result.details["rewrite_pitches"] is False
        assert result.details["notes_rewritten"] == 0
        assert ottavas[0].transposing is True
        assert _musicxml_note_pitches(ss.to_musicxml_string()) == exported_pitch

    @pytest.mark.parametrize(
        ("ottava_type", "rewritten_pitch"),
        [
            ("8va", "C3"),
            ("8vb", "C5"),
            ("15ma", "C2"),
            ("15mb", "C6"),
        ],
    )
    def test_ottava_rewrite_pitches_compensates_internal_pitch(
        self,
        ottava_type: str,
        rewritten_pitch: str,
    ) -> None:
        """Pitch rewriting stores the compensated pitch but exports equivalently."""
        ss = ScoreSpeak.create(measures=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        result = ss.add_ottava(
            ottava_type,
            1,
            1.0,
            1,
            1.0,
            rewrite_pitches=True,
        )

        note = ss.get_notes(measure=1)[0]
        ottavas = list(ss.score.parts[0].getElementsByClass(m21spanner.Ottava))

        assert note.pitch == rewritten_pitch
        assert result.details["rewrite_pitches"] is True
        assert result.details["notes_rewritten"] == 1
        assert ottavas[0].transposing is True
        assert _musicxml_note_pitches(ss.to_musicxml_string()) == ["C4"]

    def test_remove_ottava_without_rewrite_only_removes_mark(self) -> None:
        """Removing an ottava without rewrite leaves compensated pitches as-is."""
        ss = ScoreSpeak.create(measures=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss.add_ottava("8va", 1, 1.0, 1, 1.0, rewrite_pitches=True)

        result = ss.remove_ottava(1, 1.0)

        assert ss.get_notes(measure=1)[0].pitch == "C3"
        assert result.details["rewrite_pitches"] is False
        assert result.details["notes_rewritten"] == 0
        assert list(ss.score.parts[0].getElementsByClass(m21spanner.Ottava)) == []

    @pytest.mark.parametrize(
        ("ottava_type", "rewritten_pitch"),
        [
            ("8va", "C3"),
            ("8vb", "C5"),
            ("15ma", "C2"),
            ("15mb", "C6"),
        ],
    )
    def test_remove_ottava_rewrite_restores_compensated_pitches(
        self,
        ottava_type: str,
        rewritten_pitch: str,
    ) -> None:
        """Removing with rewrite restores pitches compensated during add."""
        ss = ScoreSpeak.create(measures=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss.add_ottava(
            ottava_type,
            1,
            1.0,
            1,
            1.0,
            rewrite_pitches=True,
        )
        assert ss.get_notes(measure=1)[0].pitch == rewritten_pitch

        result = ss.remove_ottava(1, 1.0, rewrite_pitches=True)

        assert ss.get_notes(measure=1)[0].pitch == "C4"
        assert result.details["rewrite_pitches"] is True
        assert result.details["notes_rewritten"] == 1
        assert list(ss.score.parts[0].getElementsByClass(m21spanner.Ottava)) == []

    def test_ottava_rewrite_pitches_transposes_chords(self) -> None:
        """Pitch rewriting compensates chord tones as well as notes."""
        ss = ScoreSpeak.create(measures=1)
        ss.add_chord(["C4", "E4", "G4"], measure=1, beat=1)
        result = ss.add_ottava("8va", 1, 1.0, 1, 1.0, rewrite_pitches=True)

        chord_notes = [
            note for note in ss.get_notes(measure=1) if note.is_chord
        ]

        assert [note.pitch for note in chord_notes] == ["C3", "E3", "G3"]
        assert all(note.is_chord for note in chord_notes)
        assert result.details["notes_rewritten"] == 1

    def test_add_ottava_rejects_old_transposing_argument(self) -> None:
        """The public add API no longer accepts the old transposing flag."""
        ss = ScoreSpeak.create(measures=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)

        with pytest.raises(TypeError, match="transposing"):
            ss.add_ottava("8va", 1, 1.0, 1, 1.0, transposing=False)

    def test_ottava_rewrite_true_matches_old_non_transposing_export(self) -> None:
        """Compensated ottavas export the old visually equivalent raw pitch."""
        cases = [
            ("8va", ["C4"]),
        ]
        for ottava_type, expected_pitches in cases:
            ss = ScoreSpeak.create(measures=1)
            ss._add_note_one("C4", "quarter", measure=1, beat=1)
            ss.add_ottava(ottava_type, 1, 1.0, 1, 1.0, rewrite_pitches=True)

            assert _musicxml_note_pitches(ss.to_musicxml_string()) == expected_pitches

    def test_glissando(self, simple_score):
        ss = simple_score
        ss.add_glissando(1, 1.0, 2, 2.0, line_type="wavy")
        ss.remove_glissando(1, 1.0)

    def test_pedal(self, simple_score):
        ss = simple_score
        ss.add_pedal(1, 1.0, 2, 2.0)
        ss.remove_pedal(1, 1.0)


class TestEndingBracket:
    def test_volta_single_measure(self):
        ss = ScoreSpeak.create(measures=3)
        ss._add_note_one("C4", measure=1)
        ss._add_note_one("D4", measure=2)
        ss._add_note_one("E4", measure=3)
        ss.add_ending_bracket(1, start_measure=1, end_measure=1)
        ss.remove_ending_bracket(1, start_measure=1)

    def test_volta_span(self):
        ss = ScoreSpeak.create(measures=4)
        for m in range(1, 5):
            ss._add_note_one("C4", measure=m)
        result = ss.add_ending_bracket(2, start_measure=2, end_measure=3)
        assert result.details["parts"] == [0]
        removed = ss.remove_ending_bracket(2, start_measure=2)
        assert removed.details["parts"] == [0]

    def test_volta_applies_to_every_part(self):
        ss = ScoreSpeak.create(parts=["violin", "cello"], measures=3)
        result = ss.add_ending_bracket(1, start_measure=1, end_measure=2)

        assert result.details["parts"] == [0, 1]
        for part in ss.score.parts:
            brackets = list(part.getElementsByClass(m21spanner.RepeatBracket))
            assert len(brackets) == 1

        removed = ss.remove_ending_bracket(1, start_measure=1)

        assert removed.details["parts"] == [0, 1]
        for part in ss.score.parts:
            brackets = list(part.getElementsByClass(m21spanner.RepeatBracket))
            assert brackets == []

    def test_volta_serializes_without_custom_display_label(self) -> None:
        """Repeat brackets should rely on the MusicXML number, not text labels."""
        ss = ScoreSpeak.create(measures=3)
        for measure_number in range(1, 4):
            ss._add_note_one("C4", measure=measure_number)

        result = ss.add_ending_bracket(1, start_measure=1, end_measure=2)

        endings = _musicxml_endings(ss.to_musicxml_string())
        start_endings = [ending for ending in endings if ending.get("type") == "start"]
        assert "label" not in result.details
        assert len(start_endings) == 1
        assert start_endings[0].get("number") == "1"
        assert start_endings[0].text is None

    def test_volta_accepts_dotted_number_without_display_label(self) -> None:
        """Dotted numbers should normalize without adding display text."""
        ss = ScoreSpeak.create(measures=2)
        ss._add_note_one("C4", measure=1)
        ss._add_note_one("D4", measure=2)

        result = ss.add_ending_bracket("1.", start_measure=1, end_measure=1)

        assert result.details["number"] == "1"
        assert "label" not in result.details
        assert result.details["parts"] == [0]
        endings = _musicxml_endings(ss.to_musicxml_string())
        start_endings = [ending for ending in endings if ending.get("type") == "start"]
        assert start_endings[0].get("number") == "1"
        assert start_endings[0].text is None
        ss.remove_ending_bracket("1.", start_measure=1)


class TestArpeggio:
    def test_arpeggio_on_chord(self, simple_score):
        ss = simple_score
        ss.add_arpeggio(measure_number=3, beat=1.0)
        ss.remove_arpeggio(3, 1.0)

    def test_arpeggio_on_note_raises(self, simple_score):
        with pytest.raises(ValueError, match="requires a chord"):
            simple_score.add_arpeggio(measure_number=1, beat=1.0)


class TestStringArticulations:
    def test_up_bow(self, simple_score):
        r = simple_score.add_articulation("up bow", 1, 1.0)
        assert r.success
        simple_score.remove_articulation("up bow", 1, 1.0)


class TestFingering:
    def test_add_remove_fingering(self, simple_score):
        ss = simple_score
        ss.add_fingering(3, measure_number=1, beat=1.0)
        ss.remove_fingering(1, 1.0)

    def test_add_fingering_has_no_substitution_argument(self, simple_score):
        """The public fingering API no longer accepts substitution markings."""
        signature = inspect.signature(ScoreSpeak.add_fingering)

        assert "substitution" not in signature.parameters
        with pytest.raises(TypeError, match="substitution"):
            simple_score.add_fingering(
                3,
                measure_number=1,
                beat=1.0,
                substitution=True,
            )

    def test_add_fingering_export_does_not_emit_substitution_yes(self, simple_score):
        """Normal fingering export must not create substitution markings."""
        ss = simple_score

        ss.add_fingering(3, measure_number=1, beat=1.0)
        musicxml = ss.to_musicxml_string()

        assert "<fingering" in musicxml
        assert 'substitution="yes"' not in musicxml


class TestMusicXMLRoundTripMarkings:
    def test_lyric_ornament_gliss_in_xml(self, simple_score):
        ss = simple_score
        ss.add_lyric("mi", 1, 1.0)
        ss.add_ornament("turn", 1, 2.0)
        ss.add_glissando(2, 1.0, 2, 2.0)
        ss.add_pedal(1, 1.0, 2, 2.0)
        xml = ss.to_musicxml_string()
        assert "<lyric" in xml
        assert "ornaments" in xml or "turn" in xml.lower()
        assert "glissando" in xml.lower()
        assert "pedal" in xml.lower()

    def test_round_trip_preserves_lyric(self):
        ss = ScoreSpeak.create(measures=1)
        ss._add_note_one("G4")
        ss.add_lyric("sol", 1, 1.0)
        fd, path = tempfile.mkstemp(suffix=".musicxml")
        os.close(fd)
        try:
            ss.to_musicxml(path)
            ss2 = ScoreSpeak.from_musicxml(path)
            lyrics = ss2.get_lyrics()
            assert any("sol" in L.text for L in lyrics)
        finally:
            os.unlink(path)
