"""Cross-cutting edge cases and defensive paths not covered elsewhere."""

import os
import tempfile
import xml.etree.ElementTree as ET

import pytest
from music21 import stream as m21stream

from scorespeak import ScoreSpeak
from scorespeak.editing.parts import resolve_instrument


MINIMAL_MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>Test</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note>
        <rest/>
        <duration>4</duration>
        <voice>1</voice>
        <type>whole</type>
      </note>
    </measure>
  </part>
</score-partwise>
"""


class TestCreateEdgeCases:
    """Invalid or unusual ScoreSpeak.create inputs."""

    def test_invalid_part_spec_type(self):
        with pytest.raises(ValueError, match="string or dict"):
            ScoreSpeak.create(parts=[123], measures=1)  # type: ignore[list-item]


class TestFromMusicXMLEdgeCases:
    """Import edge cases."""

    def test_from_string_minimal_score(self):
        ss = ScoreSpeak.from_musicxml(MINIMAL_MUSICXML)
        assert ss.part_count >= 1
        assert ss.measure_count >= 1

    def test_invalid_xml_raises(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            ScoreSpeak.from_musicxml("<<<not xml")

    def test_round_trip_xml_well_formed(self):
        ss = ScoreSpeak.from_musicxml(MINIMAL_MUSICXML)
        out = ss.to_musicxml_string()
        ET.fromstring(out)


class TestEmptyScoreAndParts:
    """Behavior with bare music21 scores and measure-less parts."""

    def test_add_part_on_completely_empty_score(self):
        raw = m21stream.Score()
        ss = ScoreSpeak(raw)
        result = ss.add_part(instrument="flute")
        assert result.success
        assert ss.part_count == 1
        assert ss.measure_count >= 1

    def test_add_part_when_reference_has_zero_measures(self):
        ss = ScoreSpeak.create(parts=["piano"], measures=0)
        ss.add_part(instrument="violin")
        assert ss.part_count == 2
        for p in ss.list_parts():
            assert p.measure_count == 0

    def test_add_measures_then_both_parts_gain(self):
        ss = ScoreSpeak.create(parts=["piano"], measures=0)
        ss.add_part(instrument="violin")
        ss.add_measures(2)
        for p in ss.list_parts():
            assert p.measure_count == 2


class TestResolvePartDefensive:
    """Part resolution errors."""

    def test_resolve_part_float_rejected(self):
        ss = ScoreSpeak.create(measures=1)
        with pytest.raises(ValueError, match="Cannot resolve part"):
            ss._add_note_one("C4", part=1.0)  # type: ignore[arg-type]

    def test_resolve_part_negative_index(self):
        ss = ScoreSpeak.create(measures=1)
        with pytest.raises(ValueError, match="out of range"):
            ss._add_note_one("C4", part=-1)

    def test_no_measures_add_note_fails(self):
        ss = ScoreSpeak.create(measures=0)
        with pytest.raises(ValueError, match="No measures"):
            ss._add_note_one("C4")


class TestBeatAndMeterEdgeCases:
    """Note entry boundary conditions."""

    def test_beat_below_one_rejected(self):
        ss = ScoreSpeak.create(measures=1)
        with pytest.raises(ValueError, match="at least 1"):
            ss._add_note_one("C4", beat=0.5)

    def test_add_measures_count_zero_rejected(self):
        ss = ScoreSpeak.create(measures=1)
        with pytest.raises(ValueError, match="at least 1"):
            ss.add_measures(0)

    def test_insert_measure_before_zero_rejected(self):
        ss = ScoreSpeak.create(measures=2)
        with pytest.raises(ValueError, match="at least 1"):
            ss.insert_measure(before=0)

    def test_compound_meter_six_eighth_note_capacity(self):
        ss = ScoreSpeak.create(time_signature="6/8", measures=1)
        ss._add_note_one("C4", duration="quarter", beat=1)
        ss._add_note_one("D4", duration="eighth", beat=3)
        with pytest.raises(ValueError, match="exceed"):
            ss._add_note_one("E4", duration="quarter", beat=3.5)


class TestTransposeTypeGuards:
    """Layout transpose interval parsing."""

    def test_transpose_rejects_float_interval(self):
        ss = ScoreSpeak.create(measures=1)
        ss._add_note_one("C4")
        with pytest.raises(ValueError, match="string or integer"):
            ss.transpose(2.5)  # type: ignore[arg-type]


class TestResolveInstrumentFallback:
    """Instrument resolution for uncommon names."""

    def test_obscure_name_falls_back_to_generic(self):
        inst = resolve_instrument("Glass Harmonica")
        assert inst is not None
        assert "Glass" in str(inst) or getattr(inst, "instrumentName", "") == "Glass Harmonica"


class TestDynamicsAndArticulationsRemoveErrors:
    """Removing absent markings raises clear errors (agent-retry friendly)."""

    def test_remove_missing_dynamic_raises(self):
        ss = ScoreSpeak.create(measures=1)
        ss._add_note_one("C4")
        with pytest.raises(ValueError, match="No dynamic found"):
            ss.remove_dynamic(1, 1.0)

    def test_remove_missing_articulation_raises(self):
        ss = ScoreSpeak.create(measures=1)
        ss._add_note_one("C4")
        with pytest.raises(ValueError, match="No staccato"):
            ss.remove_articulation("staccato", 1, 1.0)


class TestMultiVoiceMeasureFill:
    """Independent voices each respect meter."""

    def test_two_voices_full_bar_each(self):
        ss = ScoreSpeak.create(measures=1)
        ss._add_note_one("C4", voice=1)
        ss._add_note_one("E4", voice=2)
        n = ss.get_notes(measure=1)
        assert len(n) >= 2


class TestExportVariants:
    """MusicXML export flags."""

    def test_to_musicxml_make_notation_false(self):
        ss = ScoreSpeak.create(measures=1)
        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
            path = f.name
        try:
            ss.to_musicxml(path, make_notation=False)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)
