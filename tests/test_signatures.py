"""Tests for SignaturesMixin — time signatures, key signatures, clefs, barlines, repeats, pickups."""

from xml.etree import ElementTree as ET

import pytest

from music21 import bar as m21bar
from music21 import clef as m21clef
from music21 import key as m21key
from music21 import meter as m21meter
from music21 import note as m21note
from music21 import stream as m21stream

from scorespeak import ScoreSpeak


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_score(
    time_signature: str = "4/4",
    key_signature: str = "C",
    measures: int = 4,
    parts: list | None = None,
) -> ScoreSpeak:
    """Create a basic ScoreSpeak for testing."""
    return ScoreSpeak.create(
        title="Test",
        time_signature=time_signature,
        key_signature=key_signature,
        measures=measures,
        parts=parts or ["piano"],
    )


def _musicxml_attribute_measure_numbers(
    score_state: ScoreSpeak,
    attribute_name: str,
) -> list[int]:
    """Return measure numbers that export a MusicXML attribute element."""
    root = ET.fromstring(score_state.to_musicxml_string())
    measure_numbers: list[int] = []
    for measure in root.findall(".//{*}part/{*}measure"):
        attribute = measure.find(f"{{*}}attributes/{{*}}{attribute_name}")
        if attribute is None:
            continue
        measure_numbers.append(int(measure.attrib["number"]))
    return measure_numbers


def _local_time_signature_ratios(
    score_state: ScoreSpeak,
    measure_number: int,
) -> list[str]:
    """Return local time signature ratios in the first part measure."""
    part_obj, _ = score_state._resolve_part(None)
    measure = score_state._resolve_measure(part_obj, measure_number)
    return [
        time_signature.ratioString
        for time_signature in measure.getElementsByClass(m21meter.TimeSignature)
    ]


def _local_key_signature_labels(
    score_state: ScoreSpeak,
    measure_number: int,
) -> list[str]:
    """Return local key signature labels in the first part measure."""
    part_obj, _ = score_state._resolve_part(None)
    measure = score_state._resolve_measure(part_obj, measure_number)
    return [
        score_state._format_key_signature(key_signature)
        for key_signature in measure.getElementsByClass(m21key.KeySignature)
    ]


def _replace_measure_with_events(
    score_state: ScoreSpeak,
    measure_number: int,
    events: list[m21note.GeneralNote],
    part: int | None = None,
) -> m21stream.Measure:
    """Replace direct note-like content in one measure."""
    part_obj, _ = score_state._resolve_part(part)
    measure = score_state._resolve_measure(part_obj, measure_number)
    for element in list(measure.notesAndRests):
        measure.remove(element)
    for event in events:
        measure.append(event)
    return measure


def _rest_summaries(
    score_state: ScoreSpeak,
    measure_number: int,
    voice: int | None = None,
) -> list[tuple[float, float]]:
    """Return beat and duration for visible rests in a measure."""
    rests = [
        note_info
        for note_info in score_state.get_notes(measure=measure_number, voice=voice)
        if note_info.is_rest
    ]
    return [(rest.beat, rest.quarter_length) for rest in rests]


# ===================================================================
# Time Signature Tests
# ===================================================================


class TestSetTimeSignature:
    """Tests for set_time_signature."""

    def test_set_at_first_measure(self):
        ss = _make_score(time_signature="4/4", measures=4)
        result = ss.set_time_signature("3/4", 1)
        assert result.success
        assert ss.get_active_time_signature(1) == "3/4"

    def test_implied_continuity(self):
        ss = _make_score(time_signature="4/4", measures=6)
        ss.set_time_signature("6/8", 3)
        assert ss.get_active_time_signature(1) == "4/4"
        assert ss.get_active_time_signature(2) == "4/4"
        assert ss.get_active_time_signature(3) == "6/8"
        assert ss.get_active_time_signature(4) == "6/8"
        assert ss.get_active_time_signature(5) == "6/8"

    def test_multiple_changes(self):
        ss = _make_score(time_signature="4/4", measures=6)
        ss.set_time_signature("3/4", 1)
        ss.set_time_signature("6/8", 4)
        assert ss.get_active_time_signature(1) == "3/4"
        assert ss.get_active_time_signature(3) == "3/4"
        assert ss.get_active_time_signature(4) == "6/8"
        assert ss.get_active_time_signature(6) == "6/8"

    def test_all_parts_when_none(self):
        ss = _make_score(measures=4, parts=["violin", "cello"])
        result = ss.set_time_signature("3/4", 2)
        assert result.success
        assert len(result.details["parts"]) == 2
        assert ss.get_active_time_signature(2, part=0) == "3/4"
        assert ss.get_active_time_signature(2, part=1) == "3/4"

    def test_part_argument_is_removed(self):
        ss = _make_score(measures=4, parts=["violin", "cello"])
        with pytest.raises(TypeError):
            ss.set_time_signature("3/4", 2, part=0)

    def test_replaces_existing_time_signature(self):
        ss = _make_score(time_signature="4/4", measures=4)
        ss.set_time_signature("3/4", 1)
        ss.set_time_signature("6/8", 1)
        assert ss.get_active_time_signature(1) == "6/8"

    def test_invalid_time_signature(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="not a valid time signature"):
            ss.set_time_signature("banana", 1)

    def test_invalid_measure(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="does not exist"):
            ss.set_time_signature("3/4", 99)

    def test_notes_exceed_new_capacity(self):
        ss = _make_score(time_signature="4/4", measures=2)
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 1)
        for el in list(m.notesAndRests):
            m.remove(el)
        m.append(m21note.Note("C4", quarterLength=4.0))
        with pytest.raises(ValueError, match="beats of music"):
            ss.set_time_signature("2/4", 1)

    def test_multi_voice_capacity_allows_parallel_whole_notes(self):
        ss = _make_score(time_signature="4/4", measures=1)
        ss._add_note_one("C5", "whole", measure=1, beat=1, voice=1)
        ss._add_note_one("C4", "whole", measure=1, beat=1, voice=2)

        result = ss.set_time_signature("4/4", 1)

        assert result.success
        assert ss.get_active_time_signature(1) == "4/4"

    def test_multi_voice_capacity_rejects_too_small_meter(self):
        ss = _make_score(time_signature="4/4", measures=1)
        ss._add_note_one("C5", "whole", measure=1, beat=1, voice=1)
        ss._add_note_one("C4", "whole", measure=1, beat=1, voice=2)

        with pytest.raises(ValueError, match="beats of music"):
            ss.set_time_signature("1/4", 1)

    def test_result_details(self):
        ss = _make_score(measures=4)
        result = ss.set_time_signature("5/4", 2)
        assert result.details["time_signature"] == "5/4"
        assert result.details["measure"] == 2

    def test_empty_measure_rests_resize_for_exported_meter_change(self) -> None:
        """Changing meter on empty measures preserves the exported measure count."""
        ss = _make_score(time_signature="4/4", measures=8)

        ss.set_time_signature("3/4", 2)

        root = ET.fromstring(ss.to_musicxml_string())
        measures = root.findall(".//{*}measure")
        exported_times = []
        for measure in measures:
            time_element = measure.find("{*}attributes/{*}time")
            if time_element is None:
                continue
            exported_times.append(
                (
                    measure.attrib["number"],
                    time_element.findtext("{*}beats"),
                    time_element.findtext("{*}beat-type"),
                )
            )

        assert len(measures) == 8
        assert ("2", "3", "4") in exported_times

    def test_redundant_time_signature_is_noop_without_musicxml_marker(self) -> None:
        """Setting an already inherited meter does not add a local marker."""
        ss = _make_score(time_signature="4/4", measures=4)

        result = ss.set_time_signature("4/4", 3)

        assert result.success
        assert result.details["changed"] is False
        assert result.details["action"] == "already_active"
        assert _local_time_signature_ratios(ss, 3) == []
        assert _musicxml_attribute_measure_numbers(ss, "time") == [1]

    def test_time_signature_set_to_inherited_removes_middle_change(self) -> None:
        """Setting a middle meter back to the prior meter removes the change."""
        ss = _make_score(time_signature="4/4", measures=5)
        ss.set_time_signature("3/4", 3)

        result = ss.set_time_signature("4/4", 3)

        assert result.success
        assert result.details["changed"] is True
        assert result.details["action"] == "removed_local_change"
        assert _local_time_signature_ratios(ss, 3) == []
        assert [ss.get_active_time_signature(bar) for bar in range(1, 6)] == [
            "4/4",
            "4/4",
            "4/4",
            "4/4",
            "4/4",
        ]
        assert _musicxml_attribute_measure_numbers(ss, "time") == [1]

    def test_real_time_signature_return_is_preserved(self) -> None:
        """Returning to a prior meter after a real change keeps a local marker."""
        ss = _make_score(time_signature="4/4", measures=5)
        ss.set_time_signature("3/4", 2)

        result = ss.set_time_signature("4/4", 4)

        assert result.success
        assert result.details["changed"] is True
        assert _local_time_signature_ratios(ss, 4) == ["4/4"]
        assert _musicxml_attribute_measure_numbers(ss, "time") == [1, 2, 4]

    def test_shrink_valid_region_removes_trailing_rests(self) -> None:
        """Shrinking a fitting mixed bar removes stale trailing rests."""
        ss = _make_score(time_signature="4/4", measures=2)
        _replace_measure_with_events(
            ss,
            2,
            [
                m21note.Note("C4", quarterLength=1.0),
                m21note.Note("D4", quarterLength=1.0),
                m21note.Note("E4", quarterLength=1.0),
                m21note.Rest(quarterLength=1.0),
            ],
        )

        result = ss.set_time_signature("3/4", 2)

        assert result.success
        assert result.details["validated_measures"] == 1
        assert result.details["removed_overflow_rests"] == 1
        assert _rest_summaries(ss, 2) == []

    def test_shrink_invalid_region_fails_before_mutation(self) -> None:
        """A later overfull sounding bar blocks the whole meter change."""
        ss = _make_score(time_signature="4/4", measures=3)
        _replace_measure_with_events(
            ss,
            3,
            [
                m21note.Note("C4", quarterLength=1.0),
                m21note.Note("D4", quarterLength=1.0),
                m21note.Note("E4", quarterLength=1.0),
                m21note.Note("F4", quarterLength=1.0),
            ],
        )

        with pytest.raises(ValueError, match="part 0, measure 3, voice 1"):
            ss.set_time_signature("3/4", 2)

        assert ss.get_active_time_signature(2) == "4/4"
        assert _local_time_signature_ratios(ss, 2) == []

    def test_expand_incomplete_region_adds_missing_rests(self) -> None:
        """Expanding a mixed bar fills the new trailing gap with rests."""
        ss = _make_score(time_signature="3/4", measures=2)
        _replace_measure_with_events(
            ss,
            2,
            [
                m21note.Note("C4", quarterLength=1.0),
                m21note.Note("D4", quarterLength=1.0),
                m21note.Note("E4", quarterLength=1.0),
            ],
        )

        result = ss.set_time_signature("4/4", 2)

        assert result.success
        assert result.details["auto_completed_rests"] == 1
        assert _rest_summaries(ss, 2) == [(4.0, 1.0)]

    def test_expand_rest_only_region_resizes_full_measure_rests(self) -> None:
        """Expanding rest-only bars rewrites full-measure rests."""
        ss = _make_score(time_signature="3/4", measures=2)

        result = ss.set_time_signature("4/4", 2)

        assert result.success
        assert result.details["normalized_measures"] == 1
        assert _rest_summaries(ss, 2) == [(1.0, 4.0)]

    def test_region_stops_before_next_explicit_time_signature(self) -> None:
        """Normalization does not cross the next real meter boundary."""
        ss = _make_score(time_signature="3/4", measures=5)
        ss.set_time_signature("2/4", 4)

        result = ss.set_time_signature("4/4", 2)

        assert result.success
        assert result.details["validated_measures"] == 2
        assert _rest_summaries(ss, 2) == [(1.0, 4.0)]
        assert _rest_summaries(ss, 3) == [(1.0, 4.0)]
        assert _rest_summaries(ss, 4) == [(1.0, 2.0)]
        assert _rest_summaries(ss, 5) == [(1.0, 2.0)]

    def test_multi_part_region_validation_is_atomic(self) -> None:
        """An overflowing part prevents mutation in all parts."""
        ss = _make_score(time_signature="4/4", measures=2, parts=["Violin", "Cello"])
        _replace_measure_with_events(
            ss,
            2,
            [
                m21note.Note("C4", quarterLength=1.0),
                m21note.Note("D4", quarterLength=1.0),
                m21note.Note("E4", quarterLength=1.0),
            ],
            part=0,
        )
        _replace_measure_with_events(
            ss,
            2,
            [
                m21note.Note("C3", quarterLength=1.0),
                m21note.Note("D3", quarterLength=1.0),
                m21note.Note("E3", quarterLength=1.0),
                m21note.Note("F3", quarterLength=1.0),
            ],
            part=1,
        )

        with pytest.raises(ValueError, match="part 1, measure 2, voice 1"):
            ss.set_time_signature("3/4", 1)

        assert ss.get_active_time_signature(1, part=0) == "4/4"
        assert ss.get_active_time_signature(1, part=1) == "4/4"

    def test_multi_voice_region_normalizes_each_voice(self) -> None:
        """Parallel voices get independent completion rests when expanding."""
        ss = _make_score(time_signature="4/4", measures=1)
        ss._add_note_one("C5", "whole", measure=1, beat=1, voice=1)
        ss._add_note_one("C4", "whole", measure=1, beat=1, voice=2)

        result = ss.set_time_signature("5/4", 1)

        assert result.success
        assert result.details["auto_completed_rests"] == 2
        assert _rest_summaries(ss, 1, voice=1) == [(5.0, 1.0)]
        assert _rest_summaries(ss, 1, voice=2) == [(5.0, 1.0)]

    def test_hidden_rests_stay_hidden_during_region_normalization(self) -> None:
        """Hidden rest suppression is preserved when expanding a meter."""
        ss = _make_score(time_signature="3/4", measures=1)
        _replace_measure_with_events(
            ss,
            1,
            [
                m21note.Note("C4", quarterLength=1.0),
                m21note.Rest(quarterLength=1.0),
                m21note.Rest(quarterLength=1.0),
            ],
        )
        ss.remove_rests(measure=1, part=0, voice=1, beat=2.0)

        result = ss.set_time_signature("4/4", 1)

        assert result.success
        assert _rest_summaries(ss, 1) == [(3.0, 2.0)]
        part_obj, _ = ss._resolve_part(None)
        measure = ss._resolve_measure(part_obj, 1)
        hidden_rests = [
            rest
            for rest in measure.getElementsByClass(m21note.Rest)
            if bool(getattr(rest.style, "hideObjectOnPrint", False))
        ]
        hidden_ranges = [
            (measure.elementOffset(rest), rest.quarterLength)
            for rest in hidden_rests
        ]
        assert hidden_ranges == [
            (1.0, 1.0),
        ]


# ===================================================================
# Key Signature Tests
# ===================================================================


class TestSetKeySignature:
    """Tests for set_key_signature."""

    def test_set_major_key(self):
        ss = _make_score(key_signature="C", measures=4)
        result = ss.set_key_signature("G", 1)
        assert result.success
        ks = ss.get_active_key_signature(1)
        assert "G" in ks

    def test_set_minor_key(self):
        ss = _make_score(measures=4)
        ss.set_key_signature("A minor", 2)
        ks = ss.get_active_key_signature(2)
        assert "minor" in ks.lower()
        assert "A" in ks

    def test_set_key_with_shorthand_minor(self):
        ss = _make_score(measures=4)
        ss.set_key_signature("Am", 1)
        ks = ss.get_active_key_signature(1)
        assert "A" in ks
        assert "minor" in ks.lower()

    def test_set_key_with_flats(self):
        ss = _make_score(measures=4)
        ss.set_key_signature("Bb", 1)
        ks = ss.get_active_key_signature(1)
        assert "B" in ks

    def test_set_key_with_sharps(self):
        ss = _make_score(measures=4)
        ss.set_key_signature("F#", 1)
        ks = ss.get_active_key_signature(1)
        assert "F" in ks

    def test_set_key_numeric_sharps(self):
        ss = _make_score(measures=4)
        ss.set_key_signature("3", 1)
        ks = ss.get_active_key_signature(1)
        assert "3" in ks and "sharp" in ks.lower()

    def test_set_key_numeric_flats(self):
        ss = _make_score(measures=4)
        ss.set_key_signature("-2", 1)
        ks = ss.get_active_key_signature(1)
        assert "2" in ks and "flat" in ks.lower()

    def test_implied_continuity(self):
        ss = _make_score(key_signature="C", measures=6)
        ss.set_key_signature("D", 3)
        ks1 = ss.get_active_key_signature(2)
        ks3 = ss.get_active_key_signature(3)
        ks5 = ss.get_active_key_signature(5)
        assert "C" in ks1
        assert "D" in ks3
        assert "D" in ks5

    def test_all_parts_when_none(self):
        ss = _make_score(measures=4, parts=["violin", "cello"])
        result = ss.set_key_signature("Bb", 2)
        assert result.success
        assert len(result.details["parts"]) == 2

    def test_concert_key_materializes_written_key_for_transposing_part(self) -> None:
        """Global key edits use concert key and derive written part keys."""
        ss = _make_score(measures=1, parts=["flute", "clarinet"])

        result = ss.set_key_signature("F major", 1)

        assert result.success
        assert ss.get_active_key_signature(1) == "F major"
        assert ss.get_active_key_signature(1, part=0) == "F major"
        assert ss.get_active_key_signature(1, part=1) == "G major"

    def test_concert_key_uses_concert_key_for_sounding_transposing_part(self) -> None:
        """A transposing part stored at sounding pitch receives concert keys."""
        ss = _make_score(measures=1, parts=["flute", "clarinet"])
        ss.transpose_to_concert_pitch(part=1)

        result = ss.set_key_signature("F major", 1)

        assert result.success
        assert ss.get_active_key_signature(1, part=1) == "F major"

    def test_local_key_signature_survives_global_key_change(self) -> None:
        """Explicit local key overrides are not overwritten by global edits."""
        ss = _make_score(measures=1, parts=["flute", "clarinet"])
        ss.set_key_signature("F major", 1)
        ss._set_local_key_signature("open", 1, part=1)

        result = ss.set_key_signature("G major", 1)

        assert result.success
        assert ss.get_active_key_signature(1) == "G major"
        assert ss.get_active_key_signature(1, part=0) == "G major"
        assert ss.get_active_key_signature(1, part=1) == "open/atonal"
        assert result.details["part_actions"][1]["action"] == (
            "local_override_preserved"
        )

    def test_replaces_existing_key_signature(self):
        ss = _make_score(key_signature="C", measures=4)
        ss.set_key_signature("G", 1)
        ss.set_key_signature("D", 1)
        ks = ss.get_active_key_signature(1)
        assert "D" in ks

    def test_invalid_key_signature(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="Cannot parse key signature"):
            ss.set_key_signature("X Y Z", 1)

    def test_unicode_accidentals(self):
        ss = _make_score(measures=4)
        ss.set_key_signature("F\u266f minor", 1)
        ks = ss.get_active_key_signature(1)
        assert "F" in ks
        assert "minor" in ks.lower()

    def test_transpose_existing_notes(self):
        ss = _make_score(key_signature="C", measures=4)
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 2)
        for el in list(m.notesAndRests):
            m.remove(el)
        m.append(m21note.Note("C4", quarterLength=1.0))
        m.append(m21note.Note("E4", quarterLength=1.0))
        result = ss.set_key_signature("G", 2, transpose_existing=True)
        assert result.success
        assert result.details["transposed"] is True

    def test_no_transpose_by_default(self):
        ss = _make_score(key_signature="C", measures=4)
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 2)
        for el in list(m.notesAndRests):
            m.remove(el)
        m.append(m21note.Note("C4", quarterLength=1.0))
        result = ss.set_key_signature("G", 2)
        assert result.details["transposed"] is False
        notes = list(m.getElementsByClass(m21note.Note))
        assert notes[0].pitch.name == "C"

    def test_result_details(self):
        ss = _make_score(measures=4)
        result = ss.set_key_signature("Bb", 2)
        assert result.details["measure"] == 2
        assert "B" in result.details["key_signature"]

    def test_key_signature_refreshes_only_active_key_region(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Changing a key should refresh only measures before the next key."""
        ss = _make_score(key_signature="C", measures=5)
        ss.set_key_signature("D", 4)
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

        ss.set_key_signature("G", 2)

        assert calls == [2, 3]

    def test_redundant_key_signature_is_noop_without_musicxml_marker(self) -> None:
        """Setting an already inherited key does not add a local marker."""
        ss = _make_score(key_signature="C", measures=4)

        result = ss.set_key_signature("C", 3)

        assert result.success
        assert result.details["changed"] is False
        assert result.details["action"] == "already_active"
        assert _local_key_signature_labels(ss, 3) == []
        assert _musicxml_attribute_measure_numbers(ss, "key") == [1]

    def test_key_signature_set_to_inherited_removes_middle_change(self) -> None:
        """Setting a middle key back to the prior key removes the change."""
        ss = _make_score(key_signature="C", measures=5)
        ss.set_key_signature("G", 3)

        result = ss.set_key_signature("C", 3)

        assert result.success
        assert result.details["changed"] is True
        assert result.details["action"] == "removed_local_change"
        assert _local_key_signature_labels(ss, 3) == []
        assert [ss.get_active_key_signature(bar) for bar in range(1, 6)] == [
            "C major",
            "C major",
            "C major",
            "C major",
            "C major",
        ]
        assert _musicxml_attribute_measure_numbers(ss, "key") == [1]

    def test_real_key_signature_return_is_preserved(self) -> None:
        """Returning to a prior key after a real change keeps a local marker."""
        ss = _make_score(key_signature="C", measures=5)
        ss.set_key_signature("G", 2)

        result = ss.set_key_signature("C", 4)

        assert result.success
        assert result.details["changed"] is True
        assert _local_key_signature_labels(ss, 4) == ["C major"]
        assert _musicxml_attribute_measure_numbers(ss, "key") == [1, 2, 4]

    def test_redundant_key_marker_does_not_become_latent_return(self) -> None:
        """A preexisting duplicate key is removed before earlier key edits."""
        ss = _make_score(key_signature="C", measures=6)
        part_obj, _ = ss._resolve_part(None)
        measure = ss._resolve_measure(part_obj, 5)
        measure.insert(0, m21key.Key("C", "major"))

        result = ss.set_key_signature("G", 3)

        assert result.success
        assert result.details["changed"] is True
        assert _local_key_signature_labels(ss, 5) == []
        assert ss.get_active_key_signature(5) == "G major"
        assert _musicxml_attribute_measure_numbers(ss, "key") == [1, 3]


# ===================================================================
# Clef Tests
# ===================================================================


class TestSetClef:
    """Tests for set_clef."""

    def test_set_treble(self):
        ss = _make_score(measures=4)
        result = ss.set_clef("treble", 1)
        assert result.success

    def test_set_bass(self):
        ss = _make_score(measures=4)
        result = ss.set_clef("bass", 2)
        assert result.success
        assert result.details["clef"] == "bass"

    def test_set_alto(self):
        ss = _make_score(measures=4)
        result = ss.set_clef("alto", 1)
        assert result.success

    def test_set_tenor(self):
        ss = _make_score(measures=4)
        result = ss.set_clef("tenor", 3)
        assert result.success

    def test_clef_active_at_measure(self):
        ss = _make_score(measures=4)
        ss.set_clef("bass", 2)
        part_obj, _ = ss._resolve_part(None)
        clef_obj = ss._get_active_clef_obj(part_obj, 2)
        assert isinstance(clef_obj, m21clef.BassClef)

    def test_replaces_existing_clef(self):
        ss = _make_score(measures=4)
        ss.set_clef("bass", 1)
        ss.set_clef("alto", 1)
        part_obj, _ = ss._resolve_part(None)
        clef_obj = ss._get_active_clef_obj(part_obj, 1)
        assert isinstance(clef_obj, m21clef.AltoClef)

    def test_all_parts_when_none(self):
        ss = _make_score(measures=4, parts=["violin", "cello"])
        result = ss.set_clef("alto", 2)
        assert len(result.details["parts"]) == 2

    def test_single_part(self):
        ss = _make_score(measures=4, parts=["violin", "cello"])
        result = ss.set_clef("bass", 2, part=1)
        assert result.details["parts"] == [1]

    def test_invalid_clef(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="Unknown clef type"):
            ss.set_clef("kazoo", 1)

    def test_invalid_measure(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="does not exist"):
            ss.set_clef("treble", 99)


# ===================================================================
# Barline Tests
# ===================================================================


class TestSetBarline:
    """Tests for set_barline."""

    def test_set_double_barline(self):
        ss = _make_score(measures=4)
        result = ss.set_barline("double", 2)
        assert result.success
        assert result.description == (
            "Set right-edge barline of measure 2 to double."
        )
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 2)
        assert m.rightBarline is not None
        assert m.rightBarline.type == "double"

    def test_set_final_barline(self):
        ss = _make_score(measures=4)
        result = ss.set_barline("final", 4)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 4)
        assert m.rightBarline.type == "final"

    def test_set_repeat_start_barline_rejected(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="Use add_repeat"):
            ss.set_barline("repeat-start", 2)

    def test_set_repeat_end_barline_rejected(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="Use add_repeat"):
            ss.set_barline("repeat-end", 4)

    def test_add_repeat_still_sets_repeat_barlines(self):
        ss = _make_score(measures=4)
        result = ss.add_repeat(2, 4)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        start_m = ss._resolve_measure(part_obj, 2)
        end_m = ss._resolve_measure(part_obj, 4)
        assert isinstance(start_m.leftBarline, m21bar.Repeat)
        assert isinstance(end_m.rightBarline, m21bar.Repeat)

    def test_all_parts_when_none(self):
        ss = _make_score(measures=4, parts=["violin", "cello"])
        result = ss.set_barline("double", 3)
        assert len(result.details["parts"]) == 2

    def test_invalid_barline_type(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="Unknown barline type"):
            ss.set_barline("squiggly", 1)

    def test_side_argument_is_not_public(self) -> None:
        """Ordinary barline edits only expose the right measure edge."""
        ss = _make_score(measures=4)
        with pytest.raises(TypeError, match="unexpected keyword argument 'side'"):
            ss.set_barline("double", 1, side="top")

    def test_barline_changes_only_right_edge(self) -> None:
        """Left-edge changes are represented by editing the previous measure."""
        ss = _make_score(measures=4)

        ss.set_barline("double", 2)

        part_obj, _ = ss._resolve_part(None)
        measure_two = ss._resolve_measure(part_obj, 2)
        measure_three = ss._resolve_measure(part_obj, 3)
        assert measure_two.rightBarline is not None
        assert measure_two.rightBarline.type == "double"
        assert measure_three.leftBarline is None

    def test_invalid_measure(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="does not exist"):
            ss.set_barline("double", 99)

    def test_result_details(self):
        ss = _make_score(measures=4)
        result = ss.set_barline("final", 4)
        assert result.details["barline_type"] == "final"
        assert result.details["side"] == "right"
        assert result.details["measure"] == 4


# ===================================================================
# Repeat Tests
# ===================================================================


class TestAddRepeat:
    """Tests for add_repeat."""

    def test_basic_repeat(self):
        ss = _make_score(measures=4)
        result = ss.add_repeat(2, 4)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        start_m = ss._resolve_measure(part_obj, 2)
        end_m = ss._resolve_measure(part_obj, 4)
        assert isinstance(start_m.leftBarline, m21bar.Repeat)
        assert isinstance(end_m.rightBarline, m21bar.Repeat)

    def test_repeat_same_measure(self):
        ss = _make_score(measures=4)
        result = ss.add_repeat(3, 3)
        assert result.success

    def test_repeat_with_custom_times(self):
        ss = _make_score(measures=4)
        result = ss.add_repeat(1, 4, times=3)
        assert result.success
        assert result.details["times"] == 3

    def test_repeat_all_parts(self):
        ss = _make_score(measures=4, parts=["violin", "cello"])
        result = ss.add_repeat(1, 2)
        assert len(result.details["parts"]) == 2

    def test_start_after_end_raises(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="must not be after"):
            ss.add_repeat(3, 1)

    def test_times_less_than_two_raises(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="at least 2"):
            ss.add_repeat(1, 4, times=1)

    def test_invalid_measure_raises(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="does not exist"):
            ss.add_repeat(1, 99)

    def test_result_details(self):
        ss = _make_score(measures=4)
        result = ss.add_repeat(2, 3, times=2)
        assert result.details["start_measure"] == 2
        assert result.details["end_measure"] == 3
        assert result.details["times"] == 2


class TestRemoveRepeat:
    """Tests for remove_repeat."""

    def test_remove_repeat_basic(self):
        ss = _make_score(measures=4)
        ss.add_repeat(2, 4)
        result = ss.remove_repeat(2, 4)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        start_m = ss._resolve_measure(part_obj, 2)
        end_m = ss._resolve_measure(part_obj, 4)
        assert start_m.leftBarline is None
        assert end_m.rightBarline is None

    def test_remove_repeat_same_measure(self):
        ss = _make_score(measures=4)
        ss.add_repeat(3, 3)
        result = ss.remove_repeat(3, 3)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 3)
        assert m.leftBarline is None
        assert m.rightBarline is None

    def test_remove_repeat_all_parts(self):
        ss = _make_score(measures=4, parts=["violin", "cello"])
        ss.add_repeat(1, 2)
        result = ss.remove_repeat(1, 2)
        assert result.success
        assert len(result.details["parts"]) == 2

    def test_remove_repeat_missing_fails(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="No complete repeat"):
            ss.remove_repeat(1, 4)

    def test_remove_repeat_start_after_end_raises(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="must not be after"):
            ss.remove_repeat(4, 1)


# ===================================================================
# Pickup Measure Tests
# ===================================================================


class TestSetPickupMeasure:
    """Tests for set_pickup_measure."""

    def test_quarter_note_pickup(self):
        ss = _make_score(time_signature="4/4", measures=4)
        result = ss.set_pickup_measure(1.0)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 1)
        assert abs(m.paddingLeft - 3.0) < 1e-9

    def test_eighth_note_pickup(self):
        ss = _make_score(time_signature="4/4", measures=4)
        result = ss.set_pickup_measure(0.5)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 1)
        assert abs(m.paddingLeft - 3.5) < 1e-9

    def test_half_note_pickup(self):
        ss = _make_score(time_signature="4/4", measures=4)
        result = ss.set_pickup_measure(2.0)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 1)
        assert abs(m.paddingLeft - 2.0) < 1e-9

    def test_full_bar_pickup_no_padding(self):
        ss = _make_score(time_signature="4/4", measures=4)
        result = ss.set_pickup_measure(4.0)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 1)
        assert m.paddingLeft < 1e-9

    def test_all_parts_when_none(self):
        ss = _make_score(measures=4, parts=["violin", "cello"])
        result = ss.set_pickup_measure(1.0)
        assert len(result.details["parts"]) == 2

    def test_invalid_zero_duration(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="positive"):
            ss.set_pickup_measure(0.0)

    def test_invalid_negative_duration(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="positive"):
            ss.set_pickup_measure(-1.0)

    def test_duration_exceeds_bar(self):
        ss = _make_score(time_signature="3/4", measures=4)
        with pytest.raises(ValueError, match="exceeds"):
            ss.set_pickup_measure(5.0)

    def test_result_details(self):
        ss = _make_score(measures=4)
        result = ss.set_pickup_measure(1.0)
        assert result.details["duration"] == 1.0

    def test_pickup_clears_explicit_voice_content(self):
        ss = _make_score(time_signature="4/4", measures=1)
        ss._add_note_one("C5", "whole", measure=1, beat=1, voice=1)
        ss._add_note_one("C4", "whole", measure=1, beat=1, voice=2)

        ss.set_pickup_measure(1.0)

        part_obj, _ = ss._resolve_part(None)
        measure_obj = ss._resolve_measure(part_obj, 1)
        assert list(measure_obj.voices) == []
        rests = [note for note in ss.get_notes(measure=1) if note.is_rest]
        assert [(rest.beat, rest.quarter_length) for rest in rests] == [
            (1.0, 1.0)
        ]

    def test_pickup_rejects_overlong_note_insertion(self):
        ss = _make_score(time_signature="4/4", measures=1)
        ss.set_pickup_measure(1.0)

        with pytest.raises(ValueError, match="exceed"):
            ss._add_note_one("C4", "half", measure=1, beat=1)

    def test_pickup_accepts_fitting_note_insertion(self):
        ss = _make_score(time_signature="4/4", measures=1)
        ss.set_pickup_measure(1.0)

        result = ss._add_note_one("C4", "quarter", measure=1, beat=1)

        assert result.success
        assert result.details["measure_integrity"]["is_complete"] is True

    def test_pickup_rest_reshaping_uses_effective_capacity(self):
        ss = _make_score(time_signature="4/4", measures=1)
        ss.set_pickup_measure(2.0)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)

        result = ss.reshape_rests(
            measure=1,
            part=0,
            voice=1,
            start_beat=2.0,
            total_duration="quarter",
            rests=[{"duration": "quarter"}],
        )

        assert result.details["inserted_rests"] == [{
            "kind": "rest",
            "duration": "quarter",
            "quarter_length": 1.0,
            "dots": 0,
            "label": "quarter rest",
            "measure": 1,
            "part": 0,
            "voice": 1,
            "beat": 2.0,
            "offset": 1.0,
            "end": 2.0,
            "visibility": "visible",
        }]


# ===================================================================
# Integration / Edge-case Tests
# ===================================================================


class TestSignaturesIntegration:
    """Integration tests combining multiple signature operations."""

    def test_change_time_sig_and_key_together(self):
        ss = _make_score(measures=8)
        ss.set_time_signature("3/4", 5)
        ss.set_key_signature("Bb", 5)
        assert ss.get_active_time_signature(5) == "3/4"
        assert "B" in ss.get_active_key_signature(5)
        assert ss.get_active_time_signature(4) == "4/4"

    def test_clef_change_midpiece(self):
        ss = _make_score(measures=4)
        ss.set_clef("treble", 1)
        ss.set_clef("bass", 3)
        part_obj, _ = ss._resolve_part(None)
        clef_at_2 = ss._get_active_clef_obj(part_obj, 2)
        clef_at_3 = ss._get_active_clef_obj(part_obj, 3)
        assert isinstance(clef_at_2, m21clef.TrebleClef)
        assert isinstance(clef_at_3, m21clef.BassClef)

    def test_repeat_with_barlines(self):
        ss = _make_score(measures=8)
        ss.add_repeat(3, 6)
        ss.set_barline("double", 8)
        part_obj, _ = ss._resolve_part(None)
        m3 = ss._resolve_measure(part_obj, 3)
        m6 = ss._resolve_measure(part_obj, 6)
        m8 = ss._resolve_measure(part_obj, 8)
        assert isinstance(m3.leftBarline, m21bar.Repeat)
        assert isinstance(m6.rightBarline, m21bar.Repeat)
        assert m8.rightBarline.type == "double"

    def test_pickup_then_add_note(self):
        ss = _make_score(time_signature="4/4", measures=4)
        ss.set_pickup_measure(1.0)
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 1)
        for el in list(m.notesAndRests):
            m.remove(el)
        m.append(m21note.Note("C4", quarterLength=1.0))
        notes = list(m.getElementsByClass(m21note.Note))
        assert len(notes) == 1
        assert notes[0].pitch.nameWithOctave == "C4"

    def test_set_time_sig_in_34_with_notes(self):
        """Changing from 4/4 to 3/4 should fail if measure has 4 beats of notes."""
        ss = _make_score(time_signature="4/4", measures=4)
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 1)
        for el in list(m.notesAndRests):
            m.remove(el)
        m.append(m21note.Note("C4", quarterLength=4.0))
        with pytest.raises(ValueError):
            ss.set_time_signature("3/4", 1)

    def test_set_time_sig_allows_fitting_notes(self):
        """Changing to a larger time sig should be allowed."""
        ss = _make_score(time_signature="3/4", measures=4)
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 1)
        for el in list(m.notesAndRests):
            m.remove(el)
        m.append(m21note.Note("C4", quarterLength=1.0))
        result = ss.set_time_signature("4/4", 1)
        assert result.success


# ===================================================================
# Navigation Mark Tests
# ===================================================================

from music21 import repeat as m21repeat


class TestAddCoda:
    """Tests for add_coda."""

    def test_add_coda_inserts_coda_object(self):
        ss = _make_score(measures=4)
        result = ss.add_coda(2)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 2)
        codas = list(m.getElementsByClass(m21repeat.Coda))
        assert len(codas) == 1

    def test_add_coda_invalid_measure(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="does not exist"):
            ss.add_coda(99)

    def test_add_coda_result_details(self):
        ss = _make_score(measures=4)
        result = ss.add_coda(3)
        assert result.details["mark_type"] == "coda"
        assert result.details["measure"] == 3
        assert "part" not in result.details

    def test_add_coda_anchors_navigation_in_first_part_only(self) -> None:
        """Score-wide navigation marks should have one first-part carrier."""
        ss = _make_score(measures=4, parts=["Violin", "Cello"])

        ss.add_coda(2)

        first_part, _ = ss._resolve_part(0)
        second_part, _ = ss._resolve_part(1)
        first_measure = ss._resolve_measure(first_part, 2)
        second_measure = ss._resolve_measure(second_part, 2)
        assert len(list(first_measure.getElementsByClass(m21repeat.Coda))) == 1
        assert len(list(second_measure.getElementsByClass(m21repeat.Coda))) == 0


class TestAddSegno:
    """Tests for add_segno."""

    def test_add_segno_inserts_segno_object(self):
        ss = _make_score(measures=4)
        result = ss.add_segno(1)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 1)
        segnos = list(m.getElementsByClass(m21repeat.Segno))
        assert len(segnos) == 1

    def test_add_segno_invalid_measure(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="does not exist"):
            ss.add_segno(99)


class TestAddToCoda:
    """Tests for add_to_coda."""

    def test_add_to_coda_inserts_coda_text_variant(self):
        ss = _make_score(measures=4)
        result = ss.add_to_coda(2)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 2)
        marks = list(m.getElementsByClass(m21repeat.Coda))
        assert len(marks) == 1
        assert marks[0].getText() == "To Coda"

    def test_add_to_coda_result_details(self):
        ss = _make_score(measures=4)
        result = ss.add_to_coda(3)
        assert result.details["mark_type"] == "to coda"
        assert result.details["measure"] == 3


class TestAddFine:
    """Tests for add_fine."""

    def test_add_fine_inserts_fine_object(self):
        ss = _make_score(measures=4)
        result = ss.add_fine(2)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 2)
        marks = list(m.getElementsByClass(m21repeat.Fine))
        assert len(marks) == 1

    def test_add_fine_result_details(self):
        ss = _make_score(measures=4)
        result = ss.add_fine(3)
        assert result.details["mark_type"] == "fine"
        assert result.details["measure"] == 3


class TestAddDaCapo:
    """Tests for add_da_capo."""

    def test_dc_plain(self):
        ss = _make_score(measures=4)
        result = ss.add_da_capo(4)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 4)
        marks = [el for el in m.recurse() if isinstance(el, m21repeat.DaCapo)]
        assert len(marks) == 1
        assert type(marks[0]) is m21repeat.DaCapo

    def test_dc_al_fine(self):
        ss = _make_score(measures=4)
        result = ss.add_da_capo(4, al="fine")
        assert result.details["al"] == "fine"
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 4)
        marks = [el for el in m.recurse() if isinstance(el, m21repeat.DaCapoAlFine)]
        assert len(marks) == 1

    def test_dc_al_coda(self):
        ss = _make_score(measures=4)
        result = ss.add_da_capo(4, al="coda")
        assert result.details["al"] == "coda"
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 4)
        marks = [el for el in m.recurse() if isinstance(el, m21repeat.DaCapoAlCoda)]
        assert len(marks) == 1

    def test_dc_invalid_al(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="Invalid 'al' value"):
            ss.add_da_capo(4, al="allegro")


class TestAddDalSegno:
    """Tests for add_dal_segno."""

    def test_ds_plain(self):
        ss = _make_score(measures=4)
        result = ss.add_dal_segno(4)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 4)
        marks = [el for el in m.recurse() if isinstance(el, m21repeat.DalSegno)]
        assert len(marks) == 1
        assert type(marks[0]) is m21repeat.DalSegno

    def test_ds_al_fine(self):
        ss = _make_score(measures=4)
        result = ss.add_dal_segno(4, al="fine")
        assert result.details["al"] == "fine"
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 4)
        marks = [el for el in m.recurse() if isinstance(el, m21repeat.DalSegnoAlFine)]
        assert len(marks) == 1

    def test_ds_al_coda(self):
        ss = _make_score(measures=4)
        result = ss.add_dal_segno(4, al="coda")
        assert result.details["al"] == "coda"
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 4)
        marks = [el for el in m.recurse() if isinstance(el, m21repeat.DalSegnoAlCoda)]
        assert len(marks) == 1

    def test_ds_invalid_al(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="Invalid 'al' value"):
            ss.add_dal_segno(4, al="vivace")


class TestRemoveNavigationMark:
    """Tests for remove_navigation_mark."""

    def test_remove_coda(self):
        ss = _make_score(measures=4)
        ss.add_coda(2)
        result = ss.remove_navigation_mark("coda", 2)
        assert result.success
        assert result.details["removed_count"] == 1
        assert "part" not in result.details
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 2)
        assert len(list(m.getElementsByClass(m21repeat.Coda))) == 0

    def test_remove_segno(self):
        ss = _make_score(measures=4)
        ss.add_segno(3)
        result = ss.remove_navigation_mark("segno", 3)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 3)
        assert len(list(m.getElementsByClass(m21repeat.Segno))) == 0

    def test_remove_to_coda(self):
        ss = _make_score(measures=4)
        ss.add_to_coda(2)
        result = ss.remove_navigation_mark("to coda", 2)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 2)
        marks = list(m.getElementsByClass(m21repeat.Coda))
        assert len(marks) == 0

    def test_remove_to_coda_leaves_destination_coda(self):
        ss = _make_score(measures=4)
        ss.add_coda(2)
        ss.add_to_coda(2)
        result = ss.remove_navigation_mark("to coda", 2)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 2)
        marks = list(m.getElementsByClass(m21repeat.Coda))
        assert [mark.getText() for mark in marks] == ["Coda"]

    def test_remove_coda_leaves_to_coda(self):
        ss = _make_score(measures=4)
        ss.add_coda(2)
        ss.add_to_coda(2)
        result = ss.remove_navigation_mark("coda", 2)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 2)
        marks = list(m.getElementsByClass(m21repeat.Coda))
        assert [mark.getText() for mark in marks] == ["To Coda"]

    def test_remove_fine(self):
        ss = _make_score(measures=4)
        ss.add_fine(2)
        result = ss.remove_navigation_mark("fine", 2)
        assert result.success
        part_obj, _ = ss._resolve_part(None)
        m = ss._resolve_measure(part_obj, 2)
        assert len(list(m.getElementsByClass(m21repeat.Fine))) == 0

    def test_remove_da_capo(self):
        ss = _make_score(measures=4)
        ss.add_da_capo(4, al="fine")
        result = ss.remove_navigation_mark("da capo", 4)
        assert result.success

    def test_remove_dal_segno(self):
        ss = _make_score(measures=4)
        ss.add_dal_segno(4, al="coda")
        result = ss.remove_navigation_mark("dal segno", 4)
        assert result.success

    def test_remove_invalid_mark_type(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="Unknown navigation mark type"):
            ss.remove_navigation_mark("fermata", 1)

    def test_remove_nonexistent_mark(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="No .* mark found"):
            ss.remove_navigation_mark("coda", 1)

    def test_remove_invalid_measure(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError, match="does not exist"):
            ss.remove_navigation_mark("coda", 99)

    def test_remove_case_insensitive(self):
        ss = _make_score(measures=4)
        ss.add_coda(2)
        result = ss.remove_navigation_mark("Coda", 2)
        assert result.success

    def test_remove_navigation_mark_cleans_legacy_part_scoped_marks(self) -> None:
        """Score-wide removal should clean matching marks from every part."""
        ss = _make_score(measures=4, parts=["Violin", "Cello"])
        first_part, _ = ss._resolve_part(0)
        second_part, _ = ss._resolve_part(1)
        first_measure = ss._resolve_measure(first_part, 2)
        second_measure = ss._resolve_measure(second_part, 2)
        first_measure.insert(0, m21repeat.Segno())
        second_measure.insert(0, m21repeat.Segno())

        result = ss.remove_navigation_mark("segno", 2)

        assert result.success
        assert result.details["removed_count"] == 2
        assert len(list(first_measure.getElementsByClass(m21repeat.Segno))) == 0
        assert len(list(second_measure.getElementsByClass(m21repeat.Segno))) == 0

    def test_navigation_methods_no_longer_accept_part_argument(self) -> None:
        """Navigation methods should expose score-wide signatures."""
        ss = _make_score(measures=4, parts=["Violin", "Cello"])

        with pytest.raises(TypeError):
            ss.add_coda(2, part="Cello")
        with pytest.raises(TypeError):
            ss.add_segno(2, part="Cello")
        with pytest.raises(TypeError):
            ss.add_to_coda(2, part="Cello")
        with pytest.raises(TypeError):
            ss.add_fine(2, part="Cello")
        with pytest.raises(TypeError):
            ss.add_da_capo(2, part="Cello")
        with pytest.raises(TypeError):
            ss.add_dal_segno(2, part="Cello")
        with pytest.raises(TypeError):
            ss.remove_navigation_mark("coda", 2, part="Cello")
