"""Unit tests for ScoreSpeak validation helpers."""

import pytest
from music21 import clef as m21clef
from music21 import instrument as m21instrument
from music21 import key as m21key
from music21 import meter as m21meter
from music21 import pitch as m21pitch

from scorespeak.music.validation import (
    default_clef_for_instrument,
    get_key_accidentals,
    make_clef,
    needs_explicit_accidental,
    normalize_duration,
    normalize_pitch,
    validate_beat_capacity,
    validate_pitch_in_range,
)


class TestNormalizePitch:
    """Tests for normalize_pitch."""

    def test_pitch_object_passthrough(self):
        p = m21pitch.Pitch("D5")
        assert normalize_pitch(p) is p

    def test_midi_boundaries(self):
        assert normalize_pitch(0).nameWithOctave == "C-1"
        assert normalize_pitch(127).nameWithOctave == "G9"

    def test_midi_out_of_range_low(self):
        with pytest.raises(ValueError, match="out of range"):
            normalize_pitch(-1)

    def test_midi_out_of_range_high(self):
        with pytest.raises(ValueError, match="out of range"):
            normalize_pitch(128)

    def test_midi_float_truncates_to_int(self):
        p = normalize_pitch(60.9)
        assert p.midi == 60

    def test_unicode_accidentals(self):
        assert normalize_pitch("C♯4").nameWithOctave == "C#4"
        assert normalize_pitch("D♭3").nameWithOctave == "D-3"

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Cannot interpret"):
            normalize_pitch([])  # type: ignore[arg-type]

    def test_unparseable_string(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            normalize_pitch("H4")

    def test_double_flat_string(self):
        p = normalize_pitch("Dbb4")
        assert p.accidental is not None
        assert p.midi == m21pitch.Pitch("C4").midi

    def test_music21_double_flat_string(self):
        p = normalize_pitch("B--3")
        assert p.accidental is not None
        assert p.midi == m21pitch.Pitch("A3").midi

    def test_double_sharp_string(self):
        p = normalize_pitch("F##4")
        assert p.accidental is not None

    def test_mixed_sharp_flat_rejected(self):
        with pytest.raises(ValueError, match="mixed sharp and flat"):
            normalize_pitch("C#b4")


class TestNormalizeDuration:
    """Tests for normalize_duration."""

    def test_zero_quarter_length_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            normalize_duration(0)

    def test_negative_quarter_length_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            normalize_duration(-1.0)

    def test_numeric_alias_string(self):
        d = normalize_duration("4")
        assert abs(d.quarterLength - 1.0) < 1e-9

    def test_whole_via_alias(self):
        d = normalize_duration("1")
        assert abs(d.quarterLength - 4.0) < 1e-9

    def test_float_quarter_length(self):
        d = normalize_duration(1.5)
        assert abs(d.quarterLength - 1.5) < 1e-9

    def test_dots_parameter(self):
        d = normalize_duration("quarter", dots=1)
        assert d.dots == 1
        assert abs(d.quarterLength - 1.5) < 1e-9

    def test_unknown_name(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            normalize_duration("not-a-duration")

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Cannot interpret"):
            normalize_duration(None)  # type: ignore[arg-type]

    def test_zero_string_float_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            normalize_duration("0")


class TestValidateBeatCapacity:
    """Tests for validate_beat_capacity."""

    def test_overflow_at_explicit_beat(self):
        ts = m21meter.TimeSignature("4/4")
        with pytest.raises(ValueError, match="Measure 2"):
            validate_beat_capacity(
                ts,
                0.0,
                2.0,
                measure_number=2,
                beat_position=4.0,
            )

    def test_overflow_cumulative(self):
        ts = m21meter.TimeSignature("3/4")
        with pytest.raises(ValueError, match="already contains"):
            validate_beat_capacity(
                ts,
                2.5,
                1.0,
                measure_number=1,
                beat_position=None,
            )

    def test_exact_fill_allowed(self):
        ts = m21meter.TimeSignature("4/4")
        validate_beat_capacity(
            ts,
            3.0,
            1.0,
            measure_number=1,
            beat_position=None,
        )

    def test_fractional_overflow_message(self):
        ts = m21meter.TimeSignature("4/4")
        with pytest.raises(ValueError, match="0\\.25|0.25"):
            validate_beat_capacity(
                ts,
                3.75,
                0.5,
                measure_number=5,
                beat_position=None,
            )


class TestKeyAccidentals:
    """Tests for key / accidental helpers."""

    def test_c_major_empty_accidentals(self):
        ks = m21key.KeySignature(0)
        assert get_key_accidentals(ks) == {}

    def test_g_major_f_sharp(self):
        ks = m21key.KeySignature(1)
        acc = get_key_accidentals(ks)
        assert acc.get("F") == "sharp"

    def test_bb_major_two_flats(self):
        ks = m21key.KeySignature(-2)
        acc = get_key_accidentals(ks)
        assert acc.get("B") == "flat"
        assert acc.get("E") == "flat"


class TestNeedsExplicitAccidental:
    """Tests for needs_explicit_accidental."""

    def test_natural_in_sharp_key(self):
        ks = m21key.KeySignature(1)  # G major — F#
        f_nat = m21pitch.Pitch("F4")
        f_nat.accidental = m21pitch.Accidental("natural")
        assert needs_explicit_accidental(f_nat, ks) is True

    def test_diatonic_pitch_no_extra(self):
        ks = m21key.KeySignature(0)
        c = m21pitch.Pitch("C4")
        assert needs_explicit_accidental(c, ks) is False

    def test_chromatic_in_c_major(self):
        ks = m21key.KeySignature(0)
        f_sharp = m21pitch.Pitch("F#4")
        assert needs_explicit_accidental(f_sharp, ks) is True


class TestValidatePitchInRange:
    """Tests for validate_pitch_in_range."""

    def test_unknown_instrument_returns_none(self):
        p = m21pitch.Pitch("C4")
        assert validate_pitch_in_range(p, "Ocarina of Time") is None

    def test_soprano_extreme_high_warning(self):
        p = m21pitch.Pitch("C7")
        msg = validate_pitch_in_range(p, "soprano")
        assert msg is not None
        assert "outside" in msg.lower()

    def test_strict_raises(self):
        p = m21pitch.Pitch("C1")
        with pytest.raises(ValueError, match="outside"):
            validate_pitch_in_range(p, "soprano", strict=True)


class TestMakeClef:
    """Tests for make_clef."""

    def test_treble_lowercase(self):
        c = make_clef("treble")
        assert c.sign == "G"

    def test_unknown_clef(self):
        with pytest.raises(ValueError, match="Unknown clef"):
            make_clef("not-a-clef")


class TestDefaultClefForInstrument:
    """Tests for default_clef_for_instrument."""

    @pytest.mark.parametrize(
        ("instrument_obj", "clef_class"),
        [
            (m21instrument.Violin(), m21clef.TrebleClef),
            (m21instrument.Viola(), m21clef.AltoClef),
            (m21instrument.Violoncello(), m21clef.BassClef),
            (m21instrument.Bassoon(), m21clef.BassClef),
            (m21instrument.AcousticGuitar(), m21clef.Treble8vbClef),
            (m21instrument.SnareDrum(), m21clef.PercussionClef),
        ],
    )
    def test_known_instruments_use_convention_table(
        self,
        instrument_obj: m21instrument.Instrument,
        clef_class: type[m21clef.Clef],
    ) -> None:
        """Known music21 instruments use explicit default clef conventions."""
        c = default_clef_for_instrument(instrument_obj)
        assert isinstance(c, clef_class)

    def test_named_generic_instrument_uses_treble_fallback(self) -> None:
        """Generic instruments do not use name-based clef guessing."""
        inst = m21instrument.Instrument()
        inst.partName = "Cello"
        c = default_clef_for_instrument(inst)
        assert isinstance(c, m21clef.TrebleClef)

    def test_violin_treble(self) -> None:
        """Violin defaults to treble clef."""
        inst = m21instrument.Violin()
        c = default_clef_for_instrument(inst)
        assert c.sign == "G"

    def test_generic_instrument_treble_fallback(self) -> None:
        """Unknown generic instruments default to treble clef."""
        inst = m21instrument.Instrument()
        inst.partName = "Theremin"
        c = default_clef_for_instrument(inst)
        assert c.sign == "G"
