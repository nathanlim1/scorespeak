"""Tests for note, rest, chord, tie, tuplet, and grace note operations."""

from xml.etree import ElementTree as ET
from typing import Any

import pytest
from music21 import note as m21note
from music21 import chord as m21chord
from music21 import pitch as m21pitch
from music21 import spanner as m21spanner
from music21 import stream as m21stream
from music21 import tie as m21tie

from scorespeak import ScoreSpeak, TupletInfo
from scorespeak.music.validation import normalize_duration


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_score(measures=4, time_signature="4/4", key_signature="C", parts=None):
    """Create a ScoreSpeak with sensible defaults for testing."""
    kwargs = {
        "measures": measures,
        "time_signature": time_signature,
        "key_signature": key_signature,
    }
    if parts is not None:
        kwargs["parts"] = parts
    return ScoreSpeak.create(**kwargs)


def _remove_one_note(
    score_state: ScoreSpeak,
    *,
    measure: int,
    beat: float,
    part: int | str = 0,
    voice: int = 1,
    pitch: Any = None,
) -> object:
    """Call the public batch removal API for one legacy-style target."""
    item: dict[str, Any] = {"beat": beat}
    if pitch is not None:
        item["pitch"] = pitch
    return score_state.remove_notes(
        measure=measure,
        part=part,
        voice=voice,
        notes=[item],
    )


def _reshape_rest(
    score_state: ScoreSpeak,
    duration: Any = "quarter",
    *,
    measure: int,
    beat: float,
    part: int | str = 0,
    voice: int = 1,
    dots: int = 0,
) -> object:
    """Insert one visible rest spelling over already-silent time."""
    duration_obj = normalize_duration(duration, dots=dots)
    return score_state.reshape_rests(
        measure=measure,
        part=part,
        voice=voice,
        start_beat=beat,
        total_duration=duration_obj.quarterLength,
        rests=[{"duration": duration, "dots": dots}],
    )


def _measure_voice_container(
    score_state: ScoreSpeak,
    *,
    measure: int,
    part: int | str = 0,
    voice: int = 1,
    create: bool = False,
) -> m21stream.Stream:
    """Return the music21 stream used for one measure voice."""
    part_obj, _part_idx = score_state._resolve_part(part)
    measure_obj = score_state._resolve_measure(part_obj, measure)
    return score_state._get_voice_or_measure(
        measure_obj,
        voice,
        create=create,
    )


def _remove_rests_from_container(
    container: m21stream.Stream,
) -> list[m21note.Rest]:
    """Remove all rests from a test stream and return them."""
    removed_rests = list(container.getElementsByClass(m21note.Rest))
    for rest in removed_rests:
        container.remove(rest)
    return removed_rests


def _visible_rest_summary(
    score_state: ScoreSpeak,
    *,
    measure: int = 1,
) -> list[tuple[float, str]]:
    """Return visible rest beats and duration names for one measure."""
    return [
        (event.beat, event.duration_type)
        for event in score_state.get_notes(measure=measure)
        if event.is_rest
    ]


def _exported_note_accidentals(score_state: ScoreSpeak) -> list[str | None]:
    """Return visible accidental tags for exported pitched MusicXML notes."""
    root = ET.fromstring(score_state.to_musicxml_string())
    accidentals: list[str | None] = []
    for note_element in root.findall(".//{*}note"):
        if note_element.find("{*}pitch") is None:
            continue
        accidental = note_element.find("{*}accidental")
        accidentals.append(accidental.text if accidental is not None else None)
    return accidentals


def _exported_note_tuplet_types(score_state: ScoreSpeak) -> list[list[str]]:
    """Return tuplet notation type attributes for exported pitched notes."""
    root = ET.fromstring(score_state.to_musicxml_string())
    tuplet_types: list[list[str]] = []
    for note_element in root.findall(".//{*}note"):
        if note_element.find("{*}pitch") is None:
            continue
        tuplet_types.append([
            tuplet.attrib.get("type", "")
            for tuplet in note_element.findall(".//{*}tuplet")
        ])
    return tuplet_types


def _part_id_for_name(root: ET.Element, part_name: str) -> str:
    """Return the MusicXML part id for a displayed part name."""
    for score_part in root.findall(".//{*}score-part"):
        name_element = score_part.find("{*}part-name")
        if name_element is not None and name_element.text == part_name:
            part_id = score_part.attrib.get("id")
            if part_id is not None:
                return part_id
    raise AssertionError(f"No MusicXML score-part named {part_name!r}.")


def _exported_measure_note_beams(
    score_state: ScoreSpeak,
    measure_number: int,
    part_name: str | None = None,
) -> list[list[str]]:
    """Return beam text values for pitched notes in one exported measure."""
    root = ET.fromstring(score_state.to_musicxml_string())
    if part_name is None:
        part_element = root.find(".//{*}part")
    else:
        part_id = _part_id_for_name(root, part_name)
        part_element = root.find(f".//{{*}}part[@id='{part_id}']")
    if part_element is None:
        raise AssertionError("No MusicXML part found.")

    measure_element = part_element.find(f"{{*}}measure[@number='{measure_number}']")
    if measure_element is None:
        raise AssertionError(f"No MusicXML measure {measure_number}.")

    note_beams: list[list[str]] = []
    for note_element in measure_element.findall("{*}note"):
        if note_element.find("{*}pitch") is None:
            continue
        note_beams.append([
            beam.text or ""
            for beam in note_element.findall("{*}beam")
        ])
    return note_beams


def _exported_measure_grace_note_beams(
    score_state: ScoreSpeak,
    measure_number: int,
    part_name: str | None = None,
) -> list[list[str]]:
    """Return beam text values for pitched grace notes in one measure."""
    root = ET.fromstring(score_state.to_musicxml_string())
    if part_name is None:
        part_element = root.find(".//{*}part")
    else:
        part_id = _part_id_for_name(root, part_name)
        part_element = root.find(f".//{{*}}part[@id='{part_id}']")
    if part_element is None:
        raise AssertionError("No MusicXML part found.")

    measure_element = part_element.find(f"{{*}}measure[@number='{measure_number}']")
    if measure_element is None:
        raise AssertionError(f"No MusicXML measure {measure_number}.")

    note_beams: list[list[str]] = []
    for note_element in measure_element.findall("{*}note"):
        if note_element.find("{*}pitch") is None:
            continue
        if note_element.find("{*}grace") is None:
            continue
        note_beams.append([
            beam.text or ""
            for beam in note_element.findall("{*}beam")
        ])
    return note_beams


def _exported_measure_note_beams_by_voice(
    score_state: ScoreSpeak,
    measure_number: int,
) -> dict[str, list[list[str]]]:
    """Return exported beam text values grouped by MusicXML voice number."""
    root = ET.fromstring(score_state.to_musicxml_string())
    part_element = root.find(".//{*}part")
    if part_element is None:
        raise AssertionError("No MusicXML part found.")
    measure_element = part_element.find(f"{{*}}measure[@number='{measure_number}']")
    if measure_element is None:
        raise AssertionError(f"No MusicXML measure {measure_number}.")

    beams_by_voice: dict[str, list[list[str]]] = {}
    for note_element in measure_element.findall("{*}note"):
        if note_element.find("{*}pitch") is None:
            continue
        voice_element = note_element.find("{*}voice")
        voice = voice_element.text if voice_element is not None else "1"
        beams_by_voice.setdefault(voice or "1", []).append([
            beam.text or ""
            for beam in note_element.findall("{*}beam")
        ])
    return beams_by_voice


def _exported_measure_note_stems(
    score_state: ScoreSpeak,
    measure_number: int,
) -> list[tuple[str, str, str | None]]:
    """Return exported pitched note stems as ``(voice, pitch, stem)`` rows."""
    root = ET.fromstring(score_state.to_musicxml_string())
    part_element = root.find(".//{*}part")
    if part_element is None:
        raise AssertionError("No MusicXML part found.")
    measure_element = part_element.find(f"{{*}}measure[@number='{measure_number}']")
    if measure_element is None:
        raise AssertionError(f"No MusicXML measure {measure_number}.")

    stems: list[tuple[str, str, str | None]] = []
    for note_element in measure_element.findall("{*}note"):
        if note_element.find("{*}pitch") is None:
            continue
        step = note_element.findtext("{*}pitch/{*}step") or "?"
        octave = note_element.findtext("{*}pitch/{*}octave") or "?"
        voice_element = note_element.find("{*}voice")
        voice = voice_element.text if voice_element is not None else "1"
        stem_element = note_element.find("{*}stem")
        stems.append((
            voice or "1",
            f"{step}{octave}",
            stem_element.text if stem_element is not None else None,
        ))
    return stems


# ==================================================================
# add_notes
# ==================================================================

class TestAddNotes:
    """Tests for scoped batch note insertion."""

    def test_add_notes_single_note(self) -> None:
        """add_notes handles a single-item batch."""
        ss = _make_score(parts=["Piano"])

        result = ss.add_notes(
            measure=1,
            part=0,
            voice=1,
            notes=[
                {"pitch": "C4", "beat": 1.0, "duration": "quarter", "dots": 0},
            ],
        )

        assert result.success
        assert result.details["count"] == 1
        assert result.details["measure"] == 1
        assert result.details["part"] == 0
        assert result.details["voice"] == 1
        notes = ss.get_notes(measure=1, part=0, voice=1)
        assert [note.pitch for note in notes if not note.is_rest] == ["C4"]
        assert [note.beat for note in notes if note.is_rest] == [2.0]

    def test_add_notes_multiple_notes_same_scope(self) -> None:
        """add_notes inserts several notes in one measure, part, and voice."""
        ss = _make_score(parts=["Piano"], time_signature="4/4")

        result = ss.add_notes(
            measure=1,
            part=0,
            voice=1,
            notes=[
                {"pitch": "C4", "beat": 1.0, "duration": "quarter", "dots": 0},
                {"pitch": "D4", "beat": 2.0, "duration": "quarter", "dots": 0},
                {"pitch": "E4", "beat": 3.0, "duration": "quarter", "dots": 0},
            ],
        )

        assert result.details["count"] == 3
        notes = ss.get_notes(measure=1, part=0, voice=1)
        sounding = [note for note in notes if not note.is_rest]
        assert [note.pitch for note in sounding] == ["C4", "D4", "E4"]
        assert [note.beat for note in sounding] == [1.0, 2.0, 3.0]
        assert [note.beat for note in notes if note.is_rest] == [4.0]

    def test_add_notes_rejects_scoped_item_fields(self) -> None:
        """Per-note scope fields are rejected."""
        ss = _make_score(parts=["Piano"])

        with pytest.raises(ValueError, match="may not include measure"):
            ss.add_notes(
                measure=1,
                part=0,
                voice=1,
                notes=[
                    {
                        "pitch": "C4",
                        "beat": 1.0,
                        "duration": "quarter",
                        "dots": 0,
                        "measure": 2,
                    },
                ],
            )

    @pytest.mark.parametrize("missing", ["pitch", "beat", "duration", "dots"])
    def test_add_notes_requires_each_note_field(self, missing: str) -> None:
        """Every note item requires pitch, beat, duration, and dots."""
        ss = _make_score(parts=["Piano"])
        item = {
            "pitch": "C4",
            "beat": 1.0,
            "duration": "quarter",
            "dots": 0,
        }
        del item[missing]

        with pytest.raises(ValueError, match=missing):
            ss.add_notes(measure=1, part=0, voice=1, notes=[item])

    def test_add_notes_rejects_extra_item_fields(self) -> None:
        """Unsupported per-note keys are rejected."""
        ss = _make_score(parts=["Piano"])

        with pytest.raises(ValueError, match="unsupported field"):
            ss.add_notes(
                measure=1,
                part=0,
                voice=1,
                notes=[
                    {
                        "pitch": "C4",
                        "beat": 1.0,
                        "duration": "quarter",
                        "dots": 0,
                        "velocity": 90,
                    },
                ],
            )

    def test_add_notes_rejects_invalid_dots(self) -> None:
        """Dots must be an explicit non-negative integer."""
        ss = _make_score(parts=["Piano"])

        with pytest.raises(ValueError, match="dots must be"):
            ss.add_notes(
                measure=1,
                part=0,
                voice=1,
                notes=[
                    {
                        "pitch": "C4",
                        "beat": 1.0,
                        "duration": "quarter",
                        "dots": -1,
                    },
                ],
            )

    def test_add_notes_rejects_tuplet_fraction_duration(self) -> None:
        """Raw tuplet-like quarter lengths must go through add_tuplet."""
        ss = _make_score(parts=["Piano"])

        with pytest.raises(ValueError, match="Use add_tuplet"):
            ss.add_notes(
                measure=1,
                part=0,
                voice=1,
                notes=[
                    {
                        "pitch": "C4",
                        "beat": 1.0,
                        "duration": 1 / 3,
                        "dots": 0,
                    },
                ],
            )

    def test_add_notes_accepts_standard_numeric_duration(self) -> None:
        """Ordinary numeric quarter lengths remain allowed."""
        ss = _make_score(parts=["Piano"])

        result = ss.add_notes(
            measure=1,
            part=0,
            voice=1,
            notes=[
                {"pitch": "C4", "beat": 1.0, "duration": 0.5, "dots": 0},
            ],
        )

        assert result.success

    def test_add_notes_is_atomic_when_later_note_fails(self) -> None:
        """A failed later note leaves the score unchanged."""
        ss = _make_score(parts=["Piano"], time_signature="4/4")
        ss._add_note_one("G4", "quarter", measure=1, beat=1, part=0)

        with pytest.raises(ValueError, match=r"notes\[1\]"):
            ss.add_notes(
                measure=1,
                part=0,
                voice=1,
                notes=[
                    {"pitch": "C4", "beat": 2.0, "duration": "quarter", "dots": 0},
                    {"pitch": "D4", "beat": 1.0, "duration": "quarter", "dots": 0},
                ],
            )

        notes = ss.get_notes(measure=1, part=0, voice=1)
        assert [note.pitch for note in notes if not note.is_rest] == ["G4"]

    def test_add_notes_rollback_preserves_overwritten_rest(self) -> None:
        """A failed batch restores rests overwritten by earlier items."""
        ss = _make_score(parts=["Piano"], time_signature="4/4")
        _reshape_rest(ss, "half", measure=1, beat=1, part=0, voice=1)
        ss._add_note_one("G4", "quarter", measure=1, beat=3, part=0, voice=1)

        with pytest.raises(ValueError, match=r"notes\[1\]"):
            ss.add_notes(
                measure=1,
                part=0,
                voice=1,
                notes=[
                    {"pitch": "C4", "beat": 1.0, "duration": "quarter", "dots": 0},
                    {"pitch": "D4", "beat": 3.0, "duration": "quarter", "dots": 0},
                ],
            )

        notes = ss.get_notes(measure=1, part=0, voice=1)
        assert [
            (note.pitch, note.is_rest, note.beat, note.duration_type)
            for note in notes
        ] == [
            ("rest", True, 1.0, "half"),
            ("G4", False, 3.0, "quarter"),
            ("rest", True, 4.0, "quarter"),
        ]

    def test_add_notes_coalesces_same_call_chord(self) -> None:
        """Same-call same-beat notes become one chord event."""
        ss = _make_score(parts=["Piano"], time_signature="4/4")

        result = ss.add_notes(
            measure=1,
            part=0,
            voice=1,
            notes=[
                {"pitch": "C4", "beat": 1.0, "duration": "quarter", "dots": 0},
                {"pitch": "E4", "beat": 1.0, "duration": "quarter", "dots": 0},
                {"pitch": "G4", "beat": 2.0, "duration": "quarter", "dots": 0},
            ],
        )

        assert result.details["count"] == 3
        events = ss.score.parts[0].measure(1).recurse().getElementsByClass(
            m21chord.Chord,
        )
        assert len(list(events)) == 1
        notes = ss.get_notes(measure=1, part=0, voice=1)
        assert [(note.pitch, note.beat, note.is_chord) for note in notes if not note.is_rest] == [
            ("C4", 1.0, True),
            ("E4", 1.0, True),
            ("G4", 2.0, False),
        ]


# ==================================================================
# _add_note_one
# ==================================================================

class TestAddNoteOne:
    """Tests for the private single-note insertion helper."""

    def test_add_note_basic(self):
        ss = _make_score()
        result = ss._add_note_one("C4", "quarter", measure=1, beat=1)
        assert result.success
        assert result.details["pitch"] == "C4"
        assert result.details["measure"] == 1
        assert result.details["beat"] == 1.0

    def test_add_note_default_measure_is_last(self):
        ss = _make_score(measures=3)
        result = ss._add_note_one("D4")
        assert result.success
        assert result.details["measure"] == 3

    def test_add_note_default_beat_is_next_available(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        result = ss._add_note_one("D4", "quarter", measure=1)
        assert result.success
        assert result.details["beat"] == 2.0

    def test_add_two_notes_sequential(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=2)
        notes = ss.get_notes(measure=1)
        pitches = [n.pitch for n in notes]
        assert "C4" in pitches
        assert "D4" in pitches

    def test_add_note_fills_measure(self):
        ss = _make_score(time_signature="4/4")
        for i, pitch in enumerate(["C4", "D4", "E4", "F4"]):
            ss._add_note_one(pitch, "quarter", measure=1, beat=i + 1)
        notes = ss.get_notes(measure=1)
        assert len(notes) == 4

    def test_add_note_half_note(self):
        ss = _make_score()
        result = ss._add_note_one("E4", "half", measure=1, beat=1)
        assert result.details["quarter_length"] == 2.0

    def test_add_note_whole_note(self):
        ss = _make_score()
        result = ss._add_note_one("G4", "whole", measure=1, beat=1)
        assert result.details["quarter_length"] == 4.0

    def test_add_note_to_specific_part_by_index(self):
        ss = _make_score(parts=["violin", "cello"])
        result = ss._add_note_one("C4", measure=1, beat=1, part=1)
        assert result.success
        assert result.details["part"] == 1

    def test_add_note_to_specific_part_by_name(self):
        ss = _make_score(parts=["violin", "cello"])
        result = ss._add_note_one("C4", measure=1, beat=1, part="Cello")
        assert result.success
        assert result.details["part"] == 1

    def test_add_note_returns_warning_for_out_of_range(self):
        ss = _make_score(parts=["violin"])
        result = ss._add_note_one("C2", measure=1, beat=1)
        assert result.success
        assert result.details["warning"] is not None
        assert "range" in result.details["warning"].lower()

    def test_add_note_no_warning_for_in_range(self):
        ss = _make_score(parts=["piano"])
        result = ss._add_note_one("C4", measure=1, beat=1)
        assert result.details["warning"] is None


# ==================================================================
# Flexible pitch formats
# ==================================================================

class TestFlexiblePitch:
    """Pitch can be a string, MIDI int, unicode accidentals, etc."""

    def test_pitch_string_standard(self):
        ss = _make_score()
        result = ss._add_note_one("C4", measure=1, beat=1)
        assert result.details["pitch"] == "C4"

    def test_pitch_string_sharp(self):
        ss = _make_score()
        result = ss._add_note_one("C#4", measure=1, beat=1)
        assert result.details["pitch"] == "C#4"

    def test_pitch_string_lowercase(self):
        ss = _make_score()
        result = ss._add_note_one("c#4", measure=1, beat=1)
        assert result.details["pitch"] == "C#4"

    def test_pitch_string_flat(self):
        ss = _make_score()
        result = ss._add_note_one("Bb3", measure=1, beat=1)
        assert result.details["pitch"] == "B-3"

    def test_pitch_unicode_sharp(self):
        ss = _make_score()
        result = ss._add_note_one("C\u266f4", measure=1, beat=1)
        assert result.details["pitch"] == "C#4"

    def test_pitch_midi_integer(self):
        ss = _make_score()
        result = ss._add_note_one(60, measure=1, beat=1)
        assert result.success
        assert "C" in result.details["pitch"]

    def test_pitch_music21_object(self):
        ss = _make_score()
        p = m21pitch.Pitch("E5")
        result = ss._add_note_one(p, measure=1, beat=1)
        assert result.details["pitch"] == "E5"


# ==================================================================
# Accidental display
# ==================================================================

class TestAccidentalDisplay:
    """Tests for exported accidental glyph decisions."""

    def test_key_signature_flat_does_not_show_extra_accidental(self) -> None:
        """In-key flats should not export visible accidental tags."""
        ss = _make_score(key_signature="Bb")
        ss._add_note_one("Bb4", measure=1, beat=1)

        assert _exported_note_accidentals(ss) == [None]

    def test_natural_against_flat_key_shows_natural(self) -> None:
        """Natural pitches that cancel a flat key signature should be visible."""
        ss = _make_score(key_signature="Bb")
        ss._add_note_one("B4", measure=1, beat=1)

        assert _exported_note_accidentals(ss) == ["natural"]

    def test_return_to_flat_key_after_natural_shows_flat(self) -> None:
        """Returning to the key-signature flat after a natural should be visible."""
        ss = _make_score(key_signature="Bb")
        ss._add_note_one("B4", measure=1, beat=1)
        ss._add_note_one("Bb4", measure=1, beat=2)

        assert _exported_note_accidentals(ss) == ["natural", "flat"]

    def test_repeated_chromatic_flat_shows_only_once(self) -> None:
        """Repeated altered pitches in one measure should not repeat accidentals."""
        ss = _make_score(key_signature="C")
        ss._add_note_one("Bb4", measure=1, beat=1)
        ss._add_note_one("Bb4", measure=1, beat=2)

        assert _exported_note_accidentals(ss) == ["flat", None]

    def test_non_immediate_repeated_chromatic_flat_shows_only_once(self) -> None:
        """A repeated altered pitch should stay active across intervening notes."""
        ss = _make_score(key_signature="C")
        ss._add_note_one("Bb4", measure=1, beat=1)
        ss._add_note_one("C4", measure=1, beat=2)
        ss._add_note_one("Bb4", measure=1, beat=3)

        assert _exported_note_accidentals(ss) == ["flat", None, None]

    def test_natural_after_chromatic_sharp_shows_natural(self) -> None:
        """A natural that cancels a same-measure sharp should be visible."""
        ss = _make_score(key_signature="C")
        ss._add_note_one("F#4", measure=1, beat=1)
        ss._add_note_one("F4", measure=1, beat=2)

        assert _exported_note_accidentals(ss) == ["sharp", "natural"]

    def test_key_signature_sharp_does_not_show_extra_accidental(self) -> None:
        """In-key sharps should not export visible accidental tags."""
        ss = _make_score(key_signature="G")
        ss._add_note_one("F#4", measure=1, beat=1)

        assert _exported_note_accidentals(ss) == [None]

    def test_replace_note_refreshes_measure_accidentals(self) -> None:
        """Replacing a pitch should recalculate same-measure accidental state."""
        ss = _make_score(key_signature="C")
        ss._add_note_one("F#4", measure=1, beat=1)
        ss._add_note_one("G4", measure=1, beat=2)

        ss.replace_note(measure=1, beat=2, new_pitch="F4")

        assert _exported_note_accidentals(ss) == ["sharp", "natural"]

    def test_tuplet_add_refreshes_measure_accidentals(self) -> None:
        """Tuplet insertion should apply normal accidental-display rules."""
        ss = _make_score(key_signature="C")
        ss.add_tuplet(
            [("F#4", "eighth"), ("F4", "eighth"), ("G4", "eighth")],
            actual_notes=3,
            normal_notes=2,
            measure=1,
            beat=1,
        )

        assert _exported_note_accidentals(ss) == ["sharp", "natural", None]

    def test_remove_notes_refreshes_remaining_accidentals(self) -> None:
        """Removing an accidental should make the remaining first occurrence visible."""
        ss = _make_score(key_signature="C")
        ss._add_note_one("Bb4", measure=1, beat=1)
        ss._add_note_one("Bb4", measure=1, beat=2)

        _remove_one_note(ss, measure=1, beat=1)

        assert _exported_note_accidentals(ss) == ["flat"]

    def test_remove_chord_pitch_refreshes_remaining_accidentals(self) -> None:
        """Removing a chord pitch should refresh later notes in the measure."""
        ss = _make_score(key_signature="C")
        ss.add_chord(["Bb4", "D5"], measure=1, beat=1)
        ss._add_note_one("Bb4", measure=1, beat=2)

        _remove_one_note(ss, measure=1, beat=1, pitch="Bb4")

        assert _exported_note_accidentals(ss) == [None, "flat"]

    def test_tied_continuation_omits_repeated_accidental(self) -> None:
        """A tied continuation over a barline should not repeat its accidental."""
        ss = _make_score(key_signature="C", measures=2)
        ss._add_note_one("F#4", "whole", measure=1, beat=1)
        ss._add_note_one("F#4", "whole", measure=2, beat=1)
        ss.add_tie(measure=1, beat=1)

        assert _exported_note_accidentals(ss) == ["sharp", None]

    def test_untied_note_after_previous_tie_start_shows_accidental(self) -> None:
        """A next-measure note without a tie stop should show its accidental."""
        ss = _make_score(key_signature="C", measures=2)
        ss._add_note_one("F#4", "whole", measure=1, beat=1)
        part_obj = list(ss.score.parts)[0]
        measure_obj = ss._resolve_measure(part_obj, 1)
        first_note = ss._find_element_at_offset(measure_obj, 0.0)
        assert first_note is not None
        first_note.tie = m21tie.Tie("start")
        ss._refresh_measure_and_next_accidentals(part_obj, 1)
        ss._add_note_one("F#4", "whole", measure=2, beat=1)

        assert _exported_note_accidentals(ss) == ["sharp", "sharp"]

    def test_single_note_edit_refreshes_only_changed_measure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Single-note insertion should only refresh the edited measure."""
        ss = _make_score(measures=3)
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

        ss._add_note_one("C4", measure=2, beat=1)

        assert calls == [2]


# ==================================================================
# Flexible duration formats
# ==================================================================

class TestFlexibleDuration:
    """Duration can be a name, numeric alias, or float QL."""

    def test_duration_name_quarter(self):
        ss = _make_score()
        result = ss._add_note_one("C4", "quarter", measure=1, beat=1)
        assert result.details["duration"] == "quarter"

    def test_duration_name_eighth(self):
        ss = _make_score()
        result = ss._add_note_one("C4", "eighth", measure=1, beat=1)
        assert result.details["quarter_length"] == 0.5

    def test_duration_alias_8(self):
        ss = _make_score()
        result = ss._add_note_one("C4", "8", measure=1, beat=1)
        assert result.details["quarter_length"] == 0.5

    def test_duration_alias_16(self):
        ss = _make_score()
        result = ss._add_note_one("C4", "16", measure=1, beat=1)
        assert result.details["quarter_length"] == 0.25

    def test_duration_float_quarter_length(self):
        ss = _make_score()
        result = ss._add_note_one("C4", 1.0, measure=1, beat=1)
        assert result.details["quarter_length"] == 1.0

    def test_duration_float_half(self):
        ss = _make_score()
        result = ss._add_note_one("C4", 2.0, measure=1, beat=1)
        assert result.details["quarter_length"] == 2.0


# ==================================================================
# Dotted notes
# ==================================================================

class TestDottedNotes:
    """Augmentation dots increase duration by half."""

    def test_dotted_quarter(self):
        ss = _make_score()
        result = ss._add_note_one("C4", "quarter", measure=1, beat=1, dots=1)
        assert result.success
        assert result.details["quarter_length"] == 1.5
        assert result.details["dots"] == 1

    def test_double_dotted_half(self):
        ss = _make_score()
        result = ss._add_note_one("C4", "half", measure=1, beat=1, dots=2)
        assert result.success
        assert result.details["quarter_length"] == 3.5
        assert result.details["dots"] == 2

    def test_dotted_rest(self):
        ss = _make_score()
        result = _reshape_rest(ss, "quarter", measure=1, beat=1, dots=1)
        assert result.success
        assert result.details["inserted_rests"][0]["quarter_length"] == 1.5


# ==================================================================
# Beat capacity validation
# ==================================================================

class TestBeatCapacity:
    """Adding notes that exceed the measure's capacity should fail."""

    def test_overflow_at_explicit_beat(self):
        ss = _make_score(time_signature="4/4")
        with pytest.raises(ValueError, match="exceed"):
            ss._add_note_one("C4", "whole", measure=1, beat=2)

    def test_overflow_auto_beat(self):
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C4", "whole", measure=1, beat=1)
        with pytest.raises(ValueError, match="exceed"):
            ss._add_note_one("D4", "quarter", measure=1)

    def test_exact_capacity_ok(self):
        ss = _make_score(time_signature="3/4")
        ss._add_note_one("C4", "half", measure=1, beat=1)
        result = ss._add_note_one("D4", "quarter", measure=1, beat=3)
        assert result.success

    def test_beat_below_one_rejected(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="at least 1"):
            ss._add_note_one("C4", beat=0.5, measure=1)


# ==================================================================
# rhythm integrity diagnostics and visible rest completion
# ==================================================================

class TestRhythmIntegrity:
    """Tests for surgical note operations and visible rest completion."""

    def test_add_note_auto_completes_visible_rests(self):
        ss = _make_score(time_signature="4/4")

        result = ss._add_note_one("C4", "quarter", measure=1, beat=1)

        integrity = result.details["measure_integrity"]
        assert integrity["capacity_quarter_length"] == 4.0
        assert integrity["is_complete"] is True
        assert integrity["gaps"] == []
        assert "suggested_rests" not in result.details
        assert "implied_rests" not in result.details
        assert "suggested_rests" not in integrity
        assert "implied_rests" not in integrity
        assert result.details["auto_completed_rests"] == [
            {
                "kind": "rest",
                "duration": "half",
                "quarter_length": 3.0,
                "dots": 1,
                "label": "half rest",
                "measure": 1,
                "part": 0,
                "voice": 1,
                "beat": 2.0,
                "offset": 1.0,
                "end": 4.0,
                "visibility": "visible",
            }
        ]
        assert result.details["repair_hint"] is None

    def test_add_note_overwrites_visible_rest_spelling(self) -> None:
        """Note entry replaces visible rest spelling at the target beat."""
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        _reshape_rest(ss, "half", measure=1, beat=2, dots=1)

        result = ss._add_note_one("D4", "quarter", measure=1, beat=2)

        assert result.details["replaced_rests"] == [
            {
                "kind": "rest",
                "beat": 2.0,
                "offset": 1.0,
                "end": 4.0,
                "duration": "half",
                "quarter_length": 3.0,
                "dots": 1,
                "label": "half rest",
                "measure": 1,
                "part": 0,
                "voice": 1,
                "visibility": "visible",
            }
        ]
        assert result.details["auto_completed_rests"] == [
            {
                "kind": "rest",
                "duration": "half",
                "quarter_length": 2.0,
                "dots": 0,
                "label": "half rest",
                "measure": 1,
                "part": 0,
                "voice": 1,
                "beat": 3.0,
                "offset": 2.0,
                "end": 4.0,
                "visibility": "visible",
            }
        ]
        notes = ss.get_notes(measure=1)
        assert [(note.pitch, note.is_rest, note.beat) for note in notes] == [
            ("C4", False, 1.0),
            ("D4", False, 2.0),
            ("rest", True, 3.0),
        ]

    def test_add_note_inside_longer_rest_auto_fills_remaining_space(self) -> None:
        """Replacing a longer rest materializes visible rests on both sides."""
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        _reshape_rest(ss, "half", measure=1, beat=2, dots=1)

        result = ss._add_note_one("D4", "quarter", measure=1, beat=3)

        assert result.details["replaced_rests"][0]["beat"] == 2.0
        assert result.details["replaced_rests"][0]["quarter_length"] == 3.0
        assert result.details["measure_integrity"]["gaps"] == []
        assert [
            rest["beat"]
            for rest in result.details["auto_completed_rests"]
        ] == [2.0, 4.0]

    def test_add_note_rejects_occupied_beat_without_mutation(self):
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C4", "half", measure=1, beat=1)

        with pytest.raises(ValueError, match="overlap existing") as exc_info:
            ss._add_note_one("D4", "quarter", measure=1, beat=2)

        message = str(exc_info.value)
        assert "voice=2 instead of voice=1" in message
        assert "add_notes" in message
        assert "replace_note or remove_notes" in message
        notes = ss.get_notes(measure=1)
        assert [(note.pitch, note.duration_type) for note in notes if not note.is_rest] == [
            ("C4", "half")
        ]

    def test_add_note_overlap_suggests_chord_for_same_rhythm(self):
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C4", "quarter", measure=1, beat=1)

        with pytest.raises(ValueError, match="add_chord_tones") as exc_info:
            ss._add_note_one("E4", "quarter", measure=1, beat=1)

        message = str(exc_info.value)
        assert "voice=2 instead of voice=1" in message
        assert "add_chord_tones" in message

    def test_reshape_rest_and_chord_reject_overlapping_ranges(self):
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C4", "half", measure=1, beat=1)

        with pytest.raises(ValueError, match="overlaps existing"):
            _reshape_rest(ss, "quarter", measure=1, beat=2)
        with pytest.raises(ValueError, match="overlap existing"):
            ss.add_chord(["E4", "G4"], "quarter", measure=1, beat=2)

    def test_add_chord_overwrites_rest(self) -> None:
        """Chord entry mirrors note-entry rest replacement."""
        ss = _make_score(time_signature="4/4")
        _reshape_rest(ss, "quarter", measure=1, beat=1)

        result = ss.add_chord(["C4", "E4"], "quarter", measure=1, beat=1)

        assert result.details["replaced_rests"][0]["beat"] == 1.0
        assert result.details["replaced_rests"][0]["duration"] == "quarter"
        notes = ss.get_notes(measure=1)
        assert [(note.pitch, note.is_rest, note.beat) for note in notes] == [
            ("C4", False, 1.0),
            ("E4", False, 1.0),
            ("rest", True, 2.0),
        ]

    def test_replace_duration_rejects_extension_into_later_event(self):
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=2)

        with pytest.raises(ValueError, match="overlap existing"):
            ss.replace_note(measure=1, beat=1, new_duration="half")

        notes = ss.get_notes(measure=1)
        assert [(note.pitch, note.duration_type) for note in notes if not note.is_rest] == [
            ("C4", "quarter"),
            ("D4", "quarter"),
        ]

    def test_shortening_duration_reports_one_beat_gap(self):
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C4", "half", measure=1, beat=1)
        ss._add_note_one("D4", "half", measure=1, beat=3)

        result = ss.replace_note(measure=1, beat=1, new_duration="quarter")

        assert result.details["measure_integrity"]["gaps"] == []
        assert result.details["auto_completed_rests"][0]["duration"] == "quarter"

    def test_offbeat_gap_uses_reverse_greedy_rest_spelling(self) -> None:
        """Offbeat trailing silence lands its largest rest on the next beat."""
        ss = _make_score(measures=1, time_signature="2/4")

        for beat, pitch in [
            (1.0, "C4"),
            (1.25, "D4"),
            (1.5, "E4"),
        ]:
            result = ss._add_note_one(pitch, "16th", measure=1, beat=beat)

        assert [
            (rest["duration"], rest["beat"])
            for rest in result.details["auto_completed_rests"]
        ] == [
            ("16th", 1.75),
            ("quarter", 2.0),
        ]
        rests = [note for note in ss.get_notes(measure=1) if note.is_rest]
        assert [(rest.duration_type, rest.beat) for rest in rests] == [
            ("16th", 1.75),
            ("quarter", 2.0),
        ]

    def test_start_aligned_gap_keeps_forward_greedy_rest_spelling(self) -> None:
        """Start-aligned silence keeps larger rests at the gap start."""
        ss = _make_score(measures=1, time_signature="9/16")

        result = ss._add_note_one("C4", "quarter", measure=1, beat=2.25)

        assert [
            (rest["duration"], rest["beat"])
            for rest in result.details["auto_completed_rests"]
        ] == [
            ("quarter", 1.0),
            ("16th", 2.0),
        ]

    def test_hidden_rests_cover_time_during_auto_completion(self) -> None:
        """Hidden suppression remains coverage while visible gaps are filled."""
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C4", "half", measure=1, beat=1)
        ss.remove_rests(measure=1, part=0, voice=1, beat=3)

        result = ss.replace_note(measure=1, beat=1, new_duration="quarter")

        assert result.details["measure_integrity"]["gaps"] == []
        assert [
            (rest["duration"], rest["beat"], rest["visibility"])
            for rest in result.details["auto_completed_rests"]
        ] == [("quarter", 2.0, "visible")]

    def test_added_sixteenth_run_exports_beams(self) -> None:
        """Short notes added with public tools export with beam tags."""
        ss = _make_score(measures=1, time_signature="2/4")
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        for beat, pitch in [
            (2.0, "C4"),
            (2.25, "D4"),
            (2.5, "E4"),
            (2.75, "F4"),
        ]:
            ss._add_note_one(pitch, "16th", measure=1, beat=beat)

        beams = _exported_measure_note_beams(ss, 1)

        assert beams[0] == []
        assert beams[1:] == [
            ["begin", "begin"],
            ["continue", "end"],
            ["continue", "begin"],
            ["end", "end"],
        ]

    def test_imported_score_public_fill_exports_beams(self) -> None:
        """Inserted notes in imported scores get beams despite score stream status."""
        ss = ScoreSpeak.from_musicxml(
            "datasets/scores/beethoven_op125_m1_full_bars001-200.musicxml"
        )
        ss._add_note_one("C5", "quarter", measure=194, beat=1.0, part="Flute 2")
        for beat, pitch in [
            (2.0, "C5"),
            (2.25, "Eb5"),
            (2.5, "D5"),
            (2.75, "C5"),
        ]:
            ss._add_note_one(pitch, "16th", measure=194, beat=beat, part="Flute 2")

        beams = _exported_measure_note_beams(ss, 194, part_name="Flute 2")

        assert beams[0] == []
        assert beams[1:] == [
            ["begin", "begin"],
            ["continue", "end"],
            ["continue", "begin"],
            ["end", "end"],
        ]

    def test_replace_duration_refreshes_stale_beams(self) -> None:
        """Changing duration clears beams that no longer apply to an event."""
        ss = _make_score(measures=1, time_signature="2/4")
        for beat, pitch in [
            (1.0, "C4"),
            (1.25, "D4"),
            (1.5, "E4"),
            (1.75, "F4"),
        ]:
            ss._add_note_one(pitch, "16th", measure=1, beat=beat)

        ss.replace_note(measure=1, beat=1.75, new_duration="quarter")
        beams = _exported_measure_note_beams(ss, 1)

        assert beams[-1] == []

    def test_multi_voice_short_notes_export_beams_per_voice(self) -> None:
        """Automatic beaming is applied independently inside each voice."""
        ss = _make_score(measures=1, time_signature="2/4")
        voice_rows = [
            (1, ["C5", "D5", "E5", "F5"]),
            (2, ["C4", "D4", "E4", "F4"]),
        ]
        for voice, pitches in voice_rows:
            for beat, pitch in zip([1.0, 1.25, 1.5, 1.75], pitches):
                ss._add_note_one(pitch, "16th", measure=1, beat=beat, voice=voice)

        beams_by_voice = _exported_measure_note_beams_by_voice(ss, 1)

        assert beams_by_voice["1"] == [
            ["begin", "begin"],
            ["continue", "end"],
            ["continue", "begin"],
            ["end", "end"],
        ]
        assert beams_by_voice["2"] == [
            ["begin", "begin"],
            ["continue", "end"],
            ["continue", "begin"],
            ["end", "end"],
        ]

    def test_remove_notes_reports_resulting_gap(self):
        ss = _make_score(time_signature="4/4")
        for beat, pitch in enumerate(["C4", "D4", "E4", "F4"], start=1):
            ss._add_note_one(pitch, "quarter", measure=1, beat=beat)

        result = _remove_one_note(ss, measure=1, beat=2)

        assert result.details["measure_integrity"]["gaps"] == []
        assert result.details["auto_completed_rests"][0]["beat"] == 2.0

    def test_reshape_rests_fills_gap_after_duration_change(self):
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C4", "half", measure=1, beat=1)
        ss._add_note_one("D4", "half", measure=1, beat=3)
        ss.replace_note(measure=1, beat=1, new_duration="quarter")

        result = _reshape_rest(ss, "quarter", measure=1, beat=2)

        assert result.details["inserted_rests"] == [
            {
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
            }
        ]
        assert result.details["measure_integrity"]["is_complete"] is True
        rests = [note for note in ss.get_notes(measure=1) if note.is_rest]
        assert [(rest.beat, rest.duration_type) for rest in rests] == [
            (2.0, "quarter")
        ]

    def test_reshape_rests_rejects_overlaps_and_overfull(self):
        ss_overlap = _make_score(time_signature="4/4")
        part_obj, _part_idx = ss_overlap._resolve_part(None)
        measure_obj = ss_overlap._resolve_measure(part_obj, 1)
        measure_obj.insert(0.0, m21note.Note("C4", type="quarter"))
        measure_obj.insert(0.5, m21note.Note("D4", type="quarter"))

        with pytest.raises(ValueError, match="overlaps existing"):
            _reshape_rest(ss_overlap, "quarter", measure=1, beat=1)

        ss_overfull = _make_score(time_signature="4/4")

        with pytest.raises(ValueError, match="exceed"):
            _reshape_rest(ss_overfull, "whole", measure=1, beat=2)

    def test_reshape_rests_only_affects_requested_voice(self):
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C5", "quarter", measure=1, beat=1, voice=1)
        ss._add_note_one("C4", "whole", measure=1, beat=1, voice=2)

        _reshape_rest(ss, "half", measure=1, beat=2, voice=1, dots=1)

        voice_1 = ss.get_notes(measure=1, voice=1)
        voice_2 = ss.get_notes(measure=1, voice=2)
        assert [(note.pitch, note.is_rest, note.beat) for note in voice_1] == [
            ("C5", False, 1.0),
            ("rest", True, 2.0),
        ]
        assert [(note.pitch, note.duration_type, note.beat) for note in voice_2] == [
            ("C4", "whole", 1.0)
        ]


# ==================================================================
# reshape_rests / remove_rests
# ==================================================================

class TestRestSpelling:
    """Tests for visible rest spelling and hiding."""

    def test_reshape_rest_basic(self):
        ss = _make_score()
        result = _reshape_rest(ss, "quarter", measure=1, beat=1)
        assert result.success
        assert result.details["inserted_rests"][0]["duration"] == "quarter"

    def test_reshape_rests_splits_whole_silent_range(self):
        ss = _make_score(time_signature="4/4")

        result = ss.reshape_rests(
            measure=1,
            part=0,
            voice=1,
            start_beat=1.0,
            total_duration="whole",
            rests=[
                {"duration": "quarter"},
                {"duration": "quarter"},
                {"duration": "quarter"},
                {"duration": "quarter"},
            ],
        )

        assert result.success
        notes = ss.get_notes(measure=1)
        assert [
            (note.pitch, note.is_rest, note.beat, note.quarter_length)
            for note in notes
        ] == [
            ("rest", True, 1.0, 1.0),
            ("rest", True, 2.0, 1.0),
            ("rest", True, 3.0, 1.0),
            ("rest", True, 4.0, 1.0),
        ]

    def test_reshape_rests_merges_multiple_rests(self):
        ss = _make_score()
        _reshape_rest(ss, "quarter", measure=1, beat=1)
        _reshape_rest(ss, "quarter", measure=1, beat=2)

        result = ss.reshape_rests(
            measure=1,
            part=0,
            voice=1,
            start_beat=1.0,
            total_duration="half",
            rests=[{"duration": "half"}],
        )

        assert result.details["removed_rests"]
        rests = [note for note in ss.get_notes(measure=1) if note.is_rest]
        assert [(rest.beat, rest.duration_type) for rest in rests] == [
            (1.0, "half"),
            (3.0, "half"),
        ]

    def test_remove_rests_hides_one_rest_by_beat(self):
        ss = _make_score()
        _reshape_rest(ss, "quarter", measure=1, beat=1)
        _reshape_rest(ss, "quarter", measure=1, beat=2)

        result = ss.remove_rests(measure=1, part=0, voice=1, beat=2)

        assert result.details["count"] == 1
        assert result.details["hidden_rests"][0]["visibility"] == "hidden"
        assert result.details["measure_integrity"]["gaps"] == []
        rests = [note for note in ss.get_notes(measure=1) if note.is_rest]
        assert [(rest.beat, rest.duration_type) for rest in rests] == [
            (1.0, "quarter"),
            (3.0, "half"),
        ]

    def test_remove_rests_hides_all_rests_in_voice(self):
        ss = _make_score()
        _reshape_rest(ss, "quarter", measure=1, beat=1)
        _reshape_rest(ss, "quarter", measure=1, beat=2)

        result = ss.remove_rests(measure=1, part=0, voice=1)

        assert result.details["count"] == 3
        assert all(rest["visibility"] == "hidden" for rest in result.details["hidden_rests"])
        assert result.details["measure_integrity"]["gaps"] == []
        assert [note for note in ss.get_notes(measure=1) if note.is_rest] == []

    def test_remove_rests_rejects_missing_rest(self):
        ss = _make_score()
        ss.remove_rests(measure=1, part=0, voice=1, beat=1)

        with pytest.raises(ValueError, match="No visible rests found"):
            ss.remove_rests(measure=1, part=0, voice=1, beat=1)

    def test_add_rest_inserts_into_actual_gap(self) -> None:
        """add_rest inserts only the requested rest into uncovered time."""
        ss = _make_score()
        container = _measure_voice_container(ss, measure=1, part=0, voice=1)
        _remove_rests_from_container(container)

        result = ss.add_rest(
            measure=1,
            part=0,
            voice=1,
            beat=2,
            duration="quarter",
        )

        assert result.details["mode"] == "inserted"
        assert result.details["inserted_rests"][0]["beat"] == 2.0
        assert result.details["refilled_rests"] == []
        assert _visible_rest_summary(ss) == [(2.0, "quarter")]
        assert result.details["measure_integrity"]["gaps"]

    def test_add_rest_unhides_exact_hidden_rest(self) -> None:
        """add_rest makes an exact hidden rest visible again."""
        ss = _make_score()
        _reshape_rest(ss, "quarter", measure=1, beat=1)
        _reshape_rest(ss, "quarter", measure=1, beat=2)
        ss.remove_rests(measure=1, part=0, voice=1, beat=2)

        result = ss.add_rest(
            measure=1,
            part=0,
            voice=1,
            beat=2,
            duration="quarter",
        )

        assert result.details["mode"] == "unhidden"
        assert result.details["removed_rests"][0]["visibility"] == "hidden"
        assert _visible_rest_summary(ss) == [
            (1.0, "quarter"),
            (2.0, "quarter"),
            (3.0, "half"),
        ]

    def test_add_rest_inside_hidden_full_rest_refills_remainders(self) -> None:
        """add_rest respells touched hidden rest space with visible rests."""
        ss = _make_score()
        ss.remove_rests(measure=1, part=0, voice=1)

        result = ss.add_rest(
            measure=1,
            part=0,
            voice=1,
            beat=2,
            duration="quarter",
        )

        assert result.details["mode"] == "respelled"
        assert result.details["removed_rests"][0]["visibility"] == "hidden"
        assert result.details["measure_integrity"]["gaps"] == []
        assert _visible_rest_summary(ss) == [
            (1.0, "quarter"),
            (2.0, "quarter"),
            (3.0, "half"),
        ]

    def test_add_rest_inside_visible_full_rest_refills_remainders(self) -> None:
        """add_rest locally respells an overlapped visible rest."""
        ss = _make_score()

        result = ss.add_rest(
            measure=1,
            part=0,
            voice=1,
            beat=2,
            duration="quarter",
        )

        assert result.details["mode"] == "respelled"
        assert result.details["removed_rests"][0]["visibility"] == "visible"
        assert result.details["measure_integrity"]["gaps"] == []
        assert _visible_rest_summary(ss) == [
            (1.0, "quarter"),
            (2.0, "quarter"),
            (3.0, "half"),
        ]

    def test_add_rest_rejects_sounding_overlap(self) -> None:
        """add_rest refuses to overwrite notes or chords."""
        ss = _make_score()
        ss.add_notes(
            measure=1,
            part=0,
            voice=1,
            notes=[
                {
                    "pitch": "C4",
                    "beat": 1.0,
                    "duration": "quarter",
                    "dots": 0,
                }
            ],
        )

        with pytest.raises(ValueError, match="overlaps existing"):
            ss.add_rest(
                measure=1,
                part=0,
                voice=1,
                beat=1,
                duration="quarter",
            )

    def test_add_rest_rejects_over_measure_duration(self) -> None:
        """add_rest refuses a rest that exceeds measure capacity."""
        ss = _make_score()

        with pytest.raises(ValueError, match="would exceed"):
            ss.add_rest(
                measure=1,
                part=0,
                voice=1,
                beat=4,
                duration="half",
            )

    def test_fill_measure_gaps_fills_one_gap(self) -> None:
        """fill_measure_gaps fills uncovered time in one voice."""
        ss = _make_score()
        ss.add_notes(
            measure=1,
            part=0,
            voice=1,
            notes=[
                {
                    "pitch": "C4",
                    "beat": 1.0,
                    "duration": "quarter",
                    "dots": 0,
                }
            ],
        )
        container = _measure_voice_container(ss, measure=1, part=0, voice=1)
        _remove_rests_from_container(container)

        result = ss.fill_measure_gaps(measure=1, part=0, voice=1)

        assert len(result.details["filled_gaps"]) == 1
        assert result.details["count"] == len(result.details["inserted_rests"])
        assert result.details["measure_integrity"]["gaps"] == []

    def test_fill_measure_gaps_fills_multiple_gaps(self) -> None:
        """fill_measure_gaps fills each uncovered region in the voice."""
        ss = _make_score()
        ss.add_notes(
            measure=1,
            part=0,
            voice=1,
            notes=[
                {
                    "pitch": "C4",
                    "beat": 2.0,
                    "duration": "quarter",
                    "dots": 0,
                },
                {
                    "pitch": "D4",
                    "beat": 4.0,
                    "duration": "quarter",
                    "dots": 0,
                },
            ],
        )
        container = _measure_voice_container(ss, measure=1, part=0, voice=1)
        _remove_rests_from_container(container)

        result = ss.fill_measure_gaps(measure=1, part=0, voice=1)

        assert len(result.details["filled_gaps"]) == 2
        assert result.details["measure_integrity"]["gaps"] == []
        assert _visible_rest_summary(ss) == [
            (1.0, "quarter"),
            (3.0, "quarter"),
        ]

    def test_fill_measure_gaps_noops_when_no_gaps_exist(self) -> None:
        """fill_measure_gaps succeeds without mutation when the voice is full."""
        ss = _make_score()

        result = ss.fill_measure_gaps(measure=1, part=0, voice=1)

        assert result.details["count"] == 0
        assert result.details["filled_gaps"] == []
        assert result.details["inserted_rests"] == []
        assert _visible_rest_summary(ss) == [(1.0, "whole")]

    def test_fill_measure_gaps_does_not_unhide_hidden_rests(self) -> None:
        """Hidden rests occupy time and are not treated as gaps."""
        ss = _make_score()
        ss.remove_rests(measure=1, part=0, voice=1)

        result = ss.fill_measure_gaps(measure=1, part=0, voice=1)

        assert result.details["count"] == 0
        assert result.details["measure_integrity"]["gaps"] == []
        assert _visible_rest_summary(ss) == []

    def test_fill_measure_gaps_rejects_overlap_state(self) -> None:
        """fill_measure_gaps rejects malformed overlapping voices first."""
        ss = _make_score()
        container = _measure_voice_container(ss, measure=1, part=0, voice=1)
        _remove_rests_from_container(container)
        container.insert(0.0, m21note.Note("C4", type="half"))
        container.insert(1.0, m21note.Note("D4", type="half"))

        with pytest.raises(ValueError, match="fix overlaps"):
            ss.fill_measure_gaps(measure=1, part=0, voice=1)

    def test_fill_measure_gaps_respects_voice_scope(self) -> None:
        """fill_measure_gaps mutates only the requested voice."""
        ss = _make_score()
        part_obj, _part_idx = ss._resolve_part(0)
        measure_obj = ss._resolve_measure(part_obj, 1)
        voice_two = ss._get_voice_or_measure(measure_obj, 2, create=True)
        voice_one = ss._get_voice_or_measure(measure_obj, 1, create=True)
        _remove_rests_from_container(voice_one)
        voice_one.insert(1.0, m21note.Note("C4", type="quarter"))
        voice_two.insert(0.0, m21note.Rest(type="whole"))

        result = ss.fill_measure_gaps(measure=1, part=0, voice=1)

        voice_one_rest_beats = [
            float(voice_one.elementOffset(rest)) + 1.0
            for rest in voice_one.getElementsByClass(m21note.Rest)
        ]
        voice_two_rests = list(voice_two.getElementsByClass(m21note.Rest))
        assert result.details["measure_integrity"]["gaps"] == []
        assert voice_one_rest_beats == [1.0, 3.0]
        assert len(voice_two_rests) == 1
        assert voice_two_rests[0].duration.type == "whole"

    def test_reshape_rest_capacity_validation(self):
        ss = _make_score(time_signature="4/4")
        with pytest.raises(ValueError, match="exceed"):
            _reshape_rest(ss, "whole", measure=1, beat=2)


# ==================================================================
# add_chord
# ==================================================================

class TestAddChord:
    """Tests for adding chords."""

    def test_add_chord_basic(self):
        ss = _make_score()
        result = ss.add_chord(["C4", "E4", "G4"], "quarter", measure=1, beat=1)
        assert result.success
        assert "C4" in result.details["pitches"]
        assert "E4" in result.details["pitches"]
        assert "G4" in result.details["pitches"]

    def test_add_chord_appears_in_get_notes(self):
        ss = _make_score()
        ss.add_chord(["C4", "E4", "G4"], "quarter", measure=1, beat=1)
        notes = ss.get_notes(measure=1)
        chord_notes = [n for n in notes if n.is_chord]
        assert len(chord_notes) == 3
        chord_pitches = {n.pitch for n in chord_notes}
        assert chord_pitches == {"C4", "E4", "G4"}

    def test_add_chord_empty_pitches_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="at least 2"):
            ss.add_chord([], measure=1, beat=1)

    def test_add_chord_single_pitch_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="at least 2"):
            ss.add_chord(["C4"], measure=1, beat=1)

    def test_add_chord_duplicate_pitch_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="duplicate"):
            ss.add_chord(["C4", "C4"], measure=1, beat=1)

    def test_add_chord_default_position(self):
        ss = _make_score(measures=2)
        result = ss.add_chord(["C4", "E4"], "half")
        assert result.success
        assert result.details["measure"] == 2

    def test_add_chord_capacity_validation(self):
        ss = _make_score(time_signature="4/4")
        with pytest.raises(ValueError, match="exceed"):
            ss.add_chord(["C4", "E4"], "whole", measure=1, beat=2)


# ==================================================================
# add_chord_tones
# ==================================================================

class TestAddChordTones:
    """Tests for adding pitches to existing notes/chords."""

    def test_add_chord_tones_converts_note_to_chord(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)

        result = ss.add_chord_tones(["E4", "G4"], measure=1, beat=1)

        assert result.success
        assert result.details["pitches"] == ["C4", "E4", "G4"]
        notes = ss.get_notes(measure=1)
        assert [(note.pitch, note.is_chord) for note in notes if not note.is_rest] == [
            ("C4", True),
            ("E4", True),
            ("G4", True),
        ]

    def test_add_chord_tones_appends_to_chord(self):
        ss = _make_score()
        ss.add_chord(["C4", "E4"], "quarter", measure=1, beat=1)

        ss.add_chord_tones(["G4"], measure=1, beat=1)

        assert [
            note.pitch
            for note in ss.get_notes(measure=1)
            if not note.is_rest
        ] == [
            "C4", "E4", "G4",
        ]

    def test_add_chord_tones_rejects_empty_target(self):
        ss = _make_score()

        with pytest.raises(ValueError, match="Use add_chord"):
            ss.add_chord_tones(["E4"], measure=1, beat=1)

    def test_add_chord_tones_rejects_duplicate_or_present_pitch(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)

        with pytest.raises(ValueError, match="duplicate"):
            ss.add_chord_tones(["E4", "E4"], measure=1, beat=1)
        with pytest.raises(ValueError, match="already present"):
            ss.add_chord_tones(["C4"], measure=1, beat=1)


# ==================================================================
# remove_notes
# ==================================================================

class TestRemoveNotes:
    """Tests for removing notes."""

    def test_remove_notes_basic(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        result = _remove_one_note(ss, measure=1, beat=1)
        assert result.success
        notes = ss.get_notes(measure=1)
        assert [(note.pitch, note.is_rest, note.duration_type) for note in notes] == [
            ("rest", True, "whole")
        ]

    def test_remove_specific_pitch(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        result = _remove_one_note(ss, measure=1, beat=1, pitch="C4")
        assert result.success

    def test_remove_note_removes_dependent_slur(self) -> None:
        """Removing a note removes slurs that use it as an endpoint."""
        ss = _make_score(measures=2)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=2, beat=1)
        ss.add_slur(1, 1.0, 2, 1.0)

        result = _remove_one_note(ss, measure=1, beat=1, pitch="C4")

        part_obj, _ = ss._resolve_part(0)
        assert result.success
        assert result.details["spanners_removed"] == 1
        assert list(part_obj.getElementsByClass(m21spanner.Slur)) == []

    def test_remove_note_clears_connected_tie_chain(self) -> None:
        """Removing a tied note clears orphan tie markers on surviving neighbors."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=2)
        ss.add_tie(measure=1, beat=1)

        result = _remove_one_note(ss, measure=1, beat=1, pitch="C4")

        assert result.success
        assert result.details["ties_removed"] == 2
        sounding = [note for note in ss.get_notes(measure=1) if not note.is_rest]
        assert len(sounding) == 1
        assert sounding[0].pitch == "C4"
        assert not sounding[0].is_tied

    def test_remove_chord_tone_clears_connected_tie_chain(self) -> None:
        """Changing a tied chord signature clears the dependent tie chain."""
        ss = _make_score()
        ss.add_chord(["C4", "E4"], "quarter", measure=1, beat=1)
        ss.add_chord(["C4", "E4"], "quarter", measure=1, beat=2)
        ss.add_tie(measure=1, beat=1)

        result = _remove_one_note(ss, measure=1, beat=1, pitch="E4")

        assert result.success
        assert result.details["ties_removed"] == 2
        part_obj = ss.score.parts[0]
        for element in part_obj.measure(1).notes:
            assert getattr(element, "tie", None) is None

    def test_remove_wrong_pitch_fails(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        with pytest.raises(ValueError, match="Expected pitch"):
            _remove_one_note(ss, measure=1, beat=1, pitch="D4")

    def test_remove_from_empty_beat_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="Use remove_rests"):
            _remove_one_note(ss, measure=1, beat=1)

    def test_remove_notes_rejects_rest_target(self):
        ss = _make_score()
        _reshape_rest(ss, "quarter", measure=1, beat=1)
        with pytest.raises(ValueError, match="Use remove_rests"):
            _remove_one_note(ss, measure=1, beat=1)

    def test_remove_pitch_from_chord(self):
        ss = _make_score()
        ss.add_chord(["C4", "E4", "G4"], "quarter", measure=1, beat=1)
        result = _remove_one_note(ss, measure=1, beat=1, pitch="E4")
        assert result.success
        notes = ss.get_notes(measure=1)
        chord_notes = [n for n in notes if n.is_chord]
        chord_pitches = {n.pitch for n in chord_notes}
        assert "E4" not in chord_pitches
        assert "C4" in chord_pitches
        assert "G4" in chord_pitches

    def test_remove_pitch_from_chord_down_to_one(self):
        ss = _make_score()
        ss.add_chord(["C4", "E4"], "quarter", measure=1, beat=1)
        result = _remove_one_note(ss, measure=1, beat=1, pitch="E4")
        assert result.success
        notes = ss.get_notes(measure=1)
        sounding = [note for note in notes if not note.is_rest]
        assert len(sounding) == 1
        assert sounding[0].pitch == "C4"
        assert not sounding[0].is_chord

    def test_remove_nonexistent_pitch_from_chord_fails(self):
        ss = _make_score()
        ss.add_chord(["C4", "E4"], "quarter", measure=1, beat=1)
        with pytest.raises(ValueError, match="not found in the chord"):
            _remove_one_note(ss, measure=1, beat=1, pitch="G4")

    def test_remove_notes_batch_is_atomic(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=2)

        with pytest.raises(ValueError, match=r"notes\[1\]"):
            ss.remove_notes(
                measure=1,
                part=0,
                voice=1,
                notes=[
                    {"beat": 1.0, "pitch": "C4"},
                    {"beat": 2.0, "pitch": "E4"},
                ],
            )

        assert [
            note.pitch
            for note in ss.get_notes(measure=1)
            if not note.is_rest
        ] == ["C4", "D4"]

    def test_remove_notes_groups_chord_tones_and_deletes_final_pitch(self):
        ss = _make_score()
        ss.add_chord(["C4", "E4", "G4"], "quarter", measure=1, beat=1)

        first = ss.remove_notes(
            measure=1,
            part=0,
            voice=1,
            notes=[
                {"beat": 1.0, "pitch": "E4"},
                {"beat": 1.0, "pitch": "G4"},
            ],
        )

        assert first.details["count"] == 2
        remaining = ss.get_notes(measure=1)
        assert [(note.pitch, note.is_chord) for note in remaining if not note.is_rest] == [
            ("C4", False)
        ]

        second = _remove_one_note(ss, measure=1, beat=1, pitch="C4")
        assert second.details["auto_completed_rests"][0]["beat"] == 1.0
        rests = [note for note in ss.get_notes(measure=1) if note.is_rest]
        assert [(rest.beat, rest.duration_type) for rest in rests] == [
            (1.0, "whole")
        ]

    def test_remove_notes_pitchless_rejects_rest_targets(self):
        ss = _make_score()
        ss.add_chord(["C4", "E4"], "quarter", measure=1, beat=1)
        _reshape_rest(ss, "quarter", measure=1, beat=2)

        with pytest.raises(ValueError, match=r"notes\[1\].*Use remove_rests"):
            ss.remove_notes(
                measure=1,
                part=0,
                voice=1,
                notes=[{"beat": 1.0}, {"beat": 2.0}],
            )
        assert [
            note.pitch
            for note in ss.get_notes(measure=1)
            if not note.is_rest
        ] == ["C4", "E4"]

    def test_remove_notes_rejects_mixed_same_beat_items(self):
        ss = _make_score()

        with pytest.raises(ValueError, match="mixes pitchless"):
            ss.remove_notes(
                measure=1,
                part=0,
                voice=1,
                notes=[{"beat": 1.0}, {"beat": 1.0, "pitch": "C4"}],
            )

    def test_remove_notes_blocks_tuplet_events(self):
        ss = _make_score()
        ss.add_tuplet(
            [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")],
            actual_notes=3,
            normal_notes=2,
            measure=1,
            beat=1,
        )

        with pytest.raises(ValueError, match="remove_tuplet"):
            _remove_one_note(ss, measure=1, beat=1)

    def test_remove_notes_blocks_interior_tuplet_target(self):
        ss = _make_score()
        ss.add_tuplet(
            [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")],
            actual_notes=3,
            normal_notes=2,
            measure=1,
            beat=1,
        )

        with pytest.raises(ValueError, match="remove_tuplet"):
            ss.remove_notes(
                measure=1,
                part=0,
                voice=1,
                notes=[{"beat": 1.1, "pitch": "C4"}],
            )

    def test_remove_notes_rejects_grace_note_when_no_rest_masks_target(self):
        ss = _make_score()
        ss.remove_rests(measure=1, part=0, voice=1, beat=1)
        ss.add_grace_note("D4", measure=1, beat=1)

        with pytest.raises(ValueError, match="remove_grace_note"):
            _remove_one_note(ss, measure=1, beat=1)

    def test_remove_notes_reports_interior_beat_start(self):
        ss = _make_score()
        ss._add_note_one("C4", "half", measure=1, beat=1)

        with pytest.raises(ValueError, match="start beat 1"):
            _remove_one_note(ss, measure=1, beat=2)

    def test_remove_notes_rejects_multiple_pitch_targets_on_single_note(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)

        with pytest.raises(ValueError, match="single note C4"):
            ss.remove_notes(
                measure=1,
                part=0,
                voice=1,
                notes=[
                    {"beat": 1.0, "pitch": "C4"},
                    {"beat": 1.0, "pitch": "E4"},
                ],
            )

        assert [
            note.pitch
            for note in ss.get_notes(measure=1)
            if not note.is_rest
        ] == ["C4"]


# ==================================================================
# replace_note
# ==================================================================

class TestReplaceNote:
    """Tests for replacing notes."""

    def test_replace_pitch(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        result = ss.replace_note(measure=1, beat=1, new_pitch="D4")
        assert result.success
        notes = ss.get_notes(measure=1)
        assert notes[0].pitch == "D4"

    def test_replace_pitch_class_preserves_existing_octave(self):
        ss = _make_score()
        ss._add_note_one("E-6", "quarter", measure=1, beat=1)

        result = ss.replace_note(measure=1, beat=1, new_pitch="D")

        assert result.success
        notes = ss.get_notes(measure=1)
        assert notes[0].pitch == "D6"

    def test_replace_duration(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        result = ss.replace_note(measure=1, beat=1, new_duration="half")
        assert result.success
        notes = ss.get_notes(measure=1)
        assert notes[0].duration_type == "half"

    def test_replace_both_pitch_and_duration(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        result = ss.replace_note(
            measure=1, beat=1, new_pitch="E5", new_duration="half",
        )
        assert result.success
        assert len(result.details["changes"]) == 2
        notes = ss.get_notes(measure=1)
        assert notes[0].pitch == "E5"
        assert notes[0].duration_type == "half"

    def test_replace_nothing_fails(self):
        ss = _make_score()
        ss._add_note_one("C4", measure=1, beat=1)
        with pytest.raises(ValueError, match="At least one"):
            ss.replace_note(measure=1, beat=1)

    def test_replace_empty_beat_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="not a note or chord"):
            ss.replace_note(measure=1, beat=1, new_pitch="C4")

    def test_replace_duration_validates_capacity(self):
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C4", "quarter", measure=1, beat=4)
        with pytest.raises(ValueError, match="exceed"):
            ss.replace_note(measure=1, beat=4, new_duration="whole")

    def test_replace_reports_interior_beat_start(self):
        ss = _make_score()
        ss._add_note_one("C4", "half", measure=1, beat=1)

        with pytest.raises(ValueError, match="start beat 1"):
            ss.replace_note(measure=1, beat=2, new_pitch="D4")


# ==================================================================
# add_tie
# ==================================================================

class TestAddTie:
    """Tests for adding ties."""

    def test_add_tie_to_adjacent_same_pitch_event(self) -> None:
        """add_tie connects the selected note to the next matching note."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=2)

        result = ss.add_tie(measure=1, beat=1)

        assert result.success
        assert result.details["tied_positions"] == [
            {"measure": 1, "beat": 1.0},
            {"measure": 1, "beat": 2.0},
        ]
        notes = list(ss.score.parts[0].measure(1).notes)
        assert [note.tie.type if note.tie is not None else None for note in notes] == [
            "start",
            "stop",
        ]

    def test_add_tie_mismatched_pitch_fails(self) -> None:
        """add_tie rejects an adjacent note with different pitch content."""
        ss = _make_score()
        ss._add_note_one("C4", measure=1, beat=1)
        ss._add_note_one("D4", measure=1, beat=2)
        with pytest.raises(ValueError, match="adjacent matching pitches"):
            ss.add_tie(measure=1, beat=1)

    def test_add_tie_next_rest_fails(self) -> None:
        """add_tie rejects a rest between same-pitch events."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        _reshape_rest(ss, "quarter", measure=1, beat=2)
        ss._add_note_one("C4", "quarter", measure=1, beat=3)

        with pytest.raises(ValueError, match="next event is a rest"):
            ss.add_tie(measure=1, beat=1)

    def test_add_tie_selected_rest_fails(self) -> None:
        """add_tie rejects selecting a rest as the start point."""
        ss = _make_score()
        _reshape_rest(ss, "quarter", measure=1, beat=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=2)

        with pytest.raises(ValueError, match="Cannot use a rest"):
            ss.add_tie(measure=1, beat=1)

    def test_add_tie_empty_beat_fails(self) -> None:
        """add_tie rejects beats occupied only by visible rests."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        with pytest.raises(ValueError, match="Cannot use a rest"):
            ss.add_tie(measure=1, beat=2)

    def test_add_tie_selected_stop_becomes_continue(self) -> None:
        """Adding from a tie stop converts the selected note to continue."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=2)
        ss._add_note_one("C4", "quarter", measure=1, beat=3)

        ss.add_tie(measure=1, beat=1)
        result = ss.add_tie(measure=1, beat=2)

        assert result.success
        notes = list(ss.score.parts[0].measure(1).notes)
        assert [note.tie.type if note.tie is not None else None for note in notes] == [
            "start",
            "continue",
            "stop",
        ]

    def test_add_tie_selected_start_is_noop(self) -> None:
        """Adding from an existing tie start is an idempotent no-op."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=2)

        ss.add_tie(measure=1, beat=1)
        result = ss.add_tie(measure=1, beat=1)

        assert result.success
        assert result.details["already_tied"] is True
        notes = list(ss.score.parts[0].measure(1).notes)
        assert [note.tie.type if note.tie is not None else None for note in notes] == [
            "start",
            "stop",
        ]

    def test_add_tie_selected_continue_is_noop(self) -> None:
        """Adding from an existing tie continue is an idempotent no-op."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=2)
        ss._add_note_one("C4", "quarter", measure=1, beat=3)
        ss.add_tie(measure=1, beat=1)
        ss.add_tie(measure=1, beat=2)

        result = ss.add_tie(measure=1, beat=2)

        assert result.success
        assert result.details["already_tied"] is True
        notes = list(ss.score.parts[0].measure(1).notes)
        assert [note.tie.type if note.tie is not None else None for note in notes] == [
            "start",
            "continue",
            "stop",
        ]

    def test_add_tie_next_start_preserves_onward_chain(self) -> None:
        """A next note already tied onward becomes continue, preserving the chain."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=2)
        ss._add_note_one("C4", "quarter", measure=1, beat=3)
        notes = list(ss.score.parts[0].measure(1).notes)
        notes[1].tie = m21tie.Tie("start")
        notes[2].tie = m21tie.Tie("stop")

        result = ss.add_tie(measure=1, beat=1)

        assert result.success
        notes = list(ss.score.parts[0].measure(1).notes)
        assert [note.tie.type if note.tie is not None else None for note in notes] == [
            "start",
            "continue",
            "stop",
        ]

    def test_tie_reflected_in_get_notes(self) -> None:
        """get_notes reports tied events through is_tied."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=2)
        ss.add_tie(measure=1, beat=1)
        notes = ss.get_notes(measure=1)
        sounding = [note for note in notes if not note.is_rest]
        assert sounding[0].is_tied
        assert sounding[1].is_tied


class TestRemoveTie:
    """Tests for removing ties."""

    def test_remove_tie_basic(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=2)
        ss.add_tie(measure=1, beat=1)
        result = ss.remove_tie(start_measure=1, start_beat=1)
        assert result.success
        assert [item["tie_type"] for item in result.details["removed"]] == [
            "start",
            "stop",
        ]
        notes = ss.get_notes(measure=1)
        sounding = [note for note in notes if not note.is_rest]
        assert not sounding[0].is_tied
        assert not sounding[1].is_tied

    def test_remove_tie_from_chord(self):
        ss = _make_score()
        ss.add_chord(["C4", "E4"], "quarter", measure=1, beat=1)
        ss.add_chord(["C4", "E4"], "quarter", measure=1, beat=2)
        ss.add_tie(measure=1, beat=1)
        result = ss.remove_tie(start_measure=1, start_beat=1)
        assert result.success
        assert [item["tie_type"] for item in result.details["removed"]] == [
            "start",
            "stop",
        ]

    def test_remove_tie_missing_fails(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        with pytest.raises(ValueError, match="No tie found"):
            ss.remove_tie(start_measure=1, start_beat=1)

    def test_remove_tie_from_rest_fails(self):
        ss = _make_score()
        _reshape_rest(ss, "quarter", measure=1, beat=1)
        with pytest.raises(ValueError, match="Cannot use a rest"):
            ss.remove_tie(start_measure=1, start_beat=1)


# ==================================================================
# add_tuplet
# ==================================================================

class TestAddTuplet:
    """Tests for adding tuplets."""

    def test_triplet_eighth_notes(self):
        ss = _make_score(time_signature="4/4")
        triplet_notes = [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")]
        result = ss.add_tuplet(
            triplet_notes,
            actual_notes=3,
            normal_notes=2,
            measure=1,
            beat=1,
        )
        assert result.success
        assert result.details["actual_notes"] == 3
        assert result.details["normal_notes"] == 2
        total_ql = result.details["total_quarter_length"]
        assert abs(total_ql - 1.0) < 1e-6

    def test_triplet_default_position(self):
        ss = _make_score(measures=2)
        triplet_notes = [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")]
        result = ss.add_tuplet(triplet_notes, 3, 2)
        assert result.success
        assert result.details["measure"] == 2

    def test_tuplet_wrong_count_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="Expected 3"):
            ss.add_tuplet(
                [("C4", "eighth"), ("D4", "eighth")],
                actual_notes=3,
                normal_notes=2,
                measure=1,
                beat=1,
            )

    def test_tuplet_non_positive_ratio_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="positive integers"):
            ss.add_tuplet([], actual_notes=0, normal_notes=2, measure=1, beat=1)

    def test_tuplet_exceeds_capacity_fails(self):
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C4", "whole", measure=1, beat=1)
        triplet_notes = [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")]
        with pytest.raises(ValueError, match="exceed"):
            ss.add_tuplet(triplet_notes, 3, 2, measure=1)

    def test_tuplet_rejects_same_voice_overlap(self):
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C4", "half", measure=1, beat=1)
        triplet_notes = [("D4", "eighth"), ("E4", "eighth"), ("F4", "eighth")]

        with pytest.raises(ValueError, match="overlap existing"):
            ss.add_tuplet(triplet_notes, 3, 2, measure=1, beat=1)

    def test_tuplet_overwrites_visible_rest(self):
        ss = _make_score(time_signature="4/4")
        _reshape_rest(ss, "quarter", measure=1, beat=1)
        triplet_notes = [("D4", "eighth"), ("E4", "eighth"), ("F4", "eighth")]

        result = ss.add_tuplet(triplet_notes, 3, 2, measure=1, beat=1)

        assert result.details["replaced_rests"][0]["duration"] == "quarter"
        assert [
            note.pitch
            for note in ss.get_notes(measure=1)
            if not note.is_rest
        ] == [
            "D4", "E4", "F4",
        ]

    def test_tuplet_allows_different_voice_overlap(self):
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C4", "whole", measure=1, beat=1, voice=1)
        triplet_notes = [("D4", "eighth"), ("E4", "eighth"), ("F4", "eighth")]

        result = ss.add_tuplet(
            triplet_notes,
            3,
            2,
            measure=1,
            beat=1,
            voice=2,
        )

        assert result.success
        assert [
            note.pitch
            for note in ss.get_notes(measure=1, voice=2)
            if not note.is_rest
        ] == [
            "D4", "E4", "F4",
        ]

    def test_tuplet_notes_in_get_notes(self):
        ss = _make_score()
        triplet_notes = [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")]
        ss.add_tuplet(triplet_notes, 3, 2, measure=1, beat=1)
        notes = ss.get_notes(measure=1)
        assert len([note for note in notes if not note.is_rest]) == 3
        pitches = [n.pitch for n in notes]
        assert "C4" in pitches
        assert "D4" in pitches
        assert "E4" in pitches
        for note_info in [note for note in notes if not note.is_rest]:
            assert note_info.tuplets == [TupletInfo(actual_notes=3, normal_notes=2)]

    def test_tuplet_notes_have_independent_boundary_metadata(self) -> None:
        """Tuplet notes should not share mutable visual boundary metadata."""
        ss = _make_score()
        triplet_notes = [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")]
        ss.add_tuplet(triplet_notes, 3, 2, measure=1, beat=1)

        notes = list(ss.score.parts[0].measure(1).recurse().notes)
        tuplets = [note.duration.tuplets[0] for note in notes]

        assert [tuplet.type for tuplet in tuplets] == ["start", None, "stop"]
        assert len({id(tuplet) for tuplet in tuplets}) == len(tuplets)

    def test_replace_tuplet_pitch_allowed_but_duration_blocked(self) -> None:
        """Tuplet notes can change pitch but not individual duration."""
        ss = _make_score()
        triplet_notes = [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")]
        ss.add_tuplet(triplet_notes, 3, 2, measure=1, beat=1)

        ss.replace_note(measure=1, beat=1, new_pitch="F4")

        assert ss.get_notes(measure=1)[0].pitch == "F4"
        with pytest.raises(ValueError, match="tuplet"):
            ss.replace_note(measure=1, beat=1, new_duration="quarter")

    def test_adjacent_tuplets_export_separate_bracket_boundaries(self) -> None:
        """Adjacent ScoreSpeak tuplets should export as separate visual groups."""
        ss = _make_score(time_signature="4/4")
        ss.add_tuplet(
            [("C4", "quarter"), ("D4", "quarter"), ("E4", "quarter")],
            3,
            2,
            measure=1,
            beat=1,
        )
        ss.add_tuplet(
            [("F4", "eighth"), ("G4", "eighth"), ("A4", "eighth")],
            3,
            2,
            measure=1,
            beat=3,
        )
        ss.add_tuplet(
            [("B4", "eighth"), ("C5", "eighth"), ("D5", "eighth")],
            3,
            2,
            measure=1,
            beat=4,
        )

        tuplet_types = _exported_note_tuplet_types(ss)

        assert tuplet_types == [
            ["start"], [], ["stop"],
            ["start"], [], ["stop"],
            ["start"], [], ["stop"],
        ]
        flattened_types = [
            tuplet_type
            for note_tuplet_types in tuplet_types
            for tuplet_type in note_tuplet_types
        ]
        assert flattened_types.count("start") == 3
        assert flattened_types.count("stop") == 3


class TestRemoveTuplet:
    """Tests for removing tuplets."""

    def test_remove_tuplet_group(self):
        ss = _make_score()
        triplet_notes = [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")]
        ss.add_tuplet(triplet_notes, 3, 2, measure=1, beat=1)
        result = ss.remove_tuplet(measure=1, beat=1)
        assert result.success
        assert result.details["removed_notes"] == 3
        rests = [note for note in ss.get_notes(measure=1) if note.is_rest]
        assert [(rest.beat, rest.duration_type) for rest in rests] == [
            (1.0, "whole")
        ]

    def test_remove_tuplet_with_ratio_guard(self):
        ss = _make_score()
        triplet_notes = [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")]
        ss.add_tuplet(triplet_notes, 3, 2, measure=1, beat=1)
        result = ss.remove_tuplet(
            measure=1,
            beat=1,
            actual_notes=3,
            normal_notes=2,
        )
        assert result.success

    def test_remove_tuplet_wrong_ratio_fails(self):
        ss = _make_score()
        triplet_notes = [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")]
        ss.add_tuplet(triplet_notes, 3, 2, measure=1, beat=1)
        with pytest.raises(ValueError, match="actual notes"):
            ss.remove_tuplet(measure=1, beat=1, actual_notes=2)

    def test_remove_tuplet_missing_fails(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        with pytest.raises(ValueError, match="No tuplet"):
            ss.remove_tuplet(measure=1, beat=1)


# ==================================================================
# add_grace_note
# ==================================================================

class TestAddGraceNote:
    """Tests for adding grace notes."""

    def test_acciaccatura_default(self):
        ss = _make_score()
        result = ss.add_grace_note("D4", measure=1, beat=1)
        assert result.success
        assert result.details["grace_type"] == "acciaccatura"
        assert result.details["pitch"] == "D4"

    def test_appoggiatura(self):
        ss = _make_score()
        result = ss.add_grace_note("D4", measure=1, beat=1, slash=False)
        assert result.success
        assert result.details["grace_type"] == "appoggiatura"

    def test_grace_note_appears_in_get_notes(self):
        ss = _make_score()
        ss.add_grace_note("D4", measure=1, beat=1)
        notes = ss.get_notes(measure=1)
        grace_notes = [n for n in notes if n.is_grace]
        assert len(grace_notes) == 1
        assert grace_notes[0].pitch == "D4"

    def test_grace_note_zero_duration(self):
        ss = _make_score()
        ss.add_grace_note("D4", measure=1, beat=1)
        notes = ss.get_notes(measure=1)
        grace_notes = [n for n in notes if n.is_grace]
        assert grace_notes[0].quarter_length == 0.0

    def test_grace_note_default_beat(self):
        ss = _make_score()
        result = ss.add_grace_note("D4", measure=1)
        assert result.details["beat"] == 1.0

    def test_grace_note_custom_written_duration(self):
        ss = _make_score()
        result = ss.add_grace_note("D4", duration="16th", measure=1, beat=1)
        grace = [
            note
            for note in ss.score.recurse().notes
            if note.duration.isGrace
        ][0]

        assert result.details["duration"] == "16th"
        assert result.details["written_quarter_length"] == 0.25
        assert grace.duration.type == "16th"
        assert grace.duration.quarterLength == 0.0

    def test_grace_note_slur_to_principal_groups_same_beat_graces(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
        first = ss.add_grace_note(
            "B3",
            duration="16th",
            measure=1,
            beat=1,
            part=0,
            slur_to_principal=True,
        )
        second = ss.add_grace_note(
            "A3",
            duration="32nd",
            measure=1,
            beat=1,
            part=0,
            slur_to_principal=True,
        )
        slurs = list(ss.score.parts[0].getElementsByClass(m21spanner.Slur))
        spanned = list(slurs[0].getSpannedElements())

        assert first.details["slur_grace_count"] == 1
        assert second.details["slur_grace_count"] == 2
        assert len(slurs) == 1
        assert [note.pitch.nameWithOctave for note in spanned] == ["B3", "A3", "C4"]

    def test_grace_note_slur_to_principal_requires_principal_event(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="slur_to_principal=True requires"):
            ss.add_grace_note("D4", measure=1, beat=3, slur_to_principal=True)

    def test_same_beat_grace_notes_export_beams(self) -> None:
        """Same-beat grace notes with matching durations export beam tags."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)

        ss.add_grace_note("D4", duration="16th", measure=1, beat=1)
        ss.add_grace_note("E4", duration="16th", measure=1, beat=1)

        assert _exported_measure_grace_note_beams(ss, 1) == [
            ["begin", "begin"],
            ["end", "end"],
        ]

    def test_three_same_beat_grace_notes_export_continuing_beams(self) -> None:
        """Three-note grace groups export begin, continue, and end beams."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)

        for pitch in ["D4", "E4", "F4"]:
            ss.add_grace_note(pitch, duration="16th", measure=1, beat=1)

        assert _exported_measure_grace_note_beams(ss, 1) == [
            ["begin", "begin"],
            ["continue", "continue"],
            ["end", "end"],
        ]

    def test_single_grace_note_exports_no_beams(self) -> None:
        """Single grace notes remain unbeamed."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)

        ss.add_grace_note("D4", duration="16th", measure=1, beat=1)

        assert _exported_measure_grace_note_beams(ss, 1) == [[]]

    def test_different_beat_grace_notes_do_not_beam_together(self) -> None:
        """Grace notes at different anchor beats are separate groups."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=2)

        ss.add_grace_note("D4", duration="16th", measure=1, beat=1)
        ss.add_grace_note("E4", duration="16th", measure=1, beat=2)

        assert _exported_measure_grace_note_beams(ss, 1) == [[], []]

    def test_different_voice_grace_notes_do_not_beam_together(self) -> None:
        """Grace-note beams stay scoped to one voice container."""
        ss = _make_score()
        ss._add_note_one("C5", "quarter", measure=1, beat=1, voice=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=1, voice=2)

        ss.add_grace_note("D5", duration="16th", measure=1, beat=1, voice=1)
        ss.add_grace_note("D4", duration="16th", measure=1, beat=1, voice=2)

        assert _exported_measure_grace_note_beams(ss, 1) == [[], []]

    def test_mixed_duration_grace_notes_do_not_beam_together(self) -> None:
        """Mixed written grace durations are left unbeamed in v1."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)

        ss.add_grace_note("D4", duration="16th", measure=1, beat=1)
        ss.add_grace_note("E4", duration="32nd", measure=1, beat=1)

        assert _exported_measure_grace_note_beams(ss, 1) == [[], []]


class TestRemoveGraceNote:
    """Tests for removing grace notes."""

    def test_remove_grace_note_preserves_principal_note(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss.add_grace_note("D4", measure=1, beat=1)
        result = ss.remove_grace_note(measure=1, beat=1)
        assert result.success
        notes = ss.get_notes(measure=1)
        sounding = [note for note in notes if not note.is_rest]
        assert len(sounding) == 1
        assert sounding[0].pitch == "C4"
        assert not sounding[0].is_grace

    def test_remove_grace_note_by_pitch(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss.add_grace_note("D4", measure=1, beat=1)
        ss.add_grace_note("E4", measure=1, beat=1)
        result = ss.remove_grace_note(measure=1, beat=1, pitch="E4")
        assert result.success
        notes = ss.get_notes(measure=1)
        grace_pitches = [n.pitch for n in notes if n.is_grace]
        assert grace_pitches == ["D4"]

    def test_remove_grace_note_rebuilds_principal_slur(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
        ss.add_grace_note(
            "D4",
            measure=1,
            beat=1,
            part=0,
            slur_to_principal=True,
        )
        ss.add_grace_note(
            "E4",
            measure=1,
            beat=1,
            part=0,
            slur_to_principal=True,
        )

        ss.remove_grace_note(measure=1, beat=1, part=0, pitch="D4")
        slurs = list(ss.score.parts[0].getElementsByClass(m21spanner.Slur))
        spanned = list(slurs[0].getSpannedElements())

        assert len(slurs) == 1
        assert [note.pitch.nameWithOctave for note in spanned] == ["E4", "C4"]

    def test_remove_grace_note_clears_stale_beams(self) -> None:
        """Removing one grace note clears beams on the remaining singleton."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss.add_grace_note("D4", duration="16th", measure=1, beat=1)
        ss.add_grace_note("E4", duration="16th", measure=1, beat=1)

        ss.remove_grace_note(measure=1, beat=1, pitch="E4")

        assert _exported_measure_grace_note_beams(ss, 1) == [[]]

    def test_remove_grace_note_missing_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="No grace note found"):
            ss.remove_grace_note(measure=1, beat=1)


# ==================================================================
# Stem direction normalization
# ==================================================================

class TestStemDirections:
    """Tests for automatic stem direction normalization on measure edits."""

    def test_single_voice_stems_follow_pitch_and_center_line(self) -> None:
        """Single-voice notes use clef/pitch-based stem directions."""
        ss = _make_score(measures=1)

        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("B4", "quarter", measure=1, beat=2)
        ss._add_note_one("A5", "quarter", measure=1, beat=3)

        assert _exported_measure_note_stems(ss, 1) == [
            ("1", "C4", "up"),
            ("1", "B4", "down"),
            ("1", "A5", "down"),
        ]

    def test_single_voice_refresh_overwrites_existing_stems(self) -> None:
        """Measure refresh overwrites stale manual stems in touched bars."""
        ss = _make_score(measures=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        first_note = list(ss.score.recurse().getElementsByClass(m21note.Note))[0]
        first_note.stemDirection = "down"

        _reshape_rest(ss, "quarter", measure=1, beat=2)

        assert first_note.stemDirection == "up"
        assert _exported_measure_note_stems(ss, 1)[0] == ("1", "C4", "up")

    def test_multi_voice_stems_follow_voice_number(self) -> None:
        """Multi-voice measures ignore pitch and use voice-number stems."""
        ss = _make_score(measures=1)

        ss._add_note_one("A5", "quarter", measure=1, beat=1, voice=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=1, voice=2)
        ss._add_note_one("A5", "quarter", measure=1, beat=2, voice=3)
        ss._add_note_one("C4", "quarter", measure=1, beat=2, voice=4)

        assert _exported_measure_note_stems(ss, 1) == [
            ("1", "A5", "up"),
            ("2", "C4", "down"),
            ("3", "A5", "up"),
            ("4", "C4", "down"),
        ]

    def test_empty_explicit_voice_does_not_force_multi_voice_stems(self) -> None:
        """Empty Voice 2 streams do not change Voice 1 pitch-based stems."""
        ss = _make_score(measures=1)
        ss._add_note_one("A5", "quarter", measure=1, beat=1)
        part = list(ss.score.parts)[0]
        measure = part.measure(1)
        measure.insert(0, m21stream.Voice(id="2"))

        ss._refresh_measure_beams(measure)

        assert _exported_measure_note_stems(ss, 1) == [("1", "A5", "down")]

    def test_chord_stem_uses_pitch_set_in_single_voice(self) -> None:
        """Single-voice chords receive one pitch-derived stem direction."""
        ss = _make_score(measures=1)

        ss.add_chord(["C4", "E4"], "quarter", measure=1, beat=1)

        chord = list(ss.score.recurse().getElementsByClass(m21chord.Chord))[0]
        assert chord.stemDirection == "up"
        assert _exported_measure_note_stems(ss, 1)[0] == ("1", "C4", "up")

    def test_replace_and_remove_notes_refresh_stale_stems(self) -> None:
        """Replace and remove operations refresh remaining note stems."""
        ss = _make_score(measures=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=2)
        notes = list(ss.score.recurse().getElementsByClass(m21note.Note))
        for note in notes:
            note.stemDirection = "down"

        ss.replace_note(measure=1, beat=1, new_pitch="E4")

        assert [note.stemDirection for note in notes] == ["up", "up"]
        notes[1].stemDirection = "down"
        _remove_one_note(ss, measure=1, beat=1)
        remaining_note = list(ss.score.recurse().getElementsByClass(m21note.Note))[0]
        assert remaining_note.stemDirection == "up"

    def test_tuplet_add_and_remove_refresh_stems(self) -> None:
        """Tuplet insertion and removal both refresh stems in the measure."""
        ss = _make_score(measures=1)

        ss.add_tuplet(
            [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")],
            actual_notes=3,
            normal_notes=2,
            measure=1,
            beat=1,
        )
        ss._add_note_one("C4", "quarter", measure=1, beat=2)

        notes = list(ss.score.recurse().getElementsByClass(m21note.Note))
        assert [note.stemDirection for note in notes[:3]] == ["up", "up", "up"]
        notes[-1].stemDirection = "down"

        ss.remove_tuplet(measure=1, beat=1)

        assert notes[-1].stemDirection == "up"

    def test_grace_note_add_and_remove_refresh_stems(self) -> None:
        """Grace note insertion and removal refresh stems in the measure."""
        ss = _make_score(measures=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=2)

        ss.add_grace_note("A5", measure=1, beat=1)

        notes = list(ss.score.recurse().getElementsByClass(m21note.Note))
        grace_note = [note for note in notes if note.duration.isGrace][0]
        regular_note = [note for note in notes if not note.duration.isGrace][0]
        assert grace_note.stemDirection == "down"
        regular_note.stemDirection = "down"

        ss.remove_grace_note(measure=1, beat=1, pitch="A5")

        assert regular_note.stemDirection == "up"

    def test_rest_reshaping_refreshes_stems(self) -> None:
        """Rest reshaping refreshes note stems."""
        completed = _make_score(measures=1)
        completed._add_note_one("C4", "quarter", measure=1, beat=1)
        completed_note = list(
            completed.score.recurse().getElementsByClass(m21note.Note)
        )[0]
        completed_note.stemDirection = "down"
        _reshape_rest(completed, "half", measure=1, beat=2, dots=1)
        assert completed_note.stemDirection == "up"

    def test_public_voice_numbers_are_limited_to_one_through_four(self) -> None:
        """Voice-scoped note APIs reject unsupported voice values."""
        ss = _make_score(measures=1)

        with pytest.raises(ValueError, match="between 1 and 4"):
            ss._add_note_one("C4", voice=5)
        with pytest.raises(TypeError, match="integer from 1 to 4"):
            ss._add_note_one("C4", voice="2")
        with pytest.raises(ValueError, match="between 1 and 4"):
            ss.get_notes(voice=5)
        with pytest.raises(ValueError, match="between 1 and 4"):
            ss.remove_notes(measure=1, part=0, voice=5, notes=[{"beat": 1.0}])


# ==================================================================
# Voice management
# ==================================================================

class TestVoices:
    """Tests for multi-voice support."""

    def test_add_to_voice_2(self):
        ss = _make_score()
        result = ss._add_note_one("C4", measure=1, beat=1, voice=2)
        assert result.success
        assert result.details["voice"] == 2

    def test_separate_voices_independent(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1, voice=1)
        ss._add_note_one("E3", "quarter", measure=1, beat=1, voice=2)
        v1_notes = ss.get_notes(measure=1, voice=1)
        v2_notes = ss.get_notes(measure=1, voice=2)
        assert len(v1_notes) >= 1
        assert len(v2_notes) >= 1

    def test_add_note_exports_parallel_voices(self):
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C5", "whole", measure=1, beat=1, voice=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=1, voice=2)
        ss._add_note_one("D4", "quarter", measure=1, beat=2, voice=2)
        ss._add_note_one("E4", "quarter", measure=1, beat=3, voice=2)
        ss._add_note_one("F4", "quarter", measure=1, beat=4, voice=2)

        xml = ss.to_musicxml_string()

        assert "<backup>" in xml
        assert xml.count("<voice>1</voice>") == 1
        assert xml.count("<voice>2</voice>") == 4

    def test_add_note_exports_parallel_voices_when_voice_two_is_first(self):
        ss = _make_score(time_signature="4/4")
        ss._add_note_one("C4", "quarter", measure=1, beat=1, voice=2)
        ss._add_note_one("C5", "quarter", measure=1, beat=1, voice=1)

        xml = ss.to_musicxml_string()

        assert "<backup>" in xml
        assert "<voice>1</voice>" in xml
        assert "<voice>2</voice>" in xml

    def test_get_notes_voice_filter(self):
        ss = _make_score()
        ss._add_note_one("C4", measure=1, beat=1, voice=2)
        notes_v1 = ss.get_notes(measure=1, voice=1)
        notes_v2 = ss.get_notes(measure=1, voice=2)
        assert all(n.voice == 1 for n in notes_v1)
        assert any(n.voice == 2 for n in notes_v2)


# ==================================================================
# get_notes
# ==================================================================

class TestGetNotes:
    """Tests for querying notes."""

    def test_get_notes_empty_measure(self):
        ss = _make_score()
        notes = ss.get_notes(measure=1)
        assert [(note.pitch, note.is_rest, note.duration_type) for note in notes] == [
            ("rest", True, "whole")
        ]

    def test_get_notes_specific_measure(self):
        ss = _make_score(measures=3)
        ss._add_note_one("C4", measure=2, beat=1)
        notes = ss.get_notes(measure=2)
        sounding = [note for note in notes if not note.is_rest]
        assert len(sounding) == 1
        assert sounding[0].pitch == "C4"
        assert sounding[0].measure_number == 2

    def test_get_notes_all_measures(self):
        ss = _make_score(measures=3)
        ss._add_note_one("C4", measure=1, beat=1)
        ss._add_note_one("D4", measure=2, beat=1)
        ss._add_note_one("E4", measure=3, beat=1)
        notes = ss.get_notes()
        note_pitches = [n.pitch for n in notes if not n.is_rest]
        assert "C4" in note_pitches
        assert "D4" in note_pitches
        assert "E4" in note_pitches

    def test_get_notes_specific_part(self):
        ss = _make_score(parts=["violin", "cello"], measures=2)
        ss._add_note_one("G4", measure=1, beat=1, part=0)
        ss._add_note_one("C3", measure=1, beat=1, part=1)
        violin_notes = ss.get_notes(part=0)
        cello_notes = ss.get_notes(part=1)
        assert any(n.pitch == "G4" for n in violin_notes)
        assert any(n.pitch == "C3" for n in cello_notes)

    def test_get_notes_includes_rests(self):
        ss = _make_score()
        _reshape_rest(ss, "quarter", measure=1, beat=1)
        notes = ss.get_notes(measure=1)
        rests = [n for n in notes if n.is_rest]
        assert [(rest.beat, rest.duration_type) for rest in rests] == [
            (1.0, "quarter"),
            (2.0, "half"),
        ]

    def test_get_notes_includes_chords(self):
        ss = _make_score()
        ss.add_chord(["C4", "E4", "G4"], measure=1, beat=1)
        notes = ss.get_notes(measure=1)
        chord_notes = [n for n in notes if n.is_chord]
        assert len(chord_notes) == 3

    def test_note_info_fields(self):
        ss = _make_score()
        ss._add_note_one("E4", "half", measure=1, beat=1)
        notes = ss.get_notes(measure=1)
        n = notes[0]
        assert n.pitch == "E4"
        assert n.octave == 4
        assert n.duration_type == "half"
        assert n.quarter_length == 2.0
        assert n.measure_number == 1
        assert n.beat == 1.0
        assert n.part_index == 0
        assert n.voice == 1
        assert not n.is_chord
        assert not n.is_rest
        assert not n.is_tied
        assert not n.is_grace

    def test_get_notes_all_parts_default(self):
        ss = _make_score(parts=["violin", "cello"], measures=2)
        ss._add_note_one("A4", measure=1, beat=1, part=0)
        ss._add_note_one("A2", measure=1, beat=1, part=1)
        notes = ss.get_notes(measure=1)
        parts_found = {n.part_index for n in notes}
        assert 0 in parts_found
        assert 1 in parts_found


# ==================================================================
# 3/4 and 6/8 time signatures
# ==================================================================

class TestOtherTimeSignatures:
    """Verify behaviour in non-4/4 meters."""

    def test_three_four_capacity(self):
        ss = _make_score(time_signature="3/4")
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=2)
        ss._add_note_one("E4", "quarter", measure=1, beat=3)
        notes = ss.get_notes(measure=1)
        assert len(notes) == 3

    def test_three_four_overflow(self):
        ss = _make_score(time_signature="3/4")
        with pytest.raises(ValueError, match="exceed"):
            ss._add_note_one("C4", "whole", measure=1, beat=1)

    def test_six_eight_capacity(self):
        ss = _make_score(time_signature="6/8")
        result = ss._add_note_one("C4", "quarter", measure=1, beat=1)
        assert result.success


# ==================================================================
# No measures edge case
# ==================================================================

class TestNoMeasures:
    """Error handling when the score has no measures."""

    def test_add_note_no_measures(self):
        ss = ScoreSpeak.create(measures=0)
        with pytest.raises(ValueError, match="[Nn]o measures"):
            ss._add_note_one("C4")

    def test_reshape_rest_no_measures(self):
        ss = ScoreSpeak.create(measures=0)
        with pytest.raises(ValueError, match="no measures"):
            _reshape_rest(ss, "quarter", measure=1, beat=1)


# ==================================================================
# Integration: mixed operations
# ==================================================================

class TestIntegration:
    """End-to-end scenarios mixing multiple operations."""

    def test_add_then_remove_then_add(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        _remove_one_note(ss, measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=1)
        notes = ss.get_notes(measure=1)
        sounding = [note for note in notes if not note.is_rest]
        assert len(sounding) == 1
        assert sounding[0].pitch == "D4"

    def test_add_note_and_tie_across_measures(self):
        ss = _make_score(measures=2)
        ss._add_note_one("C4", "quarter", measure=1, beat=4)
        ss._add_note_one("C4", "quarter", measure=2, beat=1)
        ss.add_tie(measure=1, beat=4)
        notes_m1 = ss.get_notes(measure=1)
        notes_m2 = ss.get_notes(measure=2)
        assert any(n.is_tied for n in notes_m1)
        assert any(n.is_tied for n in notes_m2)

    def test_chord_then_remove_one_pitch(self):
        ss = _make_score()
        ss.add_chord(["C4", "E4", "G4"], "half", measure=1, beat=1)
        _remove_one_note(ss, measure=1, beat=1, pitch="G4")
        notes = ss.get_notes(measure=1)
        pitches = {n.pitch for n in notes if n.is_chord}
        assert pitches == {"C4", "E4"}
