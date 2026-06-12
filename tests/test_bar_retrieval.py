"""Tests for compact score bar retrieval."""

import pytest

from scorespeak import ScoreSpeak


def _make_score(measures: int = 2, parts: int = 2) -> ScoreSpeak:
    """Create a score with stable part names for bar retrieval tests."""
    part_specs = [
        {"instrument": "violin", "name": "Violin"},
        {"instrument": "cello", "name": "Cello"},
        {"instrument": "flute", "name": "Flute"},
    ]
    return ScoreSpeak.create(
        measures=measures,
        time_signature="4/4",
        parts=part_specs[:parts],
    )


def test_bar_result_set_no_query_returns_compact_bar_result_set():
    """No query returns all measures with compact shared schemas."""
    ss = _make_score(measures=2, parts=2)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss._add_note_one("G2", "half", measure=1, beat=1, part=1)
    ss._add_note_one("D4", "quarter", measure=2, beat=1, part=0)

    result = ss._build_bar_result_set()

    assert result["event_schema"] == [
        "kind", "beat", "pitch", "duration", "tie_status", "is_grace",
        "dots", "grace_slash", "grace_duration",
    ]
    assert result["tuplet_schema"] == [
        "ratio",
        "beat_range",
    ]
    assert result["marking_schema"] == [
        "type",
        "payload",
        "beat",
    ]
    assert result["span_schema"] == [
        "type",
        "payload",
        "flags",
        "beat_range",
    ]
    assert "active" in result["bar_notation_keys"]
    assert "changed_here" in result["bar_notation_keys"]
    assert result["part_notation_keys"] == [
        "clef",
        "key",
        "concert_key",
        "key_space",
        "key_role",
        "key_is_transposed",
        "key_label",
    ]
    assert [bar["measure_number"] for bar in result["bars"]] == [1, 2]
    assert [part["part_index"] for part in result["bars"][0]["parts"]] == [0, 1]
    assert result["bars"][0]["notation"]["active"]["time"] == "4/4"
    assert result["bars"][0]["notation"]["active"]["key"] == "C major"
    assert result["bars"][0]["notation"]["active"]["concert_key"] == "C major"
    assert result["bars"][0]["notation"]["active"]["key_space"] == "concert"
    assert result["bars"][0]["notation"]["active"]["tempo"] == 120.0
    assert "tempo" not in result["bars"][1]["notation"]["active"]


def test_bar_result_set_reports_concert_key_and_part_written_key() -> None:
    """Transposing parts expose written key while bar notation stays concert."""
    ss = ScoreSpeak.create(
        measures=1,
        key_signature="F",
        parts=["flute", "clarinet"],
    )

    result = ss._build_bar_result_set()
    bar = result["bars"][0]

    assert bar["notation"]["active"]["key"] == "F major"
    assert bar["notation"]["active"]["concert_key"] == "F major"
    assert "key" not in bar["parts"][0]["notation"]
    clarinet_key = bar["parts"][1]["notation"]
    assert clarinet_key["key"] == "G major"
    assert clarinet_key["concert_key"] == "F major"
    assert clarinet_key["key_space"] == "written pitch"
    assert clarinet_key["key_role"] == "transposed_written_key"
    assert clarinet_key["key_is_transposed"] is True
    assert clarinet_key["key_label"] == "written key: G major (concert key: F major)"


def test_bar_result_set_labels_local_key_override() -> None:
    """Local key overrides are not mislabeled as transposed written keys."""
    ss = ScoreSpeak.create(
        measures=1,
        key_signature="F",
        parts=["flute", "clarinet"],
    )
    ss._set_local_key_signature("open", 1, part=1)

    result = ss._build_bar_result_set()
    notation = result["bars"][0]["parts"][1]["notation"]

    assert notation["key"] == "open/atonal"
    assert notation["concert_key"] == "F major"
    assert notation["key_role"] == "local_staff_key"
    assert notation["key_is_transposed"] is False
    assert notation["key_label"] == "local key: open/atonal (concert key: F major)"


def test_bar_result_set_match_in_one_part_returns_all_scoped_parts():
    """A matching note qualifies the bar for all scoped parts."""
    ss = _make_score(measures=2, parts=2)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss._add_note_one("G2", "quarter", measure=1, beat=1, part=1)

    result = ss._build_bar_result_set({
        "scope": {"parts": [0, 1]},
        "match": {"sequence": [{"kind": "note", "pitch": "C4"}]},
    })

    assert [bar["measure_number"] for bar in result["bars"]] == [1]
    assert [part["part_index"] for part in result["bars"][0]["parts"]] == [0, 1]


def test_bar_result_set_chord_exact_and_contains_modes():
    """Chord queries support both exact and contains pitch-class matching."""
    ss = _make_score(measures=1, parts=1)
    ss.add_chord(["C4", "E4", "G4", "Bb4"], "half", measure=1, beat=1, part=0)

    exact_result = ss._build_bar_result_set({
        "match": {
            "sequence": [{
                "kind": "chord",
                "pitch_classes": ["C", "E", "G"],
                "duration": "half",
            }],
        },
        "options": {"chord_mode": "exact"},
    })
    contains_result = ss._build_bar_result_set({
        "match": {
            "sequence": [{
                "kind": "chord",
                "pitch_classes": ["C", "E", "G"],
                "duration": "half",
            }],
        },
        "options": {"chord_mode": "contains"},
    })

    assert exact_result["bars"] == []
    assert [bar["measure_number"] for bar in contains_result["bars"]] == [1]


def test_bar_result_set_cross_bar_match_qualifies_each_touched_measure():
    """A sequence spanning a barline returns each touched measure."""
    ss = _make_score(measures=2, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=4, part=0)
    ss._add_note_one("D4", "quarter", measure=2, beat=1, part=0)

    result = ss._build_bar_result_set({
        "match": {
            "sequence": [
                {"kind": "note", "pitch": "C4"},
                {"kind": "note", "pitch": "D4"},
            ],
        },
    })

    assert [bar["measure_number"] for bar in result["bars"]] == [1, 2]


def test_bar_result_set_voice_scope_limits_returned_voices():
    """Scoped voices restrict both matching and returned content."""
    ss = _make_score(measures=1, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0, voice=1)
    ss._add_note_one("E4", "quarter", measure=1, beat=1, part=0, voice=2)

    result = ss._build_bar_result_set({
        "scope": {"voices": [2]},
    })

    voices = result["bars"][0]["parts"][0]["voices"]
    assert [voice["voice"] for voice in voices] == [2]
    assert voices[0]["events"][0][2] == "E4"


def test_bar_result_set_voice_scope_rejects_unsupported_voice() -> None:
    """Scoped bar retrieval rejects unsupported public voice numbers."""
    ss = _make_score(measures=1, parts=1)

    with pytest.raises(ValueError, match="between 1 and 4"):
        ss._build_bar_result_set({"scope": {"voices": [5]}})


def test_bar_result_set_voice_scope_rejects_non_integer_voice() -> None:
    """Scoped bar retrieval rejects non-integer voice numbers."""
    ss = _make_score(measures=1, parts=1)

    with pytest.raises(TypeError, match="integer from 1 to 4"):
        ss._build_bar_result_set({"scope": {"voices": ["2"]}})


def test_bar_result_set_tuplets_are_preserved_as_voice_spans():
    """Tuplets are encoded once per voice as compact span rows."""
    ss = _make_score(measures=1, parts=1)
    ss.add_tuplet(
        [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")],
        actual_notes=3,
        normal_notes=2,
        measure=1,
        beat=1,
        part=0,
    )

    result = ss._build_bar_result_set()
    voice_payload = result["bars"][0]["parts"][0]["voices"][0]

    tuplet_row = voice_payload["tuplets"][0]
    assert tuplet_row[0] == [3, 2]
    assert tuplet_row[1] == [1.0, pytest.approx(1.0 + 2.0 / 3.0)]
    assert voice_payload["events"][0][3] == pytest.approx(1.0 / 3.0)


def test_bar_result_set_missing_measure_returns_empty_part_payload():
    """Parts missing a qualifying measure remain present with empty voices."""
    ss = _make_score(measures=2, parts=2)
    ss.add_measures(1)
    second_part = list(ss.score.parts)[1]
    second_part.remove(second_part.measure(3))
    ss._add_note_one("C4", "quarter", measure=3, beat=1, part=0)

    result = ss._build_bar_result_set({
        "scope": {"bar_range": (3, 3)},
    })

    assert [bar["measure_number"] for bar in result["bars"]] == [3]
    assert result["bars"][0]["parts"][0]["voices"][0]["events"][0][2] == "C4"
    assert result["bars"][0]["parts"][1]["voices"] == []


def test_bar_result_set_measure_numbers_support_disconnected_bars():
    """Structured scope may target disconnected measures directly."""
    ss = _make_score(measures=6, parts=1)
    ss._add_note_one("C4", "quarter", measure=2, beat=1, part=0)
    ss._add_note_one("D4", "quarter", measure=5, beat=1, part=0)

    result = ss._build_bar_result_set({
        "scope": {"measure_numbers": [5, 2, 5, 9]},
    })

    assert [bar["measure_number"] for bar in result["bars"]] == [2, 5]


def test_bar_result_set_validates_query_fields():
    """Invalid query shapes raise clear validation errors."""
    ss = _make_score(measures=1, parts=1)

    with pytest.raises(ValueError, match="query.scope.parts"):
        ss._build_bar_result_set({"scope": {"parts": []}})

    with pytest.raises(ValueError, match="bar_range' and 'measure_numbers"):
        ss._build_bar_result_set({"scope": {"bar_range": (1, 1), "measure_numbers": [1]}})

    with pytest.raises(ValueError, match="query.match.sequence\\[0\\]\\.kind"):
        ss._build_bar_result_set({"match": {"sequence": [{"kind": "cluster"}]}})

    with pytest.raises(ValueError, match="pitch_classes"):
        ss._build_bar_result_set({"match": {"sequence": [{"kind": "chord"}]}})

    with pytest.raises(ValueError, match="unsupported field"):
        ss._build_bar_result_set({"match": {"markings": []}})

def test_bar_result_set_is_not_public_api():
    """The bar payload builder is internal; agents use search_score."""
    ss = _make_score(measures=2, parts=1)

    assert not hasattr(ss, "find_bars")


def test_bar_result_set_rejects_string_queries():
    """The compact string DSL is no longer accepted."""
    ss = _make_score(measures=1, parts=1)

    with pytest.raises(TypeError, match="query must be a dict"):
        ss._build_bar_result_set('part:"Violin" bars:1-1 / C4:quarter /')


def test_search_score_event_filters_return_match_reasons():
    """Typed score search can locate events without the old query DSL."""
    ss = _make_score(measures=2, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss._add_note_one("D4", "half", measure=2, beat=1, part=0)

    result = ss.search_score(pitch="D4", duration="half")

    assert [bar["measure_number"] for bar in result["bars"]] == [2]
    assert result["match_schema"] == [
        "channel", "detail", "part_index", "voice", "beat", "beat_range",
    ]
    assert result["bars"][0]["matches"][0][0] == "event"
    assert result["bars"][0]["matches"][0][4] == 1.0


def test_search_score_event_sequence_preserves_pattern_search():
    """Typed event sequences replace the removed compact string pattern."""
    ss = _make_score(measures=2, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=4, part=0)
    ss._add_note_one("D4", "quarter", measure=2, beat=1, part=0)

    result = ss.search_score(event_sequence=[
        {"kind": "note", "pitch": "C4"},
        {"kind": "note", "pitch": "D4"},
    ])

    assert [bar["measure_number"] for bar in result["bars"]] == [1, 2]
    assert all(bar["matches"][0][0] == "sequence" for bar in result["bars"])


def test_search_score_limit_returns_first_matches_with_metadata():
    """Search limits cap hydrated bars without losing total match metadata."""
    ss = _make_score(measures=3, parts=1)
    for measure in range(1, 4):
        ss._add_note_one("C4", "quarter", measure=measure, beat=1, part=0)

    result = ss.search_score(pitch="C4", limit=2)

    assert [bar["measure_number"] for bar in result["bars"]] == [1, 2]
    assert result["search_metadata"] == {
        "limit": 2,
        "total_matches": 3,
        "returned_matches": 2,
        "truncated": True,
    }
    assert all(bar["matches"][0][0] == "event" for bar in result["bars"])


def test_search_score_limit_without_filters_returns_scope_prefix():
    """Unfiltered search limits avoid hydrating the whole scoped score."""
    ss = _make_score(measures=4, parts=1)

    result = ss.search_score(bar_range=(2, 4), limit=2)

    assert [bar["measure_number"] for bar in result["bars"]] == [2, 3]
    assert result["search_metadata"] == {
        "limit": 2,
        "total_matches": 3,
        "returned_matches": 2,
        "truncated": True,
    }
    assert "match_schema" not in result


# ----------------------------------------------------------------------
# Event row cleanup: chord pitch layout and tie_status
# ----------------------------------------------------------------------


def test_bar_result_set_chord_event_emits_only_pitches_with_octave():
    """Chord rows no longer duplicate pitch-class info alongside octaves."""
    ss = _make_score(measures=1, parts=1)
    ss.add_chord(["C4", "E4", "G4"], "half", measure=1, beat=1, part=0)

    result = ss._build_bar_result_set()
    chord_row = result["bars"][0]["parts"][0]["voices"][0]["events"][0]

    assert chord_row[0] == "chord"
    assert chord_row[2] == ["C4", "E4", "G4"]


def test_bar_result_set_note_tie_status_values():
    """Tied notes report ``start``/``stop`` tie_status at the event row."""
    ss = _make_score(measures=2, parts=1)
    ss._add_note_one("C4", "whole", measure=1, beat=1, part=0)
    ss._add_note_one("C4", "whole", measure=2, beat=1, part=0)
    ss.add_tie(measure=1, beat=1, part=0)

    result = ss._build_bar_result_set()

    first_event = result["bars"][0]["parts"][0]["voices"][0]["events"][0]
    second_event = result["bars"][1]["parts"][0]["voices"][0]["events"][0]
    assert first_event[4] == "start"
    assert second_event[4] == "stop"


def test_bar_result_set_untied_note_tie_status_is_none():
    """Plain notes have ``None`` tie_status so tied runs stand out."""
    ss = _make_score(measures=1, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)

    result = ss._build_bar_result_set()
    event = result["bars"][0]["parts"][0]["voices"][0]["events"][0]
    assert event[4] is None


def test_bar_result_set_grace_slash_round_trip_field():
    """Grace rows carry slash and duration state while normal rows use None."""
    ss = _make_score(measures=1, parts=1)
    ss._add_note_one("C4", "whole", measure=1, beat=1, part=0)
    ss.add_grace_note(
        "D4",
        duration="16th",
        measure=1,
        beat=1,
        part=0,
        slash=False,
    )

    result = ss._build_bar_result_set()
    events = result["bars"][0]["parts"][0]["voices"][0]["events"]
    grace_event = next(row for row in events if row[5] is True)
    normal_event = next(row for row in events if row[5] is False)

    assert grace_event[7] is False
    assert grace_event[8] == "16th"
    assert normal_event[7] is None
    assert normal_event[8] is None


# ----------------------------------------------------------------------
# Sparse markings side-channel
# ----------------------------------------------------------------------


def test_bar_result_set_markings_collect_dynamic_and_articulation():
    """Dynamics and articulations land in the per-voice markings channel."""
    ss = _make_score(measures=1, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss._add_note_one("D4", "quarter", measure=1, beat=2, part=0)
    ss.add_dynamic("ff", measure_number=1, beat=1, part=0)
    ss.add_articulation("staccato", measure_number=1, beat=2, part=0)

    result = ss._build_bar_result_set()
    voice = result["bars"][0]["parts"][0]["voices"][0]

    assert ["dynamic", "ff", 1.0] in voice["markings"]
    assert ["articulation", "staccato", 2.0] in voice["markings"]


def test_bar_result_set_measure_level_markings_keep_actual_beat():
    """Measure-level markings report their real beat, not just anchor beat."""
    ss = ScoreSpeak.create(
        measures=1,
        time_signature="2/4",
        parts=[{"instrument": "violin", "name": "Violin"}],
    )
    ss._add_note_one("C4", "half", measure=1, beat=1, part=0)
    ss.add_dynamic("p", measure_number=1, beat=1.5, part=0)

    result = ss._build_bar_result_set()
    voice = result["bars"][0]["parts"][0]["voices"][0]

    assert ["dynamic", "p", 1.5] in voice["markings"]


def test_bar_result_set_markings_absent_when_no_notations():
    """Voices without markings omit the ``markings`` key entirely."""
    ss = _make_score(measures=1, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)

    result = ss._build_bar_result_set()
    voice = result["bars"][0]["parts"][0]["voices"][0]
    assert "markings" not in voice


def test_bar_result_set_markings_ornament_and_lyric():
    """Ornaments and lyrics are reported as their own marking types."""
    ss = _make_score(measures=1, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss.add_ornament("trill", measure_number=1, beat=1, part=0)
    ss.add_lyric("la", measure_number=1, beat=1, part=0)

    result = ss._build_bar_result_set()
    voice = result["bars"][0]["parts"][0]["voices"][0]
    markings = voice["markings"]

    types_present = {row[0] for row in markings}
    assert "ornament" in types_present
    assert "lyric" in types_present


def test_bar_result_set_markings_fingering_and_chord_symbol():
    """Fingerings and chord symbols surface with their own marking types."""
    ss = _make_score(measures=1, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss.add_fingering(3, measure_number=1, beat=1, part=0)
    ss.add_chord_symbol("Cmaj7", measure_number=1, beat=1, part=0)

    result = ss._build_bar_result_set()
    voice = result["bars"][0]["parts"][0]["voices"][0]
    markings = voice["markings"]

    types_present = {row[0] for row in markings}
    assert "fingering" in types_present
    assert "chord_symbol" in types_present


def test_search_score_marking_filter_finds_articulation():
    """Typed score search can locate point markings."""
    ss = _make_score(measures=2, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss._add_note_one("D4", "quarter", measure=2, beat=1, part=0)
    ss.add_articulation("staccato", measure_number=2, beat=1, part=0)

    result = ss.search_score(marking_type="articulation", marking_value="staccato")

    assert [bar["measure_number"] for bar in result["bars"]] == [2]
    assert result["bars"][0]["matches"][0][0] == "marking"


# ----------------------------------------------------------------------
# Sparse spans side-channel with LR continuation flags
# ----------------------------------------------------------------------


def test_bar_result_set_hairpin_span_within_one_bar():
    """A hairpin that lives in one bar emits empty flags."""
    ss = _make_score(measures=1, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss._add_note_one("D4", "quarter", measure=1, beat=2, part=0)
    ss.add_hairpin(
        "crescendo",
        start_measure=1,
        start_beat=1,
        end_measure=1,
        end_beat=2,
        part=0,
    )

    result = ss._build_bar_result_set()
    spans = result["bars"][0]["parts"][0]["voices"][0]["spans"]
    hairpin_rows = [row for row in spans if row[0] == "hairpin"]

    assert len(hairpin_rows) == 1
    assert hairpin_rows[0] == ["hairpin", "crescendo", "", [1.0, 2.0]]


def test_bar_result_set_hairpin_span_lr_flags_across_scope_boundary():
    """Spans that cross scope get ``L`` / ``R`` continuation flags per bar."""
    ss = _make_score(measures=3, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss._add_note_one("D4", "quarter", measure=2, beat=1, part=0)
    ss._add_note_one("E4", "quarter", measure=3, beat=1, part=0)
    ss.add_hairpin(
        "crescendo",
        start_measure=1,
        start_beat=1,
        end_measure=3,
        end_beat=1,
        part=0,
    )

    result = ss._build_bar_result_set({"scope": {"bar_range": (2, 2)}})
    spans = result["bars"][0]["parts"][0]["voices"][0]["spans"]
    hairpin_rows = [row for row in spans if row[0] == "hairpin"]

    assert len(hairpin_rows) == 1
    assert hairpin_rows[0][2] == "LR"


def test_bar_result_set_slur_span_emits_slur_label():
    """Slurs are reported as ``slur`` spans with empty payload."""
    ss = _make_score(measures=1, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss._add_note_one("D4", "quarter", measure=1, beat=2, part=0)
    ss.add_slur(
        start_measure=1,
        start_beat=1,
        end_measure=1,
        end_beat=2,
        part=0,
    )

    result = ss._build_bar_result_set()
    spans = result["bars"][0]["parts"][0]["voices"][0]["spans"]

    slurs = [row for row in spans if row[0] == "slur"]
    assert len(slurs) == 1
    assert slurs[0] == ["slur", "", "", [1.0, 2.0]]


def test_search_score_span_filter_finds_hairpin():
    """Typed score search can locate span notation."""
    ss = _make_score(measures=2, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss._add_note_one("D4", "quarter", measure=2, beat=1, part=0)
    ss.add_hairpin(
        "crescendo",
        start_measure=1,
        start_beat=1,
        end_measure=2,
        end_beat=1,
        part=0,
    )

    result = ss.search_score(span_type="hairpin", span_value="crescendo")

    assert [bar["measure_number"] for bar in result["bars"]] == [1, 2]
    assert all(bar["matches"][0][0] == "span" for bar in result["bars"])


# ----------------------------------------------------------------------
# Per-bar notation (active + structural fields)
# ----------------------------------------------------------------------


def test_bar_result_set_bar_notation_active_time_key_on_every_bar():
    """Every returned bar carries active time and key in its notation."""
    ss = _make_score(measures=3, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)

    result = ss._build_bar_result_set()
    for bar in result["bars"]:
        assert bar["notation"]["active"]["time"] == "4/4"
        assert bar["notation"]["active"]["key"] == "C major"
        assert bar["notation"]["active"]["concert_key"] == "C major"


def test_bar_result_set_bar_notation_tempo_on_first_bar_and_change_bars():
    """Tempo appears on scope bar 1 and any bar where it changes."""
    ss = _make_score(measures=3, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss.set_tempo(90, measure_number=2, part=0)

    result = ss._build_bar_result_set()
    bar_notations = {
        bar["measure_number"]: bar["notation"]
        for bar in result["bars"]
    }
    assert bar_notations[1]["active"]["tempo"] == 120.0
    assert bar_notations[2]["active"]["tempo"] == 90.0
    assert "tempo" in bar_notations[2].get("changed_here", [])
    assert "tempo" not in bar_notations[3]["active"]


def test_bar_result_set_bar_notation_repeat_flags():
    """Repeat barlines surface as ``repeat_start`` / ``repeat_end``."""
    ss = _make_score(measures=3, parts=1)
    ss.add_repeat(start_measure=1, end_measure=2, times=2)

    result = ss._build_bar_result_set()
    bar1 = result["bars"][0]["notation"]
    bar2 = result["bars"][1]["notation"]
    assert bar1.get("repeat_start") is True
    assert bar2.get("repeat_end") is True


def test_search_score_structure_filter_finds_repeat_end():
    """Typed score search can locate structural notation."""
    ss = _make_score(measures=3, parts=1)
    ss.add_repeat(start_measure=1, end_measure=2, times=2)

    result = ss.search_score(structure="repeat_end")

    assert [bar["measure_number"] for bar in result["bars"]] == [2]
    assert result["bars"][0]["matches"][0][0] == "structure"


def test_search_score_structure_filter_finds_page_break():
    """Layout breaks are searchable as structural notation."""
    ss = _make_score(measures=3, parts=1)
    ss.add_page_break(2)

    result = ss.search_score(structure="page_break")

    assert [bar["measure_number"] for bar in result["bars"]] == [2]


def test_bar_result_set_navigation_to_coda_and_fine():
    """To Coda and Fine markers surface as navigation notation."""
    ss = _make_score(measures=3, parts=1)
    ss.add_to_coda(2)
    ss.add_fine(3)

    result = ss._build_bar_result_set()

    assert result["bars"][1]["notation"]["navigation"] == ["to_coda"]
    assert result["bars"][2]["notation"]["navigation"] == ["fine"]


def test_bar_result_set_navigation_coda_destination():
    """Destination Coda signs surface separately from To Coda markers."""
    ss = _make_score(measures=2, parts=1)
    ss.add_coda(2)

    result = ss._build_bar_result_set()

    assert result["bars"][1]["notation"]["navigation"] == ["coda"]


def test_bar_result_set_navigation_keeps_coda_and_to_coda_distinct():
    """Coda destination and To Coda jump markers keep different labels."""
    ss = _make_score(measures=3, parts=1)
    ss.add_to_coda(2)
    ss.add_coda(3)

    result = ss._build_bar_result_set()

    assert result["bars"][1]["notation"]["navigation"] == ["to_coda"]
    assert result["bars"][2]["notation"]["navigation"] == ["coda"]


def test_bar_result_set_navigation_to_coda_round_trips_musicxml():
    """A To Coda marker remains a To Coda marker after MusicXML round-trip."""
    ss = _make_score(measures=2, parts=1)
    ss.add_to_coda(2)

    round_tripped = ScoreSpeak.from_musicxml(ss.to_musicxml_string())
    result = round_tripped._build_bar_result_set()

    assert result["bars"][1]["notation"]["navigation"] == ["to_coda"]


def test_bar_result_set_ignores_legacy_to_coda_text_expression():
    """Legacy TextExpression To Coda values are not navigation marks."""
    from music21 import expressions as m21expressions

    ss = _make_score(measures=2, parts=1)
    part_obj, _ = ss._resolve_part(None)
    measure_obj = ss._resolve_measure(part_obj, 2)
    measure_obj.insert(0, m21expressions.TextExpression("To Coda"))

    result = ss._build_bar_result_set()

    assert "navigation" not in result["bars"][1]["notation"]


def test_bar_result_set_bar_notation_rehearsal_mark():
    """Rehearsal marks surface on the matching bar's notation."""
    ss = _make_score(measures=2, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss.add_rehearsal_mark("A", measure_number=2)

    result = ss._build_bar_result_set()
    bar2 = result["bars"][1]["notation"]
    assert bar2.get("rehearsal_mark") == "A"


def test_search_score_attribute_filter_finds_time_change():
    """Typed score search can locate active and changed attributes."""
    ss = _make_score(measures=3, parts=1)
    ss.set_time_signature("3/4", measure_number=2)

    result = ss.search_score(
        time_signature="3/4",
        changed_attribute="time",
        logic="all",
    )

    assert [bar["measure_number"] for bar in result["bars"]] == [2]
    assert {row[0] for row in result["bars"][0]["matches"]} == {"attribute"}


# ----------------------------------------------------------------------
# Per-part notation (clef)
# ----------------------------------------------------------------------


def test_bar_result_set_part_notation_clef_on_scope_bar_one():
    """The first bar of the scope always reports the active clef."""
    ss = _make_score(measures=2, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)

    result = ss._build_bar_result_set()
    part = result["bars"][0]["parts"][0]
    assert part.get("notation", {}).get("clef") == "treble"


def test_bar_result_set_part_notation_clef_absent_on_unchanged_bars():
    """Bars after scope bar 1 omit the clef unless it changes."""
    ss = _make_score(measures=3, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)

    result = ss._build_bar_result_set({"scope": {"bar_range": (1, 3)}})
    bar2_part = result["bars"][1]["parts"][0]
    assert "notation" not in bar2_part or "clef" not in bar2_part.get("notation", {})


# ----------------------------------------------------------------------
# Sparse-by-default guarantees
# ----------------------------------------------------------------------


def test_bar_result_set_empty_score_returns_empty_bars_list():
    """A score with no measures emits an empty ``bars`` list cleanly."""
    ss = ScoreSpeak.create(title="Empty", parts=["piano"])
    result = ss._build_bar_result_set()
    assert result["bars"] == []


def test_bar_result_set_voice_without_markings_or_spans_omits_channels():
    """Voices that carry only events have no ``markings`` or ``spans`` keys."""
    ss = _make_score(measures=1, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)

    result = ss._build_bar_result_set()
    voice = result["bars"][0]["parts"][0]["voices"][0]
    assert "markings" not in voice
    assert "spans" not in voice


# ----------------------------------------------------------------------
# Grand-staff display names and hand labels
# ----------------------------------------------------------------------


def _make_piano_score_for_bar_tests():
    """Build a minimal piano grand-staff score for bar-context assertions."""
    from music21 import clef, layout, note, stream

    rh = stream.PartStaff(); rh.partName = "Piano"
    rh.append(clef.TrebleClef())
    measure = stream.Measure(number=1); measure.append(note.Note("C4"))
    rh.append(measure)

    lh = stream.PartStaff(); lh.partName = "Piano"
    lh.append(clef.BassClef())
    measure = stream.Measure(number=1); measure.append(note.Note("C3"))
    lh.append(measure)

    score = stream.Score()
    score.insert(0, rh); score.insert(0, lh)
    score.insert(0, layout.StaffGroup([rh, lh], name="Piano", symbol="brace"))
    return ScoreSpeak(score)


def test_bar_result_set_labels_piano_rh_lh_in_part_name():
    """``BarPart.part_name`` reflects the display label for grand-staff parts."""
    ss = _make_piano_score_for_bar_tests()
    result = ss._build_bar_result_set({"scope": {"bar_range": (1, 1)}})

    parts = result["bars"][0]["parts"]
    assert parts[0]["part_name"] == "Piano RH"
    assert parts[0]["hand"] == "RH"
    assert parts[1]["part_name"] == "Piano LH"
    assert parts[1]["hand"] == "LH"


def test_bar_result_set_omits_hand_field_for_non_grouped_parts():
    """Parts outside any brace group keep their raw name and omit ``hand``."""
    ss = _make_score(measures=1, parts=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    result = ss._build_bar_result_set({"scope": {"bar_range": (1, 1)}})

    part = result["bars"][0]["parts"][0]
    assert part["part_name"] == "Violin"
    assert "hand" not in part
