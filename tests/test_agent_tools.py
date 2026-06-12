"""
Tests for :mod:`scorespeak.agent.overview` and :mod:`scorespeak.agent.tools`.

These tests deliberately avoid hitting any LLM or network.  They exercise
the helpers that build the per-turn context and wrap ScoreSpeak methods
into LangChain tools, plus the turn-summarization helper in ``graph``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scorespeak import ScoreSpeak
from scorespeak.agent.context_renderers import (
    render_exact_context,
    render_summary_context,
)
from scorespeak.agent.graph import (
    _extract_final_text,
    _prepare_agent_turn,
    AgentTurnRuntime,
    build_system_prompt,
    build_agent_tool_bundle,
    ScoreSpeakAgentMiddleware,
    run_turn_stream,
    stream_events_from_update,
    summarize_turn_context,
    tool_progress_label,
)
from scorespeak.agent.memory import AgentMemoryStore
from scorespeak.agent.overview import (
    ScoreOverview,
    build_score_overview,
    format_overview_for_prompt,
)
from scorespeak.retrieval import ExtractedContextScope
from scorespeak.agent.tools import (
    make_inspect_score_attributes_tool,
    make_inspect_score_region_tool,
    make_search_score_tool,
    make_tool_from_record,
    make_tools_from_records,
)
from scorespeak.retrieval import (
    LexicalContextRetriever,
    LexicalRetriever,
    MethodIndex,
)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def test_build_score_overview_reports_parts_and_bars():
    """Overview reflects parts, total bar count, and bar-1 signatures."""
    ss = ScoreSpeak.create(
        title="Demo",
        composer="Tester",
        time_signature="3/4",
        key_signature="G",
        tempo=100.0,
        parts=["Piano", "Violin"],
        measures=6,
    )

    overview = build_score_overview(ss)

    assert isinstance(overview, ScoreOverview)
    assert overview.title == "Demo"
    assert overview.composer == "Tester"
    assert overview.total_bars == 6
    assert overview.time_signature_at_bar_1 == "3/4"
    assert overview.key_signature_at_bar_1.startswith("G")
    assert overview.tempo_at_bar_1 == 100.0
    assert [p.name for p in overview.parts] == ["Piano", "Violin"]


def test_build_score_overview_handles_empty_score():
    """An empty score returns a usable overview rather than raising."""
    ss = ScoreSpeak.create(parts=[], measures=0)
    overview = build_score_overview(ss)

    assert overview.total_bars == 0
    assert overview.parts == []
    assert overview.time_signature_at_bar_1 is None
    assert overview.key_signature_at_bar_1 is None
    assert overview.tempo_at_bar_1 is None
    assert overview.pickup is False


def test_format_overview_for_prompt_contains_key_fields():
    """The rendered prompt block mentions parts, bars, and active signatures."""
    ss = ScoreSpeak.create(
        title="T",
        time_signature="4/4",
        key_signature="C",
        parts=["Violin"],
        measures=2,
    )

    text = format_overview_for_prompt(build_score_overview(ss))

    assert "Violin" in text
    assert "Total bars: 2" in text
    assert "time=4/4" in text
    assert "concert_key=C" in text


# ---------------------------------------------------------------------------
# Tool wrapping
# ---------------------------------------------------------------------------


def _record_for(ss: ScoreSpeak, method_name: str):
    """Return the MethodRecord for ``method_name`` on ``ss``."""
    index = MethodIndex(ss)
    for record in index.records:
        if record.name == method_name:
            return record
    raise AssertionError(f"method {method_name} not found in index")


def test_set_key_signature_tool_description_explains_concert_key_semantics() -> None:
    """``set_key_signature`` should expose its concert-key/all-parts behavior."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=1)

    record = _record_for(ss, "set_key_signature")
    tool = make_tool_from_record(ss, record)
    assert tool is not None

    description = " ".join(tool.description.split())
    assert "score-level concert key signature" in description
    assert "apply it to all parts" in description
    assert "concert-pitch parts get the concert key" in description
    assert "written-pitch transposing parts get the derived written key" in description
    assert "does not toggle stored pitch space" in description


def test_make_tool_from_record_builds_callable_tool():
    """``add_notes`` wraps into a StructuredTool that mutates the score."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=4)

    record = _record_for(ss, "add_notes")
    tool = make_tool_from_record(ss, record)

    assert tool is not None
    assert tool.name == "add_notes"
    assert "measure" in tool.args
    assert "part" in tool.args
    assert "voice" in tool.args
    assert "notes" in tool.args
    schema = tool.args_schema.model_json_schema()
    note_schema = schema["$defs"]["AddNotesNoteItem"]["properties"]
    assert {"pitch", "beat", "duration", "dots"}.issubset(note_schema)

    result = tool.invoke(
        {
            "measure": 1,
            "part": 0,
            "voice": 1,
            "notes": [
                {"pitch": "C4", "beat": 1.0, "duration": "quarter", "dots": 0}
            ],
        }
    )

    assert isinstance(result, str)
    assert result.startswith("OK:")
    assert "1 note" in result or "1 note(s)" in result

    notes = ss.get_notes(measure=1)
    assert len(notes) >= 1


def test_set_tempo_tool_schema_exposes_referent_description() -> None:
    """The agent-facing tempo tool exposes accepted beat-unit referents."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=1)

    record = _record_for(ss, "set_tempo")
    tool = make_tool_from_record(ss, record)

    assert tool is not None
    assert "referent" in tool.args
    assert "referent='quarter'" in tool.description

    schema = tool.args_schema.model_json_schema()
    referent_schema = schema["properties"]["referent"]
    referent_description = referent_schema["description"]

    assert referent_schema["default"] == "quarter"
    assert "Note value that" in referent_description
    assert "dotted half" in referent_description
    assert "dotted quarter" in referent_description


def test_get_active_key_signature_tool_annotates_pitch_space() -> None:
    """Agent tool output identifies concert and written key signatures."""
    ss = ScoreSpeak.create(
        parts=["flute", "clarinet"],
        measures=1,
        key_signature="F",
    )
    record = _record_for(ss, "get_active_key_signature")
    tool = make_tool_from_record(ss, record)

    assert tool is not None
    assert tool.invoke({"measure_number": 1}) == "concert key: F major"
    assert (
        tool.invoke({"measure_number": 1, "part": 1})
        == "written key: G major (concert key: F major)"
    )
    assert (
        tool.invoke({"measure_number": 1, "part": 0})
        == "part key: F major (non-transposing; concert key: F major)"
    )

    ss._set_local_key_signature("open", 1, part=1)
    assert (
        tool.invoke({"measure_number": 1, "part": 1})
        == "local key: open/atonal (concert key: F major)"
    )


def test_add_notes_tool_schema_rejects_note_scope_fields():
    """``add_notes`` item schema exposes note fields and forbids scope keys."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=1, time_signature="4/4")

    record = _record_for(ss, "add_notes")
    tool = make_tool_from_record(ss, record)
    assert tool is not None
    assert "single measure, part, and rhythmic voice" in tool.description
    assert "multiple ``add_notes`` calls" in tool.description
    assert "notes" in tool.args
    schema = tool.args_schema.model_json_schema()
    note_schema = schema["$defs"]["AddNotesNoteItem"]
    assert note_schema["additionalProperties"] is False
    assert "measure" not in note_schema["properties"]
    assert "part" not in note_schema["properties"]
    assert "voice" not in note_schema["properties"]

    result = tool.invoke({
        "measure": 1,
        "part": 0,
        "voice": 1,
        "notes": [
            {"pitch": "C4", "beat": 1.0, "duration": "quarter", "dots": 0},
            {"pitch": "D4", "beat": 2.0, "duration": "quarter", "dots": 0},
            {"pitch": "E4", "beat": 3.0, "duration": "quarter", "dots": 0},
        ],
    })

    assert result.startswith("OK:")
    notes = ss.get_notes(measure=1)
    assert [n.pitch for n in notes if not n.is_rest] == ["C4", "D4", "E4"]
    assert [n.beat for n in notes if n.is_rest] == [4.0]


def test_add_tuplet_tool_schema_explains_written_durations() -> None:
    """``add_tuplet`` should guide agents to use written note values."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=1, time_signature="4/4")

    record = _record_for(ss, "add_tuplet")
    tool = make_tool_from_record(ss, record)
    assert tool is not None

    tool_description = " ".join(tool.description.split())
    assert "written/base durations before the tuplet ratio" in tool_description
    schema = tool.args_schema.model_json_schema()
    pitch_duration_schema = schema["properties"]["pitches_and_durations"]
    description = pitch_duration_schema["description"]
    assert "not the already scaled performed length" in description
    assert "normal_notes / actual_notes" in description
    assert "0.5" in description
    assert "do not use ``1/3`` durations" in description
    assert pitch_duration_schema["items"]["minItems"] == 2
    assert pitch_duration_schema["items"]["maxItems"] == 2

    result = tool.invoke({
        "pitches_and_durations": [
            ["C4", 0.5],
            ["D4", 0.5],
            ["E4", 0.5],
        ],
        "actual_notes": 3,
        "normal_notes": 2,
        "measure": 1,
        "beat": 1,
        "part": 0,
        "voice": 1,
    })

    assert result.startswith("OK:")
    assert "'total_quarter_length': 1.0" in result


def test_remove_notes_tool_schema_rejects_note_scope_fields():
    """``remove_notes`` item schema exposes beat/pitch and forbids scope keys."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=1, time_signature="4/4")
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)

    record = _record_for(ss, "remove_notes")
    tool = make_tool_from_record(ss, record)
    assert tool is not None
    assert "single measure, part, and voice" in tool.description
    assert "notes" in tool.args
    schema = tool.args_schema.model_json_schema()
    note_schema = schema["$defs"]["RemoveNotesNoteItem"]
    assert note_schema["additionalProperties"] is False
    assert "measure" not in note_schema["properties"]
    assert "part" not in note_schema["properties"]
    assert "voice" not in note_schema["properties"]

    result = tool.invoke({
        "measure": 1,
        "part": 0,
        "voice": 1,
        "notes": [{"beat": 1.0, "pitch": "C4"}],
    })

    assert result.startswith("OK:")
    rests = [note for note in ss.get_notes(measure=1) if note.is_rest]
    assert [(rest.beat, rest.duration_type) for rest in rests] == [
        (1.0, "whole")
    ]


def test_add_grace_note_tool_schema_documents_optional_principal_slur():
    """The grace-note tool should document optional same-beat principal slurs."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=1)

    record = _record_for(ss, "add_grace_note")
    tool = make_tool_from_record(ss, record)
    assert tool is not None

    assert "requires a non-grace note" not in tool.description
    assert "duration" in tool.args
    assert "slur_to_principal" in tool.args
    assert "principal note or chord" in json.dumps(tool.args)


def test_agent_tool_schemas_reject_unknown_arguments() -> None:
    """Agent tools should reject parameters that are not in their schema."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    record = _record_for(ss, "set_barline")
    tool = make_tool_from_record(ss, record)
    assert tool is not None

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        tool.invoke({
            "barline_type": "double",
            "measure_number": 2,
            "side": "left",
        })


def test_add_fingering_tool_has_no_substitution_argument() -> None:
    """The agent-facing fingering tool should not expose substitutions."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=1)
    record = _record_for(ss, "add_fingering")
    tool = make_tool_from_record(ss, record)
    assert tool is not None

    schema = tool.args_schema.model_json_schema()

    assert "substitution" not in tool.args
    assert "substitution" not in schema["properties"]
    assert "substitution" not in tool.description.lower()
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        tool.invoke({
            "finger_number": 3,
            "measure_number": 1,
            "beat": 1.0,
            "substitution": True,
        })


def test_add_tie_tool_is_start_point_based() -> None:
    """The agent-facing tie tool should expose one selected event, not markers."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    record = _record_for(ss, "add_tie")
    tool = make_tool_from_record(ss, record)
    assert tool is not None

    schema = tool.args_schema.model_json_schema()
    properties = schema["properties"]

    assert {"measure", "beat", "part", "voice"}.issubset(properties)
    assert "tie_type" not in properties
    assert "start_measure" not in properties
    assert "start_beat" not in properties
    assert "end_measure" not in properties
    assert "end_beat" not in properties


def test_single_note_prompt_does_not_route_to_rest_completion():
    """Surgical note prompts should not surface automatic rest completion."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=1)
    retriever = LexicalRetriever(MethodIndex(ss), threshold=0.5)

    names = {
        record.name
        for record, _ in retriever.query("add a single quarter note C4 at beat 1")
    }

    assert "add_notes" in names
    assert "add_note" not in names
    assert "complete_measure_with_rests" not in names


def test_make_tool_from_record_records_successful_mutations():
    """Tool wrappers can report successful OperationResult mutations."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    recorded = []

    record = _record_for(ss, "add_notes")
    tool = make_tool_from_record(
        ss,
        record,
        lambda name, kwargs, result: recorded.append((name, kwargs, result)),
    )
    assert tool is not None

    tool.invoke({
        "measure": 2,
        "part": 0,
        "voice": 1,
        "notes": [
            {"pitch": "D4", "beat": 1.0, "duration": "quarter", "dots": 0}
        ],
    })

    assert recorded
    assert recorded[0][0] == "add_notes"
    assert recorded[0][1]["measure"] == 2
    assert recorded[0][2].success is True


def test_make_tool_from_record_skips_changed_false_noops() -> None:
    """Tool wrappers should not record successful no-op signature edits."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2, time_signature="4/4")
    recorded: list[tuple[str, dict[str, Any], Any]] = []

    record = _record_for(ss, "set_time_signature")
    tool = make_tool_from_record(
        ss,
        record,
        lambda name, kwargs, result: recorded.append((name, kwargs, result)),
    )
    assert tool is not None

    result = tool.invoke({"time_signature": "4/4", "measure_number": 2})

    assert "OK:" in result
    assert "no score change made" in result
    assert "'changed': False" in result
    assert recorded == []


def test_tool_captures_value_error_as_string():
    """Bad args produce an ERROR string instead of propagating the exception."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)

    record = _record_for(ss, "add_notes")
    tool = make_tool_from_record(ss, record)
    assert tool is not None

    result = tool.invoke(
        {
            "measure": 99,
            "part": 0,
            "voice": 1,
            "notes": [
                {"pitch": "C4", "beat": 1.0, "duration": "quarter", "dots": 0}
            ],
        }
    )

    assert result.startswith("ERROR")
    assert "99" in result or "exist" in result


def test_add_dynamic_tool_reports_existing_dynamic_replacement_instruction() -> None:
    """Agent-facing add_dynamic errors should explain the replacement workflow."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=1)
    ss.add_dynamic("p", measure_number=1, beat=1, part=0)
    record = _record_for(ss, "add_dynamic")
    tool = make_tool_from_record(ss, record)
    assert tool is not None

    result = tool.invoke({
        "level": "f",
        "measure_number": 1,
        "beat": 1,
        "part": 0,
    })
    description = " ".join(tool.description.split())

    assert "only one dynamic can exist" in description
    assert result.startswith("ERROR (ValueError):")
    assert "A p dynamic already exists" in result
    assert "remove_dynamic" in result


def test_dynamic_family_remove_tool_descriptions_explain_blanket_scope() -> None:
    """Agent-facing remove tools should expose omitted-part skip semantics."""
    ss = ScoreSpeak.create(parts=["Violin", "Cello"], measures=2)

    for tool_name, mark_name in [
        ("remove_dynamic", "dynamic"),
        ("remove_hairpin", "hairpin"),
    ]:
        record = _record_for(ss, tool_name)
        tool = make_tool_from_record(ss, record)
        assert tool is not None

        description = " ".join(tool.description.split())

        assert "If part is None" in description
        assert "skip parts without one" in description
        assert f"fail only if no parts have a {mark_name} there" in description
        assert "Explicit part removal remains strict" in description


def test_make_tools_from_records_deduplicates_and_skips_failures():
    """Duplicate records collapse to one tool; None results are filtered."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    index = MethodIndex(ss)

    add_notes = _record_for(ss, "add_notes")

    tools = make_tools_from_records(ss, [add_notes, add_notes])

    assert len(tools) == 1
    assert tools[0].name == "add_notes"

    larger = make_tools_from_records(ss, index.records[:5])
    assert len(larger) <= 5
    assert all(t.name for t in larger)


def test_search_score_tool_returns_summary_context_not_exact_payload():
    """Agent-facing ``search_score`` has typed args and returns summaries."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss.add_articulation("staccato", measure_number=1, beat=1, part=0)

    tool = make_search_score_tool(ss)

    assert "query" not in tool.args
    assert "marking_type" in tool.args
    assert "event_kind" in tool.args
    assert "limit" in tool.args

    result = tool.invoke({
        "bar_range": [1, 1],
        "marking_type": "articulation",
        "marking_value": "staccato",
    })

    assert "Violin [0]" in result
    assert "quarter note" in result
    assert "matches=marking:" in result
    assert "event_schema" not in result
    assert "marking_schema" not in result


def test_search_score_tool_reports_limit_truncation():
    """Agent-facing search tells the model when a result limit was hit."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=3)
    for measure in range(1, 4):
        ss._add_note_one("C4", "quarter", measure=measure, beat=1, part=0)

    tool = make_search_score_tool(ss)

    result = tool.invoke({"pitch": "C4", "limit": 2})

    assert "search_score limit reached" in result
    assert "returned first 2 of 3 matching bars" in result
    assert "m1" in result
    assert "m2" in result
    assert "m3" not in result


def test_search_score_tool_description_lists_supported_filters():
    """Agent-facing ``search_score`` describes what can be searched."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=1)
    tool = make_search_score_tool(ss)
    description = tool.description

    assert "Supported filters:" in description
    assert "event_kind ('note', 'rest', 'chord')" in description
    assert "tie_status ('start', 'continue', 'stop', 'none', 'untied')" in description
    assert "marking_type ('dynamic', 'articulation', 'ornament'" in description
    assert "span_type ('hairpin', 'slur', 'ottava'" in description
    assert "two_note_tremolo" not in description
    assert "structure ('barline_start', 'barline_end', 'repeat_start'" in description
    assert "logic is 'all' or 'any'" in description
    assert "limit caps the number of matching bars returned" in description
    assert "cannot directly search arbitrary musical concepts" in description


def test_add_hairpin_agent_tool_explains_noninclusive_end_beat():
    """Agent-facing ``add_hairpin`` exposes the arrival-end contract."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=1)
    record = _record_for(ss, "add_hairpin")
    tool = make_tool_from_record(ss, record)

    assert tool is not None
    assert "[start, end)" in tool.description
    assert "non-inclusive arrival" in tool.description
    schema = tool.args_schema.model_json_schema()
    assert "Non-inclusive arrival beat" in schema["properties"]["end_beat"]["description"]


def test_search_score_tool_reports_empty_results_as_negative_search():
    """Empty ``search_score`` results should not imply unknown locations."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    tool = make_search_score_tool(ss)

    result = tool.invoke({"pitch": "F#4"})

    assert "No bars matched search_score() filters" in result
    assert "(no scoped bars)" not in result


def test_inspect_score_region_tool_returns_exact_core_by_default():
    """Exact inspection is a dedicated tool with minimal core defaults."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=1)
    ss.add_tuplet(
        [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")],
        actual_notes=3,
        normal_notes=2,
        measure=1,
        beat=1,
        part=0,
    )
    ss.add_dynamic("ff", measure_number=1, beat=1, part=0)

    tool = make_inspect_score_region_tool(ss)
    payload = json.loads(tool.invoke({"bar_range": [1, 1]}))

    voice = payload["bars"][0]["parts"][0]["voices"][0]
    assert payload["retrieval"] == "explicit inspection"
    assert payload["event_schema"]
    assert voice["events"]
    assert voice["tuplets"][0][0] == [3, 2]
    assert voice["tuplets"][0][1] == [1.0, pytest.approx(1.0 + 2.0 / 3.0)]
    assert "markings" not in voice


def test_inspect_score_region_tool_includes_requested_optional_channels():
    """Exact inspection includes optional channels through include categories."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss.add_dynamic("ff", measure_number=1, beat=1, part=0)

    tool = make_inspect_score_region_tool(ss)
    payload = json.loads(tool.invoke({
        "bar_range": [1, 1],
        "include": ["dynamics"],
    }))

    voice = payload["bars"][0]["parts"][0]["voices"][0]
    assert payload["marking_schema"] == [
        "type",
        "payload",
        "beat",
    ]
    assert ["dynamic", "ff", 1.0] in voice["markings"]


def test_inspect_score_region_tool_reports_mixed_hairpin_types() -> None:
    """Exact inspection reports crescendo and diminuendo spans in one bar."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=1, time_signature="4/4")
    for beat, pitch in [
        (1, "C4"),
        (2, "D4"),
        (3, "E4"),
        (4, "F4"),
    ]:
        ss._add_note_one(pitch, "quarter", measure=1, beat=beat, part=0)
    ss.add_hairpin(
        "crescendo",
        start_measure=1,
        start_beat=1,
        end_measure=1,
        end_beat=3,
        part=0,
    )
    ss.add_hairpin(
        "diminuendo",
        start_measure=1,
        start_beat=3,
        end_measure=1,
        end_beat=4,
        part=0,
    )

    tool = make_inspect_score_region_tool(ss)
    payload = json.loads(tool.invoke({
        "bar_range": [1, 1],
        "include": ["hairpins"],
    }))

    voice = payload["bars"][0]["parts"][0]["voices"][0]
    assert voice["spans"] == [
        ["hairpin", "crescendo", "", [1.0, 3.0]],
        ["hairpin", "diminuendo", "", [3.0, 4.0]],
    ]


def test_inspect_score_region_tool_text_expression_reports_exact_beat():
    """Text-expression inspection includes the removable beat location."""
    ss = ScoreSpeak.create(
        parts=["Violin"],
        measures=1,
        time_signature="2/4",
    )
    ss._add_note_one("C4", "half", measure=1, beat=1, part=0)
    ss.add_text_expression("cresc.", measure_number=1, beat=1.5, part=0)

    tool = make_inspect_score_region_tool(ss)
    payload = json.loads(tool.invoke({
        "bar_range": [1, 1],
        "include": ["text_expression"],
    }))

    voice = payload["bars"][0]["parts"][0]["voices"][0]
    assert "warnings" not in payload
    assert payload["marking_schema"] == [
        "type",
        "payload",
        "beat",
    ]
    assert voice["markings"] == [["text_expression", "cresc.", 1.5]]
    assert "span_schema" not in payload
    assert "spans" not in voice


def test_inspect_score_region_tool_beethoven_text_expression_regression():
    """Beethoven m45 text-expression inspection should not expose event indices."""
    fixture = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "scores"
        / "beethoven_op125_m1_violin1_bars001-200.musicxml"
    )
    ss = ScoreSpeak.from_musicxml(fixture)

    tool = make_inspect_score_region_tool(ss)
    payload = json.loads(tool.invoke({
        "parts": [0],
        "bar_range": [45, 45],
        "include": ["text_expression"],
    }))

    voice = payload["bars"][0]["parts"][0]["voices"][0]
    serialized = json.dumps(payload)
    assert "warnings" not in payload
    assert payload["marking_schema"] == ["type", "payload", "beat"]
    assert ["text_expression", "cresc.", 1.875] in voice["markings"]
    assert "[3, 3]" not in serialized
    assert "event_index" not in serialized


def test_inspect_score_region_tool_rejects_unsupported_voice() -> None:
    """Agent inspection rejects unsupported voice scope values."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=1)
    tool = make_inspect_score_region_tool(ss)

    result = tool.invoke({"bar_range": [1, 1], "voices": [5]})

    assert result.startswith("ERROR (ValueError):")
    assert "between 1 and 4" in result


def test_inspect_score_attributes_tool_returns_attributes_only():
    """Attribute inspection projects scoped bars without event rows."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=2, time_signature="4/4")
    ss.set_time_signature("3/4", measure_number=2)

    tool = make_inspect_score_attributes_tool(ss)
    payload = json.loads(tool.invoke({"bar_range": [1, 2]}))

    assert payload["retrieval"] == "attribute inspection"
    assert [bar["measure_number"] for bar in payload["bars"]] == [1, 2]
    assert payload["bars"][0]["notation"]["active"]["time"] == "4/4"
    assert payload["bars"][1]["notation"]["active"]["time"] == "3/4"
    assert "events" not in json.dumps(payload)


def test_inspect_score_attributes_labels_part_written_key() -> None:
    """Attribute inspection shows written part keys with concert-key context."""
    ss = ScoreSpeak.create(
        parts=["flute", "clarinet"],
        measures=1,
        key_signature="F",
    )
    tool = make_inspect_score_attributes_tool(ss)
    payload = json.loads(tool.invoke({"bar_range": [1, 1]}))

    bar = payload["bars"][0]
    assert bar["notation"]["active"]["key"] == "F major"
    assert bar["notation"]["active"]["concert_key"] == "F major"
    assert bar["notation"]["active"]["key_space"] == "concert"
    assert "key" not in bar["parts"][0]["notation"]
    clarinet_notation = bar["parts"][1]["notation"]
    assert clarinet_notation["key"] == "G major"
    assert clarinet_notation["concert_key"] == "F major"
    assert clarinet_notation["key_space"] == "written pitch"
    assert clarinet_notation["key_role"] == "transposed_written_key"
    assert clarinet_notation["key_label"] == (
        "written key: G major (concert key: F major)"
    )


def test_tool_search_returns_matches_and_loads_returned_tools():
    """``tool_search`` searches capabilities and loads returned tools."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    payload = json.loads(bundle.tools_by_name["tool_search"].invoke({
        "query": "forte dynamic",
        "tool_names": ["add_dynamic", "not_a_tool"],
    }))

    names = {match["name"] for match in payload["matches"]}
    assert "add_dynamic" in names
    assert "add_dynamic" in payload["loaded_tools"]
    assert payload["invalid_tool_names"] == ["not_a_tool"]
    assert "schema" not in json.dumps(payload)
    assert "signature" not in json.dumps(payload)
    assert bundle.expansion_requests.loaded_tool_names
    assert "add_dynamic" in bundle.expansion_requests.loaded_tool_names
    assert "include_schemas" not in bundle.tools_by_name["tool_search"].args
    assert "request_tools" not in bundle.tools_by_name["tool_search"].args


def test_add_part_tool_exposes_grand_staff_option() -> None:
    """The existing add_part tool carries the grand-staff option."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=1)

    record = _record_for(ss, "add_part")
    tool = make_tool_from_record(ss, record)

    assert tool is not None
    assert "grand_staff" in tool.args
    assert "RH/LH" in tool.args["grand_staff"]["description"]


def test_set_score_parts_tool_exposes_ordered_part_specs() -> None:
    """The set_score_parts tool exposes only an ordered parts list."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=1)

    record = _record_for(ss, "set_score_parts")
    tool = make_tool_from_record(ss, record)

    assert tool is not None
    assert set(tool.args) == {"parts"}
    assert "clef_type" not in tool.args
    assert "grand_staff" not in tool.args
    assert "index" not in tool.args
    schema = tool.args_schema.model_json_schema()
    instrument_description = (
        schema["$defs"]["SetScorePartsPartItem"]["properties"]["instrument"][
            "description"
        ]
    )
    assert "B♭ clarinet" in instrument_description
    assert "E♭ horn" in instrument_description
    assert "C trumpet" in instrument_description
    assert "displayed staff label" in instrument_description

    result = tool.invoke({
        "parts": [
            {"instrument": "flute", "name": "Flute"},
            {"instrument": "piano", "name": "Piano"},
        ],
    })

    assert result.startswith("OK: Set score parts")
    assert [part.display_name for part in ss.list_parts()] == [
        "Flute",
        "Piano RH",
        "Piano LH",
    ]


def test_tool_search_finds_add_part_for_grand_staff_request() -> None:
    """Grand-staff creation should route through add_part, not a new tool."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    payload = json.loads(bundle.tools_by_name["tool_search"].invoke({
        "query": "add a piano grand staff",
    }))

    names = {match["name"] for match in payload["matches"]}
    assert "add_part" in names
    assert "add_part" in payload["loaded_tools"]
    assert "add_grand_staff" not in names


def test_tool_search_prefers_set_score_parts_for_initial_setup_request() -> None:
    """Initial instrumentation requests should surface set_score_parts first."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    payload = json.loads(bundle.tools_by_name["tool_search"].invoke({
        "query": "initialize the score with flute, cello, and piano",
    }))
    names = [match["name"] for match in payload["matches"]]

    assert names[0] == "set_score_parts"
    assert "set_score_parts" in payload["loaded_tools"]


def test_tool_search_still_finds_add_part_for_incremental_request() -> None:
    """A single append-style request should still surface add_part."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    payload = json.loads(bundle.tools_by_name["tool_search"].invoke({
        "query": "add a flute part",
    }))
    names = [match["name"] for match in payload["matches"]]

    assert "add_part" in names
    assert "add_part" in payload["loaded_tools"]


def test_tool_search_exact_core_lookup_uses_agent_facing_descriptions():
    """Core tools returned by ``tool_search`` should not show stale signatures."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    payload = json.loads(bundle.tools_by_name["tool_search"].invoke({
        "tool_names": ["search_score", "inspect_score_region"],
    }))
    descriptions = {
        match["name"]: match["description"]
        for match in payload["matches"]
    }

    assert "Supported filters:" in descriptions["search_score"]
    assert "cannot directly search arbitrary musical concepts" in descriptions["search_score"]
    assert "Return exact compact symbolic rows" in descriptions["inspect_score_region"]
    assert not descriptions["inspect_score_region"].startswith("inspect_score_region(")
    assert "search_score" not in bundle.catalog.names


def test_local_key_signature_helper_is_not_agent_tool() -> None:
    """Local key overrides stay private and are not exposed to the agent."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=1)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())
    indexed_names = {record.name for record in retr.method_records}

    assert "set_local_key_signature" not in indexed_names
    assert "_set_local_key_signature" not in indexed_names
    assert "set_local_key_signature" not in bundle.tools_by_name
    assert "_set_local_key_signature" not in bundle.tools_by_name


def test_tool_search_add_hairpin_summary_mentions_noninclusive_end():
    """``tool_search`` should expose hairpin endpoint semantics before loading."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    payload = json.loads(bundle.tools_by_name["tool_search"].invoke({
        "tool_names": ["add_hairpin"],
    }))
    descriptions = {
        match["name"]: match["description"]
        for match in payload["matches"]
    }

    assert "[start, end)" in descriptions["add_hairpin"]
    assert "non-inclusive arrival" in descriptions["add_hairpin"]
    assert "add_hairpin" in payload["loaded_tools"]


def test_tool_search_add_articulation_summary_lists_supported_markings():
    """``tool_search`` should show what add_articulation can actually add."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    payload = json.loads(bundle.tools_by_name["tool_search"].invoke({
        "query": (
            "Find the tool to add a caesura marking to a score, ideally at a "
            "specific bar/beat or before a measure."
        ),
    }))
    descriptions = {
        match["name"]: match["description"]
        for match in payload["matches"]
    }

    assert "add_articulation" in descriptions
    description = descriptions["add_articulation"]
    assert "caesura" in description
    assert "breath mark" in description
    assert "staccatissimo" in description
    assert "part/measure/beat/voice" in description
    assert "present" in description
    assert "appear after" in description
    assert "add_articulation" in payload["loaded_tools"]


def test_agent_tool_bundle_preregisters_all_tools_and_core_tools():
    """The modernized runtime registers all tools once and deduplicates names."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)

    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())
    names = [tool.name for tool in bundle.tools]

    assert len(names) == len(set(names))
    assert "add_dynamic" in names
    assert "remove_notes" in names
    assert "remove_note" not in names
    assert "search_score" in names
    assert "find_bars" not in names
    assert "inspect_score_region" in names
    assert "inspect_score_attributes" in names
    assert "tool_search" in names


def test_ottava_generated_tool_args_have_common_descriptions_for_sparse_docstrings():
    """Sparse public docstrings should not expose ``parameter x`` guidance."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)

    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())
    args = bundle.tools_by_name["add_ottava"].args
    remove_args = bundle.tools_by_name["remove_ottava"].args

    assert args["start_measure"]["description"].startswith("1-based measure number")
    assert args["start_beat"]["description"].startswith("1-based beat position")
    assert "0-based part index" in args["part"]["description"]
    assert "simultaneous rhythmic line" in args["voice"]["description"]
    assert args["ottava_type"]["description"].startswith("Ottava type")
    assert "move written notes" in args["rewrite_pitches"]["description"]
    assert "reduce ledger lines" in args["rewrite_pitches"]["description"]
    assert "adds or removes the octave-shift mark" in args["rewrite_pitches"]["description"]
    assert "transposing" not in args
    assert "rewrite_pitches" in remove_args
    assert "transposing" not in remove_args
    assert not any(
        arg.get("description", "").startswith("parameter ")
        for arg in args.values()
    )


def test_agent_tool_bundle_excludes_lifecycle_methods():
    """Runtime lifecycle helpers remain public API but are not agent tools."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)

    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    hidden = {"create", "from_musicxml", "to_musicxml", "to_musicxml_string"}
    assert not hidden.intersection(bundle.tools_by_name)


def test_tool_search_treats_lifecycle_methods_as_unavailable():
    """``tool_search`` should not return or expand hidden lifecycle methods."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    payload = json.loads(bundle.tools_by_name["tool_search"].invoke({
        "tool_names": ["to_musicxml_string"],
    }))

    names = {match["name"] for match in payload["matches"]}
    assert "to_musicxml_string" not in names
    assert "to_musicxml_string" not in payload["loaded_tools"]
    assert payload["invalid_tool_names"] == ["to_musicxml_string"]


def test_tool_search_treats_remove_note_as_unavailable():
    """The old singular note-removal tool should not be agent-visible."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    payload = json.loads(bundle.tools_by_name["tool_search"].invoke({
        "tool_names": ["remove_note", "remove_notes"],
    }))

    assert "remove_notes" in payload["loaded_tools"]
    assert "remove_note" not in payload["loaded_tools"]
    assert payload["invalid_tool_names"] == ["remove_note"]


def test_tool_search_uses_default_limit_and_max_clamp():
    """``tool_search`` defaults to 5 matches and clamps large limits to 20."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())
    tool = bundle.tools_by_name["tool_search"]

    default_payload = json.loads(tool.invoke({"query": "add"}))
    clamped_payload = json.loads(tool.invoke({"query": "add", "limit": 200}))

    assert len(default_payload["matches"]) <= 5
    assert len(clamped_payload["matches"]) <= 20


def test_summarize_turn_context_filters_lifecycle_method_hits():
    """Debug summaries should show only methods the agent can actually use."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss, threshold=0.5)

    summary = summarize_turn_context(ss, retr, "export musicxml string")

    assert "to_musicxml_string" not in summary["method_hits"]


def test_cleaned_score_wide_tool_schemas_omit_part_argument():
    """Score-wide agent tools inherit public signatures without ``part``."""
    ss = ScoreSpeak.create(parts=["Piano", "Violin"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    score_wide_tools = {
        "add_measures",
        "insert_measure",
        "delete_measure",
        "delete_measures",
        "set_time_signature",
        "set_key_signature",
        "set_barline",
        "add_repeat",
        "remove_repeat",
        "set_pickup_measure",
        "add_system_break",
        "remove_system_break",
        "add_page_break",
        "remove_page_break",
        "add_ending_bracket",
        "remove_ending_bracket",
        "add_coda",
        "add_segno",
        "add_to_coda",
        "add_fine",
        "add_da_capo",
        "add_dal_segno",
        "remove_navigation_mark",
    }
    for name in score_wide_tools:
        assert "part" not in bundle.tools_by_name[name].args
    assert "side" not in bundle.tools_by_name["set_barline"].args


def test_set_barline_tool_describes_right_edge_contract():
    """The set_barline tool should make right-edge targeting explicit."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())
    tool = bundle.tools_by_name["set_barline"]

    assert "right edge" in tool.description
    assert "previous measure" in tool.description
    assert "right edge" in tool.args["measure_number"]["description"]
    assert "side" not in tool.args


def test_layout_break_tools_describe_measure_as_new_start():
    """Page/system break tools should identify the target as the new start."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    expectations = {
        "add_system_break": "Make a specific measure start a new system.",
        "remove_system_break": (
            "Remove the system break that makes a measure start a new system."
        ),
        "add_page_break": "Make a specific measure start a new page.",
        "remove_page_break": (
            "Remove the page break that makes a measure start a new page."
        ),
    }
    for tool_name, expected_text in expectations.items():
        tool = bundle.tools_by_name[tool_name]
        assert expected_text in tool.description
        assert "start" in tool.args["measure_number"]["description"]


def test_clear_measures_tool_schema_is_part_scoped():
    """Clearing bars can target parts and scoped voices."""
    ss = ScoreSpeak.create(parts=["Piano", "Violin"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    assert "clear_measures" in bundle.tools_by_name
    assert "start" in bundle.tools_by_name["clear_measures"].args
    assert "end" in bundle.tools_by_name["clear_measures"].args
    assert "part" in bundle.tools_by_name["clear_measures"].args
    assert "voice" in bundle.tools_by_name["clear_measures"].args
    assert "all_voices" in bundle.tools_by_name["clear_measures"].args


def test_copy_measure_contents_tool_schema_is_part_scoped():
    """Copy/paste can target one part, cross parts, or all parts."""
    ss = ScoreSpeak.create(parts=["Piano", "Violin"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    tool_args = bundle.tools_by_name["copy_measure_contents"].args
    assert "source_start" in tool_args
    assert "target_start" in tool_args
    assert "count" in tool_args
    assert "source_part" in tool_args
    assert "target_part" in tool_args


def test_tool_search_finds_clear_measures_for_blank_bars():
    """Tool search should route blank/empty bar wording to clear_measures."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    payload = json.loads(bundle.tools_by_name["tool_search"].invoke({
        "query": "blank bars",
    }))

    names = {match["name"] for match in payload["matches"]}
    assert "clear_measures" in names


def test_tool_search_finds_copy_measure_contents_for_paste_bars():
    """Tool search should route copy/paste bar wording to copy_measure_contents."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    payload = json.loads(bundle.tools_by_name["tool_search"].invoke({
        "query": "paste bar 1 into bar 2",
    }))

    names = {match["name"] for match in payload["matches"]}
    assert "copy_measure_contents" in names


def test_pedal_tool_schemas_omit_voice_argument():
    """Pedal tools should be part-scoped but not voice-scoped."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    assert "voice" not in bundle.tools_by_name["add_pedal"].args
    assert "voice" not in bundle.tools_by_name["remove_pedal"].args


def test_rest_spelling_tools_are_exposed_without_rest_completion() -> None:
    """Expose rest spelling tools while hiding old rest-completion tools."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    assert "add_rest" in bundle.tools_by_name
    assert "fill_measure_gaps" in bundle.tools_by_name
    assert "add_rests" not in bundle.tools_by_name
    assert "complete_measure_with_rests" not in bundle.tools_by_name
    assert "remove_rests" in bundle.tools_by_name
    assert "reshape_rests" in bundle.tools_by_name
    assert "duration" in bundle.tools_by_name["add_rest"].args
    assert "rests" not in bundle.tools_by_name["add_rest"].args
    assert "measure" in bundle.tools_by_name["fill_measure_gaps"].args
    assert "duration" not in bundle.tools_by_name["fill_measure_gaps"].args
    assert "beat" in bundle.tools_by_name["remove_rests"].args
    assert "rests" in bundle.tools_by_name["reshape_rests"].args
    assert "inspect_measure_capacity" not in bundle.tools_by_name


@pytest.mark.parametrize(
    "query",
    [
        "unhide rests in bar 2",
        "show hidden rests in bar 2",
        "make rest visible",
        "reveal rest",
    ],
)
def test_tool_search_prefers_add_rest_for_unhide_rest_wording(query: str) -> None:
    """Tool search should route unhide/show-rest wording to add_rest."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    payload = json.loads(bundle.tools_by_name["tool_search"].invoke({
        "query": query,
    }))

    names = [match["name"] for match in payload["matches"]]
    assert names[0] == "add_rest"


@pytest.mark.parametrize(
    "query",
    [
        "fill gaps in bar 2",
        "fill missing rests in bar 2",
        "complete the measure with rests",
    ],
)
def test_tool_search_prefers_fill_measure_gaps_for_gap_wording(query: str) -> None:
    """Tool search should route true-gap wording to fill_measure_gaps."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    payload = json.loads(bundle.tools_by_name["tool_search"].invoke({
        "query": query,
    }))

    names = [match["name"] for match in payload["matches"]]
    assert names[0] == "fill_measure_gaps"


@pytest.mark.parametrize(
    "query",
    [
        "split rests in bar 2",
        "merge rests in bar 2",
        "reformat rests in bar 2",
        "change rest spelling in bar 2",
    ],
)
def test_tool_search_prefers_reshape_rests_for_rest_reformat_wording(
    query: str,
) -> None:
    """Tool search should keep spelling transformations on reshape_rests."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    payload = json.loads(bundle.tools_by_name["tool_search"].invoke({
        "query": query,
    }))

    names = [match["name"] for match in payload["matches"]]
    assert names[0] == "reshape_rests"


def test_tool_search_prefers_remove_rests_for_hide_rest_wording() -> None:
    """Tool search should keep hide-rest wording on remove_rests."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())

    payload = json.loads(bundle.tools_by_name["tool_search"].invoke({
        "query": "hide rests in bar 2",
    }))

    names = [match["name"] for match in payload["matches"]]
    assert names[0] == "remove_rests"


def test_dynamic_tool_filter_has_no_fallback_write_tools():
    """Empty lexical hits expose only core tools until ``tool_search`` expands."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    retrieval = retr.query("xyzzy nonsense")
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())
    middleware = ScoreSpeakAgentMiddleware(
        ss,
        retrieval.context_bars,
        retrieval.scope,
        bundle.expansion_requests,
        bundle.core_tool_names,
    )

    visible = middleware.visible_tool_names({"candidate_tool_names": []})
    assert visible == bundle.core_tool_names
    assert "add_dynamic" not in visible

    bundle.tools_by_name["tool_search"].invoke({
        "query": "dynamic",
    })
    visible_after_search = middleware.visible_tool_names({"candidate_tool_names": []})
    assert "add_dynamic" in visible_after_search


def test_dynamic_prompt_rebuilds_overview_after_score_mutation():
    """The middleware rebuilds the overview before each model call."""
    from langchain.agents.middleware import ModelRequest, ModelResponse
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)
    retrieval = retr.query("xyzzy nonsense")
    bundle = build_agent_tool_bundle(ss, retr.method_records, AgentMemoryStore())
    middleware = ScoreSpeakAgentMiddleware(
        ss,
        retrieval.context_bars,
        retrieval.scope,
        bundle.expansion_requests,
        bundle.core_tool_names,
    )
    captured_prompts = []

    def handler(request: ModelRequest) -> ModelResponse:
        """Capture the dynamically generated system prompt."""
        captured_prompts.append(request.system_message.content)
        return ModelResponse([])

    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[],
        tools=bundle.tools,
        state={"candidate_tool_names": []},
    )
    middleware.wrap_model_call(request, handler)
    ss.add_part(name="Violin", instrument="violin")
    middleware.wrap_model_call(request, handler)

    assert "Piano" in captured_prompts[0]
    assert "Violin" not in captured_prompts[0]
    assert "Violin" in captured_prompts[1]


def test_prepare_agent_turn_serializes_tool_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent runtime config prevents parallel tool calls on the live score."""
    class FakeGraph:
        """Placeholder graph returned by the patched agent factory."""

    def fake_create_agent(
        llm: Any,
        tools: list[Any],
        middleware: list[Any],
        state_schema: Any,
    ) -> FakeGraph:
        """Return a graph without constructing a real LangGraph agent."""
        return FakeGraph()

    monkeypatch.setattr("scorespeak.agent.graph.create_agent", fake_create_agent)
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    retr = LexicalContextRetriever(ss)

    runtime = _prepare_agent_turn(
        ss,
        retr,
        object(),
        "copy measure 1 to measure 2",
        AgentMemoryStore(),
        recursion_limit=13,
    )

    assert not isinstance(runtime, str)
    assert runtime.config == {"recursion_limit": 13, "max_concurrency": 1}


# ---------------------------------------------------------------------------
# Turn summarization
# ---------------------------------------------------------------------------


def test_summarize_turn_context_reports_hits_and_scope():
    """summarize_turn_context surfaces retrieval hits and rendered prompt."""
    ss = ScoreSpeak.create(
        parts=["Piano", "Violin"],
        measures=8,
    )
    retr = LexicalContextRetriever(ss, threshold=0.5)

    summary = summarize_turn_context(ss, retr, "add a forte to bar 3 of piano")

    assert "add_dynamic" in summary["method_hits"]
    assert "add_dynamic" in summary["candidate_tool_names"]
    assert summary["scope"]["bar_range"] == (3, 3)
    assert summary["scope"]["part_indices"] == [0]
    assert summary["scope"]["bar_context_status"] == "explicit"
    assert "SCORE OVERVIEW:" in summary["system_prompt"]
    assert "SCORE SUMMARY CONTEXT" in summary["system_prompt"]
    assert "AUTO-SELECTED TOOL CANDIDATES" in summary["system_prompt"]
    assert "CONTEXT BARS" not in summary["system_prompt"]
    assert "event_schema" not in summary["system_prompt"]
    assert summary["inspection_tool"] == "inspect_score_region"
    assert summary["search_tool"] == "search_score"
    assert "Piano [0]" in summary["summary_context"]


def test_summarize_turn_context_has_no_bars_when_no_bar_is_mentioned():
    """Generic missing-bar queries do not receive final-bar context."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=5)
    retr = LexicalContextRetriever(ss, threshold=0.5)

    summary = summarize_turn_context(ss, retr, "make the music louder")

    scope = summary["scope"]
    assert scope["bar_range"] is None
    assert scope["bar_context_status"] == "missing"
    assert summary["summary_context"] == "(no scoped bars)"
    assert "bars: (none) (status: missing)" in summary["system_prompt"]


# ---------------------------------------------------------------------------
# Streaming progress
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_name", "label"),
    [
        ("inspect_score_region", "Examining the score..."),
        ("search_score", "Examining the score..."),
        ("get_active_key_signature", "Examining the score..."),
        ("memory_search", "Checking session memory..."),
        ("tool_search", "Choosing score-editing tools..."),
        ("copy_measure_contents", "Copying notation..."),
        ("add_notes", "Adding notation..."),
        ("add_dynamic", "Adding notation..."),
        ("set_time_signature", "Updating score settings..."),
        ("transpose", "Transposing the score..."),
        ("delete_measure", "Removing notation..."),
        ("unknown_tool", "Working..."),
        (None, "Working..."),
    ],
)
def test_tool_progress_label_maps_known_groups(tool_name, label):
    """Tool names are mapped to stable user-facing progress labels."""
    assert tool_progress_label(tool_name) == label


def test_stream_events_from_update_detects_tool_start():
    """AI tool calls become tool_start progress events."""
    from langchain_core.messages import AIMessage

    update = {
        "model": {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "add_notes",
                        "args": {"measure": 1, "part": 0, "voice": 1},
                        "id": "call_1",
                    }],
                )
            ]
        }
    }

    assert stream_events_from_update(update) == [{
        "type": "tool_start",
        "tool": "add_notes",
        "label": "Adding notation...",
    }]


def test_stream_events_from_update_detects_tool_completion():
    """Tool messages become tool_end progress events."""
    from langchain_core.messages import ToolMessage

    update = {
        "tools": {
            "messages": [
                ToolMessage(
                    content="ok",
                    name="add_notes",
                    tool_call_id="call_1",
                )
            ]
        }
    }

    assert stream_events_from_update(update) == [{
        "type": "tool_end",
        "tool": "add_notes",
        "ok": True,
    }]


def test_stream_events_from_update_marks_tool_errors():
    """Tool errors are surfaced as unsuccessful completions."""
    from langchain_core.messages import ToolMessage

    update = {
        "tools": {
            "messages": [
                ToolMessage(
                    content="bad args",
                    name="add_notes",
                    tool_call_id="call_1",
                    status="error",
                )
            ]
        }
    }

    assert stream_events_from_update(update) == [{
        "type": "tool_end",
        "tool": "add_notes",
        "ok": False,
    }]


def test_run_turn_stream_emits_phases_and_final_text(monkeypatch):
    """The streaming turn wraps graph updates with phase and final events."""
    from langchain_core.messages import AIMessage

    class FakeGraph:
        def stream(self, input_state, config, stream_mode):
            assert input_state == {"messages": []}
            assert config == {"recursion_limit": 30}
            assert stream_mode == "updates"
            yield {"model": {"messages": [AIMessage(content="Done.")]}}

    monkeypatch.setattr(
        "scorespeak.agent.graph._prepare_agent_turn",
        lambda *args, **kwargs: AgentTurnRuntime(
            graph=FakeGraph(),
            input_state={"messages": []},
            config={"recursion_limit": 30},
        ),
    )

    events = list(run_turn_stream(None, None, None, "add a note", AgentMemoryStore()))

    assert events[0] == {"type": "phase", "label": "Examining the score..."}
    assert {"type": "phase", "label": "Planning edits..."} in events
    assert {"type": "phase", "label": "Rendering the updated score..."} in events
    assert events[-1] == {"type": "final", "response": "Done."}


def test_run_turn_stream_emits_error_when_setup_fails(monkeypatch):
    """A setup error is streamed as an error event after the initial phase."""
    monkeypatch.setattr(
        "scorespeak.agent.graph._prepare_agent_turn",
        lambda *args, **kwargs: "ERROR: no usable tools",
    )

    events = list(run_turn_stream(None, None, None, "add a note", AgentMemoryStore()))

    assert events == [
        {"type": "phase", "label": "Examining the score..."},
        {"type": "error", "error": "ERROR: no usable tools"},
    ]


def test_run_turn_stream_emits_error_when_graph_raises(monkeypatch):
    """Graph exceptions are streamed as error events."""
    class FakeGraph:
        def stream(self, input_state, config, stream_mode):
            raise RuntimeError("boom")
            yield

    monkeypatch.setattr(
        "scorespeak.agent.graph._prepare_agent_turn",
        lambda *args, **kwargs: AgentTurnRuntime(
            graph=FakeGraph(),
            input_state={},
            config={"recursion_limit": 30},
        ),
    )

    events = list(run_turn_stream(None, None, None, "add a note", AgentMemoryStore()))

    assert events[-1]["type"] == "error"
    assert "RuntimeError: boom" in events[-1]["error"]


# ---------------------------------------------------------------------------
# Final-message extraction
# ---------------------------------------------------------------------------


class _StubAIMessage:
    """Minimal stand-in compatible enough with ``isinstance`` checks fail.

    We intentionally do NOT import AIMessage here; the extractor should not
    crash on arbitrary message shapes, only recognize real AIMessage.
    """

    def __init__(self, content):
        self.content = content


def test_extract_final_text_handles_missing_ai_message():
    """When no AIMessage is present the extractor returns a placeholder."""
    assert _extract_final_text({"messages": []}) == "(no response produced)"


def test_extract_final_text_returns_ai_message_content():
    """A plain AIMessage string is returned verbatim."""
    from langchain_core.messages import AIMessage, HumanMessage

    result = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(content="hello!"),
        ]
    }
    assert _extract_final_text(result) == "hello!"


def test_extract_final_text_joins_structured_content():
    """Content blocks of the form ``[{type:text, text:...}]`` are concatenated."""
    from langchain_core.messages import AIMessage

    result = {
        "messages": [
            AIMessage(
                content=[
                    {"type": "text", "text": "part one "},
                    {"type": "text", "text": "part two"},
                ]
            ),
        ]
    }
    assert _extract_final_text(result) == "part one part two"


# ---------------------------------------------------------------------------
# Overview enrichments (voice_count, transposition, signature_timeline)
# ---------------------------------------------------------------------------


def test_overview_part_snapshot_includes_voice_count_default_one():
    """Single-voice parts report ``voice_count = 1``."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)

    overview = build_score_overview(ss)
    assert overview.parts[0].voice_count == 1


def test_overview_part_snapshot_records_multiple_voices():
    """When a bar has multiple voices, the max voice count is surfaced."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=2)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0, voice=1)
    ss._add_note_one("E4", "quarter", measure=1, beat=1, part=0, voice=2)

    overview = build_score_overview(ss)
    assert overview.parts[0].voice_count >= 2


def test_overview_signature_timeline_captures_mid_score_changes():
    """Time/key/tempo changes past bar 1 show up in ``signature_timeline``."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=6, time_signature="4/4")
    ss.set_time_signature("3/4", measure_number=3)
    ss.set_tempo(90, measure_number=4, part=0)
    ss.set_key_signature("G major", measure_number=5)

    overview = build_score_overview(ss)
    timeline = overview.signature_timeline

    assert (1, "4/4") in timeline.time
    assert (3, "3/4") in timeline.time
    assert (4, 90.0) in timeline.tempo
    assert any(bar == 5 for bar, _ in timeline.key)


def test_overview_renders_transposing_part_pitch_space() -> None:
    """Transposing parts show both interval and stored pitch space."""
    ss = ScoreSpeak.create(parts=["Clarinet"], measures=1)

    text = format_overview_for_prompt(build_score_overview(ss))

    assert "transp=M-2" in text
    assert "stored=written pitch" in text


def test_format_overview_for_prompt_renders_changes_line_when_present():
    """Non-trivial timelines render as a compact ``Changes:`` line."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=4, time_signature="4/4")
    ss.set_time_signature("3/4", measure_number=3)

    text = format_overview_for_prompt(build_score_overview(ss))
    assert "Changes:" in text
    assert "3/4" in text


def test_format_overview_for_prompt_omits_changes_line_when_trivial():
    """A score with no mid-score changes has no ``Changes:`` line."""
    ss = ScoreSpeak.create(parts=["Piano"], measures=3)

    text = format_overview_for_prompt(build_score_overview(ss))
    assert "Changes:" not in text


def _make_grand_staff_score():
    """Build a minimal score with one piano grand staff for overview tests."""
    from music21 import clef, layout, note, stream

    rh = stream.PartStaff(); rh.partName = "Piano"
    rh.append(clef.TrebleClef())
    m = stream.Measure(number=1); m.append(note.Note("C4")); rh.append(m)

    lh = stream.PartStaff(); lh.partName = "Piano"
    lh.append(clef.BassClef())
    m = stream.Measure(number=1); m.append(note.Note("C3")); lh.append(m)

    score = stream.Score()
    score.insert(0, rh); score.insert(0, lh)
    score.insert(0, layout.StaffGroup([rh, lh], name="Piano", symbol="brace"))
    return ScoreSpeak(score)


def test_overview_populates_display_name_and_hand_for_grand_staff():
    """Piano grand-staff parts carry display_name=\"Piano RH\"/... and hand."""
    ss = _make_grand_staff_score()
    overview = build_score_overview(ss)

    rh_snapshot = next(p for p in overview.parts if p.index == 0)
    lh_snapshot = next(p for p in overview.parts if p.index == 1)

    assert rh_snapshot.display_name == "Piano RH"
    assert rh_snapshot.hand == "RH"
    assert lh_snapshot.display_name == "Piano LH"
    assert lh_snapshot.hand == "LH"


def test_overview_format_emits_groups_line_for_grand_staff():
    """``format_overview_for_prompt`` emits a ``Groups:`` line with RH/LH."""
    ss = _make_grand_staff_score()
    text = format_overview_for_prompt(build_score_overview(ss))

    assert "Piano RH" in text
    assert "Piano LH" in text
    assert "Groups: Piano [0, 1] (RH, LH)" in text


def test_overview_no_groups_line_when_no_brace_groups_exist():
    """Scores without any brace groups omit the ``Groups:`` line."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=2)
    text = format_overview_for_prompt(build_score_overview(ss))
    assert "Groups:" not in text


# ---------------------------------------------------------------------------
# Prompt provenance block (CONTEXT SCOPE)
# ---------------------------------------------------------------------------


def test_build_system_prompt_emits_context_scope_block_with_explicit_bars():
    """An explicit bar range is reflected in the scope block."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=4)
    overview = build_score_overview(ss)
    scope = ExtractedContextScope(
        part_indices=[0],
        measure_numbers=[2, 3],
        bar_range=(2, 3),
        matched_part_names=["Violin"],
        explicit_part_mention=True,
        explicit_bar_mention=True,
        bar_context_status="explicit",
    )

    prompt = build_system_prompt(overview, {"bars": []}, scope)

    assert "CONTEXT SCOPE:" in prompt
    assert "retrieval: automatic" in prompt
    assert "bars: 2-3" in prompt
    assert "status: explicit" in prompt
    assert "Violin" in prompt


def test_build_system_prompt_scope_block_shows_end_fallback_status():
    """End-oriented final-bar fallback is labeled distinctly."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=4)
    overview = build_score_overview(ss)
    scope = ExtractedContextScope(
        used_fallback_bar=True,
        bar_context_status="end_fallback",
    )
    context_bars = {"bars": [{"measure_number": 4, "parts": []}]}

    prompt = build_system_prompt(overview, context_bars, scope)

    assert "CONTEXT SCOPE:" in prompt
    assert "status: end_fallback" in prompt
    assert "bars: 4" in prompt


def test_build_system_prompt_scope_block_shows_missing_bar_status():
    """Missing automatic bar context is rendered as no scoped bars."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=4)
    overview = build_score_overview(ss)
    scope = ExtractedContextScope(bar_context_status="missing")

    prompt = build_system_prompt(overview, {"bars": []}, scope)

    assert "CONTEXT SCOPE:" in prompt
    assert "bars: (none) (status: missing)" in prompt
    assert "(no scoped bars)" in prompt


def test_build_system_prompt_scope_block_includes_ambiguity_messages():
    """Ambiguity warnings are attached to the scope block verbatim."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=2)
    overview = build_score_overview(ss)
    scope = ExtractedContextScope(
        bar_range=(1, 1),
        ambiguity_messages=["Part mention 'vn' is ambiguous"],
        bar_context_status="explicit",
    )

    prompt = build_system_prompt(overview, {"bars": []}, scope)

    assert "ambiguity: Part mention 'vn' is ambiguous" in prompt


def test_build_system_prompt_includes_candidates_and_truncation_messages():
    """Candidate tool names and context truncation provenance are rendered."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=2)
    overview = build_score_overview(ss)
    scope = ExtractedContextScope(
        bar_range=(1, 2),
        context_truncation_messages=["automatic context limited to bars 1-1"],
        bar_context_status="explicit",
    )

    prompt = build_system_prompt(
        overview,
        {"bars": []},
        scope,
        candidate_tool_names=["add_dynamic"],
    )

    assert "AUTO-SELECTED TOOL CANDIDATES" in prompt
    assert "- add_dynamic" in prompt
    assert "truncated: automatic context limited to bars 1-1" in prompt


def test_build_system_prompt_includes_surgical_note_contract():
    """The system prompt tells the agent how surgical note tools behave."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=2)
    overview = build_score_overview(ss)

    prompt = build_system_prompt(overview, {"bars": []}, ExtractedContextScope())

    assert "Individual note/chord/rest-spelling tools are surgical" in prompt
    assert "visible rests in the target voice" in prompt
    assert "must not overlap existing" in prompt
    assert "A voice is an independent rhythmic timeline" in prompt
    assert "chords, not separate voices" in prompt
    assert "remove a sharp, flat, or natural" in prompt
    assert "rather than deleting the note" in prompt
    assert "add_rest" in prompt
    assert "fill_measure_gaps" in prompt
    assert "reshape_rests" in prompt
    assert "remove_rests" in prompt
    assert "add_rests" not in prompt
    assert "complete_measure_with_rests" not in prompt
    assert "copy_measure_contents" in prompt
    assert "target structure" in prompt


def test_build_system_prompt_omits_prompt_split_instruction_by_default() -> None:
    """Prompt split guidance is absent for normal single-turn prompts."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=2)
    overview = build_score_overview(ss)

    prompt = build_system_prompt(overview, {"bars": []}, ExtractedContextScope())

    assert "These messages are chunks of one original request" not in prompt
    assert "These messages may be chunks" not in prompt


def test_build_system_prompt_includes_prompt_split_instruction_when_enabled() -> None:
    """Prompt split mode inserts concrete no-clarification guidance."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=2)
    overview = build_score_overview(ss)

    prompt = build_system_prompt(
        overview,
        {"bars": []},
        ExtractedContextScope(),
        prompt_split_mode=True,
    )

    assert "These messages are chunks of one original request" in prompt
    assert "Do not ask clarifying" in prompt
    assert "current chunk is ambiguous" in prompt


def test_build_system_prompt_omits_scope_block_when_scope_absent():
    """No scope object means no scope block is emitted."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=2)
    overview = build_score_overview(ss)

    prompt_without_scope = build_system_prompt(overview, {"bars": []})
    assert "CONTEXT SCOPE:" not in prompt_without_scope


# ---------------------------------------------------------------------------
# Score context renderers
# ---------------------------------------------------------------------------


def test_render_summary_context_uses_counts_range_and_present_markings():
    """Summary context reports count buckets, range, and present markings only."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss._add_note_one("G4", "quarter", measure=1, beat=2, part=0)
    ss.add_dynamic("ff", measure_number=1, beat=1, part=0)

    summary = render_summary_context(ss._build_bar_result_set({"scope": {"bar_range": (1, 1)}}))

    assert "m1" in summary
    assert "Violin [0]" in summary
    assert "2 quarter notes" in summary
    assert "range=C4-G4" in summary
    assert "dynamic(ff)" in summary
    assert "event_schema" not in summary
    assert "articulation" not in summary


def test_render_summary_context_distinguishes_tuplet_buckets():
    """Tuplet summaries bucket by notated value and ratio."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=1)
    ss.add_tuplet(
        [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")],
        actual_notes=3,
        normal_notes=2,
        measure=1,
        beat=1,
        part=0,
    )

    summary = render_summary_context(ss._build_bar_result_set({"scope": {"bar_range": (1, 1)}}))

    assert "3 eighth-note tuplets (3:2)" in summary


def test_render_summary_context_labels_transposed_written_key() -> None:
    """Summary context distinguishes bar concert key from part written key."""
    ss = ScoreSpeak.create(
        parts=["flute", "clarinet"],
        measures=1,
        key_signature="F",
    )

    summary = render_summary_context(
        ss._build_bar_result_set({"scope": {"bar_range": (1, 1)}})
    )

    assert "concert_key=F major" in summary
    assert "written_key=G major (concert_key=F major)" in summary


def test_render_exact_context_minimal_core_omits_optional_markings():
    """Minimal exact context keeps events and tuplets but omits optional rows."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=1)
    ss.add_tuplet(
        [("C4", "eighth"), ("D4", "eighth"), ("E4", "eighth")],
        actual_notes=3,
        normal_notes=2,
        measure=1,
        beat=1,
        part=0,
    )
    ss.add_dynamic("ff", measure_number=1, beat=1, part=0)

    exact = render_exact_context(ss._build_bar_result_set({"scope": {"bar_range": (1, 1)}}))
    voice = exact["bars"][0]["parts"][0]["voices"][0]

    assert voice["events"]
    assert voice["tuplets"][0][0] == [3, 2]
    assert voice["tuplets"][0][1] == [1.0, pytest.approx(1.0 + 2.0 / 3.0)]
    assert "markings" not in voice


def test_render_exact_context_include_filters_optional_channels():
    """Include categories add matching optional exact rows only."""
    ss = ScoreSpeak.create(parts=["Violin"], measures=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
    ss._add_note_one("D4", "quarter", measure=1, beat=2, part=0)
    ss.add_dynamic("ff", measure_number=1, beat=1, part=0)
    ss.add_articulation("staccato", measure_number=1, beat=2, part=0)

    exact = render_exact_context(
        ss._build_bar_result_set({"scope": {"bar_range": (1, 1)}}),
        include=["dynamics"],
    )
    voice = exact["bars"][0]["parts"][0]["voices"][0]

    assert ["dynamic", "ff", 1.0] in voice["markings"]
    assert all(row[0] == "dynamic" for row in voice["markings"])
