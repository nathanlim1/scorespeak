"""Tests for public ScoreSpeak benchmark datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pytest import MonkeyPatch

from scorespeak import ScoreSpeak
from scorespeak.evaluation import (
    BenchmarkEditAction,
    PreciseEditAction,
    apply_benchmark_edit_actions,
    apply_precise_edit_actions,
    apply_precise_edit_case,
    benchmark_action_names,
    benchmark_action_schema_hash,
    benchmark_action_schemas,
    extract_long_task_facts,
    load_precise_edit_cases,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
PRECISE_COLUMNS = [
    "public_case_id",
    "base_score_id",
    "base_musicxml_path",
    "prompt",
    "expected_edit_actions_json",
    "tags_json",
    "difficulty",
]
LONG_CASE_IDS = [
    "lt_20260602_235150_mozart_k156_bars001_040",
    "lt_20260602_235150_mozart_k545_bars001_012",
    "lt_20260602_235150_beethoven_op59no1_m4_bars001_040",
    "lt_20260602_235150_haydn_op74no1_m2_bars001_040",
    "lt_20260602_235150_mozart_k155_m1_bars001_040",
    "lt_20260602_235150_beethoven_op67_m1_full_bars001_020",
]


def _read_precise_rows() -> list[dict[str, str]]:
    """Load public precise-edit CSV rows."""
    with (REPO_ROOT / "datasets/precise_edit/cases.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == PRECISE_COLUMNS
        return list(reader)


def _load_long_task_manifest() -> dict[str, Any]:
    """Load the long-task reconstruction manifest."""
    manifest_path = REPO_ROOT / "datasets/long_task_reconstruction/manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_precise_edit_csv_is_public_shape() -> None:
    """The precise-edit benchmark should expose exactly the public CSV schema."""
    rows = _read_precise_rows()
    public_ids = [row["public_case_id"] for row in rows]

    assert len(rows) == 752
    assert len(set(public_ids)) == 752
    assert public_ids[0] == "pe_0001"
    assert public_ids[-1] == "pe_0752"
    assert "inspection_requirement" not in rows[0]


def test_precise_edit_csv_references_existing_scores() -> None:
    """Every precise-edit case should reference an included base MusicXML file."""
    rows = _read_precise_rows()

    for row in rows:
        base_path = REPO_ROOT / row["base_musicxml_path"]
        actions = json.loads(row["expected_edit_actions_json"])
        tags = json.loads(row["tags_json"])

        assert base_path.is_file()
        assert base_path.is_relative_to(REPO_ROOT / "datasets/scores")
        assert isinstance(actions, list)
        assert actions
        assert isinstance(tags, list)
        assert row["difficulty"] in {"easy", "medium", "hard"}
        assert "llm_generated" not in tags


def test_precise_edit_cases_load_as_replayable_actions() -> None:
    """The public precise-edit CSV should load into typed replay cases."""
    cases = load_precise_edit_cases(REPO_ROOT / "datasets/precise_edit/cases.csv")
    action_names = {
        action.name
        for case in cases
        for action in case.expected_edit_actions
    }

    assert len(cases) == 752
    assert cases[0].public_case_id == "pe_0001"
    assert cases[0].expected_edit_actions
    assert "add_note" in action_names
    assert "remove_note" in action_names


def test_precise_edit_action_replay_supports_legacy_note_actions() -> None:
    """Legacy single-note benchmark actions should replay through their runner."""
    score_state = ScoreSpeak.create(measures=1, parts=["Piano"])
    results = apply_precise_edit_actions(
        score_state,
        [
            PreciseEditAction(
                name="add_note",
                args={
                    "measure": 1,
                    "part": "Piano",
                    "voice": 1,
                    "pitch": "C4",
                    "beat": 1.0,
                    "duration": "quarter",
                },
            ),
            PreciseEditAction(
                name="remove_note",
                args={
                    "measure": 1,
                    "part": "Piano",
                    "voice": 1,
                    "pitch": "C4",
                    "beat": 1.0,
                },
            ),
        ],
    )

    assert len(results) == 2
    assert "Added C4 quarter note" in results[0]
    assert "Removed" in results[1]


def test_precise_edit_action_runner_is_separate_from_scorespeak_note_helper(
    monkeypatch: MonkeyPatch,
) -> None:
    """Benchmark v1 note actions should not call ScoreSpeak note-entry helpers."""
    score_state = ScoreSpeak.create(measures=1, parts=["Piano"])

    def fail_note_helper(*args: object, **kwargs: object) -> object:
        """Fail if benchmark application calls the ScoreSpeak helper."""
        raise AssertionError("ScoreSpeak note helper called")

    monkeypatch.setattr(ScoreSpeak, "_add_note_one", fail_note_helper)

    apply_benchmark_edit_actions(
        score_state,
        [
            BenchmarkEditAction(
                "add_note",
                {
                    "pitch": "C4",
                    "duration": "quarter",
                    "measure": 1,
                    "beat": 1,
                },
            ),
        ],
    )

    assert score_state.get_notes(measure=1)[0].pitch == "C4"


def test_precise_edit_action_runner_keeps_legacy_overlap_semantics() -> None:
    """Benchmark v1 note actions should allow legacy overlapping insertions."""
    score_state = ScoreSpeak.create(measures=1, time_signature="4/4")

    apply_benchmark_edit_actions(
        score_state,
        [
            BenchmarkEditAction(
                "add_note",
                {
                    "pitch": "C4",
                    "duration": "quarter",
                    "measure": 1,
                    "beat": 1,
                },
            ),
            BenchmarkEditAction(
                "add_note",
                {
                    "pitch": "D4",
                    "duration": "quarter",
                    "measure": 1,
                    "beat": 1,
                },
            ),
        ],
    )

    notes = score_state.get_notes(measure=1)
    assert [note.pitch for note in notes if not note.is_rest] == ["C4", "D4"]
    assert [
        (note.pitch, note.beat, note.quarter_length)
        for note in notes
        if note.is_rest
    ] == [("rest", 2.0, 3.0)]


def test_precise_edit_action_replay_supports_legacy_tie_stop() -> None:
    """Legacy tie stop actions should not attempt to tie the stop note onward."""
    score_state = ScoreSpeak.create(measures=1, parts=["Piano"])
    score_state.add_notes(
        measure=1,
        part="Piano",
        voice=1,
        notes=[
            {"pitch": "G4", "beat": 1.0, "duration": "quarter", "dots": 0},
            {"pitch": "G4", "beat": 2.0, "duration": "quarter", "dots": 0},
        ],
    )

    results = apply_precise_edit_actions(
        score_state,
        [
            PreciseEditAction(
                name="add_tie",
                args={
                    "measure": 1,
                    "part": "Piano",
                    "voice": 1,
                    "beat": 1.0,
                    "tie_type": "start",
                },
            ),
            PreciseEditAction(
                name="add_tie",
                args={
                    "measure": 1,
                    "part": "Piano",
                    "voice": 1,
                    "beat": 2.0,
                    "tie_type": "stop",
                },
            ),
        ],
    )
    notes = score_state.get_notes(measure=1, part="Piano")

    assert len(results) == 2
    assert [note.is_tied for note in notes if not note.is_rest] == [True, True]


def test_precise_edit_action_inventory_is_supported() -> None:
    """Every public precise-edit action matches the benchmark action contract."""
    cases = load_precise_edit_cases(REPO_ROOT / "datasets/precise_edit/cases.csv")
    action_names = benchmark_action_names()
    accepted_args_by_name = {
        schema["name"]: {parameter["name"] for parameter in schema["parameters"]}
        for schema in benchmark_action_schemas()
    }

    for case in cases:
        for action in case.expected_edit_actions:
            assert action.name in action_names
            assert set(action.args) <= accepted_args_by_name[action.name]


def test_precise_edit_action_schema_matches_original_contract() -> None:
    """The public precise-edit action schema should preserve the thesis contract."""
    assert benchmark_action_schema_hash() == "36d1dfa93be2"


def test_precise_edit_case_replay_loads_included_base_score() -> None:
    """A public precise-edit case should load its base score and replay actions."""
    cases = load_precise_edit_cases(REPO_ROOT / "datasets/precise_edit/cases.csv")
    score_state, results = apply_precise_edit_case(
        cases[1],
        repository_root=REPO_ROOT,
    )

    assert isinstance(score_state, ScoreSpeak)
    assert results
    assert all(isinstance(result, str) and result for result in results)


def test_long_task_manifest_references_public_files() -> None:
    """The long-task manifest should expose the six thesis cases and files."""
    dataset_root = REPO_ROOT / "datasets/long_task_reconstruction"
    manifest = _load_long_task_manifest()
    cases = manifest["cases"]

    assert manifest["case_count"] == 6
    assert [case["case_id"] for case in cases] == LONG_CASE_IDS

    for case in cases:
        prompt_path = dataset_root / case["prompt_path"]
        source_path = dataset_root / case["source_musicxml_path"]
        target_path = dataset_root / case["target_musicxml_path"]

        assert set(case) == {
            "case_id",
            "title",
            "source_musicxml_path",
            "target_musicxml_path",
            "prompt_path",
            "tags",
            "difficulty",
        }
        assert prompt_path.is_file()
        assert source_path.is_file()
        assert target_path.is_file()
        assert prompt_path.read_text(encoding="utf-8").startswith("Setup:")
        assert case["difficulty"] == "hard"


def test_long_task_fact_extraction_reads_included_target() -> None:
    """Symbolic fact extraction should work on an included long-task target."""
    target_path = (
        REPO_ROOT
        / "datasets/long_task_reconstruction/targets/mozart_k545_bars001_012.musicxml"
    )

    result = extract_long_task_facts(target_path)
    channels = {fact.channel for fact in result.facts}

    assert result.target_fact_count_supported > 200
    assert "event" in channels
    assert "part" in channels
