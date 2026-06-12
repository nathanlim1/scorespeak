"""Public evaluation helpers for ScoreSpeak datasets."""

from scorespeak.evaluation.long_task_facts import (
    DependencyEdge,
    FactExtractionResult,
    SymbolicFact,
    extract_long_task_facts,
    fact_identity_payload,
    stable_fact_id,
)
from scorespeak.evaluation.precise_edit_actions import (
    BenchmarkEditAction,
    PreciseEditAction,
    PreciseEditCase,
    apply_benchmark_edit_actions,
    apply_precise_edit_actions,
    apply_precise_edit_case,
    benchmark_action_names,
    benchmark_action_schema_hash,
    benchmark_action_schemas,
    load_precise_edit_cases,
    materialize_precise_edit_musicxml,
)

__all__ = [
    "DependencyEdge",
    "FactExtractionResult",
    "BenchmarkEditAction",
    "PreciseEditAction",
    "PreciseEditCase",
    "SymbolicFact",
    "apply_benchmark_edit_actions",
    "apply_precise_edit_actions",
    "apply_precise_edit_case",
    "benchmark_action_names",
    "benchmark_action_schema_hash",
    "benchmark_action_schemas",
    "extract_long_task_facts",
    "fact_identity_payload",
    "load_precise_edit_cases",
    "materialize_precise_edit_musicxml",
    "stable_fact_id",
]
