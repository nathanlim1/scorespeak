"""LangChain agent loop over the ScoreSpeak API.

This subpackage wires lexical retrieval in :mod:`scorespeak.retrieval` to a
chat model through LangChain's graph-backed ``create_agent`` runtime.  All
ScoreSpeak tools are pre-registered, while middleware dynamically filters the
model-visible tools to the always-available inspection/tool-search core plus
tools loaded by ``tool_search``.
"""

from __future__ import annotations

from .prompt_split import (
    DEFAULT_PROMPT_SPLIT_CONFIG,
    PromptSplitConfig,
    llm_splitter_messages,
    should_use_prompt_split,
    split_sentences,
)
from .graph import (
    AgentPromptRunResult,
    build_agent_tool_bundle,
    run_prompt,
    run_prompt_collect,
    run_prompt_stream,
    run_turn,
    run_turn_stream,
    summarize_turn_context,
)
from .memory import (
    AgentMemoryStore,
    ExecutionMemoryEntry,
    InstructionMemoryEntry,
    MemoryToolCallTrace,
    format_memory_context_for_prompt,
    make_memory_search_tool,
)
from .overview import ScoreOverview, build_score_overview, format_overview_for_prompt
from .tool_catalog import ToolCatalog
from .tools import (
    ToolExpansionRequests,
    make_inspect_score_attributes_tool,
    make_inspect_score_region_tool,
    make_search_score_tool,
    make_tool_search_tool,
    make_tool_from_record,
    make_tools_from_records,
)

__all__ = [
    "ScoreOverview",
    "AgentMemoryStore",
    "ExecutionMemoryEntry",
    "InstructionMemoryEntry",
    "MemoryToolCallTrace",
    "ToolCatalog",
    "ToolExpansionRequests",
    "AgentPromptRunResult",
    "PromptSplitConfig",
    "DEFAULT_PROMPT_SPLIT_CONFIG",
    "build_agent_tool_bundle",
    "build_score_overview",
    "format_overview_for_prompt",
    "format_memory_context_for_prompt",
    "make_inspect_score_attributes_tool",
    "make_inspect_score_region_tool",
    "make_search_score_tool",
    "make_memory_search_tool",
    "make_tool_search_tool",
    "make_tool_from_record",
    "make_tools_from_records",
    "llm_splitter_messages",
    "run_prompt",
    "run_prompt_collect",
    "run_prompt_stream",
    "run_turn",
    "run_turn_stream",
    "should_use_prompt_split",
    "split_sentences",
    "summarize_turn_context",
]


def run_repl(argv: list[str] | None = None) -> int:
    """Lazy-import wrapper around :func:`scorespeak.agent.repl.main`.

    Importing :mod:`scorespeak.agent.repl` at package import time produces a
    ``runpy`` warning when the module is executed via ``python -m``.  By
    routing callers through this helper we keep the REPL off the import
    path unless it is actually needed.
    """
    from .repl import main

    return main(argv)
