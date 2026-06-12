"""Tests for mandatory session-scoped agent memory."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from scorespeak import ScoreSpeak
from scorespeak.agent.graph import (
    ScoreSpeakAgentMiddleware,
    build_agent_tool_bundle,
    run_turn,
    run_turn_stream,
)
from scorespeak.agent.memory import (
    AgentMemoryStore,
    MemoryToolCallTrace,
    format_memory_context_for_prompt,
)
from scorespeak.retrieval import LexicalContextRetriever
from web.server import AgentSession


def _stamp(offset_seconds: int) -> datetime:
    """Return a deterministic UTC timestamp for memory tests."""
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=offset_seconds)


def test_web_agent_session_always_owns_memory_store() -> None:
    """Web agent sessions create memory before any turns run."""
    session = AgentSession(ScoreSpeak.create(measures=2))

    assert isinstance(session.memory_store, AgentMemoryStore)
    assert not session.memory_store.has_entries()


def test_repl_creates_and_passes_memory_store(monkeypatch: Any, tmp_path: Any) -> None:
    """The CLI REPL creates one memory store and passes it into turns."""
    from scorespeak.agent import repl

    captured: dict[str, Any] = {}
    prompts = iter(["remember this", "quit"])

    def fake_input(prompt: str) -> str:
        """Return scripted REPL input."""
        return next(prompts)

    def fake_run_turn(
        score_state: ScoreSpeak,
        retriever: LexicalContextRetriever,
        llm: Any,
        user_text: str,
        memory_store: AgentMemoryStore,
    ) -> str:
        """Capture the memory store passed by the REPL."""
        captured["memory_store"] = memory_store
        captured["user_text"] = user_text
        return "Done."

    monkeypatch.setattr(repl, "_require_openai_key", lambda: "test-key")
    monkeypatch.setattr(repl, "_build_llm", lambda model, api_key: object())
    monkeypatch.setattr(repl, "_load_score", lambda musicxml_path, start_new: ScoreSpeak.create())
    monkeypatch.setattr(repl, "_print_intro", lambda score_state, output_path, model: None)
    monkeypatch.setattr(repl, "_autosave", lambda score_state, output_path: None)
    monkeypatch.setattr(repl, "run_turn", fake_run_turn)
    monkeypatch.setattr("builtins.input", fake_input)

    result = repl.main(["--new", "--output", str(tmp_path / "out.musicxml")])

    assert result == 0
    assert captured["user_text"] == "remember this"
    assert isinstance(captured["memory_store"], AgentMemoryStore)


def test_memory_search_hidden_until_store_has_entries() -> None:
    """The memory tool is invisible for empty stores and core-visible after memory exists."""
    score_state = ScoreSpeak.create(parts=["Piano"], measures=2)
    retriever = LexicalContextRetriever(score_state)
    memory_store = AgentMemoryStore()

    empty_bundle = build_agent_tool_bundle(score_state, retriever.method_records, memory_store)
    assert "memory_search" not in empty_bundle.tools_by_name
    assert "memory_search" not in empty_bundle.core_tool_names

    memory_store.append_instruction("turn-1", "previous prompt about measure 2")
    filled_bundle = build_agent_tool_bundle(score_state, retriever.method_records, memory_store)
    retrieval = retriever.query("nonsense")
    middleware = ScoreSpeakAgentMiddleware(
        score_state,
        retrieval.context_bars,
        retrieval.scope,
        filled_bundle.expansion_requests,
        filled_bundle.core_tool_names,
    )

    assert "memory_search" in filled_bundle.tools_by_name
    assert "memory_search" in filled_bundle.core_tool_names
    assert "memory_search" in middleware.visible_tool_names({"candidate_tool_names": []})
    description = filled_bundle.tools_by_name["memory_search"].description
    assert "prior completed turns" in description
    assert "DIRECT PREVIOUS TURN MEMORY" in description
    assert "Blank query returns newest entries" in description


def test_blank_memory_search_returns_newest_entries_and_pool_filters() -> None:
    """Blank queries retrieve newest entries across both pools and by pool."""
    memory_store = AgentMemoryStore()
    memory_store.append_instruction("turn-1", "first prompt", timestamp=_stamp(1))
    memory_store.append_execution(
        "turn-1",
        [],
        "first final",
        "ok",
        timestamp=_stamp(2),
    )
    memory_store.append_instruction("turn-2", "second prompt", timestamp=_stamp(3))
    memory_store.append_execution(
        "turn-2",
        [MemoryToolCallTrace("add_notes", {"measure": 2}, "OK", True)],
        "second final",
        "ok",
        timestamp=_stamp(4),
    )

    all_recent = memory_store.search(limit=3)
    instruction_recent = memory_store.search(pools=["instruction"], limit=2)
    execution_recent = memory_store.search(pools=["execution"], limit=2)

    assert [(item["pool"], item["turn_id"]) for item in all_recent] == [
        ("execution", "turn-2"),
        ("instruction", "turn-2"),
        ("execution", "turn-1"),
    ]
    assert [item["user_prompt"] for item in instruction_recent] == [
        "second prompt",
        "first prompt",
    ]
    assert [item["final_response"] for item in execution_recent] == [
        "second final",
        "first final",
    ]


def test_memory_search_uses_lexical_and_structured_fields() -> None:
    """Search finds measure, part, prompt, response, and tool-name matches."""
    memory_store = AgentMemoryStore()
    memory_store.append_instruction(
        "turn-1",
        "Add forte in measure 12 for violin",
        timestamp=_stamp(1),
    )
    memory_store.append_execution(
        "turn-1",
        [
            MemoryToolCallTrace(
                "add_dynamic",
                {"measure": 12, "part": "Violin", "value": "f"},
                "OK: added dynamic in measure 12",
                True,
            ),
            MemoryToolCallTrace(
                "inspect_score_region",
                {"bar_range": [12, 14], "part": "Violin"},
                "OK",
                True,
            )
        ],
        "Added forte to violin.",
        "ok",
        timestamp=_stamp(2),
    )
    memory_store.append_instruction("turn-2", "Set the title", timestamp=_stamp(3))
    memory_store.append_execution(
        "turn-2",
        [MemoryToolCallTrace("set_title", {"title": "Demo"}, "OK", True)],
        "Updated the title.",
        "ok",
        timestamp=_stamp(4),
    )

    assert memory_store.search("measure 12", limit=1)[0]["turn_id"] == "turn-1"
    assert memory_store.search("measure 13", pools=["execution"], limit=1)[0]["turn_id"] == "turn-1"
    assert memory_store.search("violin", pools=["instruction"], limit=1)[0]["turn_id"] == "turn-1"
    assert memory_store.search("forte", pools=["instruction"], limit=1)[0]["turn_id"] == "turn-1"
    assert memory_store.search("Added forte", pools=["execution"], limit=1)[0]["turn_id"] == "turn-1"
    assert memory_store.search("add_dynamic", pools=["execution"], limit=1)[0]["turn_id"] == "turn-1"
    assert memory_store.search("set_title", pools=["execution"], limit=1)[0]["turn_id"] == "turn-2"


def test_current_prompt_is_not_retrievable_during_its_own_turn(monkeypatch: Any) -> None:
    """The current user prompt is appended only after the agent run finishes."""
    score_state = ScoreSpeak.create(parts=["Piano"], measures=2)
    retriever = LexicalContextRetriever(score_state)
    memory_store = AgentMemoryStore()
    memory_store.record_turn("turn-0", "older prompt", [], "older response", "ok")

    def fake_create_agent(
        llm: Any,
        tools: list[Any],
        middleware: list[Any],
        state_schema: Any,
    ) -> Any:
        """Return a graph that queries memory during the turn."""
        memory_tool = next(tool for tool in tools if tool.name == "memory_search")

        class FakeGraph:
            """Minimal graph that asserts current prompt is absent."""

            def invoke(self, input_state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
                """Search memory before returning a final response."""
                payload = json.loads(memory_tool.invoke({"query": "secret current"}))
                assert payload["matches"] == []
                return {"messages": [AIMessage(content="Done.")]}

        return FakeGraph()

    monkeypatch.setattr("scorespeak.agent.graph.create_agent", fake_create_agent)

    response = run_turn(
        score_state,
        retriever,
        object(),
        "secret current prompt for measure 99",
        memory_store,
    )

    assert response == "Done."
    assert memory_store.search("secret current", pools=["instruction"], limit=1)[0]["turn_id"] == "turn-1"


def test_non_streamed_and_streamed_turns_record_memory_once(monkeypatch: Any) -> None:
    """Both run modes append one instruction and one execution entry per turn."""
    score_state = ScoreSpeak.create(parts=["Piano"], measures=2)
    retriever = LexicalContextRetriever(score_state)

    class FakeGraph:
        """Graph with deterministic invoke and stream outputs."""

        def invoke(self, input_state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            """Return one tool call, one tool result, and one final message."""
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "add_notes",
                            "args": {"measure": 1, "part": 0, "voice": 1},
                            "id": "call_1",
                        }],
                    ),
                    ToolMessage(
                        content="OK: added note",
                        name="add_notes",
                        tool_call_id="call_1",
                    ),
                    AIMessage(content="Added the note."),
                ]
            }

        def stream(
            self,
            input_state: dict[str, Any],
            config: dict[str, Any],
            stream_mode: str,
        ) -> Any:
            """Yield one streamed tool call, result, and final message."""
            yield {
                "model": {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[{
                                "name": "add_dynamic",
                                "args": {"measure": 2, "value": "f"},
                                "id": "call_2",
                            }],
                        )
                    ]
                }
            }
            yield {
                "tools": {
                    "messages": [
                        ToolMessage(
                            content="OK: added dynamic",
                            name="add_dynamic",
                            tool_call_id="call_2",
                        )
                    ]
                }
            }
            yield {"model": {"messages": [AIMessage(content="Added the dynamic.")]}}

    def fake_create_agent(
        llm: Any,
        tools: list[Any],
        middleware: list[Any],
        state_schema: Any,
    ) -> FakeGraph:
        """Return the deterministic fake graph."""
        return FakeGraph()

    monkeypatch.setattr("scorespeak.agent.graph.create_agent", fake_create_agent)

    non_stream_store = AgentMemoryStore()
    response = run_turn(score_state, retriever, object(), "add a note", non_stream_store)

    assert response == "Added the note."
    assert len(non_stream_store.instruction_entries) == 1
    assert len(non_stream_store.execution_entries) == 1
    assert non_stream_store.execution_entries[0].tool_trace[0].name == "add_notes"
    assert non_stream_store.execution_entries[0].tool_trace[0].result == "OK: added note"

    stream_store = AgentMemoryStore()
    events = list(run_turn_stream(score_state, retriever, object(), "add a dynamic", stream_store))

    assert events[-1] == {"type": "final", "response": "Added the dynamic."}
    assert len(stream_store.instruction_entries) == 1
    assert len(stream_store.execution_entries) == 1
    assert stream_store.execution_entries[0].tool_trace[0].name == "add_dynamic"
    assert stream_store.execution_entries[0].tool_trace[0].result == "OK: added dynamic"


def test_prompt_memory_context_uses_only_direct_previous_turn() -> None:
    """Prompt memory includes the direct previous turn and omits older details."""
    memory_store = AgentMemoryStore()
    memory_store.record_turn(
        "turn-1",
        "older instruction measure 7",
        [],
        "older response",
        "ok",
    )
    memory_store.record_turn(
        "turn-2",
        "direct previous instruction measure 8",
        [MemoryToolCallTrace("add_dynamic", {"measure": 8}, "OK", True)],
        "direct previous response",
        "ok",
    )

    context = format_memory_context_for_prompt(memory_store)

    assert "DIRECT PREVIOUS TURN MEMORY" in context
    assert "Use memory_search" in context
    assert "current user message is not stored" in context
    assert "missing part/bar scope" in context
    assert "direct previous instruction" in context
    assert "direct previous response" in context
    assert "add_dynamic(ok)" in context
    assert "args=" not in context
    assert "older instruction" not in context
    assert "older response" not in context


def test_full_prompt_memory_context_includes_previous_tool_details() -> None:
    """Full previous-turn memory includes args and compact result summaries."""
    memory_store = AgentMemoryStore()
    memory_store.record_turn(
        "turn-1",
        "direct previous instruction measure 8",
        [
            MemoryToolCallTrace(
                "add_dynamic",
                {"measure": 8},
                "OK: Added f dynamic | details={'measure': 8, 'beat': 1.0}",
                True,
            )
        ],
        "direct previous response",
        "ok",
    )

    context = format_memory_context_for_prompt(
        memory_store,
        detail="full_previous_turn",
    )

    assert "Detail mode: full_previous_turn" in context
    assert "Previous tool calls/actions" in context
    assert "args={\"measure\":8}" in context
    assert "result=OK: Added f dynamic" in context
    assert "details=" not in context


def test_full_prompt_memory_omits_noisy_retrieval_tool_results() -> None:
    """Full memory keeps retrieval tool args but omits bulky result payloads."""
    memory_store = AgentMemoryStore()
    memory_store.record_turn(
        "turn-1",
        "inspect and pick tools",
        [
            MemoryToolCallTrace(
                "tool_search",
                {"query": "add notes"},
                '{"matches": [{"name": "add_notes", "description": "large"}]}',
                True,
            ),
            MemoryToolCallTrace(
                "inspect_score_region",
                {"bar_range": [1, 2]},
                '{"bars": [{"measure_number": 1, "large": "payload"}]}',
                True,
            ),
            MemoryToolCallTrace(
                "add_dynamic",
                {"measure_number": 1, "level": "f"},
                "OK: Added f dynamic | details={'measure': 1}",
                True,
            ),
        ],
        "direct previous response",
        "ok",
    )

    context = format_memory_context_for_prompt(
        memory_store,
        detail="full_previous_turn",
    )

    assert "1. tool_search" in context
    assert "args={\"query\":\"add notes\"}" in context
    assert "matches" not in context
    assert "2. inspect_score_region" in context
    assert "args={\"bar_range\":[1,2]}" in context
    assert "payload" not in context
    assert "3. add_dynamic" in context
    assert "result=OK: Added f dynamic" in context


def test_previous_turn_memory_context_is_brief_when_scope_is_complete() -> None:
    """Complete current scope injects brief previous-turn memory."""
    score_state = ScoreSpeak.create(parts=["Piano"], measures=2)
    retriever = LexicalContextRetriever(score_state)
    retrieval = retriever.query("add a note in bar 1 of piano")
    memory_store = AgentMemoryStore()
    memory_store.record_turn(
        "turn-1",
        "secret prior instruction measure 7",
        [MemoryToolCallTrace("add_dynamic", {"measure": 7}, "OK", True)],
        "secret prior response",
        "ok",
    )
    bundle = build_agent_tool_bundle(score_state, retriever.method_records, memory_store)
    middleware = ScoreSpeakAgentMiddleware(
        score_state,
        retrieval.context_bars,
        retrieval.scope,
        bundle.expansion_requests,
        bundle.core_tool_names,
        memory_store,
    )
    captured: dict[str, str] = {}

    def handler(request: ModelRequest) -> ModelResponse:
        """Capture the generated system prompt."""
        captured["prompt"] = request.system_message.content
        return ModelResponse([])

    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[],
        tools=bundle.tools,
        state={"candidate_tool_names": []},
    )
    middleware.wrap_model_call(request, handler)

    assert "memory_search" in middleware.visible_tool_names({"candidate_tool_names": []})
    assert "DIRECT PREVIOUS TURN MEMORY" in captured["prompt"]
    assert "Detail mode: brief" in captured["prompt"]
    assert "Use memory_search" in captured["prompt"]
    assert "secret prior instruction" in captured["prompt"]
    assert "secret prior response" in captured["prompt"]
    assert "args={\"measure\":7}" not in captured["prompt"]


def test_missing_scope_uses_full_previous_turn_memory_in_system_prompt() -> None:
    """Missing part or bar scope expands previous-turn memory detail."""
    score_state = ScoreSpeak.create(parts=["Piano"], measures=2)
    retriever = LexicalContextRetriever(score_state)
    retrieval = retriever.query("add a note")
    memory_store = AgentMemoryStore()
    memory_store.record_turn(
        "turn-1",
        "secret prior instruction measure 7",
        [MemoryToolCallTrace("add_dynamic", {"measure": 7, "part": 0}, "OK", True)],
        "secret prior response",
        "ok",
    )
    bundle = build_agent_tool_bundle(score_state, retriever.method_records, memory_store)
    middleware = ScoreSpeakAgentMiddleware(
        score_state,
        retrieval.context_bars,
        retrieval.scope,
        bundle.expansion_requests,
        bundle.core_tool_names,
        memory_store,
    )
    captured: dict[str, str] = {}

    def handler(request: ModelRequest) -> ModelResponse:
        """Capture the generated system prompt."""
        captured["prompt"] = request.system_message.content
        return ModelResponse([])

    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[],
        tools=bundle.tools,
        state={"candidate_tool_names": []},
    )
    middleware.wrap_model_call(request, handler)

    assert "Detail mode: full_previous_turn" in captured["prompt"]
    assert "Previous tool calls/actions" in captured["prompt"]
    assert "args={\"measure\":7,\"part\":0}" in captured["prompt"]
    assert "result=OK" in captured["prompt"]


def test_single_part_missing_part_scope_still_uses_brief_memory() -> None:
    """One-part scores do not treat absent part wording as missing scope."""
    score_state = ScoreSpeak.create(parts=["Piano"], measures=2)
    retriever = LexicalContextRetriever(score_state)
    retrieval = retriever.query("add a dynamic to bar 1")
    memory_store = AgentMemoryStore()
    memory_store.record_turn(
        "turn-1",
        "secret prior instruction measure 7",
        [MemoryToolCallTrace("add_dynamic", {"measure": 7, "part": 0}, "OK", True)],
        "secret prior response",
        "ok",
    )
    bundle = build_agent_tool_bundle(score_state, retriever.method_records, memory_store)
    middleware = ScoreSpeakAgentMiddleware(
        score_state,
        retrieval.context_bars,
        retrieval.scope,
        bundle.expansion_requests,
        bundle.core_tool_names,
        memory_store,
    )
    captured: dict[str, str] = {}

    def handler(request: ModelRequest) -> ModelResponse:
        """Capture the generated system prompt."""
        captured["prompt"] = request.system_message.content
        return ModelResponse([])

    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[],
        tools=bundle.tools,
        state={"candidate_tool_names": []},
    )
    middleware.wrap_model_call(request, handler)

    assert retrieval.scope.part_indices is None
    assert retrieval.scope.bar_context_status == "explicit"
    assert "Detail mode: brief" in captured["prompt"]
    assert "args={\"measure\":7,\"part\":0}" not in captured["prompt"]


def test_multi_part_missing_part_scope_uses_full_memory() -> None:
    """Multi-part scores still expand memory when part scope is absent."""
    score_state = ScoreSpeak.create(parts=["Piano", "Violin"], measures=2)
    retriever = LexicalContextRetriever(score_state)
    retrieval = retriever.query("add a dynamic to bar 1")
    memory_store = AgentMemoryStore()
    memory_store.record_turn(
        "turn-1",
        "secret prior instruction measure 7",
        [MemoryToolCallTrace("add_dynamic", {"measure": 7, "part": 0}, "OK", True)],
        "secret prior response",
        "ok",
    )
    bundle = build_agent_tool_bundle(score_state, retriever.method_records, memory_store)
    middleware = ScoreSpeakAgentMiddleware(
        score_state,
        retrieval.context_bars,
        retrieval.scope,
        bundle.expansion_requests,
        bundle.core_tool_names,
        memory_store,
    )
    captured: dict[str, str] = {}

    def handler(request: ModelRequest) -> ModelResponse:
        """Capture the generated system prompt."""
        captured["prompt"] = request.system_message.content
        return ModelResponse([])

    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[],
        tools=bundle.tools,
        state={"candidate_tool_names": []},
    )
    middleware.wrap_model_call(request, handler)

    assert retrieval.scope.part_indices is None
    assert retrieval.scope.bar_context_status == "explicit"
    assert "Detail mode: full_previous_turn" in captured["prompt"]
    assert "args={\"measure\":7,\"part\":0}" in captured["prompt"]


def test_first_turn_missing_scope_has_no_memory_prompt_or_tool() -> None:
    """No previous memory means missing scope does not fabricate memory context."""
    score_state = ScoreSpeak.create(parts=["Piano"], measures=2)
    retriever = LexicalContextRetriever(score_state)
    retrieval = retriever.query("add a note")
    memory_store = AgentMemoryStore()
    bundle = build_agent_tool_bundle(score_state, retriever.method_records, memory_store)
    middleware = ScoreSpeakAgentMiddleware(
        score_state,
        retrieval.context_bars,
        retrieval.scope,
        bundle.expansion_requests,
        bundle.core_tool_names,
        memory_store,
    )
    captured: dict[str, str] = {}

    def handler(request: ModelRequest) -> ModelResponse:
        """Capture the generated system prompt."""
        captured["prompt"] = request.system_message.content
        return ModelResponse([])

    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[],
        tools=bundle.tools,
        state={"candidate_tool_names": []},
    )
    middleware.wrap_model_call(request, handler)

    assert "memory_search" not in middleware.visible_tool_names({"candidate_tool_names": []})
    assert "DIRECT PREVIOUS TURN MEMORY" not in captured["prompt"]
