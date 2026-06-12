"""Tests for automatic prompt splitting and execution."""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from scorespeak import ScoreSpeak
from scorespeak.agent.prompt_split import (
    PromptSplitConfig,
    llm_splitter_messages,
    make_prompt_split_chunks,
    should_use_prompt_split,
    split_sentences,
)
from scorespeak.agent import AgentMemoryStore
from scorespeak.agent.graph import AgentTurnRuntime, run_prompt_collect
from scorespeak.retrieval import LexicalContextRetriever


class FakeSplitter:
    """Fake splitter model returning deterministic split indices."""

    def __init__(self, responses: list[str]) -> None:
        """Store response texts in call order."""
        self.responses = list(responses)
        self.calls: list[tuple[list[Any], dict[str, Any] | None]] = []

    def invoke(
        self,
        messages: list[Any],
        config: dict[str, Any] | None = None,
    ) -> AIMessage:
        """Return the next configured response."""
        self.calls.append((messages, config))
        response = self.responses.pop(0) if self.responses else "2"
        return AIMessage(content=response)


def test_should_use_prompt_split_activates_above_threshold() -> None:
    """Prompts activate only when sentence count is strictly above threshold."""
    six_sentences = "One. Two. Three. Four. Five. Six."
    seven_sentences = f"{six_sentences} Seven."

    assert len(split_sentences(six_sentences)) == 6
    assert not should_use_prompt_split(six_sentences)
    assert should_use_prompt_split(seven_sentences)


def test_llm_splitter_messages_uses_rolling_window_and_metadata() -> None:
    """The splitter asks for rolling local split points with LangSmith metadata."""
    splitter = FakeSplitter(["3", "2"])
    prompt = "One. Two. Three. Four. Five."

    chunks = llm_splitter_messages(prompt, splitter, window_sentences=4)

    assert chunks == ["One. Two. Three.", "Four. Five."]
    assert len(splitter.calls) == 2
    assert splitter.calls[0][1] is not None
    assert splitter.calls[0][1]["run_name"] == "prompt_splitter"


def test_llm_splitter_messages_falls_back_on_bad_output() -> None:
    """Bad splitter output falls back to deterministic four-sentence chunks."""
    splitter = FakeSplitter(["not an integer"])
    prompt = "One. Two. Three. Four. Five."

    chunks = llm_splitter_messages(prompt, splitter, window_sentences=5)

    assert chunks == ["One. Two. Three. Four.", "Five."]


def test_prompt_split_stream_returns_first_chunk_before_remaining_done() -> None:
    """The first chunk is available while the background splitter keeps working."""
    first_call_done = threading.Event()
    second_call_started = threading.Event()
    release_second_call = threading.Event()

    class BlockingSplitter:
        """Splitter that blocks the second split decision."""

        def __init__(self) -> None:
            """Initialize call counter."""
            self.call_count = 0

        def invoke(
            self,
            messages: list[Any],
            config: dict[str, Any] | None = None,
        ) -> AIMessage:
            """Return two-sentence chunks, blocking on the second call."""
            del messages, config
            self.call_count += 1
            if self.call_count == 1:
                first_call_done.set()
                return AIMessage(content="2")
            second_call_started.set()
            release_second_call.wait(timeout=1)
            return AIMessage(content="2")

    splitter = BlockingSplitter()
    stream = make_prompt_split_chunks(
        "One. Two. Three. Four. Five.",
        splitter,
        PromptSplitConfig(window_sentences=5),
    )
    assert first_call_done.is_set()

    iterator = iter(stream)
    assert next(iterator) == "One. Two."
    assert second_call_started.wait(timeout=1)
    stream.close()
    release_second_call.set()


def test_run_prompt_collect_records_each_chunk_in_same_memory(
    monkeypatch: Any,
) -> None:
    """Prompt split execution stores every chunk in one memory store."""
    score_state = ScoreSpeak.create(parts=["Violin"], measures=1)
    retriever = LexicalContextRetriever(score_state)
    memory_store = AgentMemoryStore()
    splitter = FakeSplitter(["2", "2", "2", "2"])

    class FakeGraph:
        """Fake graph that echoes the user chunk."""

        def invoke(
            self,
            input_state: dict[str, Any],
            config: dict[str, Any],
        ) -> dict[str, Any]:
            """Return a single final AI message."""
            del config
            user_text = input_state["messages"][0].content
            return {"messages": [AIMessage(content=f"Done: {user_text}")]}

    def fake_prepare_agent_turn(
        score_state: ScoreSpeak,
        retriever: LexicalContextRetriever,
        llm: Any,
        user_text: str,
        memory_store: AgentMemoryStore,
        *,
        recursion_limit: int,
        mutation_recorder: Any | None = None,
        prompt_split_mode: bool = False,
    ) -> AgentTurnRuntime:
        """Return a fake runtime and verify prompt split mode is enabled."""
        del score_state, retriever, llm, memory_store
        del recursion_limit, mutation_recorder
        assert prompt_split_mode
        return AgentTurnRuntime(
            graph=FakeGraph(),
            input_state={"messages": [HumanMessage(content=user_text)]},
            config={},
        )

    monkeypatch.setattr(
        "scorespeak.agent.graph._prepare_agent_turn",
        fake_prepare_agent_turn,
    )

    result = run_prompt_collect(
        score_state,
        retriever,
        object(),
        "One. Two. Three. Four. Five. Six. Seven.",
        memory_store,
        splitter_llm=splitter,
    )

    assert result.prompt_split_mode
    assert result.chunks == ["One. Two.", "Three. Four.", "Five. Six.", "Seven."]
    assert len(memory_store.instruction_entries) == 4
    assert memory_store.instruction_entries[0].user_prompt == "One. Two."
    assert memory_store.instruction_entries[-1].user_prompt == "Seven."
