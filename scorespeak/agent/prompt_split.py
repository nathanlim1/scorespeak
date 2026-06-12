"""Prompt splitting for multi-turn agent execution."""

from __future__ import annotations

import logging
import queue
import re
import threading
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptSplitConfig:
    """Configuration for automatic prompt splitting."""

    sentence_threshold: int = 6
    window_sentences: int = 10
    fallback_split_sentences: int = 4


DEFAULT_PROMPT_SPLIT_CONFIG = PromptSplitConfig()
_DONE = object()
_SPLITTER_RUN_CONFIG = {
    "run_name": "prompt_splitter",
    "tags": ["scorespeak", "prompt_split", "splitter"],
    "metadata": {"scorespeak_component": "prompt_splitter"},
}


class PromptSplitChunkStream:
    """Iterable chunk stream that splits the remaining prompt in the background."""

    def __init__(
        self,
        sentences: Sequence[str],
        splitter_llm: Any,
        config: PromptSplitConfig,
    ) -> None:
        """Prepare the first chunk synchronously and queue the rest later."""
        self._sentences = list(sentences)
        self._splitter_llm = splitter_llm
        self._config = config
        self._queue: queue.Queue[object] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._first_chunk, self._next_position = _next_chunk_from_position(
            self._sentences,
            0,
            self._splitter_llm,
            self._config,
        )

    def start(self) -> None:
        """Start background splitting for chunks after the first one."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._produce_remaining_chunks,
            name="scorespeak-prompt-splitter",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Request that background splitting stop as soon as possible."""
        self._stop_event.set()

    def __iter__(self) -> Iterator[str]:
        """Yield the first chunk, then chunks produced by the background thread."""
        self.start()
        if self._first_chunk:
            yield self._first_chunk
        while True:
            item = self._queue.get()
            if item is _DONE:
                break
            if isinstance(item, BaseException):
                raise item
            yield str(item)

    def _produce_remaining_chunks(self) -> None:
        """Split and enqueue all chunks after the first chunk."""
        try:
            position = self._next_position
            while position < len(self._sentences) and not self._stop_event.is_set():
                chunk, position = _next_chunk_from_position(
                    self._sentences,
                    position,
                    self._splitter_llm,
                    self._config,
                )
                if self._stop_event.is_set():
                    break
                if chunk:
                    self._queue.put(chunk)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Prompt splitting failed in the background")
            self._queue.put(exc)
        finally:
            self._queue.put(_DONE)


def split_sentences(text: str) -> list[str]:
    """Return rough sentence-like units while preserving score headings."""
    normalized = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9-])", normalized)
    return [part.strip() for part in parts if part.strip()]


def should_use_prompt_split(
    prompt: str,
    config: PromptSplitConfig | None = None,
) -> bool:
    """Return whether a prompt should activate prompt split mode."""
    safe_config = config or DEFAULT_PROMPT_SPLIT_CONFIG
    return len(split_sentences(prompt)) > max(0, safe_config.sentence_threshold)


def make_prompt_split_chunks(
    prompt: str,
    splitter_llm: Any,
    config: PromptSplitConfig | None = None,
) -> PromptSplitChunkStream:
    """Return a background chunk stream for one split prompt."""
    safe_config = config or DEFAULT_PROMPT_SPLIT_CONFIG
    sentences = split_sentences(prompt)
    return PromptSplitChunkStream(sentences, splitter_llm, safe_config)


def llm_splitter_messages(
    prompt: str,
    splitter_llm: Any,
    window_sentences: int = 10,
) -> list[str]:
    """Return all splitter chunks synchronously for experiments and tests."""
    config = PromptSplitConfig(
        sentence_threshold=0,
        window_sentences=window_sentences,
    )
    sentences = split_sentences(prompt)
    chunks = []
    position = 0
    while position < len(sentences):
        chunk, position = _next_chunk_from_position(
            sentences,
            position,
            splitter_llm,
            config,
        )
        if chunk:
            chunks.append(chunk)
    return chunks


def _next_chunk_from_position(
    sentences: Sequence[str],
    position: int,
    splitter_llm: Any,
    config: PromptSplitConfig,
) -> tuple[str, int]:
    """Choose and render the next chunk starting at ``position``."""
    if position >= len(sentences):
        return "", len(sentences)
    safe_window = max(2, int(config.window_sentences))
    window = sentences[position:position + safe_window]
    split_index = _choose_llm_split_index(window, splitter_llm, config)
    split_index = max(1, min(split_index, len(window)))
    chunk = " ".join(window[:split_index]).strip()
    return chunk, position + split_index


def _choose_llm_split_index(
    sentences: Sequence[str],
    splitter_llm: Any,
    config: PromptSplitConfig,
) -> int:
    """Ask the splitter model for a 1-based split index."""
    if len(sentences) <= 1:
        return 1
    try:
        prompt = _llm_splitter_prompt(sentences)
        response = _invoke_splitter(splitter_llm, prompt)
        text = _llm_text(response)
        match = re.search(r"\d+", text)
        if match is None:
            logger.warning("Splitter returned no integer: %r", text)
            return _fallback_split_index(sentences, config)
        return int(match.group(0))
    except Exception:  # noqa: BLE001
        logger.exception("Splitter failed; using fallback split size")
        return _fallback_split_index(sentences, config)


def _fallback_split_index(
    sentences: Sequence[str],
    config: PromptSplitConfig,
) -> int:
    """Return a clamped deterministic fallback split size."""
    fallback = max(1, int(config.fallback_split_sentences))
    return min(len(sentences), fallback)


def _invoke_splitter(splitter_llm: Any, prompt: str) -> Any:
    """Invoke a chat model with LangSmith metadata when supported."""
    message = HumanMessage(content=prompt)
    try:
        return splitter_llm.invoke([message], config=_SPLITTER_RUN_CONFIG)
    except TypeError:
        return splitter_llm.invoke([message])


def _llm_splitter_prompt(sentences: Sequence[str]) -> str:
    """Build a concise prompt for selecting one split point."""
    lines = [
        "Choose the next split point for a long user request that will be "
        "sent to an editing agent over multiple sequential turns.",
        "Return only one integer N, meaning: send sentence-like units 1..N "
        "as the next chunk.",
        "Optimize for a concrete, scoped chunk the agent can complete before "
        "the next chunk arrives.",
        "Do not assume any fixed prompt format. Use headings, lists, measure "
        "numbers, or section labels only as clues when they are present.",
        "Keep together setup, definitions, locators, constraints, exceptions, "
        "and follow-up references needed to make the requested work "
        "unambiguous.",
        "Avoid ending right after introducing a measure number, part, voice, "
        "beat, target object, range, ID, file path, or named section if later "
        "units in the window still refer to it.",
        "Prefer to split after one complete objective, location, phase, "
        "object, part, voice, or operation group. If a measure or section is "
        "small, keep it together.",
        "It is okay to split inside a large measure, section, or objective "
        "when the earlier units form a complete sub-task and later units can "
        "be handled from agent memory.",
        "Avoid splitting between an instruction and its validation, cleanup, "
        "ordering requirement, exception, or correction.",
        "Prefer 3-6 units when possible. Use 1-2 for standalone or very large "
        "units, and use up to 8-10 only when needed to avoid dangling context.",
        "If no semantic boundary is clearly better, choose the nearest "
        "complete task boundary.",
        "",
    ]
    for index, sentence in enumerate(sentences, start=1):
        lines.append(f"[{index}] {sentence}")
    return "\n".join(lines)


def _llm_text(message: Any) -> str:
    """Return plain text from a chat model response."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    return str(content).strip()
