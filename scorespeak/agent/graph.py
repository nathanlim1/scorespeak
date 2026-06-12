"""LangChain agent loop for editing a live ``ScoreSpeak``."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from typing_extensions import NotRequired

from ..core import ScoreSpeak
from ..retrieval import (
    BarContextStatus,
    ExtractedContextScope,
    LexicalContextRetriever,
    MethodRecord,
)
from .context_renderers import render_summary_context
from .defaults import DEFAULT_AUTO_TOOL_CANDIDATE_LIMIT
from .prompt_split import (
    DEFAULT_PROMPT_SPLIT_CONFIG,
    PromptSplitConfig,
    make_prompt_split_chunks,
    should_use_prompt_split,
)
from .memory import (
    AgentMemoryStore,
    MemoryToolCallTrace,
    format_memory_context_for_prompt,
    make_memory_search_tool,
)
from .overview import ScoreOverview, build_score_overview, format_overview_for_prompt
from .tool_catalog import ToolCatalog
from .tools import (
    MutationRecorder,
    ToolExpansionRequests,
    filter_agent_method_records,
    make_inspect_score_attributes_tool,
    make_inspect_score_region_tool,
    make_search_score_tool,
    make_tool_search_tool,
    make_tools_from_records,
)

logger = logging.getLogger(__name__)


_CORE_TOOL_NAMES = frozenset({
    "search_score",
    "inspect_score_region",
    "inspect_score_attributes",
    "tool_search",
})

_TOOL_EXECUTION_MAX_CONCURRENCY = 1


_SYSTEM_PROMPT_TEMPLATE = """You are a ScoreSpeak agent, a music-editing assistant.

You edit a single MusicXML score through the provided tools.  Every tool
mutates a shared, live score object; you do not need to save, export, or
load files.  The score is always valid music21 after each call.

User-facing behavior:
- Do not mention internal tools, tool names, retrieval, schemas, API calls,
  or implementation details in ordinary replies to the user.
- Take initiative before asking for missing details. Use available score
  context, session memory, score search, and score inspection to resolve where
  something is or what the user likely means.
- Ask the user only when the target remains genuinely ambiguous or the edit
  would be unsafe after checking the available evidence.
{prompt_split_instruction}- If you cannot complete something, say plainly that you cannot do that in this
  editor yet, then suggest the closest practical alternative when one exists.
- Before ending the turn, verify that EVERY part of the user's request has
  either been completed or explicitly called out as not completed if you
  genuinely lack the capability to do it.

Music terminology:
- Notes are sounding pitched events. Rests are silence. Beats are metric
  positions in a measure. In user-facing language, rests and beats are not
  notes.
- "The first note of a measure" means the first played note or chord in the
  relevant part/voice, skipping leading rests and regardless of whether it
  starts on beat 1. "The first beat" means beat 1.0. "The first rest" means the
  first rest event.

Conventions:
- Measure numbers and beats are 1-based (beat 1 = start of the measure); in
  an N/4 measure, the measure spans beat 1.0 up to beat N+1.0, where N+1.0 
  is the exclusive end boundary / next measure downbeat. For example, in 4/4, 
  beat 4.5 is still inside the bar, and beat 5.0 is the end boundary.
- Fractional beat positions and numeric durations use quarter-note units:
  whole=4.0, half=2.0, quarter/crotchet=1.0, eighth/quaver=0.5,
  16th/semiquaver=0.25, and 32nd/demisemiquaver=0.125. Add these
  values to the 1-based beat position for subdivisions; for example,
  beat 2.25 is the second 16th-note slot after beat 2.
- Tools that accept ``part`` use an integer index or part name string for
  local part targeting. Score-wide tools omit ``part`` and affect all parts.
- If a tool returns a string beginning with ``ERROR:``, read the message
  and try again with corrected arguments. Do not invent tools; call only
  the tools currently available to you.
- Automatic retrieval lists candidate tool names only. If a needed write/edit
  tool is missing, call tool_search with a natural language query or exact
  tool_names=[...]. Returned tools become available on the next model step.
- Prefer retrieved context bars only when CONTEXT SCOPE says bars are explicit
  or end_fallback. If bars are missing, do not infer a final/all-bars target.
- When the user's request is referential or omits part/bar scope, use DIRECT
  PREVIOUS TURN MEMORY first unless the current request gives a conflicting
  explicit target. Then search or inspect the score when that can resolve the
  scope. If no memory/search/inspection evidence resolves the scope, ask in
  your final reply before making changes.
- NEVER remove anything that already exists in the score unless the user 
  explicitly asks you to do so or if you are performing a user requested 
  explicit replacement. Never assume that the user intends to remove/replace
  something if they don't explicitly ask you to do so.
- In this system, empty space within a measure is automatically completed with
  rests. Rest spelling refers to the makeup of silent gaps; for example, two
  quarter rests are an equivalent rest spelling of one half rest. Do not add,
  complete, fill, or reshape rests unless the user explicitly asks for visible
  rest notation or rest spelling that differs from what the system already
  creates.
- Use clear_measures when the user asks to empty, blank, clear, or erase
  existing bars while keeping the bar structure. Use delete_measures only when
  the user asks to remove the bars themselves.
- Use copy_measure_contents when the user asks to copy, paste, duplicate, 
  clone, or otherwise create "the same" content of bars/measures. It replaces 
  target musical contents while preserving target structure; it does not merge 
  with existing target music. Contents copied cannot be filtered, but any 
  necessary edits can be made after the copy.
- Keep your final natural-language reply short: confirm what you changed
  and flag anything you could not complete.
- Default score context below is a compact summary. Exact symbolic rows are
  intentionally omitted; call inspect_score_region when precise notes, ties,
  tuplets, or optional markings/spans are needed. Call search_score for typed
  semantic search and inspect_score_attributes for scoped time/key/tempo/clef/
  structure attributes without event rows.
- A voice is an independent rhythmic timeline inside the same part/instrument.
  Change voice only for simultaneous overlapping material with a different
  rhythm. Same-start, same-duration pitches are chords, not separate voices.
  Unless the user explicitly asks for a different voice or you must write in a
  different voice to carry out the user's request, always write in voice 1.
- Individual note/chord/rest-spelling tools are surgical. Note/chord entry may
  replace visible rests in the target voice, but must not overlap existing
  notes/chords.
- Use add_chord_tones when an existing note/chord should gain pitches in the
  same voice. Use remove_tuplet for tuplet deletion and remove_grace_note for
  grace notes. Rhythm edits auto-complete active voices with visible rests.
  Use add_rest when the user asks to add, show, reveal, unhide, or make one
  visible rest at a specific beat/duration. Hidden rests are not shown as
  normal score events; remove_rests hides visible rest notation, and add_rest
  can make the requested rest span visible again. Use fill_measure_gaps only
  for actual uncovered rhythmic gaps. Use reshape_rests when the user asks to
  split, merge, reformat, or change rest spelling.
- Accidentals are visual signs rendered next to pitches when needed for the
  current key signature. If the user asks to remove a sharp, flat, or natural
  from a note, usually use replace_note to change that note to the in-key
  version of the same pitch letter/octave rather than deleting the note.

SCORE OVERVIEW:
{overview_block}
{candidate_tools_block}
{scope_block}
{memory_block}
SCORE SUMMARY CONTEXT (retrieved for this turn):
{context_block}
"""


_PROMPT_SPLIT_INSTRUCTION = (
    "- These messages are chunks of one original request. Do not ask clarifying\n"
    "  questions. If the current chunk is ambiguous, use session memory and\n"
    "  memory_search to infer the missing scope, then proceed.\n"
)


def _format_prompt_split_instruction(enabled: bool) -> str:
    """Return the optional prompt split behavior instruction."""
    if not enabled:
        return ""
    return _PROMPT_SPLIT_INSTRUCTION


def _resolve_prompt_split_config(
    prompt_split_config: PromptSplitConfig | None,
) -> PromptSplitConfig:
    """Return the configured prompt split settings or defaults."""
    return prompt_split_config or DEFAULT_PROMPT_SPLIT_CONFIG


def _format_context_block(context_bars: Any) -> str:
    """Render a ``BarResultSet`` as default summary prompt context."""
    return render_summary_context(context_bars)


def _format_candidate_tools_block(candidate_tool_names: list[str] | None) -> str:
    """Render names-only automatic tool candidates for the system prompt."""
    if not candidate_tool_names:
        return ""

    lines = [
        "",
        "AUTO-SELECTED TOOL CANDIDATES (names only):",
    ]
    for name in candidate_tool_names:
        lines.append(f"- {name}")
    lines.append(
        "- Call tool_search(query=...) or tool_search(tool_names=[...]) "
        "to load needed edit tools before using them."
    )
    lines.append("")
    return "\n".join(lines)


def _format_bar_range_label(
    scope: ExtractedContextScope,
    context_bars: Any,
) -> str:
    """Return a human-readable string for the scope's bar range.

    When an explicit range was matched, uses the scope directly; when
    the retrieval fell back to the last bar, reads the resolved bar
    numbers out of ``context_bars`` so the provenance block reports the
    exact bar the agent is looking at.
    """
    status = _effective_bar_context_status(scope)
    if status == "missing":
        return "(none)"
    if scope.bar_range is not None:
        start, end = scope.bar_range
        if start == end:
            return str(start)
        return f"{start}-{end}"
    if scope.measure_numbers:
        sorted_bars = sorted(set(scope.measure_numbers))
        return ", ".join(str(bar) for bar in sorted_bars)
    if scope.used_fallback_bar and isinstance(context_bars, dict):
        bars = context_bars.get("bars") or []
        numbers = [
            bar.get("measure_number")
            for bar in bars
            if isinstance(bar, dict) and bar.get("measure_number") is not None
        ]
        if numbers:
            if len(numbers) == 1:
                return str(numbers[0])
            return f"{min(numbers)}-{max(numbers)}"
    return "(all)"


def _effective_bar_context_status(scope: ExtractedContextScope) -> BarContextStatus:
    """Return a status compatible with older manually built scope objects."""
    if scope.bar_context_status != "missing":
        return scope.bar_context_status
    if scope.bar_range is not None or scope.measure_numbers:
        return "explicit"
    if scope.used_fallback_bar:
        return "end_fallback"
    return "missing"


def _format_parts_label(scope: ExtractedContextScope) -> str:
    """Return a human-readable string for the scope's matched parts."""
    if scope.part_indices:
        indices_str = ", ".join(str(idx) for idx in scope.part_indices)
        if scope.matched_part_names:
            names_str = ", ".join(scope.matched_part_names)
            return f"[{indices_str}] (matched: {names_str})"
        return f"[{indices_str}]"
    return "(all; no specific part resolved)"


def _format_scope_block(
    scope: Optional[ExtractedContextScope],
    context_bars: Any,
) -> str:
    """Render a ``CONTEXT SCOPE`` provenance block or empty string.

    Emits the block whenever any explicit scope cue, fallback, or
    ambiguity exists so the agent can tell the difference between
    "user said bar 5" and "we guessed".
    """
    if scope is None:
        return ""

    has_content = bool(
        scope.part_indices
        or scope.measure_numbers
        or scope.bar_range
        or scope.used_fallback_bar
        or _effective_bar_context_status(scope) == "missing"
        or scope.ambiguity_messages
        or scope.context_truncation_messages
    )
    if not has_content:
        return ""

    lines: list[str] = ["", "CONTEXT SCOPE:"]
    lines.append("- retrieval: automatic")
    lines.append(f"- parts: {_format_parts_label(scope)}")
    status = _effective_bar_context_status(scope)
    lines.append(
        f"- bars: {_format_bar_range_label(scope, context_bars)} "
        f"(status: {status})"
    )
    for message in scope.ambiguity_messages:
        lines.append(f"- ambiguity: {message}")
    for message in scope.context_truncation_messages:
        lines.append(f"- truncated: {message}")
    lines.append("")
    return "\n".join(lines)


def build_system_prompt(
    overview: ScoreOverview,
    context_bars: Any,
    scope: Optional[ExtractedContextScope] = None,
    memory_store: AgentMemoryStore | None = None,
    candidate_tool_names: list[str] | None = None,
    *,
    prompt_split_mode: bool = False,
) -> str:
    """Assemble the per-turn system prompt from the overview and bar context.

    Args:
        overview: The pre-computed score overview block.
        context_bars: The ``BarResultSet`` returned by lexical retrieval.
        scope: The resolved ``ExtractedContextScope`` from this turn's
            retrieval. When provided and non-trivial, its content is
            rendered as a short ``CONTEXT SCOPE`` provenance block so the
            agent can see which parts/bars were inferred, whether a
            fallback bar was used, and any ambiguity warnings.
        memory_store: Optional session memory. When non-empty, the prompt
            receives direct previous turn memory. Missing part or bar scope
            gets fuller previous-turn detail; complete scope gets a brief
            summary plus a reminder to use ``memory_search`` for older or
            exact details.
        candidate_tool_names: Names-only automatic tool candidates. These are
            not callable until loaded by ``tool_search``.
        prompt_split_mode: Whether the current user message is one chunk of
            a larger submitted prompt.
    """
    overview_block = format_overview_for_prompt(overview)
    context_block = _format_context_block(context_bars)
    scope_block = _format_scope_block(scope, context_bars)
    memory_block = _format_memory_prompt_block(memory_store, scope, overview)
    candidate_tools_block = _format_candidate_tools_block(candidate_tool_names)
    return _SYSTEM_PROMPT_TEMPLATE.format(
        overview_block=overview_block,
        candidate_tools_block=candidate_tools_block,
        scope_block=scope_block,
        memory_block=memory_block,
        context_block=context_block,
        prompt_split_instruction=_format_prompt_split_instruction(
            prompt_split_mode
        ),
    )


def _format_memory_prompt_block(
    memory_store: AgentMemoryStore | None,
    scope: Optional[ExtractedContextScope],
    overview: ScoreOverview,
) -> str:
    """Return an optional session-memory prompt block with trailing spacing."""
    if memory_store is None:
        return ""
    detail = (
        "full_previous_turn"
        if _scope_needs_full_previous_turn_memory(scope, len(overview.parts))
        else "brief"
    )
    memory_context = format_memory_context_for_prompt(memory_store, detail=detail)
    if not memory_context:
        return ""
    return f"{memory_context}\n\n"


def _scope_needs_full_previous_turn_memory(
    scope: Optional[ExtractedContextScope],
    part_count: int,
) -> bool:
    """Return whether missing current scope should expand prompt memory."""
    if scope is None:
        return False
    missing_part_scope = scope.part_indices is None and part_count != 1
    status = _effective_bar_context_status(scope)
    missing_bar_scope = (
        status == "missing"
        and scope.bar_range is None
        and scope.measure_numbers is None
    )
    return missing_part_scope or missing_bar_scope


class ScoreSpeakAgentState(AgentState):
    """Agent state fields used for dynamic ScoreSpeak tool narrowing."""

    candidate_tool_names: NotRequired[list[str]]


@dataclass
class AgentToolBundle:
    """All pre-registered tools plus metadata for dynamic filtering."""

    tools: list[StructuredTool]
    tools_by_name: dict[str, StructuredTool]
    catalog: ToolCatalog
    expansion_requests: ToolExpansionRequests
    core_tool_names: set[str]


@dataclass
class AgentTurnRuntime:
    """Prepared graph inputs for one agent turn."""

    graph: Any
    input_state: dict[str, Any]
    config: dict[str, Any]


@dataclass
class AgentTurnExecution:
    """Completed single-turn execution details."""

    response: str
    messages: list[Any]
    status: str
    error: str | None


@dataclass
class AgentPromptRunResult:
    """Completed prompt execution details, including prompt split chunks."""

    response: str
    messages: list[Any]
    chunks: list[str]
    chunk_responses: list[str]
    prompt_split_mode: bool
    errors: list[str]


class ScoreSpeakAgentMiddleware(AgentMiddleware):
    """Refresh prompt context and filter model-visible tools each model call."""

    state_schema = ScoreSpeakAgentState

    def __init__(
        self,
        score_state: ScoreSpeak,
        context_bars: Any,
        scope: Optional[ExtractedContextScope],
        expansion_requests: ToolExpansionRequests,
        core_tool_names: set[str],
        memory_store: AgentMemoryStore | None = None,
        candidate_tool_names: list[str] | None = None,
        prompt_split_mode: bool = False,
    ) -> None:
        self._score_state = score_state
        self._context_bars = context_bars
        self._scope = scope
        self._expansion_requests = expansion_requests
        self._core_tool_names = set(core_tool_names)
        self._memory_store = memory_store
        self._candidate_tool_names = list(candidate_tool_names or [])
        self._prompt_split_mode = bool(prompt_split_mode)

    def visible_tool_names(self, state: dict[str, Any] | None) -> set[str]:
        """Return the names exposed to the model for this model call."""
        del state
        names = set(self._core_tool_names)
        names.update(self._expansion_requests.loaded_tool_names)
        return names

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse:
        """Apply dynamic prompt and tool filtering before each model call."""
        overview = build_score_overview(self._score_state)
        system_prompt = build_system_prompt(
            overview,
            self._context_bars,
            self._scope,
            self._memory_store,
            self._candidate_tool_names,
            prompt_split_mode=self._prompt_split_mode,
        )
        visible_names = self.visible_tool_names(request.state)
        tools = _filter_tools_by_name(request.tools or [], visible_names)
        return handler(
            request.override(
                system_message=SystemMessage(content=system_prompt),
                tools=tools,
            )
        )


def _tool_name(tool: Any) -> str | None:
    """Return a tool name from a LangChain tool or tool-schema dict."""
    if isinstance(tool, dict):
        name = tool.get("name")
        return str(name) if name else None
    name = getattr(tool, "name", None)
    return str(name) if name else None


def _filter_tools_by_name(tools: list[Any], names: set[str]) -> list[Any]:
    """Keep only tools whose names are in ``names``."""
    filtered = []
    for tool in tools:
        name = _tool_name(tool)
        if name is not None and name in names:
            filtered.append(tool)
    return filtered


def _candidate_tool_names(
    records: list[MethodRecord],
    limit: int = DEFAULT_AUTO_TOOL_CANDIDATE_LIMIT,
) -> list[str]:
    """Return capped names-only automatic tool candidates."""
    safe_limit = max(0, int(limit))
    return [record.name for record in records[:safe_limit]]


def build_agent_tool_bundle(
    score_state: ScoreSpeak,
    all_records: list[MethodRecord],
    memory_store: AgentMemoryStore,
    mutation_recorder: MutationRecorder | None = None,
) -> AgentToolBundle:
    """Pre-register all ScoreSpeak tools plus the always-available core."""
    agent_records = filter_agent_method_records(all_records)
    catalog = ToolCatalog(agent_records)
    expansion_requests = ToolExpansionRequests()
    core_tool_names = set(_CORE_TOOL_NAMES)
    tools_by_name = {
        tool.name: tool
        for tool in make_tools_from_records(
            score_state,
            agent_records,
            mutation_recorder,
        )
    }

    search_score_tool = make_search_score_tool(score_state)
    tools_by_name[search_score_tool.name] = search_score_tool
    inspect_tool = make_inspect_score_region_tool(score_state)
    tools_by_name[inspect_tool.name] = inspect_tool
    attributes_tool = make_inspect_score_attributes_tool(score_state)
    tools_by_name[attributes_tool.name] = attributes_tool
    if memory_store.has_entries():
        memory_tool = make_memory_search_tool(memory_store)
        tools_by_name[memory_tool.name] = memory_tool
        core_tool_names.add(memory_tool.name)
    search_tool = make_tool_search_tool(
        catalog,
        expansion_requests,
        tools_by_name,
        core_tool_names=core_tool_names,
    )
    tools_by_name[search_tool.name] = search_tool

    return AgentToolBundle(
        tools=list(tools_by_name.values()),
        tools_by_name=tools_by_name,
        catalog=catalog,
        expansion_requests=expansion_requests,
        core_tool_names=core_tool_names,
    )


def _extract_final_text(result: dict) -> str:
    """Extract the final assistant message text from a ReAct agent result."""
    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                joined = "".join(parts).strip()
                if joined:
                    return joined
    return "(no response produced)"


def tool_progress_label(tool_name: str | None) -> str:
    """Return a user-facing progress label for an agent tool call."""
    if not tool_name:
        return "Working..."

    name = str(tool_name)
    if name in {
        "search_score",
        "inspect_score_region",
        "inspect_score_attributes",
        "get_metadata",
        "list_parts",
    } or name.startswith("get_active_"):
        return "Examining the score..."
    if name == "memory_search":
        return "Checking session memory..."
    if name == "tool_search":
        return "Choosing score-editing tools..."
    if name.startswith("copy_"):
        return "Copying notation..."
    if name.startswith("transpose"):
        return "Transposing the score..."
    if name.startswith("set_"):
        return "Updating score settings..."
    if name.startswith(("remove_", "delete_")):
        return "Removing notation..."
    if (
        name.startswith(("add_", "insert_"))
        or "note" in name
        or "rest" in name
        or "chord" in name
        or "dynamic" in name
        or "lyric" in name
        or "slur" in name
        or "marking" in name
        or "articulation" in name
    ):
        return "Adding notation..."
    return "Working..."


def _prepare_agent_turn(
    score_state: ScoreSpeak,
    retriever: LexicalContextRetriever,
    llm: BaseChatModel,
    user_text: str,
    memory_store: AgentMemoryStore,
    *,
    recursion_limit: int,
    mutation_recorder: MutationRecorder | None = None,
    prompt_split_mode: bool = False,
) -> AgentTurnRuntime | str:
    """Build the graph and input state for a single agent turn."""
    retrieval = retriever.query(user_text)
    method_records = filter_agent_method_records(
        rec for rec, _ in retrieval.methods
    )

    candidate_tool_names = _candidate_tool_names(method_records)
    bundle = build_agent_tool_bundle(
        score_state,
        retriever.method_records,
        memory_store,
        mutation_recorder,
    )
    if not bundle.tools:
        return (
            "ERROR: no usable tools could be built for this turn. "
            "Please rephrase your request or check the retrieval configuration."
        )

    middleware = ScoreSpeakAgentMiddleware(
        score_state,
        retrieval.context_bars,
        retrieval.scope,
        bundle.expansion_requests,
        bundle.core_tool_names,
        memory_store,
        candidate_tool_names,
        prompt_split_mode=prompt_split_mode,
    )

    graph = create_agent(
        llm,
        tools=bundle.tools,
        middleware=[middleware],
        state_schema=ScoreSpeakAgentState,
    )

    return AgentTurnRuntime(
        graph=graph,
        input_state={
            "messages": [HumanMessage(content=user_text)],
            "candidate_tool_names": candidate_tool_names,
        },
        config={
            "recursion_limit": recursion_limit,
            # Tool calls share one mutable music21 score, so same-message
            # tool calls must run serially even when the model requests more
            # than one at once.
            "max_concurrency": _TOOL_EXECUTION_MAX_CONCURRENCY,
        },
    )


def _messages_from_stream_update(update: Any) -> list[Any]:
    """Extract LangChain messages from a LangGraph stream update."""
    if isinstance(update, tuple) and len(update) == 2:
        update = update[1]

    messages: list[Any] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            direct_messages = value.get("messages")
            if isinstance(direct_messages, list):
                messages.extend(direct_messages)
            for nested in value.values():
                if nested is not direct_messages:
                    visit(nested)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(update)
    return messages


def stream_events_from_update(update: Any) -> list[dict[str, Any]]:
    """Convert a LangGraph update into public progress events."""
    events: list[dict[str, Any]] = []
    for message in _messages_from_stream_update(update):
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                tool_name = tool_call.get("name")
                events.append({
                    "type": "tool_start",
                    "tool": tool_name,
                    "label": tool_progress_label(tool_name),
                })
        elif isinstance(message, ToolMessage):
            status = getattr(message, "status", None)
            events.append({
                "type": "tool_end",
                "tool": getattr(message, "name", None),
                "ok": status in (None, "success"),
            })
    return events


def _message_content_to_text(content: Any) -> str:
    """Convert a LangChain message content payload into compact text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return " ".join(parts)
    return str(content)


def _tool_trace_from_messages(messages: list[Any]) -> list[MemoryToolCallTrace]:
    """Extract compact tool calls and results from LangChain messages."""
    trace_items: list[dict[str, Any]] = []
    by_call_id: dict[str, int] = {}

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                name = str(tool_call.get("name") or "")
                args = tool_call.get("args")
                if not isinstance(args, dict):
                    args = {}
                item = {
                    "name": name,
                    "arguments": dict(args),
                    "result": None,
                    "ok": None,
                }
                trace_items.append(item)
                call_id = tool_call.get("id")
                if call_id:
                    by_call_id[str(call_id)] = len(trace_items) - 1
        elif isinstance(message, ToolMessage):
            status = getattr(message, "status", None)
            result_text = _message_content_to_text(message.content)
            ok = status in (None, "success") and not result_text.startswith("ERROR")
            call_id = getattr(message, "tool_call_id", None)
            index = by_call_id.get(str(call_id)) if call_id else None
            if index is None:
                trace_items.append({
                    "name": str(getattr(message, "name", "") or ""),
                    "arguments": {},
                    "result": result_text,
                    "ok": ok,
                })
            else:
                trace_items[index]["result"] = result_text
                trace_items[index]["ok"] = ok

    return [
        MemoryToolCallTrace(
            name=str(item["name"]),
            arguments=dict(item["arguments"]),
            result=item["result"],
            ok=item["ok"],
        )
        for item in trace_items
    ]


def _record_turn_memory(
    memory_store: AgentMemoryStore,
    turn_id: str,
    user_text: str,
    messages: list[Any],
    final_response: str,
    status: str,
    error: str | None,
) -> None:
    """Append instruction and execution memory for a completed agent turn."""
    try:
        memory_store.record_turn(
            turn_id,
            user_text,
            _tool_trace_from_messages(messages),
            final_response,
            status,
            error=error,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to record agent memory")


def _run_turn_collect(
    score_state: ScoreSpeak,
    retriever: LexicalContextRetriever,
    llm: BaseChatModel,
    user_text: str,
    memory_store: AgentMemoryStore,
    *,
    recursion_limit: int = 30,
    mutation_recorder: MutationRecorder | None = None,
    prompt_split_mode: bool = False,
) -> AgentTurnExecution:
    """Run one edit turn and return response, messages, and error state."""
    turn_id = memory_store.next_turn_id()
    messages: list[Any] = []
    final_response = ""
    status = "error"
    error: str | None = None
    try:
        prepare_kwargs: dict[str, Any] = {
            "recursion_limit": recursion_limit,
            "mutation_recorder": mutation_recorder,
        }
        if prompt_split_mode:
            prepare_kwargs["prompt_split_mode"] = True
        runtime = _prepare_agent_turn(
            score_state,
            retriever,
            llm,
            user_text,
            memory_store,
            **prepare_kwargs,
        )
        if isinstance(runtime, str):
            final_response = runtime
            error = runtime
            return AgentTurnExecution(final_response, messages, status, error)

        result = runtime.graph.invoke(runtime.input_state, config=runtime.config)
        raw_messages = result.get("messages", [])
        if isinstance(raw_messages, list):
            messages = raw_messages
        final_response = _extract_final_text(result)
        status = "ok"
        return AgentTurnExecution(final_response, messages, status, error)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent turn failed")
        final_response = f"ERROR: agent turn failed: {type(exc).__name__}: {exc}"
        error = final_response
        return AgentTurnExecution(final_response, messages, status, error)
    finally:
        if final_response:
            _record_turn_memory(
                memory_store,
                turn_id,
                user_text,
                messages,
                final_response,
                status,
                error,
            )


def run_turn(
    score_state: ScoreSpeak,
    retriever: LexicalContextRetriever,
    llm: BaseChatModel,
    user_text: str,
    memory_store: AgentMemoryStore,
    *,
    recursion_limit: int = 30,
    mutation_recorder: MutationRecorder | None = None,
    prompt_split_mode: bool = False,
) -> str:
    """Run a single edit turn against ``score_state`` and return the agent reply.

    Args:
        score_state: The live score.  Tool calls mutate this instance.
        retriever: Configured lexical context retriever wrapping the same
            ``score_state``.
        llm: The chat model used by the agent (e.g. ``ChatOpenAI``).
        user_text: The raw natural-language request from the user.
        memory_store: Session-scoped memory store. The current prompt is
            written only after the graph completes.
        recursion_limit: Maximum number of LangGraph steps per turn.  Acts
            as a guard against runaway tool loops.

    Returns:
        The final assistant message content. If the agent graph raises, the
        error is stringified and returned so the REPL can keep looping.
    """
    result = _run_turn_collect(
        score_state,
        retriever,
        llm,
        user_text,
        memory_store,
        recursion_limit=recursion_limit,
        mutation_recorder=mutation_recorder,
        prompt_split_mode=prompt_split_mode,
    )
    return result.response


def run_prompt_collect(
    score_state: ScoreSpeak,
    retriever: LexicalContextRetriever,
    llm: BaseChatModel,
    user_text: str,
    memory_store: AgentMemoryStore,
    *,
    recursion_limit: int = 30,
    mutation_recorder: MutationRecorder | None = None,
    splitter_llm: BaseChatModel | None = None,
    prompt_split_config: PromptSplitConfig | None = None,
) -> AgentPromptRunResult:
    """Run one submitted prompt, automatically chunking split prompts."""
    config = _resolve_prompt_split_config(prompt_split_config)
    if not should_use_prompt_split(user_text, config):
        turn_result = _run_turn_collect(
            score_state,
            retriever,
            llm,
            user_text,
            memory_store,
            recursion_limit=recursion_limit,
            mutation_recorder=mutation_recorder,
            prompt_split_mode=False,
        )
        errors = [turn_result.error] if turn_result.error is not None else []
        return AgentPromptRunResult(
            response=turn_result.response,
            messages=list(turn_result.messages),
            chunks=[user_text],
            chunk_responses=[turn_result.response] if turn_result.response else [],
            prompt_split_mode=False,
            errors=errors,
        )

    chunk_stream = make_prompt_split_chunks(user_text, splitter_llm or llm, config)
    messages: list[Any] = []
    chunks: list[str] = []
    chunk_responses: list[str] = []
    errors: list[str] = []
    final_response = ""
    try:
        for chunk in chunk_stream:
            if not chunk:
                continue
            chunks.append(chunk)
            turn_result = _run_turn_collect(
                score_state,
                retriever,
                llm,
                chunk,
                memory_store,
                recursion_limit=recursion_limit,
                mutation_recorder=mutation_recorder,
                prompt_split_mode=True,
            )
            messages.extend(turn_result.messages)
            final_response = turn_result.response
            if turn_result.response:
                chunk_responses.append(turn_result.response)
            if turn_result.error is not None:
                errors.append(turn_result.error)
                break
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prompt split run failed")
        final_response = (
            f"ERROR: prompt split failed: {type(exc).__name__}: {exc}"
        )
        errors.append(final_response)
    finally:
        chunk_stream.close()

    return AgentPromptRunResult(
        response=final_response,
        messages=messages,
        chunks=chunks,
        chunk_responses=chunk_responses,
        prompt_split_mode=True,
        errors=errors,
    )


def run_prompt(
    score_state: ScoreSpeak,
    retriever: LexicalContextRetriever,
    llm: BaseChatModel,
    user_text: str,
    memory_store: AgentMemoryStore,
    *,
    recursion_limit: int = 30,
    mutation_recorder: MutationRecorder | None = None,
    splitter_llm: BaseChatModel | None = None,
    prompt_split_config: PromptSplitConfig | None = None,
) -> str:
    """Run one submitted prompt and return the final agent reply."""
    result = run_prompt_collect(
        score_state,
        retriever,
        llm,
        user_text,
        memory_store,
        recursion_limit=recursion_limit,
        mutation_recorder=mutation_recorder,
        splitter_llm=splitter_llm,
        prompt_split_config=prompt_split_config,
    )
    return result.response


def run_turn_stream(
    score_state: ScoreSpeak,
    retriever: LexicalContextRetriever,
    llm: BaseChatModel,
    user_text: str,
    memory_store: AgentMemoryStore,
    *,
    recursion_limit: int = 30,
    mutation_recorder: MutationRecorder | None = None,
    prompt_split_mode: bool = False,
) -> Iterator[dict[str, Any]]:
    """Run an agent turn and yield user-facing progress events."""
    turn_id = memory_store.next_turn_id()
    messages: list[Any] = []
    final_response = ""
    status = "error"
    error: str | None = None
    recorded = False

    def record_once() -> None:
        """Record the turn memory exactly once."""
        nonlocal recorded
        if recorded:
            return
        recorded = True
        _record_turn_memory(
            memory_store,
            turn_id,
            user_text,
            messages,
            final_response,
            status,
            error,
        )

    try:
        yield {"type": "phase", "label": "Examining the score..."}
        prepare_kwargs: dict[str, Any] = {
            "recursion_limit": recursion_limit,
            "mutation_recorder": mutation_recorder,
        }
        if prompt_split_mode:
            prepare_kwargs["prompt_split_mode"] = True
        runtime = _prepare_agent_turn(
            score_state,
            retriever,
            llm,
            user_text,
            memory_store,
            **prepare_kwargs,
        )
        if isinstance(runtime, str):
            final_response = runtime
            error = runtime
            record_once()
            yield {"type": "error", "error": runtime}
            return

        yield {"type": "phase", "label": "Choosing score-editing tools..."}
        yield {"type": "phase", "label": "Planning edits..."}
        for update in runtime.graph.stream(
            runtime.input_state,
            config=runtime.config,
            stream_mode="updates",
        ):
            update_messages = _messages_from_stream_update(update)
            messages.extend(update_messages)
            yield from stream_events_from_update(update)

        yield {"type": "phase", "label": "Rendering the updated score..."}
        final_response = _extract_final_text({"messages": messages})
        status = "ok"
        record_once()
        yield {
            "type": "final",
            "response": final_response,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Streaming agent turn failed")
        final_response = f"ERROR: agent turn failed: {type(exc).__name__}: {exc}"
        error = final_response
        record_once()
        yield {
            "type": "error",
            "error": final_response,
        }


def run_prompt_stream(
    score_state: ScoreSpeak,
    retriever: LexicalContextRetriever,
    llm: BaseChatModel,
    user_text: str,
    memory_store: AgentMemoryStore,
    *,
    recursion_limit: int = 30,
    mutation_recorder: MutationRecorder | None = None,
    splitter_llm: BaseChatModel | None = None,
    prompt_split_config: PromptSplitConfig | None = None,
) -> Iterator[dict[str, Any]]:
    """Run one submitted prompt and stream progress events."""
    config = _resolve_prompt_split_config(prompt_split_config)
    if not should_use_prompt_split(user_text, config):
        yield from run_turn_stream(
            score_state,
            retriever,
            llm,
            user_text,
            memory_store,
            recursion_limit=recursion_limit,
            mutation_recorder=mutation_recorder,
            prompt_split_mode=False,
        )
        return

    chunk_stream = None
    last_response = ""
    try:
        yield {"type": "phase", "label": "Splitting prompt..."}
        chunk_stream = make_prompt_split_chunks(
            user_text,
            splitter_llm or llm,
            config,
        )
        for index, chunk in enumerate(chunk_stream, start=1):
            if not chunk:
                continue
            yield {
                "type": "phase",
                "label": f"Working through prompt split chunk {index}...",
            }
            for event in run_turn_stream(
                score_state,
                retriever,
                llm,
                chunk,
                memory_store,
                recursion_limit=recursion_limit,
                mutation_recorder=mutation_recorder,
                prompt_split_mode=True,
            ):
                if event.get("type") == "final":
                    last_response = str(event.get("response") or "")
                    continue
                yield event
                if event.get("type") == "error":
                    return
        yield {
            "type": "final",
            "response": last_response,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Streaming prompt split failed")
        yield {
            "type": "error",
            "error": f"ERROR: prompt split failed: {type(exc).__name__}: {exc}",
        }
    finally:
        if chunk_stream is not None:
            chunk_stream.close()


def summarize_turn_context(
    score_state: ScoreSpeak,
    retriever: LexicalContextRetriever,
    user_text: str,
    memory_store: AgentMemoryStore | None = None,
) -> dict:
    """Return a dict describing what a turn would see, without calling the LLM.

    Useful for debugging and tests: returns the retrieved method names,
    the bar scope, the overview, and the rendered system prompt.
    """
    retrieval = retriever.query(user_text)
    overview = build_score_overview(score_state)
    method_records = filter_agent_method_records(
        rec for rec, _ in retrieval.methods
    )
    candidate_tool_names = _candidate_tool_names(method_records)
    system_prompt = build_system_prompt(
        overview,
        retrieval.context_bars,
        retrieval.scope,
        memory_store,
        candidate_tool_names,
    )
    core_tool_names = set(_CORE_TOOL_NAMES)
    if memory_store is not None and memory_store.has_entries():
        core_tool_names.add("memory_search")

    return {
        "overview": asdict(overview),
        "method_hits": candidate_tool_names,
        "candidate_tool_names": candidate_tool_names,
        "always_available_tools": sorted(core_tool_names),
        "inspection_tool": "inspect_score_region",
        "search_tool": "search_score",
        "summary_context": render_summary_context(retrieval.context_bars),
        "scope": {
            "part_indices": retrieval.scope.part_indices,
            "measure_numbers": retrieval.scope.measure_numbers,
            "bar_range": retrieval.scope.bar_range,
            "bar_context_status": retrieval.scope.bar_context_status,
        },
        "system_prompt": system_prompt,
    }
