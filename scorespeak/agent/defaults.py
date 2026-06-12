"""Shared defaults for the ScoreSpeak agent runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentModelOption:
    """Metadata for one selectable ScoreSpeak agent model."""

    id: str
    label: str


@dataclass(frozen=True)
class AgentReasoningEffortOption:
    """Metadata for one selectable OpenAI reasoning effort."""

    id: str
    label: str


DEFAULT_AGENT_MODEL = "gpt-5.4-mini"
AGENT_MODEL_OPTIONS = (
    AgentModelOption("gpt-5.4-mini", "GPT-5.4 Mini"),
    AgentModelOption("gpt-5.4", "GPT-5.4"),
    AgentModelOption("gpt-5.4-nano", "GPT-5.4 Nano"),
)
SUPPORTED_AGENT_MODEL_IDS = frozenset(option.id for option in AGENT_MODEL_OPTIONS)
AGENT_REASONING_EFFORT_API_DEFAULT = "api_default"
DEFAULT_AGENT_REASONING_EFFORT = "low"
OPENAI_RESPONSES_OUTPUT_VERSION = "responses/v1"
AGENT_REASONING_EFFORT_OPTIONS = (
    AgentReasoningEffortOption(AGENT_REASONING_EFFORT_API_DEFAULT, "API default"),
    AgentReasoningEffortOption("none", "None"),
    AgentReasoningEffortOption("minimal", "Minimal"),
    AgentReasoningEffortOption("low", "Low"),
    AgentReasoningEffortOption("medium", "Medium"),
    AgentReasoningEffortOption("high", "High"),
)
SUPPORTED_AGENT_REASONING_EFFORT_IDS = frozenset(
    option.id for option in AGENT_REASONING_EFFORT_OPTIONS
)
DEFAULT_RETRIEVAL_THRESHOLD = 0.5
DEFAULT_RECURSION_LIMIT = 60
DEFAULT_AUTO_TOOL_CANDIDATE_LIMIT = 10
DEFAULT_TOOL_SEARCH_LIMIT = 5
MAX_TOOL_SEARCH_LIMIT = 20
DEFAULT_AUTO_CONTEXT_MEASURE_LIMIT = 8
DEFAULT_AUTO_CONTEXT_PART_BAR_LIMIT = 16


def normalize_agent_model(model: str | None) -> str:
    """Return a supported agent model id or raise ``ValueError``."""
    selected_model = DEFAULT_AGENT_MODEL if model is None else str(model).strip()
    if not selected_model:
        selected_model = DEFAULT_AGENT_MODEL
    if selected_model not in SUPPORTED_AGENT_MODEL_IDS:
        supported = ", ".join(option.id for option in AGENT_MODEL_OPTIONS)
        raise ValueError(
            f"Unsupported agent model '{selected_model}'. "
            f"Supported models: {supported}."
        )
    return selected_model


def normalize_agent_reasoning_effort(reasoning_effort: object | None) -> str:
    """Return a supported reasoning effort id or raise ``ValueError``."""
    selected_effort = (
        DEFAULT_AGENT_REASONING_EFFORT
        if reasoning_effort is None
        else str(reasoning_effort).strip()
    )
    if not selected_effort:
        selected_effort = DEFAULT_AGENT_REASONING_EFFORT
    if selected_effort not in SUPPORTED_AGENT_REASONING_EFFORT_IDS:
        supported = ", ".join(option.id for option in AGENT_REASONING_EFFORT_OPTIONS)
        raise ValueError(
            f"Unsupported reasoning_effort '{selected_effort}'. "
            f"Supported reasoning efforts: {supported}."
        )
    return selected_effort


def chat_openai_reasoning_kwargs(reasoning_effort: object | None) -> dict[str, object]:
    """Return ChatOpenAI kwargs for OpenAI agent reasoning via Responses API."""
    selected_effort = normalize_agent_reasoning_effort(reasoning_effort)
    kwargs: dict[str, object] = {
        "use_responses_api": True,
        "output_version": OPENAI_RESPONSES_OUTPUT_VERSION,
    }
    if selected_effort == AGENT_REASONING_EFFORT_API_DEFAULT:
        return kwargs
    kwargs["reasoning"] = {"effort": selected_effort}
    return kwargs


def agent_model_options_payload() -> list[dict[str, str]]:
    """Return JSON-serializable metadata for supported agent models."""
    return [
        {
            "id": option.id,
            "label": option.label,
        }
        for option in AGENT_MODEL_OPTIONS
    ]


def agent_reasoning_effort_options_payload() -> list[dict[str, str]]:
    """Return UI-selectable reasoning efforts."""
    return [
        {
            "id": option.id,
            "label": option.label,
        }
        for option in AGENT_REASONING_EFFORT_OPTIONS
        if option.id != AGENT_REASONING_EFFORT_API_DEFAULT
    ]
