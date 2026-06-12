"""
Interactive CLI REPL that drives the ScoreSpeak agent loop.

Run with::

    python -m scorespeak.agent.repl --new --output edits.musicxml
    python -m scorespeak.agent.repl --musicxml path/to/score.musicxml --output out.musicxml

Each prompt line is sent through :func:`scorespeak.agent.graph.run_turn`.
After every turn the current score is auto-exported to ``--output`` so the
user can open the file in MuseScore between queries.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from ..core import ScoreSpeak
from ..retrieval import LexicalContextRetriever
from .defaults import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_RETRIEVAL_THRESHOLD,
    chat_openai_reasoning_kwargs,
)
from .graph import run_prompt as run_turn
from .memory import AgentMemoryStore

_DEFAULT_MODEL = DEFAULT_AGENT_MODEL
_DEFAULT_THRESHOLD = DEFAULT_RETRIEVAL_THRESHOLD
_DEFAULT_OUTPUT = "agent_output.musicxml"
_PROMPT = "edit> "
_QUIT_WORDS = {"quit", "exit", "q"}


logger = logging.getLogger(__name__)


def _load_dotenv_if_available() -> None:
    """Load variables from a ``.env`` file when ``python-dotenv`` is present."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _require_openai_key() -> str:
    """Return the OpenAI API key or exit with a helpful message."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print(
            "ERROR: OPENAI_API_KEY is not set. "
            "Add it to your environment or to a .env file in the project root.",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def _load_score(
    musicxml_path: Optional[str],
    start_new: bool,
) -> ScoreSpeak:
    """Build a starting ``ScoreSpeak`` from CLI flags."""
    if musicxml_path and start_new:
        raise SystemExit("Choose either --new or --musicxml PATH, not both.")
    if musicxml_path:
        path = Path(musicxml_path)
        if not path.exists():
            raise SystemExit(f"MusicXML file not found: {path}")
        return ScoreSpeak.from_musicxml(path)
    return ScoreSpeak.create(measures=8)


def _autosave(score_state: ScoreSpeak, output_path: Path) -> None:
    """Export the score to ``output_path``; log but do not raise on failure."""
    try:
        score_state.to_musicxml(output_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[warn] could not export to {output_path}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def _build_llm(model: str, api_key: str):
    """Construct the chat model.  Imported lazily so tests can stub it."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        **chat_openai_reasoning_kwargs(None),
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the agent REPL."""
    parser = argparse.ArgumentParser(
        prog="scorespeak-agent",
        description="Chat-based editor for ScoreSpeak, powered by LangGraph.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--new",
        action="store_true",
        help="Start from an empty 8-bar score (default when no source is given).",
    )
    source.add_argument(
        "--musicxml",
        metavar="PATH",
        help="Load the starting score from a MusicXML file.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=_DEFAULT_OUTPUT,
        help=f"Auto-export destination after each turn (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"OpenAI chat model id (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=_DEFAULT_THRESHOLD,
        help=f"Lexical retrieval threshold in [0, 1] (default: {_DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def _print_intro(score_state: ScoreSpeak, output_path: Path, model: str) -> None:
    """Print a short welcome block before the REPL starts."""
    from .overview import build_score_overview, format_overview_for_prompt

    overview = build_score_overview(score_state)
    print("ScoreSpeak agent REPL (type 'quit' to exit)")
    print(f"Model: {model}")
    print(f"Output: {output_path}")
    print("--- initial score overview ---")
    print(format_overview_for_prompt(overview))
    print("------------------------------")


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for the agent REPL."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    _load_dotenv_if_available()
    api_key = _require_openai_key()

    score_state = _load_score(args.musicxml, args.new)
    retriever = LexicalContextRetriever(score_state, threshold=args.threshold)
    memory_store = AgentMemoryStore()
    llm = _build_llm(args.model, api_key)

    output_path = Path(args.output)
    _print_intro(score_state, output_path, args.model)
    _autosave(score_state, output_path)

    while True:
        try:
            raw = input(_PROMPT)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        text = raw.strip()
        if not text:
            continue
        if text.lower() in _QUIT_WORDS:
            break

        reply = run_turn(score_state, retriever, llm, text, memory_store)
        print(reply)
        _autosave(score_state, output_path)

    print("goodbye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
