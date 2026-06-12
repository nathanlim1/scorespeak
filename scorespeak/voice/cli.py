"""
Command-line interface for file-based voice preprocessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

from .pipeline import VoiceInputProcessor


def _json_default(value: Any) -> Any:
    """Return a JSON-friendly representation for voice CLI output."""
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the speech-only voice CLI parser."""
    parser = argparse.ArgumentParser(
        prog="scorespeak-voice",
        description="Preprocess speech audio into structured JSON.",
    )
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--prompt", dest="speech_prompt", default=None)
    parser.add_argument("--language", default="en")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the speech-only voice CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    processor = VoiceInputProcessor()
    result = processor.process(
        args.audio_path,
        speech_prompt=args.speech_prompt,
        language=args.language,
    )
    print(json.dumps(asdict(result), default=_json_default, indent=2, sort_keys=True))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
