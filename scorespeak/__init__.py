"""
ScoreSpeak — Music-theory-aware sheet music editing framework.

Provides a stateful ScoreSpeak class that wraps a single score and exposes
editing operations as methods. Built on top of music21 for MusicXML I/O and
music theory primitives.
"""

from .core import ScoreSpeak
from .types import (
    ArticulationType,
    BarEventRow,
    BarGroup,
    BarlineType,
    BarPart,
    BarQuery,
    BarQueryEvent,
    BarQueryInput,
    BarQueryMatch,
    BarQueryOptions,
    BarQueryScope,
    BarResultSet,
    BarVoice,
    ClefType,
    DynamicLevel,
    HairpinType,
    LyricInfo,
    MeasureInfo,
    NoteInfo,
    OperationResult,
    PartInfo,
    RepeatDirection,
    SearchMatchRow,
    ScorePartSpec,
    TupletInfo,
    TupletSpanRow,
)

__all__ = [
    "ScoreSpeak",
    "ArticulationType",
    "BarEventRow",
    "BarGroup",
    "BarlineType",
    "BarPart",
    "BarQuery",
    "BarQueryEvent",
    "BarQueryInput",
    "BarQueryMatch",
    "BarQueryOptions",
    "BarQueryScope",
    "BarResultSet",
    "BarVoice",
    "ClefType",
    "DynamicLevel",
    "HairpinType",
    "LyricInfo",
    "MeasureInfo",
    "NoteInfo",
    "OperationResult",
    "PartInfo",
    "RepeatDirection",
    "SearchMatchRow",
    "ScorePartSpec",
    "TupletInfo",
    "TupletSpanRow",
]
