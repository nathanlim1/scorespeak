"""Composite lyric, ornament, span-marking, ending, and technique mixin."""

from __future__ import annotations

from .endings import EndingBracketEditingMixin
from .lyrics import LyricEditingMixin
from .marking_common import *
from .marking_shared import MarkingSharedMixin
from .ornaments import OrnamentEditingMixin
from .spanner_markings import SpannerMarkingEditingMixin
from .techniques import TechniqueEditingMixin


class MarkingsMixin(
    LyricEditingMixin,
    OrnamentEditingMixin,
    SpannerMarkingEditingMixin,
    EndingBracketEditingMixin,
    TechniqueEditingMixin,
    MarkingSharedMixin,
):
    """Mixin providing extended notational marking operations."""

    add_lyric = LyricEditingMixin.add_lyric
    remove_lyric = LyricEditingMixin.remove_lyric
    get_lyrics = LyricEditingMixin.get_lyrics
    add_ornament = OrnamentEditingMixin.add_ornament
    remove_ornament = OrnamentEditingMixin.remove_ornament
    add_ottava = SpannerMarkingEditingMixin.add_ottava
    remove_ottava = SpannerMarkingEditingMixin.remove_ottava
    add_glissando = SpannerMarkingEditingMixin.add_glissando
    remove_glissando = SpannerMarkingEditingMixin.remove_glissando
    add_pedal = SpannerMarkingEditingMixin.add_pedal
    remove_pedal = SpannerMarkingEditingMixin.remove_pedal
    add_ending_bracket = EndingBracketEditingMixin.add_ending_bracket
    remove_ending_bracket = EndingBracketEditingMixin.remove_ending_bracket
    add_arpeggio = TechniqueEditingMixin.add_arpeggio
    add_fingering = TechniqueEditingMixin.add_fingering
    remove_fingering = TechniqueEditingMixin.remove_fingering
    remove_arpeggio = TechniqueEditingMixin.remove_arpeggio


__all__ = ["MarkingsMixin"]
