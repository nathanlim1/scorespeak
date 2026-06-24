"""Composite bar retrieval mixin for ScoreSpeak."""

from __future__ import annotations

from .marking_payloads import BarMarkingPayloadMixin
from .matching import BarSearchMatchingMixin
from .notation import BarNotationMixin
from .parsing import BarQueryParsingMixin
from .payloads import BarPayloadAssemblyMixin
from .result import BarRetrievalResultMixin


class BarRetrievalMixin(
    BarRetrievalResultMixin,
    BarSearchMatchingMixin,
    BarQueryParsingMixin,
    BarPayloadAssemblyMixin,
    BarMarkingPayloadMixin,
    BarNotationMixin,
):
    """Mixin providing compact bar-first retrieval over score contents."""

    search_score = BarRetrievalResultMixin.search_score


__all__ = ["BarRetrievalMixin"]
