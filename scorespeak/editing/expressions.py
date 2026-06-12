"""Composite dynamics, articulation, slur, text, tempo, and chord-symbol mixin."""

from __future__ import annotations

from .articulations import ArticulationEditingMixin
from .dynamics import DynamicsEditingMixin
from .expression_common import *
from .expression_shared import ExpressionSharedMixin
from .text_expressions import TextExpressionEditingMixin


class ExpressionsMixin(
    DynamicsEditingMixin,
    ArticulationEditingMixin,
    TextExpressionEditingMixin,
    ExpressionSharedMixin,
):
    """Mixin providing expressive marking operations."""

    add_dynamic = DynamicsEditingMixin.add_dynamic
    remove_dynamic = DynamicsEditingMixin.remove_dynamic
    add_hairpin = DynamicsEditingMixin.add_hairpin
    remove_hairpin = DynamicsEditingMixin.remove_hairpin
    add_articulation = ArticulationEditingMixin.add_articulation
    remove_articulation = ArticulationEditingMixin.remove_articulation
    add_slur = ArticulationEditingMixin.add_slur
    remove_slur = ArticulationEditingMixin.remove_slur
    add_text_expression = TextExpressionEditingMixin.add_text_expression
    remove_text_expression = TextExpressionEditingMixin.remove_text_expression
    set_tempo = TextExpressionEditingMixin.set_tempo
    add_rehearsal_mark = TextExpressionEditingMixin.add_rehearsal_mark
    remove_rehearsal_mark = TextExpressionEditingMixin.remove_rehearsal_mark
    add_chord_symbol = TextExpressionEditingMixin.add_chord_symbol
    remove_chord_symbol = TextExpressionEditingMixin.remove_chord_symbol
    get_chord_symbols = TextExpressionEditingMixin.get_chord_symbols


__all__ = ["ExpressionsMixin"]
