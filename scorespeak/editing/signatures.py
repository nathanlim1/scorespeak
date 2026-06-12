"""Composite time/key signature, clef, barline, and navigation mixin."""

from __future__ import annotations

from .barlines import BarlineEditingMixin
from .key_signatures import KeySignatureEditingMixin
from .navigation import NavigationMarkEditingMixin
from .signature_common import *
from .signature_shared import SignatureSharedMixin
from .time_signatures import TimeSignatureEditingMixin


class SignaturesMixin(
    TimeSignatureEditingMixin,
    KeySignatureEditingMixin,
    BarlineEditingMixin,
    NavigationMarkEditingMixin,
    SignatureSharedMixin,
):
    """Mixin providing signature, clef, barline, repeat, and navigation operations."""

    set_time_signature = TimeSignatureEditingMixin.set_time_signature
    set_key_signature = KeySignatureEditingMixin.set_key_signature
    set_clef = BarlineEditingMixin.set_clef
    set_barline = BarlineEditingMixin.set_barline
    add_repeat = BarlineEditingMixin.add_repeat
    remove_repeat = BarlineEditingMixin.remove_repeat
    set_pickup_measure = BarlineEditingMixin.set_pickup_measure
    add_coda = NavigationMarkEditingMixin.add_coda
    add_segno = NavigationMarkEditingMixin.add_segno
    add_to_coda = NavigationMarkEditingMixin.add_to_coda
    add_fine = NavigationMarkEditingMixin.add_fine
    add_da_capo = NavigationMarkEditingMixin.add_da_capo
    add_dal_segno = NavigationMarkEditingMixin.add_dal_segno
    remove_navigation_mark = NavigationMarkEditingMixin.remove_navigation_mark


__all__ = ["SignaturesMixin"]
