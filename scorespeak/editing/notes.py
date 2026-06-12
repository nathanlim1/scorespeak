"""Composite note, rest, chord, tie, tuplet, and grace-note editing mixin."""

from __future__ import annotations

from .chords import ChordEditingMixin
from .grace_notes import GraceNoteEditingMixin
from .note_events import NoteEventsMixin
from .rests import RestEditingMixin
from .shared import NotesSharedMixin
from .ties import TieEditingMixin
from .tuplets import TupletEditingMixin


class NotesMixin(
    NoteEventsMixin,
    RestEditingMixin,
    ChordEditingMixin,
    TieEditingMixin,
    TupletEditingMixin,
    GraceNoteEditingMixin,
    NotesSharedMixin,
):
    """Mixin providing note, rest, chord, tie, tuplet, and grace note operations."""

    add_notes = NoteEventsMixin.add_notes
    remove_notes = NoteEventsMixin.remove_notes
    replace_note = NoteEventsMixin.replace_note
    get_notes = NoteEventsMixin.get_notes
    add_rest = RestEditingMixin.add_rest
    fill_measure_gaps = RestEditingMixin.fill_measure_gaps
    reshape_rests = RestEditingMixin.reshape_rests
    remove_rests = RestEditingMixin.remove_rests
    add_chord = ChordEditingMixin.add_chord
    add_chord_tones = ChordEditingMixin.add_chord_tones
    add_tie = TieEditingMixin.add_tie
    remove_tie = TieEditingMixin.remove_tie
    add_tuplet = TupletEditingMixin.add_tuplet
    remove_tuplet = TupletEditingMixin.remove_tuplet
    add_grace_note = GraceNoteEditingMixin.add_grace_note
    remove_grace_note = GraceNoteEditingMixin.remove_grace_note


__all__ = ["NotesMixin"]
