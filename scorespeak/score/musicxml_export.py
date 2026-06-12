"""MusicXML export helpers for ScoreSpeak."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Callable, Iterator

from music21 import instrument as m21instrument
from music21 import stream as m21stream
from music21.musicxml import m21ToXml

from ..music.pitch_space import part_stores_sounding_pitch, part_transposition_interval


_MIDI_CHANNEL_WARNING = "we are out of midi channels! help!"
_WARN_FILTER_LOCK = RLock()


class _MusicXmlWarningFilter:
    """Callable wrapper that filters one known noisy music21 warning."""

    def __init__(self, original_warn: Callable[..., None]) -> None:
        """Store the music21 warning callable to forward to."""
        self._original_warn = original_warn

    def __call__(self, msg: object, header: object | None = None) -> None:
        """Forward all music21 warnings except MIDI-channel exhaustion noise."""
        if _is_midi_channel_warning(msg):
            return
        self._original_warn(msg, header=header)


def write_musicxml_file(
    score: m21stream.Stream,
    path: str | Path,
    *,
    make_notation: bool = True,
) -> None:
    """Write ``score`` to MusicXML while suppressing known MIDI-channel noise."""
    path_str = str(path)
    with (
        _suppress_music21_midi_channel_warning(),
        _suppress_transpose_for_sounding_pitch_parts(score),
    ):
        if make_notation:
            score.write("musicxml", fp=path_str)
        else:
            score.write("musicxml", fp=path_str, makeNotation=False)


@contextmanager
def _suppress_music21_midi_channel_warning() -> Iterator[None]:
    """Temporarily filter music21's harmless MIDI-channel export warning."""
    with _WARN_FILTER_LOCK:
        original_warn = m21ToXml.environLocal.warn
        m21ToXml.environLocal.warn = _MusicXmlWarningFilter(original_warn)
        try:
            yield
        finally:
            m21ToXml.environLocal.warn = original_warn


@contextmanager
def _suppress_transpose_for_sounding_pitch_parts(
    score: m21stream.Stream,
) -> Iterator[None]:
    """Prevent music21 from re-exporting sounding-pitch parts as written."""
    changed_instruments: list[
        tuple[m21instrument.Instrument, object]
    ] = []
    for part in score.parts:
        if not isinstance(part, m21stream.Part):
            continue
        if not part_stores_sounding_pitch(part):
            continue
        if part_transposition_interval(part) is None:
            continue
        instruments = list(part.getElementsByClass(m21instrument.Instrument))
        if not instruments:
            instrument = part.getInstrument(returnDefault=False)
            if isinstance(instrument, m21instrument.Instrument):
                instruments = [instrument]

        for instrument in instruments:
            if getattr(instrument, "transposition", None) is None:
                continue
            changed_instruments.append((instrument, instrument.transposition))
            instrument.transposition = None

    try:
        yield
    finally:
        for instrument, transposition in changed_instruments:
            instrument.transposition = transposition


def _is_midi_channel_warning(msg: object) -> bool:
    """Return whether ``msg`` is music21's MIDI-channel exhaustion warning."""
    if isinstance(msg, str):
        return _MIDI_CHANNEL_WARNING in msg
    if isinstance(msg, Exception):
        return _MIDI_CHANNEL_WARNING in str(msg)
    if isinstance(msg, (list, tuple)):
        return any(_is_midi_channel_warning(item) for item in msg)
    return False
