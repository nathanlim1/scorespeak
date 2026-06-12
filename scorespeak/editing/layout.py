"""
Layout, metadata, and transposition operations for ScoreSpeak.

Provides methods for score metadata (title, subtitle, composer),
layout breaks (system/page), and transposition (by interval,
to concert pitch, to written pitch).
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from music21 import chord as m21chord
from music21 import interval as m21interval
from music21 import key as m21key
from music21 import layout as m21layout
from music21 import metadata as m21metadata
from music21 import note as m21note
from music21 import stream as m21stream

from ..music.pitch_space import (
    concert_key_signature_for_stored_key,
    copy_key_signature,
    part_stores_sounding_pitch,
    part_transposition_interval,
    set_part_stores_sounding_pitch,
    stored_key_signature_for_concert_key,
)
from ..types import OperationResult


class LayoutMixin:
    """Mixin providing layout, metadata, and transposition operations."""

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def set_title(self, title: str) -> OperationResult:
        """Set or change the score title.

        Args:
            title: The new title string.

        Returns:
            OperationResult confirming the change.
        """
        self._ensure_metadata()
        old_title = self._score.metadata.title or ""
        self._score.metadata.title = title
        return OperationResult(
            success=True,
            description=f"Score title set to '{title}'.",
            details={"old_title": old_title, "new_title": title},
        )

    def set_subtitle(self, subtitle: str) -> OperationResult:
        """Set or change the score subtitle.

        Uses movementName on the score's Metadata object, which is the
        standard field renderers use for subtitles.

        Args:
            subtitle: The new subtitle string.

        Returns:
            OperationResult confirming the change.
        """
        self._ensure_metadata()
        old_subtitle = self._score.metadata.movementName or ""
        self._score.metadata.movementName = subtitle
        return OperationResult(
            success=True,
            description=f"Score subtitle set to '{subtitle}'.",
            details={"old_subtitle": old_subtitle, "new_subtitle": subtitle},
        )

    def set_composer(self, composer: str) -> OperationResult:
        """Set or change the score composer.

        Args:
            composer: The new composer string.

        Returns:
            OperationResult confirming the change.
        """
        self._ensure_metadata()
        old_composer = self._score.metadata.composer or ""
        self._score.metadata.composer = composer
        return OperationResult(
            success=True,
            description=f"Composer set to '{composer}'.",
            details={"old_composer": old_composer, "new_composer": composer},
        )

    def get_metadata(self) -> dict:
        """Return all score metadata as a dictionary.

        Returns:
            Dictionary with keys: title, subtitle, composer.
        """
        self._ensure_metadata()
        md = self._score.metadata
        return {
            "title": md.title or "",
            "subtitle": md.movementName or "",
            "composer": md.composer or "",
        }

    def _ensure_metadata(self) -> None:
        """Create metadata on the score if it doesn't already exist."""
        if self._score.metadata is None:
            self._score.metadata = m21metadata.Metadata()

    # ------------------------------------------------------------------
    # System / Page breaks
    # ------------------------------------------------------------------

    def add_system_break(
        self,
        measure_number: int,
    ) -> OperationResult:
        """Make a specific measure start a new system.

        A system break tells the renderer that this measure is the first
        measure on a new system (line of music).

        Args:
            measure_number: 1-based measure number that should start on a new
                system.

        Returns:
            OperationResult confirming the addition.
        """
        targets = self._resolve_parts_or_all()
        affected_parts = []

        for part_obj, part_idx in targets:
            measure = self._resolve_measure(part_obj, measure_number)

            existing = list(measure.getElementsByClass(m21layout.SystemLayout))
            already_has_break = any(sl.isNew for sl in existing)
            if already_has_break:
                affected_parts.append(part_idx)
                continue

            sl = m21layout.SystemLayout(isNew=True)
            measure.insert(0, sl)
            affected_parts.append(part_idx)

        return OperationResult(
            success=True,
            description=(
                f"System break added: measure {measure_number} now starts "
                "a new system."
            ),
            details={
                "measure_number": measure_number,
                "parts": affected_parts,
            },
        )

    def remove_system_break(
        self,
        measure_number: int,
    ) -> OperationResult:
        """Remove the system break that makes a measure start a new system.

        Args:
            measure_number: 1-based measure number that starts on a new
                system.

        Returns:
            OperationResult describing what was removed.
        """
        targets = self._resolve_parts_or_all()
        removed_count = 0

        for part_obj, part_idx in targets:
            measure = self._resolve_measure(part_obj, measure_number)
            system_layouts = list(
                measure.getElementsByClass(m21layout.SystemLayout)
            )
            for sl in system_layouts:
                if sl.isNew:
                    measure.remove(sl)
                    removed_count += 1

        if removed_count == 0:
            return OperationResult(
                success=False,
                description=(
                    f"No system break found at measure {measure_number} "
                    f"— nothing to remove."
                ),
                details={"measure_number": measure_number},
            )

        return OperationResult(
            success=True,
            description=(
                f"Removed system break(s) that made measure {measure_number} "
                "start a new system."
            ),
            details={
                "measure_number": measure_number,
                "removed_count": removed_count,
            },
        )

    def add_page_break(
        self,
        measure_number: int,
    ) -> OperationResult:
        """Make a specific measure start a new page.

        A page break tells the renderer that this measure is the first
        measure on a new page.

        Args:
            measure_number: 1-based measure number that should start on a new
                page.

        Returns:
            OperationResult confirming the addition.
        """
        targets = self._resolve_parts_or_all()
        affected_parts = []

        for part_obj, part_idx in targets:
            measure = self._resolve_measure(part_obj, measure_number)

            existing = list(measure.getElementsByClass(m21layout.PageLayout))
            already_has_break = any(pl.isNew for pl in existing)
            if already_has_break:
                affected_parts.append(part_idx)
                continue

            pl = m21layout.PageLayout(isNew=True)
            measure.insert(0, pl)
            affected_parts.append(part_idx)

        return OperationResult(
            success=True,
            description=(
                f"Page break added: measure {measure_number} now starts "
                "a new page."
            ),
            details={
                "measure_number": measure_number,
                "parts": affected_parts,
            },
        )

    def remove_page_break(
        self,
        measure_number: int,
    ) -> OperationResult:
        """Remove the page break that makes a measure start a new page.

        Args:
            measure_number: 1-based measure number that starts on a new page.

        Returns:
            OperationResult describing what was removed.
        """
        targets = self._resolve_parts_or_all()
        removed_count = 0

        for part_obj, part_idx in targets:
            measure = self._resolve_measure(part_obj, measure_number)
            page_layouts = list(
                measure.getElementsByClass(m21layout.PageLayout)
            )
            for pl in page_layouts:
                if pl.isNew:
                    measure.remove(pl)
                    removed_count += 1

        if removed_count == 0:
            return OperationResult(
                success=False,
                description=(
                    f"No page break found at measure {measure_number} "
                    f"— nothing to remove."
                ),
                details={"measure_number": measure_number},
            )

        return OperationResult(
            success=True,
            description=(
                f"Removed page break(s) that made measure {measure_number} "
                "start a new page."
            ),
            details={
                "measure_number": measure_number,
                "removed_count": removed_count,
            },
        )

    # ------------------------------------------------------------------
    # Transposition
    # ------------------------------------------------------------------

    def transpose(
        self,
        interval: Union[str, int],
        part: Optional[Union[int, str]] = None,
        start_measure: Optional[int] = None,
        end_measure: Optional[int] = None,
    ) -> OperationResult:
        """Transpose part(s) by a given interval.

        Args:
            interval: Interval string (e.g. "P5", "-m3", "M2") or integer
                semitones (e.g. 7 for a perfect fifth up, -3 for a minor
                third down).
            part: Part index (0-based), name, or None for all parts.
            start_measure: First measure to transpose (1-based, inclusive).
                None means start from the beginning.
            end_measure: Last measure to transpose (1-based, inclusive).
                None means through the end.

        Returns:
            OperationResult confirming the transposition.
        """
        interval_obj = _parse_interval(interval)

        targets = self._resolve_parts_or_all(part)
        affected_parts = []
        total_notes = 0

        for part_obj, part_idx in targets:
            max_measure = self._get_measure_count(part_obj)
            first = start_measure if start_measure is not None else 1
            last = end_measure if end_measure is not None else max_measure

            if first < 1 or first > max_measure:
                raise ValueError(
                    f"start_measure {first} is out of range for part {part_idx} "
                    f"(valid: 1–{max_measure})."
                )
            if last < first or last > max_measure:
                raise ValueError(
                    f"end_measure {last} is out of range for part {part_idx} "
                    f"(valid: {first}–{max_measure})."
                )

            notes_transposed = 0
            for m_num in range(first, last + 1):
                measure = self._resolve_measure(part_obj, m_num)
                notes_transposed += _transpose_elements_in_stream(
                    measure, interval_obj
                )
                self._refresh_measure_accidentals(part_obj, m_num)

            affected_parts.append(part_idx)
            total_notes += notes_transposed

        interval_name = str(interval_obj)
        range_desc = ""
        if start_measure is not None or end_measure is not None:
            range_desc = (
                f" (measures {start_measure or 1}–"
                f"{end_measure or 'end'})"
            )

        return OperationResult(
            success=True,
            description=(
                f"Transposed {total_notes} note(s) by {interval_name}"
                f"{range_desc}."
            ),
            details={
                "interval": interval_name,
                "semitones": interval_obj.semitones,
                "parts": affected_parts,
                "notes_transposed": total_notes,
                "start_measure": start_measure,
                "end_measure": end_measure,
            },
        )

    def transpose_to_concert_pitch(
        self,
        part: Optional[Union[int, str]] = None,
    ) -> OperationResult:
        """Transpose part(s) from written pitch to concert (sounding) pitch.

        For transposing instruments (e.g. Bb clarinet, F horn), this converts
        their written notes to how they actually sound. Non-transposing
        instruments are left unchanged.

        Args:
            part: Part index (0-based), name, or None for all parts.

        Returns:
            OperationResult describing what was transposed.
        """
        targets = self._resolve_parts_or_all(part)
        transposed_parts = []

        for part_obj, part_idx in targets:
            transposition = part_transposition_interval(part_obj)
            if transposition is None:
                continue
            if part_stores_sounding_pitch(part_obj):
                continue

            notes_transposed = 0
            for measure in part_obj.getElementsByClass(m21stream.Measure):
                notes_transposed += _transpose_elements_in_stream(
                    measure,
                    transposition,
                )
                if measure.number is not None:
                    self._refresh_measure_accidentals(part_obj, measure.number)
            key_signatures_changed = self._convert_part_key_signatures(
                part_obj,
                lambda key_signature: concert_key_signature_for_stored_key(
                    part_obj,
                    key_signature,
                ),
            )
            set_part_stores_sounding_pitch(part_obj, True)
            self._refresh_part_accidentals(part_obj)

            transposed_parts.append({
                "part_index": part_idx,
                "part_name": part_obj.partName or f"Part {part_idx}",
                "transposition": str(transposition),
                "notes_transposed": notes_transposed,
                "key_signatures_changed": key_signatures_changed,
                "stored_pitch": "sounding",
            })

        if not transposed_parts:
            return OperationResult(
                success=True,
                description=(
                    "No written-pitch transposing parts found — "
                    "score is already at concert pitch."
                ),
                details={"transposed_parts": []},
            )

        names = [p["part_name"] for p in transposed_parts]
        return OperationResult(
            success=True,
            description=(
                f"Transposed {', '.join(names)} to concert pitch."
            ),
            details={"transposed_parts": transposed_parts},
        )

    def transpose_to_written_pitch(
        self,
        part: Optional[Union[int, str]] = None,
    ) -> OperationResult:
        """Transpose part(s) from concert (sounding) pitch to written pitch.

        Reverses the instrument's transposition so that the notation shows
        what the player reads. Non-transposing instruments are left unchanged.

        Args:
            part: Part index (0-based), name, or None for all parts.

        Returns:
            OperationResult describing what was transposed.
        """
        targets = self._resolve_parts_or_all(part)
        transposed_parts = []

        for part_obj, part_idx in targets:
            transposition = part_transposition_interval(part_obj)
            if transposition is None:
                continue
            if not part_stores_sounding_pitch(part_obj):
                continue

            reverse_transposition = transposition.reverse()
            notes_transposed = 0
            for measure in part_obj.getElementsByClass(m21stream.Measure):
                notes_transposed += _transpose_elements_in_stream(
                    measure,
                    reverse_transposition,
                )
                if measure.number is not None:
                    self._refresh_measure_accidentals(part_obj, measure.number)
            set_part_stores_sounding_pitch(part_obj, False)
            key_signatures_changed = self._convert_part_key_signatures(
                part_obj,
                lambda key_signature: stored_key_signature_for_concert_key(
                    part_obj,
                    key_signature,
                ),
            )
            self._refresh_part_accidentals(part_obj)

            transposed_parts.append({
                "part_index": part_idx,
                "part_name": part_obj.partName or f"Part {part_idx}",
                "transposition": str(reverse_transposition),
                "notes_transposed": notes_transposed,
                "key_signatures_changed": key_signatures_changed,
                "stored_pitch": "written",
            })

        if not transposed_parts:
            return OperationResult(
                success=True,
                description=(
                    "No sounding-pitch transposing parts found — "
                    "score is already at written pitch."
                ),
                details={"transposed_parts": []},
            )

        names = [p["part_name"] for p in transposed_parts]
        return OperationResult(
            success=True,
            description=(
                f"Transposed {', '.join(names)} to written pitch."
            ),
            details={"transposed_parts": transposed_parts},
        )

    def _refresh_part_accidentals(self, part_obj: m21stream.Part) -> None:
        """Refresh accidentals in every numbered measure of ``part_obj``."""
        for measure in part_obj.getElementsByClass(m21stream.Measure):
            if measure.number is not None:
                self._refresh_measure_accidentals(part_obj, measure.number)

    def _convert_part_key_signatures(
        self,
        part_obj: m21stream.Part,
        converter: Callable[[m21key.KeySignature], m21key.KeySignature],
    ) -> int:
        """Replace explicit key signatures in ``part_obj`` with converted ones."""
        changed = 0
        for measure in part_obj.getElementsByClass(m21stream.Measure):
            for key_signature in self._local_key_signatures(measure):
                offset = float(measure.elementOffset(key_signature))
                replacement = converter(key_signature)
                if self._key_signatures_equal(key_signature, replacement):
                    if key_signature is not replacement:
                        replacement = copy_key_signature(key_signature)
                    else:
                        continue
                measure.remove(key_signature)
                measure.insert(offset, replacement)
                changed += 1
        return changed


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _parse_interval(raw: Union[str, int]) -> m21interval.Interval:
    """Parse a flexible interval specification into a music21 Interval.

    Accepts:
        - String: "P5", "-m3", "M2", "A4", "d5"
        - Integer: semitone count (positive = up, negative = down)

    Raises:
        ValueError: If the interval cannot be parsed.
    """
    if isinstance(raw, int):
        try:
            return m21interval.Interval(raw)
        except Exception as exc:
            raise ValueError(
                f"Cannot create interval from {raw} semitones: {exc}"
            ) from exc

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError(
                "Interval string cannot be empty. "
                "Expected something like 'P5', '-m3', 'M2', or an integer."
            )
        try:
            return m21interval.Interval(text)
        except Exception as exc:
            raise ValueError(
                f"Cannot parse interval '{raw}'. "
                f"Expected a standard interval abbreviation like 'P5' "
                f"(perfect fifth), '-m3' (minor third down), 'M2' "
                f"(major second up), or an integer semitone count. "
                f"Details: {exc}"
            ) from exc

    raise ValueError(
        f"Interval must be a string or integer, got {type(raw).__name__}."
    )


def _transpose_elements_in_stream(
    stream: m21stream.Stream,
    interval_obj: m21interval.Interval,
) -> int:
    """Transpose all notes and chords in a stream by the given interval.

    Recurses into voices. Returns the number of elements transposed.
    """
    count = 0

    for voice in stream.getElementsByClass(m21stream.Voice):
        count += _transpose_elements_in_stream(voice, interval_obj)

    for n in stream.getElementsByClass(m21note.Note):
        n.transpose(interval_obj, inPlace=True)
        count += 1

    for ch in stream.getElementsByClass(m21chord.Chord):
        ch.transpose(interval_obj, inPlace=True)
        count += 1

    return count
