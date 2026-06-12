"""Internal marking-editing implementation slice."""

from __future__ import annotations

from .marking_common import *


class TechniqueEditingMixin:
    """Internal mixin for ScoreSpeak marking operations."""

    def add_arpeggio(
        self,
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
    ) -> OperationResult:
        """Add an arpeggiate marking to the chord at the given beat."""
        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure_number)
        ts = self._get_active_time_signature_obj(part_obj, measure_number)
        _validate_beat_in_measure(beat, ts, measure_number)

        container = self._get_voice_or_measure(measure_obj, voice)
        offset = beat - 1.0
        el = _find_note_at_offset(container, offset)
        if el is None or isinstance(el, m21note.Rest):
            raise ValueError(
                f"No note or chord at beat {beat} in measure {measure_number}."
            )
        if not isinstance(el, m21chord.Chord):
            raise ValueError(
                f"Arpeggio requires a chord at measure {measure_number}, "
                f"beat {beat} (found a single note)."
            )

        el.expressions.append(m21expressions.ArpeggioMark())
        return OperationResult(
            success=True,
            description=(
                f"Added arpeggio at measure {measure_number}, beat {beat}"
            ),
            details={
                "measure": measure_number,
                "beat": beat,
                "part": part_idx,
                "voice": voice,
            },
        )


    def add_fingering(
        self,
        finger_number: Union[int, str],
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
    ) -> OperationResult:
        """Add a fingering digit (or letter) above/below a note or chord.

        Args:
            finger_number: Typically 0–5 or a string such as ``"T"`` (thumb).
        """
        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure_number)
        ts = self._get_active_time_signature_obj(part_obj, measure_number)
        _validate_beat_in_measure(beat, ts, measure_number)

        container = self._get_voice_or_measure(measure_obj, voice)
        offset = beat - 1.0
        el = _find_note_at_offset(container, offset)
        if el is None or isinstance(el, m21note.Rest):
            raise ValueError(
                f"No note or chord at beat {beat} in measure {measure_number}."
            )

        fn: Union[int, str]
        if isinstance(finger_number, str):
            s = finger_number.strip()
            if s.isdigit():
                fn = int(s)
            else:
                fn = s
        else:
            fn = int(finger_number)

        fg = m21articulations.Fingering(fn)
        el.articulations.append(fg)

        return OperationResult(
            success=True,
            description=(
                f"Added fingering {fn} at measure {measure_number}, beat {beat}"
            ),
            details={
                "finger": fn,
                "measure": measure_number,
                "beat": beat,
                "part": part_idx,
                "voice": voice,
            },
        )


    def remove_fingering(
        self,
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
    ) -> OperationResult:
        """Remove the first fingering articulation at the given note."""
        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure_number)
        container = self._get_voice_or_measure(measure_obj, voice)
        offset = beat - 1.0
        el = _find_note_at_offset(container, offset)
        if el is None or isinstance(el, m21note.Rest):
            raise ValueError(
                f"No note or chord at beat {beat} in measure {measure_number}."
            )
        new_art: list = []
        removed = False
        for a in el.articulations:
            if not removed and isinstance(a, m21articulations.Fingering):
                removed = True
                continue
            new_art.append(a)
        if not removed:
            raise ValueError(
                f"No fingering at measure {measure_number}, beat {beat}."
            )
        el.articulations = new_art
        return OperationResult(
            success=True,
            description=(
                f"Removed fingering at measure {measure_number}, beat {beat}"
            ),
            details={
                "measure": measure_number,
                "beat": beat,
                "part": part_idx,
                "voice": voice,
            },
        )


    def remove_arpeggio(
        self,
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
    ) -> OperationResult:
        """Remove an ArpeggioMark from the chord at the beat."""
        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure_number)
        container = self._get_voice_or_measure(measure_obj, voice)
        offset = beat - 1.0
        el = _find_note_at_offset(container, offset)
        if el is None or not isinstance(el, m21chord.Chord):
            raise ValueError(
                f"No chord at measure {measure_number}, beat {beat}."
            )
        new_ex = [
            ex
            for ex in el.expressions
            if not isinstance(ex, m21expressions.ArpeggioMark)
        ]
        if len(new_ex) == len(el.expressions):
            raise ValueError(
                f"No arpeggio at measure {measure_number}, beat {beat}."
            )
        el.expressions = new_ex
        return OperationResult(
            success=True,
            description=(
                f"Removed arpeggio at measure {measure_number}, beat {beat}"
            ),
            details={
                "measure": measure_number,
                "beat": beat,
                "part": part_idx,
                "voice": voice,
            },
        )
