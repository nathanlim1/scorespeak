"""Internal marking-editing implementation slice."""

from __future__ import annotations

from .marking_common import *


class OrnamentEditingMixin:
    """Internal mixin for ScoreSpeak marking operations."""

    def add_ornament(
        self,
        ornament_type: str,
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
        tremolo_marks: int = 1,
    ) -> OperationResult:
        """Add an ornament expression to a note or chord.

        Args:
            ornament_type: One of: trill, inverted trill, turn, inverted turn,
                mordent, inverted mordent, whole step trill, half step trill,
                whole step mordent, half step mordent, or ``tremolo``
                (slash marks on the note).
            tremolo_marks: For ``tremolo`` only — number of slashes (1–4).
        """
        key = ornament_type.strip().lower()
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

        if key == "tremolo":
            if not 1 <= tremolo_marks <= 4:
                raise ValueError(
                    f"tremolo_marks must be 1–4, got {tremolo_marks}."
                )
            orn = m21expressions.Tremolo()
            orn.numberOfMarks = tremolo_marks
        else:
            cls = _ORNAMENT_MAP.get(key)
            if cls is None:
                opts = ", ".join(sorted({*list(_ORNAMENT_MAP.keys()), "tremolo"}))
                raise ValueError(
                    f"Unknown ornament type '{ornament_type}'. Options: {opts}."
                )
            orn = cls()

        el.expressions.append(orn)

        return OperationResult(
            success=True,
            description=(
                f"Added {key} at measure {measure_number}, beat {beat}"
            ),
            details={
                "ornament": key,
                "measure": measure_number,
                "beat": beat,
                "part": part_idx,
                "voice": voice,
                "tremolo_marks": tremolo_marks if key == "tremolo" else None,
            },
        )


    def remove_ornament(
        self,
        ornament_type: str,
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
    ) -> OperationResult:
        """Remove the first matching ornament class at the given note/chord."""
        key = ornament_type.strip().lower()
        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure_number)
        container = self._get_voice_or_measure(measure_obj, voice)
        offset = beat - 1.0
        el = _find_note_at_offset(container, offset)
        if el is None or isinstance(el, m21note.Rest):
            raise ValueError(
                f"No note or chord at beat {beat} in measure {measure_number}."
            )

        if key == "tremolo":
            target_cls = m21expressions.Tremolo
        else:
            target_cls = _ORNAMENT_MAP.get(key)
            if target_cls is None:
                raise ValueError(f"Unknown ornament type '{ornament_type}'.")

        removed = False
        new_expr = []
        for ex in el.expressions:
            if not removed and isinstance(ex, target_cls):
                removed = True
                continue
            new_expr.append(ex)
        if not removed:
            raise ValueError(
                f"No {key} ornament at measure {measure_number}, beat {beat}."
            )
        el.expressions = new_expr

        return OperationResult(
            success=True,
            description=f"Removed {key} from measure {measure_number}, beat {beat}",
            details={
                "ornament": key,
                "measure": measure_number,
                "beat": beat,
                "part": part_idx,
                "voice": voice,
            },
        )
