"""Internal marking-editing implementation slice."""

from __future__ import annotations

from .marking_common import *


class SpannerMarkingEditingMixin:
    """Internal mixin for ScoreSpeak marking operations."""

    def add_ottava(
        self,
        ottava_type: str,
        start_measure: int,
        start_beat: float,
        end_measure: int,
        end_beat: float,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
        rewrite_pitches: bool = False,
        placement: str = "above",
    ) -> OperationResult:
        """Add an octave-shift bracket over an inclusive note range.

        The exported MusicXML ottava is always transposing. With
        ``rewrite_pitches=True``, covered notes/chords are rewritten in the
        opposite octave direction so the rendered staff position is preserved
        while the score stores the easier-to-read written pitches.
        """
        otype = _normalize_ottava_type(ottava_type)
        if placement not in ("above", "below"):
            raise ValueError("placement must be 'above' or 'below'.")

        part_obj, part_idx = self._resolve_part(part)
        start_m = self._resolve_measure(part_obj, start_measure)
        end_m = self._resolve_measure(part_obj, end_measure)
        ts_s = self._get_active_time_signature_obj(part_obj, start_measure)
        ts_e = self._get_active_time_signature_obj(part_obj, end_measure)
        _validate_beat_in_measure(start_beat, ts_s, start_measure)
        _validate_beat_in_measure(end_beat, ts_e, end_measure)
        if end_measure < start_measure or (
            end_measure == start_measure and end_beat < start_beat
        ):
            raise ValueError("Ottava end must be at or after the ottava start.")

        start_c = self._get_voice_or_measure(start_m, voice)
        end_c = self._get_voice_or_measure(end_m, voice)
        n1 = _find_note_at_offset(start_c, start_beat - 1.0)
        n2 = _find_note_at_offset(end_c, end_beat - 1.0)
        if n1 is None or isinstance(n1, m21note.Rest):
            raise ValueError(
                f"No note at ottava start (measure {start_measure}, "
                f"beat {start_beat})."
            )
        if n2 is None or isinstance(n2, m21note.Rest):
            raise ValueError(
                f"No note at ottava end (measure {end_measure}, "
                f"beat {end_beat})."
            )

        spanned_notes: list[m21note.NotRest] = []
        for measure_number in range(start_measure, end_measure + 1):
            if measure_number == start_measure:
                measure_obj = start_m
                first_offset = start_beat - 1.0
            else:
                measure_obj = self._resolve_measure(part_obj, measure_number)
                first_offset = float("-inf")

            if measure_number == end_measure:
                last_offset = end_beat - 1.0
            else:
                last_offset = float("inf")

            container = self._get_voice_or_measure(measure_obj, voice)
            for el in container.getElementsByClass(m21note.NotRest):
                offset = float(container.elementOffset(el))
                if first_offset - 1e-9 <= offset <= last_offset + 1e-9:
                    spanned_notes.append(el)

        notes_rewritten = 0
        if rewrite_pitches:
            rewrite_interval = _ottava_rewrite_interval(otype, adding=True)
            notes_rewritten = _transpose_spanned_pitch_elements(
                spanned_notes,
                rewrite_interval,
            )
            for measure_number in range(start_measure, end_measure + 1):
                self._refresh_measure_accidentals(part_obj, measure_number)

        ott = m21spanner.Ottava(
            *spanned_notes,
            type=otype,
            transposing=True,
            placement=placement,
        )
        part_obj.insert(0, ott)

        return OperationResult(
            success=True,
            description=(
                f"Added {otype} from m.{start_measure} b.{start_beat} "
                f"to m.{end_measure} b.{end_beat}"
            ),
            details={
                "ottava_type": otype,
                "start_measure": start_measure,
                "start_beat": start_beat,
                "end_measure": end_measure,
                "end_beat": end_beat,
                "part": part_idx,
                "voice": voice,
                "rewrite_pitches": rewrite_pitches,
                "notes_rewritten": notes_rewritten,
            },
        )


    def remove_ottava(
        self,
        start_measure: int,
        start_beat: float,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
        rewrite_pitches: bool = False,
    ) -> OperationResult:
        """Remove an Ottava spanner whose first anchor matches the beat."""
        part_obj, part_idx = self._resolve_part(part)
        found = _find_spanner_by_first_anchor(
            part_obj,
            m21spanner.Ottava,
            start_measure,
            start_beat,
            voice,
        )
        if found is None:
            raise ValueError(
                f"No ottava starting at measure {start_measure}, "
                f"beat {start_beat}."
            )
        notes_rewritten = 0
        if rewrite_pitches:
            otype = _normalize_ottava_type(getattr(found, "type", "8va"))
            rewrite_interval = _ottava_rewrite_interval(otype, adding=False)
            spanned_elements = list(found.getSpannedElements())
            notes_rewritten = _transpose_spanned_pitch_elements(
                spanned_elements,
                rewrite_interval,
            )
            for measure_number in _spanned_measure_numbers(spanned_elements):
                self._refresh_measure_accidentals(part_obj, measure_number)
        part_obj.remove(found)
        return OperationResult(
            success=True,
            description=(
                f"Removed ottava starting at measure {start_measure}, "
                f"beat {start_beat}"
            ),
            details={
                "start_measure": start_measure,
                "start_beat": start_beat,
                "part": part_idx,
                "voice": voice,
                "rewrite_pitches": rewrite_pitches,
                "notes_rewritten": notes_rewritten,
            },
        )


    def add_glissando(
        self,
        start_measure: int,
        start_beat: float,
        end_measure: int,
        end_beat: float,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
        line_type: str = "wavy",
        label: Optional[str] = None,
    ) -> OperationResult:
        """Add a glissando between two notes (wavy or solid line)."""
        lt = line_type.strip().lower()
        if lt not in ("wavy", "solid"):
            raise ValueError("line_type must be 'wavy' or 'solid'.")

        part_obj, part_idx = self._resolve_part(part)
        start_m = self._resolve_measure(part_obj, start_measure)
        end_m = self._resolve_measure(part_obj, end_measure)
        ts_s = self._get_active_time_signature_obj(part_obj, start_measure)
        ts_e = self._get_active_time_signature_obj(part_obj, end_measure)
        _validate_beat_in_measure(start_beat, ts_s, start_measure)
        _validate_beat_in_measure(end_beat, ts_e, end_measure)

        start_c = self._get_voice_or_measure(start_m, voice)
        end_c = self._get_voice_or_measure(end_m, voice)
        n1 = _find_note_at_offset(start_c, start_beat - 1.0)
        n2 = _find_note_at_offset(end_c, end_beat - 1.0)
        if n1 is None or isinstance(n1, m21note.Rest):
            raise ValueError(
                f"No note at glissando start (measure {start_measure})."
            )
        if n2 is None or isinstance(n2, m21note.Rest):
            raise ValueError(
                f"No note at glissando end (measure {end_measure})."
            )

        gl = m21spanner.Glissando(n1, n2, lineType=lt, label=label)
        part_obj.insert(0, gl)

        return OperationResult(
            success=True,
            description=(
                f"Added {lt} glissando from m.{start_measure} b.{start_beat} "
                f"to m.{end_measure} b.{end_beat}"
            ),
            details={
                "line_type": lt,
                "label": label,
                "start_measure": start_measure,
                "start_beat": start_beat,
                "end_measure": end_measure,
                "end_beat": end_beat,
                "part": part_idx,
                "voice": voice,
            },
        )


    def remove_glissando(
        self,
        start_measure: int,
        start_beat: float,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
    ) -> OperationResult:
        """Remove a glissando whose first anchor matches the given beat."""
        part_obj, part_idx = self._resolve_part(part)
        found = _find_spanner_by_first_anchor(
            part_obj,
            m21spanner.Glissando,
            start_measure,
            start_beat,
            voice,
        )
        if found is None:
            raise ValueError(
                f"No glissando starting at measure {start_measure}, "
                f"beat {start_beat}."
            )
        part_obj.remove(found)
        return OperationResult(
            success=True,
            description=(
                f"Removed glissando at measure {start_measure}, beat {start_beat}"
            ),
            details={
                "start_measure": start_measure,
                "start_beat": start_beat,
                "part": part_idx,
                "voice": voice,
            },
        )


    def add_pedal(
        self,
        start_measure: int,
        start_beat: float,
        end_measure: int,
        end_beat: float,
        part: Optional[Union[int, str]] = None,
    ) -> OperationResult:
        """Add a damper (sustain) pedal line between two rhythmic positions."""
        part_obj, part_idx = self._resolve_part(part)
        start_m = self._resolve_measure(part_obj, start_measure)
        end_m = self._resolve_measure(part_obj, end_measure)
        ts_s = self._get_active_time_signature_obj(part_obj, start_measure)
        ts_e = self._get_active_time_signature_obj(part_obj, end_measure)
        _validate_beat_in_measure(start_beat, ts_s, start_measure)
        _validate_beat_in_measure(end_beat, ts_e, end_measure)

        start_c = self._get_voice_or_measure(start_m, 1)
        end_c = self._get_voice_or_measure(end_m, 1)
        n1 = _find_note_at_offset(start_c, start_beat - 1.0)
        n2 = _find_note_at_offset(end_c, end_beat - 1.0)
        if n1 is None or isinstance(n1, m21note.Rest):
            raise ValueError(
                f"No note at pedal start (measure {start_measure}, "
                f"beat {start_beat})."
            )
        if n2 is None or isinstance(n2, m21note.Rest):
            raise ValueError(
                f"No note at pedal end (measure {end_measure}, beat {end_beat})."
            )

        ped = m21expressions.PedalMark(n1, n2)
        part_obj.insert(0, ped)

        return OperationResult(
            success=True,
            description=(
                f"Added pedal from m.{start_measure} b.{start_beat} "
                f"to m.{end_measure} b.{end_beat}"
            ),
            details={
                "start_measure": start_measure,
                "start_beat": start_beat,
                "end_measure": end_measure,
                "end_beat": end_beat,
                "part": part_idx,
            },
        )


    def remove_pedal(
        self,
        start_measure: int,
        start_beat: float,
        part: Optional[Union[int, str]] = None,
    ) -> OperationResult:
        """Remove a pedal spanner starting at the given measure and beat."""
        part_obj, part_idx = self._resolve_part(part)
        found = _find_spanner_by_first_anchor(
            part_obj,
            m21expressions.PedalMark,
            start_measure,
            start_beat,
            1,
        )
        if found is None:
            raise ValueError(
                f"No pedal line starting at measure {start_measure}, "
                f"beat {start_beat}."
            )
        part_obj.remove(found)
        return OperationResult(
            success=True,
            description=(
                f"Removed pedal at measure {start_measure}, beat {start_beat}"
            ),
            details={
                "start_measure": start_measure,
                "start_beat": start_beat,
                "part": part_idx,
            },
        )
