"""Internal marking-editing implementation slice."""

from __future__ import annotations

from .marking_common import *


class LyricEditingMixin:
    """Internal mixin for ScoreSpeak marking operations."""

    def add_lyric(
        self,
        text: str,
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
        lyric_number: int = 1,
        syllabic: Optional[str] = None,
        identifier: Optional[str] = None,
    ) -> OperationResult:
        """Attach a lyric syllable to the note or chord at a beat.

        Args:
            text: Lyric text (may be empty only if you intend a spacer;
                normally non-empty).
            measure_number: 1-based measure number.
            beat: 1-based beat position.
            part: Part index or name (default first part).
            voice: 1-based voice number.
            lyric_number: Verse / lyric line number (MusicXML ``number``).
            syllabic: ``single``, ``begin``, ``middle``, ``end``, or
                ``composite``; default lets music21 choose.
            identifier: Optional MusicXML lyric name / label.

        Returns:
            OperationResult describing the lyric attachment.
        """
        if lyric_number < 1:
            raise ValueError(f"lyric_number must be >= 1, got {lyric_number}.")
        if syllabic is not None and syllabic not in _VALID_SYLLABICS:
            raise ValueError(
                f"Invalid syllabic '{syllabic}'. "
                f"Use single, begin, middle, end, or composite."
            )

        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure_number)
        ts = self._get_active_time_signature_obj(part_obj, measure_number)
        _validate_beat_in_measure(beat, ts, measure_number)

        container = self._get_voice_or_measure(measure_obj, voice)
        offset = beat - 1.0
        el = _find_note_at_offset(container, offset)
        if el is None:
            raise ValueError(
                f"No note or chord at beat {beat} in measure {measure_number}."
            )
        if isinstance(el, m21note.Rest):
            raise ValueError(
                f"Cannot attach a lyric to a rest at beat {beat} "
                f"in measure {measure_number}."
            )

        kept = [ly for ly in el.lyrics if ly.number != lyric_number]
        lyric_obj = m21note.Lyric(
            text,
            number=lyric_number,
            syllabic=syllabic,
            identifier=identifier,
        )
        kept.append(lyric_obj)
        el.lyrics = kept

        return OperationResult(
            success=True,
            description=(
                f"Added lyric (verse {lyric_number}) '{text}' at "
                f"measure {measure_number}, beat {beat}"
            ),
            details={
                "text": text,
                "measure": measure_number,
                "beat": beat,
                "part": part_idx,
                "voice": voice,
                "lyric_number": lyric_number,
                "syllabic": syllabic,
            },
        )


    def remove_lyric(
        self,
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
        lyric_number: int = 1,
    ) -> OperationResult:
        """Remove a lyric line from the note or chord at the given beat."""
        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure_number)
        container = self._get_voice_or_measure(measure_obj, voice)
        offset = beat - 1.0
        el = _find_note_at_offset(container, offset)
        if el is None or isinstance(el, m21note.Rest):
            raise ValueError(
                f"No note or chord at beat {beat} in measure {measure_number}."
            )

        kept = [ly for ly in el.lyrics if ly.number != lyric_number]
        if len(kept) == len(el.lyrics):
            raise ValueError(
                f"No lyric number {lyric_number} at measure {measure_number}, "
                f"beat {beat}."
            )
        el.lyrics = kept

        return OperationResult(
            success=True,
            description=(
                f"Removed lyric {lyric_number} from measure {measure_number}, "
                f"beat {beat}"
            ),
            details={
                "measure": measure_number,
                "beat": beat,
                "part": part_idx,
                "voice": voice,
                "lyric_number": lyric_number,
            },
        )


    def get_lyrics(
        self,
        measure_number: Optional[int] = None,
        part: Optional[Union[int, str]] = None,
        voice: Optional[int] = None,
    ) -> list[LyricInfo]:
        """Return all lyrics in the given scope (defaults: entire score)."""
        if voice is not None:
            voice = validate_voice_number(voice)

        targets = self._resolve_parts_or_all(part)
        out: list[LyricInfo] = []

        for part_obj, pidx in targets:
            measures = sorted(
                part_obj.getElementsByClass(m21stream.Measure),
                key=lambda m: m.number,
            )
            for m_obj in measures:
                if measure_number is not None and m_obj.number != measure_number:
                    continue
                for vnum, stream_like in _iter_voice_streams(m_obj):
                    if voice is not None and vnum != voice:
                        continue
                    for el in stream_like.getElementsByClass(m21note.GeneralNote):
                        if isinstance(el, m21note.Rest):
                            if (
                                hasattr(el.style, "hideObjectOnPrint")
                                and el.style.hideObjectOnPrint
                            ):
                                continue
                        off = stream_like.elementOffset(el)
                        beat = off + 1.0
                        for ly in el.lyrics:
                            syl = ly.syllabic
                            if syl is not None:
                                syl = str(syl)
                            out.append(
                                LyricInfo(
                                    text=ly.text or "",
                                    measure_number=m_obj.number,
                                    beat=beat,
                                    part_index=pidx,
                                    voice=vnum,
                                    lyric_number=ly.number,
                                    syllabic=syl,
                                    pitch_or_chord=_pitch_label(el),
                                )
                            )
        return out
