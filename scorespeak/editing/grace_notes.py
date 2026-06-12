"""
Note, rest, and chord operations for ScoreSpeak.

Provides methods for adding, removing, replacing, and querying notes,
rests, chords, ties, tuplets, and grace notes in a score.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional, Union

from music21 import beam as m21beam
from music21 import chord as m21chord
from music21 import clef as m21clef
from music21 import duration as m21duration
from music21 import instrument as m21instrument
from music21 import key as m21key
from music21 import meter as m21meter
from music21 import note as m21note
from music21 import pitch as m21pitch
from music21 import spanner as m21spanner
from music21 import stream as m21stream
from music21 import tie as m21tie

from ..types import (
    DurationInput,
    NoteInfo,
    OperationResult,
    PitchInput,
    TupletInfo,
)
from ..music.validation import (
    normalize_duration,
    normalize_pitch,
    validate_voice_number,
    validate_pitch_in_range,
)


_RHYTHM_EPSILON = 1e-9
_ADD_NOTES_REQUIRED_FIELDS = frozenset({"pitch", "beat", "duration", "dots"})
_ADD_NOTES_FORBIDDEN_FIELDS = frozenset({"measure", "part", "voice"})
_ADD_NOTES_ALLOWED_FIELDS = _ADD_NOTES_REQUIRED_FIELDS
_REMOVE_NOTES_REQUIRED_FIELDS = frozenset({"beat"})
_REMOVE_NOTES_FORBIDDEN_FIELDS = frozenset({"measure", "part", "voice"})
_REMOVE_NOTES_ALLOWED_FIELDS = frozenset({"beat", "pitch"})
_STANDARD_DURATION_BASES = (
    4.0,
    2.0,
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
)
_STANDARD_DURATION_VALUES = frozenset(
    round(base * sum(0.5 ** dot for dot in range(dots + 1)), 9)
    for base in _STANDARD_DURATION_BASES
    for dots in range(4)
)
_BEAMABLE_GRACE_DURATION_TYPES = frozenset(
    {
        "eighth",
        "16th",
        "32nd",
        "64th",
        "128th",
        "256th",
        "512th",
        "1024th",
        "2048th",
    }
)


class GraceNoteEditingMixin:
    """Internal mixin for ScoreSpeak note-editing operations."""

    def add_grace_note(
        self,
        pitch: PitchInput,
        duration: DurationInput = "eighth",
        measure: Optional[int] = None,
        beat: Optional[float] = None,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
        slash: bool = True,
        slur_to_principal: bool = False,
    ) -> OperationResult:
        """Add one grace note in one voice at a target beat. Inspect multi-voice
        bars first so the grace note attaches to the intended layer.

        Args:
            pitch: Grace note pitch.
            duration: Written grace-note duration. The sounding duration remains
                zero, but the exported notation type uses this value, such as
                ``"eighth"`` or ``"16th"``.
            measure: 1-based measure number.  *None* → last measure.
            beat: 1-based beat position for the grace note. *None* → beat 1.
            part: Part identifier.
            voice: 1-based rhythmic timeline inside this part. Defaults to
                voice 1. Use another voice only when the grace note belongs to
                a different simultaneous rhythmic line.
            slash: If *True* (default), creates an acciaccatura (slashed).
                If *False*, creates an appoggiatura.
            slur_to_principal: If *True*, add or extend a slur containing every
                grace note at this beat that requested the slur plus the
                principal note or chord at the same beat. Only include if
                specifically requested by the user.

        Returns:
            OperationResult with grace note details.
        """
        voice = validate_voice_number(voice)
        pitch_obj = normalize_pitch(pitch)
        duration_obj = normalize_duration(duration)

        part_obj, part_idx = self._resolve_part(part)
        measure_number = self._resolve_measure_number(part_obj, measure)
        measure_obj = self._resolve_measure(part_obj, measure_number)
        container = self._get_voice_or_measure(measure_obj, voice, create=True)

        if beat is not None:
            offset = beat - 1.0
        else:
            offset = 0.0
        beat_pos = offset + 1.0

        principal = None
        if slur_to_principal:
            principal = self._find_principal_at_offset(container, offset)
            if principal is None:
                raise ValueError(
                    "slur_to_principal=True requires a non-grace note or chord "
                    f"at beat {beat_pos} in measure {measure_number}, voice {voice}."
                )

        grace = m21note.Note(pitch=pitch_obj)
        grace.duration = m21duration.GraceDuration(duration_obj.type)
        grace.duration.slash = slash
        setattr(grace, "scorespeak_slur_to_principal", bool(slur_to_principal))

        container.insert(offset, grace)
        slur_grace_count = 0
        if slur_to_principal and principal is not None:
            slur_grace_count = self._replace_grace_to_principal_slur(
                part_obj,
                container,
                principal,
                offset,
            )
        self._refresh_measure_beams(measure_obj)
        self._refresh_measure_accidentals(part_obj, measure_number)

        grace_type = "acciaccatura" if slash else "appoggiatura"
        return OperationResult(
            success=True,
            description=(
                f"Added {grace_type} grace note {pitch_obj.nameWithOctave} "
                f"at measure {measure_number}, beat {beat_pos}"
            ),
            details={
                "pitch": pitch_obj.nameWithOctave,
                "duration": duration_obj.type,
                "written_quarter_length": duration_obj.quarterLength,
                "grace_type": grace_type,
                "measure": measure_number,
                "beat": beat_pos,
                "part": part_idx,
                "voice": voice,
                "slur_to_principal": bool(slur_to_principal),
                "slur_grace_count": slur_grace_count,
            },
        )


    def _find_principal_at_offset(
        self,
        container: m21stream.Stream,
        offset: float,
    ) -> Optional[m21note.GeneralNote]:
        """Return the non-grace note or chord that starts at ``offset``."""
        for element in container.getElementsByClass(m21note.GeneralNote):
            if getattr(element.duration, "isGrace", False):
                continue
            if isinstance(element, m21note.Rest):
                continue
            if not isinstance(element, (m21note.Note, m21chord.Chord)):
                continue
            if abs(float(container.elementOffset(element)) - offset) < 1e-9:
                return element
        return None


    def _replace_grace_to_principal_slur(
        self,
        part_obj: m21stream.Part,
        container: m21stream.Stream,
        principal: m21note.GeneralNote,
        offset: float,
    ) -> int:
        """Replace same-beat grace/principal slurs with one shared slur."""
        grace_notes = self._slurred_grace_notes_at_offset(container, offset)
        self._remove_grace_to_principal_slurs(part_obj, principal)
        if not grace_notes:
            return 0

        slur = m21spanner.Slur()
        slur.addSpannedElements(*grace_notes, principal)
        setattr(slur, "scorespeak_grace_to_principal", True)
        part_obj.insert(0, slur)
        return len(grace_notes)


    def _slurred_grace_notes_at_offset(
        self,
        container: m21stream.Stream,
        offset: float,
    ) -> list[m21note.GeneralNote]:
        """Return grace notes at ``offset`` that requested principal slurring."""
        grace_notes: list[m21note.GeneralNote] = []
        for element in container.getElementsByClass(m21note.GeneralNote):
            if not getattr(element.duration, "isGrace", False):
                continue
            if not getattr(element, "scorespeak_slur_to_principal", False):
                continue
            if isinstance(element, m21note.Rest):
                continue
            if abs(float(container.elementOffset(element)) - offset) < 1e-9:
                grace_notes.append(element)
        return grace_notes


    def _remove_grace_to_principal_slurs(
        self,
        part_obj: m21stream.Part,
        principal: m21note.GeneralNote,
    ) -> None:
        """Remove generated same-beat grace slurs ending at ``principal``."""
        for slur in list(part_obj.getElementsByClass(m21spanner.Slur)):
            if not bool(getattr(slur, "scorespeak_grace_to_principal", False)):
                continue
            spanned = list(slur.getSpannedElements())
            if not spanned or spanned[-1] is not principal:
                continue
            active_site = getattr(slur, "activeSite", None)
            if active_site is not None:
                active_site.remove(slur)
            else:
                part_obj.remove(slur)


    def remove_grace_note(
        self,
        measure: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
        pitch: Optional[PitchInput] = None,
    ) -> OperationResult:
        """Remove one grace note from one voice. Use only after the exact
        measure, beat, part, and voice are known; inspect multi-voice bars first
        when the target layer is unclear.

        Args:
            measure: 1-based measure number.
            beat: 1-based beat position where the grace note is attached.
            part: Part identifier.
            voice: 1-based rhythmic timeline inside this part. Defaults to
                voice 1. Inspect first and use another voice only when the grace
                note belongs to a different simultaneous rhythmic line.
            pitch: Optional pitch guard when multiple grace notes share a beat.

        Returns:
            OperationResult describing the removed grace note.
        """
        part_obj, part_idx = self._resolve_part(part)
        measure_obj = self._resolve_measure(part_obj, measure)
        container = self._get_voice_or_measure(measure_obj, voice)
        offset = beat - 1.0
        pitch_obj = normalize_pitch(pitch) if pitch is not None else None

        found = None
        for el in container.getElementsByClass(m21note.GeneralNote):
            if not getattr(el.duration, "isGrace", False):
                continue
            if abs(container.elementOffset(el) - offset) > 1e-9:
                continue
            if pitch_obj is not None:
                if not isinstance(el, m21note.Note):
                    continue
                if el.pitch.nameWithOctave != pitch_obj.nameWithOctave:
                    continue
            found = el
            break

        if found is None:
            raise ValueError(
                f"No grace note found at beat {beat} in measure {measure}, "
                f"voice {voice}. Inspect the measure to find the grace note's "
                "voice and attachment beat; include pitch if several grace "
                "notes share the beat."
            )

        principal = self._find_principal_at_offset(container, offset)
        should_rebuild_principal_slur = (
            principal is not None
            and (
                bool(getattr(found, "scorespeak_slur_to_principal", False))
                or self._is_in_grace_to_principal_slur(part_obj, found)
            )
        )
        if should_rebuild_principal_slur and principal is not None:
            self._remove_grace_to_principal_slurs(part_obj, principal)

        desc = self._describe_element(found)
        container.remove(found)
        if should_rebuild_principal_slur and principal is not None:
            self._replace_grace_to_principal_slur(
                part_obj,
                container,
                principal,
                offset,
            )
        self._refresh_measure_beams(measure_obj)
        self._refresh_measure_accidentals(part_obj, measure)

        return OperationResult(
            success=True,
            description=(
                f"Removed grace {desc} from measure {measure}, beat {beat}"
            ),
            details={
                "measure": measure,
                "beat": beat,
                "part": part_idx,
                "voice": voice,
                "pitch": (
                    found.pitch.nameWithOctave
                    if isinstance(found, m21note.Note)
                    else None
                ),
            },
        )


    def _is_in_grace_to_principal_slur(
        self,
        part_obj: m21stream.Part,
        element: m21note.GeneralNote,
    ) -> bool:
        """Return whether ``element`` belongs to a generated grace slur."""
        for slur in part_obj.getElementsByClass(m21spanner.Slur):
            if not bool(getattr(slur, "scorespeak_grace_to_principal", False)):
                continue
            if element in list(slur.getSpannedElements()):
                return True
        return False


    def _refresh_measure_grace_beams(
        self,
        measure_obj: m21stream.Measure,
    ) -> None:
        """Recompute automatic grace-note beams inside one measure."""
        for _voice, container in self._measure_voice_containers(measure_obj):
            self._refresh_container_grace_beams(container)


    def _refresh_container_grace_beams(
        self,
        container: m21stream.Stream,
    ) -> None:
        """Beam same-offset grace-note groups within one voice container."""
        groups: dict[
            tuple[float, str],
            list[m21note.Note | m21chord.Chord],
        ] = {}
        for element in self._direct_general_notes(container):
            if not isinstance(element, (m21note.Note, m21chord.Chord)):
                continue
            if not getattr(element.duration, "isGrace", False):
                continue

            element.beams = m21beam.Beams()
            duration_type = self._beamable_grace_duration_type(element)
            if duration_type is None:
                continue

            offset = round(float(container.elementOffset(element)), 9)
            key = (offset, duration_type)
            groups.setdefault(key, []).append(element)

        for group in groups.values():
            self._apply_grace_beam_group(group)


    @staticmethod
    def _beamable_grace_duration_type(
        element: m21note.Note | m21chord.Chord,
    ) -> str | None:
        """Return the written grace duration type when it can be beamed."""
        duration_type = getattr(element.duration, "type", None)
        if duration_type in _BEAMABLE_GRACE_DURATION_TYPES:
            return str(duration_type)
        return None


    @staticmethod
    def _apply_grace_beam_group(
        group: list[m21note.Note | m21chord.Chord],
    ) -> None:
        """Apply start/continue/stop beam tags to one grace-note group."""
        if len(group) < 2:
            return

        final_index = len(group) - 1
        for index, element in enumerate(group):
            if index == 0:
                beam_type = "start"
            elif index == final_index:
                beam_type = "stop"
            else:
                beam_type = "continue"
            element.beams.fill(element.duration.type, beam_type)
