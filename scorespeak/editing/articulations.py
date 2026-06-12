"""Internal expression-editing implementation slice."""

from __future__ import annotations

from .expression_common import *


class ArticulationEditingMixin:
    """Internal mixin for ScoreSpeak expression operations."""

    def add_articulation(
        self,
        articulation_type: Union[str, ArticulationType],
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
    ) -> OperationResult:
        """Add staccato/staccatissimo, breath mark/caesura, etc. to target
        present at requested part/measure/beat/voice; between-note marks
        appear after it; part=None applies to all parts.

        Supports staccato/staccatissimo, accent/marcato/strong accent,
        tenuto, fermata, breath mark/caesura, up/down bow,
        harmonic/string harmonic, and stopped. Fermatas may be attached to
        notes, chords, or rests; other articulations require notes or chords.

        Args:
            articulation_type: ``"staccato"``, ``"accent"``, ``"tenuto"``,
                ``"fermata"``, ``"breath mark"``, ``"caesura"``,
                ``"up bow"``, ``"down bow"``, ``"harmonic"``,
                ``"string harmonic"``, ``"stopped"``, etc., or an
                ArticulationType enum member.
            measure_number: 1-based measure number.
            beat: 1-based beat position (default 1.0).
            part: Part index or name, or None for all parts.
            voice: 1-based voice number (default 1).

        Returns:
            OperationResult confirming the articulation was added.
        """
        type_str = (
            articulation_type.value
            if isinstance(articulation_type, ArticulationType)
            else str(articulation_type).strip().lower()
        )
        art_cls = _ARTICULATION_MAP.get(type_str)
        if art_cls is None:
            raise ValueError(
                f"Unknown articulation type '{type_str}'. "
                f"Valid types: {', '.join(sorted(_ARTICULATION_MAP.keys()))}"
            )

        offset = beat - 1.0
        allows_rest = _articulation_allows_rest(type_str)

        if part is not None:
            part_obj, part_idx = self._resolve_part(part)
            measure_obj = self._resolve_measure(part_obj, measure_number)
            container = self._get_voice_or_measure(measure_obj, voice)
            element = _find_general_note_at_offset(
                container,
                offset,
                include_rests=True,
            )
            if element is None:
                target_desc = (
                    "note, chord, or rest" if allows_rest else "note or chord"
                )
                raise ValueError(
                    f"No {target_desc} found at beat {beat} in "
                    f"measure {measure_number}. Articulations must be "
                    f"attached to a valid target."
                )
            if isinstance(element, m21note.Rest) and not allows_rest:
                raise ValueError(
                    f"Cannot add a {type_str} articulation to a rest "
                    f"at beat {beat} in measure {measure_number}. "
                    f"Articulations can only be placed on notes or chords."
                )

            _add_articulation_marking(element, art_cls)
            target_label = _articulation_target_label(element)

            return OperationResult(
                success=True,
                description=(
                    f"Added {type_str} to {target_label} at "
                    f"measure {measure_number}, beat {beat} in part {part_idx}"
                ),
                details={
                    "articulation": type_str,
                    "measure": measure_number,
                    "beat": beat,
                    "part": part_idx,
                    "voice": voice,
                },
            )

        targets = self._resolve_parts_or_all(part)
        insertions = []
        skipped_parts = []
        already_present_parts = []
        for part_obj, part_idx in targets:
            measure_obj = self._resolve_measure(part_obj, measure_number)
            container = self._get_voice_or_measure(measure_obj, voice)
            element = _find_general_note_at_offset(
                container,
                offset,
                include_rests=True,
            )
            if element is None:
                skipped_parts.append(part_idx)
                continue
            if isinstance(element, m21note.Rest) and not allows_rest:
                skipped_parts.append(part_idx)
                continue
            if _has_articulation_marking(element, art_cls):
                already_present_parts.append(part_idx)
                continue
            insertions.append((element, part_idx))

        if not insertions and not already_present_parts:
            target_desc = (
                "note, chord, or rest" if allows_rest else "note or chord"
            )
            raise ValueError(
                f"No valid {target_desc} found for {type_str} at beat {beat} "
                f"in measure {measure_number} in any part."
            )

        for element, _part_idx in insertions:
            _add_articulation_marking(element, art_cls)

        added_parts = [part_idx for _element, part_idx in insertions]
        target_parts = [part_idx for _part_obj, part_idx in targets]
        if added_parts:
            description = (
                f"Added {type_str} at measure {measure_number}, beat {beat} "
                f"in parts {added_parts}"
            )
        else:
            description = (
                f"{type_str.capitalize()} already present at measure "
                f"{measure_number}, beat {beat} in parts "
                f"{already_present_parts}"
            )
        if added_parts and already_present_parts:
            description += (
                f"; already present in parts {already_present_parts}"
            )
        if skipped_parts:
            description += (
                f"; skipped parts without a valid target {skipped_parts}"
            )

        return OperationResult(
            success=True,
            description=description,
            details={
                "articulation": type_str,
                "measure": measure_number,
                "beat": beat,
                "voice": voice,
                "parts": target_parts,
                "added_parts": added_parts,
                "skipped_parts": skipped_parts,
                "already_present_parts": already_present_parts,
            },
        )


    def remove_articulation(
        self,
        articulation_type: Union[str, ArticulationType],
        measure_number: int,
        beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
    ) -> OperationResult:
        """Remove articulation marking(s) such as staccato/staccatissimo,
        accent/marcato/strong accent, tenuto, fermata, breath mark/caesura,
        up/down bow, harmonic, string harmonic, or stopped at the requested
        measure/beat/voice; part=None applies to all parts.

        Supports staccato/staccatissimo, accent/marcato/strong accent,
        tenuto, fermata, breath mark/caesura, up/down bow,
        harmonic/string harmonic, and stopped. Fermatas may be removed from
        notes, chords, or rests; other articulations target notes or chords.

        Args:
            articulation_type: The articulation to remove.
            measure_number: 1-based measure number.
            beat: 1-based beat position (default 1.0).
            part: Part index or name, or None for all parts.
            voice: 1-based voice number (default 1).

        Returns:
            OperationResult confirming removal.
        """
        type_str = (
            articulation_type.value
            if isinstance(articulation_type, ArticulationType)
            else str(articulation_type).strip().lower()
        )
        art_cls = _ARTICULATION_MAP.get(type_str)
        if art_cls is None:
            raise ValueError(
                f"Unknown articulation type '{type_str}'. "
                f"Valid types: {', '.join(sorted(_ARTICULATION_MAP.keys()))}"
            )

        offset = beat - 1.0
        allows_rest = _articulation_allows_rest(type_str)

        if part is not None:
            part_obj, part_idx = self._resolve_part(part)
            measure_obj = self._resolve_measure(part_obj, measure_number)
            container = self._get_voice_or_measure(measure_obj, voice)
            element = _find_general_note_at_offset(
                container,
                offset,
                include_rests=allows_rest,
            )
            if element is None:
                target_desc = (
                    "note, chord, or rest" if allows_rest else "note or chord"
                )
                raise ValueError(
                    f"No {target_desc} found at beat {beat} in "
                    f"measure {measure_number}."
                )

            removed = _remove_articulation_marking(element, art_cls)
            if not removed:
                target_label = _articulation_target_label(element)
                raise ValueError(
                    f"No {type_str} articulation found on the "
                    f"{target_label} at beat {beat} in measure "
                    f"{measure_number}."
                )

            target_label = _articulation_target_label(element)
            return OperationResult(
                success=True,
                description=(
                    f"Removed {type_str} from {target_label} at "
                    f"measure {measure_number}, beat {beat} in part {part_idx}"
                ),
                details={
                    "articulation": type_str,
                    "measure": measure_number,
                    "beat": beat,
                    "part": part_idx,
                    "voice": voice,
                },
            )

        targets = self._resolve_parts_or_all(part)
        removals = []
        skipped_parts = []
        missing_parts = []
        for part_obj, part_idx in targets:
            measure_obj = self._resolve_measure(part_obj, measure_number)
            container = self._get_voice_or_measure(measure_obj, voice)
            element = _find_general_note_at_offset(
                container,
                offset,
                include_rests=allows_rest,
            )
            if element is None:
                skipped_parts.append(part_idx)
                continue
            if not _has_articulation_marking(element, art_cls):
                missing_parts.append(part_idx)
                continue
            removals.append((element, part_idx))

        if not removals:
            raise ValueError(
                f"No {type_str} articulation found at beat {beat} "
                f"in measure {measure_number} in any part."
            )

        for element, _part_idx in removals:
            _remove_articulation_marking(element, art_cls)

        removed_parts = [part_idx for _element, part_idx in removals]
        target_parts = [part_idx for _part_obj, part_idx in targets]
        description = (
            f"Removed {type_str} from measure {measure_number}, beat {beat} "
            f"in parts {removed_parts}"
        )
        if missing_parts:
            description += f"; missing in parts {missing_parts}"
        if skipped_parts:
            description += (
                f"; skipped parts without a valid target {skipped_parts}"
            )

        return OperationResult(
            success=True,
            description=description,
            details={
                "articulation": type_str,
                "measure": measure_number,
                "beat": beat,
                "voice": voice,
                "parts": target_parts,
                "removed_parts": removed_parts,
                "skipped_parts": skipped_parts,
                "missing_parts": missing_parts,
            },
        )


    def add_slur(
        self,
        start_measure: int,
        start_beat: float,
        end_measure: int,
        end_beat: float,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
    ) -> OperationResult:
        """Add a slur spanning from one note to another.

        Args:
            start_measure: 1-based measure where the slur begins.
            start_beat: Beat where the slur begins.
            end_measure: 1-based measure where the slur ends.
            end_beat: Beat where the slur ends.
            part: Part index, name, or None for first part.
            voice: 1-based voice number (default 1).

        Returns:
            OperationResult confirming the slur was placed.
        """
        part_obj, part_idx = self._resolve_part(part)

        start_m = self._resolve_measure(part_obj, start_measure)
        end_m = self._resolve_measure(part_obj, end_measure)
        start_container = self._get_voice_or_measure(start_m, voice)
        end_container = self._get_voice_or_measure(end_m, voice)

        start_offset = start_beat - 1.0
        end_offset = end_beat - 1.0

        start_note = _find_note_at_offset(start_container, start_offset)
        if start_note is None:
            raise ValueError(
                f"No note found at beat {start_beat} in "
                f"measure {start_measure} for slur start."
            )
        if isinstance(start_note, m21note.Rest):
            raise ValueError(
                f"Cannot start a slur on a rest at beat {start_beat} "
                f"in measure {start_measure}."
            )

        end_note = _find_note_at_offset(end_container, end_offset)
        if end_note is None:
            raise ValueError(
                f"No note found at beat {end_beat} in "
                f"measure {end_measure} for slur end."
            )
        if isinstance(end_note, m21note.Rest):
            raise ValueError(
                f"Cannot end a slur on a rest at beat {end_beat} "
                f"in measure {end_measure}."
            )

        slur = m21spanner.Slur()
        slur.addSpannedElements(start_note, end_note)
        part_obj.insert(0, slur)

        return OperationResult(
            success=True,
            description=(
                f"Added slur from measure {start_measure} beat {start_beat} "
                f"to measure {end_measure} beat {end_beat}"
            ),
            details={
                "start_measure": start_measure,
                "start_beat": start_beat,
                "end_measure": end_measure,
                "end_beat": end_beat,
                "part": part_idx,
                "voice": voice,
            },
        )


    def remove_slur(
        self,
        start_measure: int,
        start_beat: float = 1.0,
        part: Optional[Union[int, str]] = None,
        voice: int = 1,
    ) -> OperationResult:
        """Remove a slur starting at the given position.

        Args:
            start_measure: 1-based measure where the slur begins.
            start_beat: Beat where the slur begins (default 1.0).
            part: Part index, name, or None for first part.
            voice: 1-based voice number (default 1).

        Returns:
            OperationResult confirming removal.
        """
        part_obj, part_idx = self._resolve_part(part)
        start_m = self._resolve_measure(part_obj, start_measure)
        start_container = self._get_voice_or_measure(start_m, voice)
        start_offset = start_beat - 1.0

        found = None
        for sp in part_obj.getElementsByClass(m21spanner.Slur):
            spanned = sp.getSpannedElements()
            if not spanned:
                continue
            first_el = spanned[0]
            container = first_el.getContextByClass(m21stream.Measure)
            if container is None:
                for parent_container in (start_container, start_m):
                    try:
                        el_off = parent_container.elementOffset(first_el)
                        if abs(el_off - start_offset) < 1e-9:
                            found = sp
                            break
                    except Exception:
                        continue
                if found is not None:
                    break
                continue
            if container.number != start_measure:
                continue
            el_offset = container.elementOffset(first_el)
            if abs(el_offset - start_offset) < 1e-9:
                found = sp
                break

        if found is None:
            raise ValueError(
                f"No slur found starting at beat {start_beat} "
                f"in measure {start_measure}."
            )

        part_obj.remove(found)

        return OperationResult(
            success=True,
            description=(
                f"Removed slur starting at "
                f"measure {start_measure}, beat {start_beat}"
            ),
            details={
                "start_measure": start_measure,
                "start_beat": start_beat,
                "part": part_idx,
                "voice": voice,
            },
        )
