"""Internal signature-editing implementation slice."""

from __future__ import annotations

from .signature_common import *


class KeySignatureEditingMixin:
    """Internal mixin for ScoreSpeak signature operations."""

    def set_key_signature(
        self,
        key_signature: str,
        measure_number: int,
        transpose_existing: bool = False,
    ) -> OperationResult:
        """Set the score-level concert key signature at a measure and apply it to all parts: concert-pitch parts get the concert key, written-pitch transposing parts get the derived written key, and non-transposing parts get the same key. It does not toggle stored pitch space.

        This is a concert/sounding-key operation. Parts currently in concert
        pitch get the concert key directly. Written-pitch transposing parts
        get the derived written key for their instrument. Non-transposing
        parts get the same key in either pitch space. This method uses each
        part's current pitch-space state; it does not toggle stored pitch
        space or remove explicit local key overrides.

        The new score-level concert key signature takes effect at the given
        measure and propagates forward until another score-level concert key
        signature is encountered. If the requested value matches the
        inherited value from before the measure, any local score-level change
        at the measure is removed so the timeline inherits cleanly.

        Args:
            key_signature: Flexible key string such as "C", "G", "Bb",
                "F# minor", "Am", "3" (sharps), "-2" (flats).
            measure_number: 1-based measure number.
            transpose_existing: If True, transpose all notes from the old
                concert key to the new concert key starting at the given
                measure. This is musical transposition of material, not a
                conversion between written and concert pitch storage.

        Returns:
            OperationResult describing the outcome.

        Raises:
            ValueError: If the key signature string is invalid or the
                measure does not exist.
        """
        concert_ks = _parse_key_signature(key_signature)
        targets = self._resolve_parts_or_all()
        changed_parts: list[int] = []
        part_actions: list[dict[str, object]] = []
        active_concert_before = self._get_active_concert_key_signature_obj(
            measure_number,
        )

        for part_obj, part_idx in targets:
            if self._preserve_local_key_override_for_global_key(
                part_obj,
                part_idx,
                measure_number,
                active_concert_before,
            ):
                part_action = self._local_key_override_part_action(
                    part_obj,
                    part_idx,
                    measure_number,
                )
            else:
                stored_ks = stored_key_signature_for_concert_key(
                    part_obj,
                    concert_ks,
                )
                part_action = self._set_part_key_signature(
                    part_obj,
                    part_idx,
                    stored_ks,
                    measure_number,
                    transpose_existing,
                )

            part_changed = bool(part_action["changed"])
            if part_changed:
                changed_parts.append(part_idx)
            part_actions.append(part_action)

        formatted = self._format_key_signature(concert_ks)
        changed = bool(changed_parts)
        action = self._combined_signature_action(part_actions)
        if changed:
            description = (
                f"Set concert key signature timeline to {formatted} "
                f"at measure {measure_number}"
            )
        else:
            description = (
                f"Concert key signature is already {formatted} at measure "
                f"{measure_number}; no score change made"
            )
        return OperationResult(
            success=True,
            description=description,
            details={
                "key_signature": formatted,
                "measure": measure_number,
                "parts": changed_parts,
                "transposed": transpose_existing,
                "changed": changed,
                "action": action,
                "active_before": self._common_part_action_value(
                    part_actions,
                    "active_before",
                ),
                "active_after": self._common_part_action_value(
                    part_actions,
                    "active_after",
                ),
                "inherited_before": self._common_part_action_value(
                    part_actions,
                    "inherited_before",
                ),
                "part_actions": part_actions,
            },
        )


    def _set_local_key_signature(
        self,
        key_signature: str,
        measure_number: int,
        part: Union[int, str],
        transpose_existing: bool = False,
    ) -> OperationResult:
        """Set a written/local key signature on one part only.

        Use this for staff-level exceptions such as open/atonal horn parts or
        imported staff-level overrides. This is intentionally private so the
        agent does not use it for ordinary key changes.
        """
        part_obj, part_idx = self._resolve_part(part)
        was_marked = has_marked_local_key_override(part_obj)
        target_ks = _parse_key_signature(key_signature)
        part_action = self._set_part_key_signature(
            part_obj,
            part_idx,
            target_ks,
            measure_number,
            transpose_existing,
        )
        mark_local_key_override(part_obj)

        changed = bool(part_action["changed"] or not was_marked)
        formatted = self._format_key_signature(target_ks)
        return OperationResult(
            success=True,
            description=(
                f"Set local key signature for part {part_idx} to {formatted} "
                f"at measure {measure_number}"
            ),
            details={
                "key_signature": formatted,
                "measure": measure_number,
                "part": part_idx,
                "parts": [part_idx] if changed else [],
                "local": True,
                "transposed": transpose_existing,
                "changed": changed,
                "action": (
                    part_action["action"]
                    if was_marked
                    else "marked_local_override"
                ),
                "part_actions": [part_action],
            },
        )


    def _set_part_key_signature(
        self,
        part_obj: m21stream.Part,
        part_idx: int,
        new_ks: m21key.KeySignature,
        measure_number: int,
        transpose_existing: bool,
    ) -> dict[str, object]:
        """Apply a stored key signature to one part and return action details."""
        measure = self._resolve_measure(part_obj, measure_number)
        old_ks = self._get_active_key_signature_obj(part_obj, measure_number)
        pre_removed = self._canonicalize_key_signatures_from(
            part_obj,
            measure_number,
        )
        inherited_before = self._get_key_signature_before_measure(
            part_obj,
            measure_number,
        )
        current_ks = self._get_active_key_signature_obj(
            part_obj,
            measure_number,
        )
        local_ks_list = self._local_key_signatures(measure)

        edit_changed = False
        action = "already_active"
        if not self._key_signatures_equal(current_ks, new_ks):
            self._remove_local_key_signatures(measure)
            if (
                inherited_before is not None
                and self._key_signatures_equal(new_ks, inherited_before)
            ):
                action = "removed_local_change"
            else:
                measure.insert(0, self._copy_key_signature(new_ks))
                action = "set"
            edit_changed = True
        elif (
            local_ks_list
            and inherited_before is not None
            and self._key_signatures_equal(current_ks, inherited_before)
        ):
            self._remove_local_key_signatures(measure)
            action = "removed_redundant_local_change"
            edit_changed = True
        elif pre_removed:
            action = "canonicalized"

        post_removed = self._canonicalize_key_signatures_from(
            part_obj,
            measure_number,
        )
        if edit_changed:
            if transpose_existing:
                self._transpose_from_measure(
                    part_obj,
                    measure_number,
                    old_ks,
                    new_ks,
                )
            self._refresh_accidentals_until_next_key_change(
                part_obj,
                measure_number,
            )

        active_after = self._get_active_key_signature_obj(
            part_obj,
            measure_number,
        )
        part_changed = bool(edit_changed or pre_removed or post_removed)
        return {
            "part": part_idx,
            "action": action,
            "changed": part_changed,
            "active_before": self._format_key_signature(old_ks),
            "active_after": self._format_key_signature(active_after),
            "inherited_before": (
                self._format_key_signature(inherited_before)
                if inherited_before is not None
                else None
            ),
            "target_key": self._format_key_signature(new_ks),
            "removed_redundant": pre_removed + post_removed,
        }


    def _get_active_concert_key_signature_obj(
        self,
        measure_number: int,
    ) -> m21key.KeySignature:
        """Return the score-level concert key at ``measure_number``."""
        reference_part, _ = self._resolve_part(None)
        stored_ks = self._get_active_key_signature_obj(
            reference_part,
            measure_number,
        )
        return concert_key_signature_for_stored_key(reference_part, stored_ks)


    def _preserve_local_key_override_for_global_key(
        self,
        part_obj: m21stream.Part,
        part_idx: int,
        measure_number: int,
        active_concert_before: m21key.KeySignature,
    ) -> bool:
        """Return whether a global key edit should leave ``part_obj`` alone."""
        if has_marked_local_key_override(part_obj):
            return True
        if part_idx == 0:
            return False

        active_stored = self._get_active_key_signature_obj(
            part_obj,
            measure_number,
        )
        expected_stored = stored_key_signature_for_concert_key(
            part_obj,
            active_concert_before,
        )
        return not self._key_signatures_equal(active_stored, expected_stored)


    def _local_key_override_part_action(
        self,
        part_obj: m21stream.Part,
        part_idx: int,
        measure_number: int,
    ) -> dict[str, object]:
        """Return no-op action details for a preserved local-key part."""
        active_ks = self._get_active_key_signature_obj(part_obj, measure_number)
        inherited_before = self._get_key_signature_before_measure(
            part_obj,
            measure_number,
        )
        return {
            "part": part_idx,
            "action": "local_override_preserved",
            "changed": False,
            "active_before": self._format_key_signature(active_ks),
            "active_after": self._format_key_signature(active_ks),
            "inherited_before": (
                self._format_key_signature(inherited_before)
                if inherited_before is not None
                else None
            ),
            "target_key": "preserved",
            "removed_redundant": 0,
        }


    @staticmethod
    def _copy_key_signature(
        key_signature: m21key.KeySignature,
    ) -> m21key.KeySignature:
        """Return a fresh key signature preserving key or fifths identity."""
        return copy_key_signature(key_signature)


    @staticmethod
    def _key_signature_identity(
        key_signature: m21key.KeySignature,
    ) -> tuple[object, ...]:
        """Return the normalized identity used for timeline comparisons."""
        if is_open_key_signature(key_signature):
            return ("open",)
        if isinstance(key_signature, m21key.Key) and key_signature.tonic is not None:
            return (
                "key",
                key_signature.tonic.name,
                str(key_signature.mode).lower(),
            )
        return ("fifths", int(key_signature.sharps or 0))


    def _key_signatures_equal(
        self,
        left: m21key.KeySignature,
        right: m21key.KeySignature,
    ) -> bool:
        """Return whether two key signatures have the same timeline identity."""
        return (
            self._key_signature_identity(left)
            == self._key_signature_identity(right)
        )


    @staticmethod
    def _local_key_signatures(
        measure: m21stream.Measure,
    ) -> list[m21key.KeySignature]:
        """Return explicit key signatures stored directly on ``measure``."""
        local_key_signatures: list[m21key.KeySignature] = []
        seen_ids: set[int] = set()
        for class_name in (m21key.KeySignature, m21key.Key):
            for key_signature in measure.getElementsByClass(class_name):
                key_id = id(key_signature)
                if key_id in seen_ids:
                    continue
                seen_ids.add(key_id)
                local_key_signatures.append(key_signature)
        return local_key_signatures


    def _remove_local_key_signatures(
        self,
        measure: m21stream.Measure,
    ) -> int:
        """Remove explicit key signatures from ``measure`` and count them."""
        local_key_signatures = self._local_key_signatures(measure)
        for key_signature in local_key_signatures:
            measure.remove(key_signature)
        return len(local_key_signatures)


    def _canonicalize_key_signatures_from(
        self,
        part_obj: m21stream.Part,
        start_measure: int,
    ) -> int:
        """Remove redundant local key signatures from ``start_measure`` on."""
        previous_key_signature = self._get_key_signature_before_measure(
            part_obj,
            start_measure,
        )
        removed_count = 0
        for measure in self._sorted_measures_for_signature_scan(part_obj):
            measure_number = measure.number
            if measure_number is None or measure_number < start_measure:
                continue

            local_key_signatures = self._local_key_signatures(measure)
            if not local_key_signatures:
                continue

            effective_key_signature = local_key_signatures[0]
            if (
                previous_key_signature is not None
                and self._key_signatures_equal(
                    effective_key_signature,
                    previous_key_signature,
                )
            ):
                removed_count += self._remove_local_key_signatures(measure)
                continue

            for extra_key_signature in local_key_signatures[1:]:
                measure.remove(extra_key_signature)
                removed_count += 1
            previous_key_signature = effective_key_signature
        return removed_count


    def _get_key_signature_before_measure(
        self,
        part_obj: m21stream.Part,
        measure_number: int,
    ) -> Optional[m21key.KeySignature]:
        """Return the active key signature before ``measure_number``."""
        previous_measure = self._previous_signature_measure(part_obj, measure_number)
        if previous_measure is None or previous_measure.number is None:
            return None
        return self._get_active_key_signature_obj(part_obj, previous_measure.number)


    @staticmethod
    def _sorted_measures_for_signature_scan(
        part_obj: m21stream.Part,
    ) -> list[m21stream.Measure]:
        """Return part measures sorted by numeric measure number."""
        return sorted(
            part_obj.getElementsByClass(m21stream.Measure),
            key=lambda measure: measure.number or 0,
        )


    def _previous_signature_measure(
        self,
        part_obj: m21stream.Part,
        measure_number: int,
    ) -> Optional[m21stream.Measure]:
        """Return the closest measure before ``measure_number``."""
        previous_measure: Optional[m21stream.Measure] = None
        for measure in self._sorted_measures_for_signature_scan(part_obj):
            if measure.number is None:
                continue
            if measure.number >= measure_number:
                break
            previous_measure = measure
        return previous_measure


    @staticmethod
    def _combined_signature_action(
        part_actions: list[dict[str, object]],
    ) -> object:
        """Return one action value when all parts agree, else ``mixed``."""
        actions = {part_action.get("action") for part_action in part_actions}
        if len(actions) == 1:
            return next(iter(actions))
        return "mixed"


    @staticmethod
    def _common_part_action_value(
        part_actions: list[dict[str, object]],
        key: str,
    ) -> object:
        """Return a common per-part value or ``mixed`` when values differ."""
        values = [part_action.get(key) for part_action in part_actions]
        if not values:
            return None
        first_value = values[0]
        if all(value == first_value for value in values):
            return first_value
        return "mixed"


    def _transpose_from_measure(
        self,
        part_obj: m21stream.Part,
        start_measure_number: int,
        old_ks: m21key.KeySignature,
        new_ks: m21key.KeySignature,
    ) -> None:
        """Transpose notes from old key to new key, starting at a measure.

        Walks forward from start_measure_number through all subsequent
        measures until a *different* explicit key signature is found (which
        would indicate a separate key region not governed by this change).
        """
        from music21 import interval as m21interval

        old_tonic = self._ks_tonic(old_ks)
        new_tonic = self._ks_tonic(new_ks)

        ivl = m21interval.Interval(old_tonic, new_tonic)

        measures = sorted(
            part_obj.getElementsByClass(m21stream.Measure),
            key=lambda m: m.number,
        )

        for m in measures:
            if m.number < start_measure_number:
                continue

            if m.number > start_measure_number:
                local_ks_list = list(m.getElementsByClass(m21key.KeySignature))
                if local_ks_list:
                    break

            for n in m.recurse().getElementsByClass(m21note.NotRest):
                if hasattr(n, "pitches"):
                    for p in n.pitches:
                        new_p = p.transpose(ivl)
                        p.name = new_p.name
                        p.octave = new_p.octave
                elif hasattr(n, "pitch"):
                    new_p = n.pitch.transpose(ivl)
                    n.pitch.name = new_p.name
                    n.pitch.octave = new_p.octave


    @staticmethod
    def _ks_tonic(ks: m21key.KeySignature) -> "m21note.Note":
        """Extract a representative tonic pitch from a key signature."""
        from music21 import pitch as m21pitch

        if isinstance(ks, m21key.Key) and ks.tonic is not None:
            return ks.tonic
        sharp_major_tonics = ["C", "G", "D", "A", "E", "B", "F#", "C#"]
        flat_major_tonics = ["C", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"]
        sharps = ks.sharps
        if sharps >= 0:
            idx = min(sharps, len(sharp_major_tonics) - 1)
            return m21pitch.Pitch(sharp_major_tonics[idx])
        else:
            idx = min(-sharps, len(flat_major_tonics) - 1)
            return m21pitch.Pitch(flat_major_tonics[idx])
