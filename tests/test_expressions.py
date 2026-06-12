"""Tests for dynamics, articulations, hairpins, slurs, text expressions,
tempo, and rehearsal mark operations."""

import pytest
from lxml import etree
from music21 import articulations as m21articulations
from music21 import dynamics as m21dynamics
from music21 import expressions as m21expressions
from music21 import note as m21note
from music21 import spanner as m21spanner
from music21 import stream as m21stream
from music21 import tempo as m21tempo

from scorespeak import ScoreSpeak
from scorespeak.types import ArticulationType, DynamicLevel, HairpinType


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_score(measures=4, time_signature="4/4", key_signature="C", parts=None):
    """Create a ScoreSpeak with sensible defaults for testing."""
    kwargs = {
        "measures": measures,
        "time_signature": time_signature,
        "key_signature": key_signature,
    }
    if parts is not None:
        kwargs["parts"] = parts
    return ScoreSpeak.create(**kwargs)


def _get_measure(ss, measure_number, part=0):
    """Directly retrieve a music21 Measure for inspection."""
    part_obj = list(ss.score.parts)[part]
    return part_obj.measure(measure_number)


# ==================================================================
# add_dynamic / remove_dynamic
# ==================================================================

class TestAddDynamic:
    """Tests for adding dynamic markings."""

    def test_add_dynamic_basic(self):
        ss = _make_score()
        result = ss.add_dynamic("mf", measure_number=1, beat=1)
        assert result.success
        assert result.details["level"] == "mf"
        assert result.details["measure"] == 1
        assert result.details["beat"] == 1.0

    def test_add_dynamic_with_enum(self):
        ss = _make_score()
        result = ss.add_dynamic(DynamicLevel.FF, measure_number=1, beat=1)
        assert result.success
        assert result.details["level"] == "ff"

    def test_add_dynamic_pp(self):
        ss = _make_score()
        result = ss.add_dynamic("pp", measure_number=1)
        assert result.success
        assert result.details["level"] == "pp"

    def test_add_dynamic_sfz(self):
        ss = _make_score()
        result = ss.add_dynamic("sfz", measure_number=2, beat=3)
        assert result.success
        assert result.details["level"] == "sfz"

    def test_add_dynamic_at_non_default_beat(self):
        ss = _make_score()
        result = ss.add_dynamic("f", measure_number=1, beat=3)
        assert result.success
        assert result.details["beat"] == 3.0

    def test_add_dynamic_placed_in_measure(self):
        ss = _make_score()
        ss.add_dynamic("mf", measure_number=2, beat=1)
        m = _get_measure(ss, 2)
        dyns = list(m.getElementsByClass(m21dynamics.Dynamic))
        assert len(dyns) == 1
        assert dyns[0].value == "mf"

    def test_add_dynamic_rejects_existing_dynamic_at_same_position(
        self: "TestAddDynamic",
    ) -> None:
        """Adding a dynamic over an existing same-position dynamic should fail."""
        ss = _make_score()
        ss.add_dynamic("p", measure_number=1, beat=1)

        with pytest.raises(ValueError, match="Remove the existing dynamic first"):
            ss.add_dynamic("f", measure_number=1, beat=1)

        m = _get_measure(ss, 1)
        dyns = list(m.getElementsByClass(m21dynamics.Dynamic))
        assert [dynamic.value for dynamic in dyns] == ["p"]

    def test_add_dynamic_invalid_level(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="not a valid dynamic"):
            ss.add_dynamic("xyz", measure_number=1)

    def test_add_dynamic_invalid_beat_below_one(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="at least 1"):
            ss.add_dynamic("mf", measure_number=1, beat=0.5)

    def test_add_dynamic_beat_beyond_measure(self):
        ss = _make_score(time_signature="3/4")
        with pytest.raises(ValueError, match="beyond the end"):
            ss.add_dynamic("f", measure_number=1, beat=5)

    def test_add_dynamic_to_specific_part(self):
        ss = _make_score(parts=["violin", "cello"])
        result = ss.add_dynamic("p", measure_number=1, part=1)
        assert result.success
        assert result.details["part"] == 1
        first_part = _get_measure(ss, 1, part=0)
        second_part = _get_measure(ss, 1, part=1)
        assert list(first_part.getElementsByClass(m21dynamics.Dynamic)) == []
        assert [
            dynamic.value
            for dynamic in second_part.getElementsByClass(m21dynamics.Dynamic)
        ] == ["p"]

    def test_add_dynamic_without_part_targets_all_parts(self) -> None:
        """Omitting part should add the dynamic to every score part."""
        ss = _make_score(parts=["violin", "viola", "cello"])

        result = ss.add_dynamic("mf", measure_number=1, beat=1)

        assert result.success
        assert result.details["parts"] == [0, 1, 2]
        assert result.details["added_parts"] == [0, 1, 2]
        assert result.details["skipped_parts"] == []
        for part_index in range(3):
            measure = _get_measure(ss, 1, part=part_index)
            dynamics = list(measure.getElementsByClass(m21dynamics.Dynamic))
            assert [dynamic.value for dynamic in dynamics] == ["mf"]

    def test_add_dynamic_without_part_is_idempotent_for_matching_marks(
        self: "TestAddDynamic",
    ) -> None:
        """An all-parts add should skip already matching dynamics."""
        ss = _make_score(parts=["violin", "cello"])
        ss.add_dynamic("mf", measure_number=1, beat=1)

        result = ss.add_dynamic("mf", measure_number=1, beat=1)

        assert result.success
        assert result.details["added_parts"] == []
        assert result.details["skipped_parts"] == [0, 1]
        for part_index in range(2):
            measure = _get_measure(ss, 1, part=part_index)
            dynamics = list(measure.getElementsByClass(m21dynamics.Dynamic))
            assert [dynamic.value for dynamic in dynamics] == ["mf"]

    def test_add_dynamic_without_part_conflict_is_atomic(
        self: "TestAddDynamic",
    ) -> None:
        """A conflicting part should prevent all all-parts insertions."""
        ss = _make_score(parts=["violin", "cello"])
        ss.add_dynamic("p", measure_number=1, beat=1, part=1)

        with pytest.raises(ValueError, match="part 1"):
            ss.add_dynamic("f", measure_number=1, beat=1)

        first_part = _get_measure(ss, 1, part=0)
        second_part = _get_measure(ss, 1, part=1)
        assert list(first_part.getElementsByClass(m21dynamics.Dynamic)) == []
        assert [
            dynamic.value
            for dynamic in second_part.getElementsByClass(m21dynamics.Dynamic)
        ] == ["p"]

    def test_add_dynamic_allows_same_beat_in_different_parts(
        self: "TestAddDynamic",
    ) -> None:
        """Dynamics at the same beat in different parts should both succeed."""
        ss = _make_score(parts=["violin", "cello"])

        ss.add_dynamic("p", measure_number=1, beat=1, part=0)
        ss.add_dynamic("f", measure_number=1, beat=1, part=1)

        first_part = _get_measure(ss, 1, part=0)
        second_part = _get_measure(ss, 1, part=1)
        first_part_dynamics = list(
            first_part.getElementsByClass(m21dynamics.Dynamic)
        )
        second_part_dynamics = list(
            second_part.getElementsByClass(m21dynamics.Dynamic)
        )
        assert [dynamic.value for dynamic in first_part_dynamics] == ["p"]
        assert [dynamic.value for dynamic in second_part_dynamics] == ["f"]

    def test_add_multiple_dynamics_same_measure(self):
        ss = _make_score()
        ss.add_dynamic("p", measure_number=1, beat=1)
        ss.add_dynamic("f", measure_number=1, beat=3)
        m = _get_measure(ss, 1)
        dyns = list(m.getElementsByClass(m21dynamics.Dynamic))
        assert len(dyns) == 2


class TestRemoveDynamic:
    """Tests for removing dynamic markings."""

    def test_remove_dynamic_basic(self):
        ss = _make_score()
        ss.add_dynamic("mf", measure_number=1, beat=1)
        result = ss.remove_dynamic(measure_number=1, beat=1)
        assert result.success
        m = _get_measure(ss, 1)
        dyns = list(m.getElementsByClass(m21dynamics.Dynamic))
        assert len(dyns) == 0

    def test_remove_dynamic_returns_level(self):
        ss = _make_score()
        ss.add_dynamic("ff", measure_number=1, beat=1)
        result = ss.remove_dynamic(measure_number=1, beat=1)
        assert result.details["level"] == "ff"

    def test_remove_dynamic_not_found(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="No dynamic found"):
            ss.remove_dynamic(measure_number=1, beat=1)

    def test_remove_dynamic_explicit_part_not_found_is_strict(self) -> None:
        """Explicit part removal should still fail if that part has no mark."""
        ss = _make_score(parts=["violin", "cello"])
        ss.add_dynamic("mf", measure_number=1, beat=1, part=0)

        with pytest.raises(ValueError, match="No dynamic found"):
            ss.remove_dynamic(measure_number=1, beat=1, part=1)

        first_part = _get_measure(ss, 1, part=0)
        first_part_dynamics = list(
            first_part.getElementsByClass(m21dynamics.Dynamic)
        )
        assert [dynamic.value for dynamic in first_part_dynamics] == ["mf"]

    def test_remove_dynamic_wrong_beat(self):
        ss = _make_score()
        ss.add_dynamic("mf", measure_number=1, beat=1)
        with pytest.raises(ValueError, match="No dynamic found"):
            ss.remove_dynamic(measure_number=1, beat=3)

    def test_remove_one_of_multiple_dynamics(self):
        ss = _make_score()
        ss.add_dynamic("p", measure_number=1, beat=1)
        ss.add_dynamic("f", measure_number=1, beat=3)
        ss.remove_dynamic(measure_number=1, beat=1)
        m = _get_measure(ss, 1)
        dyns = list(m.getElementsByClass(m21dynamics.Dynamic))
        assert len(dyns) == 1
        assert dyns[0].value == "f"

    def test_remove_dynamic_without_part_targets_all_parts(self) -> None:
        """Omitting part should remove matching-position dynamics everywhere."""
        ss = _make_score(parts=["violin", "viola", "cello"])
        ss.add_dynamic("mf", measure_number=1, beat=1)

        result = ss.remove_dynamic(measure_number=1, beat=1)

        assert result.success
        assert result.details["parts"] == [0, 1, 2]
        assert result.details["removed_parts"] == [0, 1, 2]
        assert result.details["level"] == "mf"
        for part_index in range(3):
            measure = _get_measure(ss, 1, part=part_index)
            assert list(measure.getElementsByClass(m21dynamics.Dynamic)) == []

    def test_remove_dynamic_without_part_skips_missing_targets(
        self: "TestRemoveDynamic",
    ) -> None:
        """A missing dynamic in one part should be reported and skipped."""
        ss = _make_score(parts=["violin", "cello"])
        ss.add_dynamic("mf", measure_number=1, beat=1, part=0)

        result = ss.remove_dynamic(measure_number=1, beat=1)

        assert result.success
        assert result.details["parts"] == [0, 1]
        assert result.details["removed_parts"] == [0]
        assert result.details["skipped_parts"] == [1]
        first_part = _get_measure(ss, 1, part=0)
        second_part = _get_measure(ss, 1, part=1)
        assert list(first_part.getElementsByClass(m21dynamics.Dynamic)) == []
        assert list(second_part.getElementsByClass(m21dynamics.Dynamic)) == []

    def test_remove_dynamic_without_part_all_missing_fails(
        self: "TestRemoveDynamic",
    ) -> None:
        """Blanket dynamic removal should fail when no target part has a mark."""
        ss = _make_score(parts=["violin", "cello"])

        with pytest.raises(ValueError, match="No dynamic found"):
            ss.remove_dynamic(measure_number=1, beat=1)


# ==================================================================
# add_hairpin / remove_hairpin
# ==================================================================

class TestAddHairpin:
    """Tests for adding hairpins (crescendo / diminuendo)."""

    def test_add_crescendo_basic(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=4)
        result = ss.add_hairpin(
            "crescendo", start_measure=1, start_beat=1,
            end_measure=1, end_beat=4,
        )
        assert result.success
        assert result.details["type"] == "crescendo"

    def test_add_diminuendo_basic(self):
        ss = _make_score()
        result = ss.add_hairpin(
            "diminuendo", start_measure=1, start_beat=1,
            end_measure=2, end_beat=1,
        )
        assert result.success
        assert result.details["type"] == "diminuendo"

    def test_decrescendo_alias(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=3)
        result = ss.add_hairpin(
            "decrescendo", start_measure=1, start_beat=1,
            end_measure=1, end_beat=3,
        )
        assert result.success
        assert result.details["type"] == "diminuendo"

    def test_add_hairpin_with_enum(self):
        ss = _make_score()
        result = ss.add_hairpin(
            HairpinType.CRESCENDO, start_measure=1, start_beat=1,
            end_measure=2, end_beat=1,
        )
        assert result.success

    def test_add_hairpin_invalid_type(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="Invalid hairpin type"):
            ss.add_hairpin(
                "sforzando", start_measure=1, start_beat=1,
                end_measure=1, end_beat=4,
            )

    def test_add_hairpin_spanning_multiple_measures(self):
        ss = _make_score()
        result = ss.add_hairpin(
            "crescendo", start_measure=1, start_beat=1,
            end_measure=3, end_beat=1,
        )
        assert result.success
        assert result.details["start_measure"] == 1
        assert result.details["end_measure"] == 3

    def test_add_hairpin_downbeat_endpoint_stops_before_arrival_note(self) -> None:
        """A next-measure downbeat endpoint should not span through that note."""
        ss = _make_score(measures=3, time_signature="2/4")
        ss._add_note_one("C4", "half", measure=1, beat=1)
        ss._add_note_one("D4", "half", measure=2, beat=1)

        result = ss.add_hairpin(
            "crescendo",
            start_measure=1,
            start_beat=1,
            end_measure=2,
            end_beat=1,
        )

        part_obj = list(ss.score.parts)[0]
        wedges = list(part_obj.getElementsByClass(m21dynamics.DynamicWedge))
        end_anchor = wedges[-1].getSpannedElements()[-1]
        end_measure = end_anchor.getContextByClass(m21stream.Measure)

        assert result.success
        assert isinstance(end_anchor, m21spanner.SpannerAnchor)
        assert end_measure is not None
        assert end_measure.number == 1
        assert end_measure.elementOffset(end_anchor) == 2.0

    def test_add_hairpin_downbeat_endpoint_uses_previous_bar_final_note(self) -> None:
        """A downbeat endpoint should prefer the previous barline-ending note."""
        ss = _make_score(measures=3, time_signature="2/4")
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=2)
        ss._add_note_one("E4", "half", measure=2, beat=1)

        result = ss.add_hairpin(
            "crescendo",
            start_measure=1,
            start_beat=1,
            end_measure=2,
            end_beat=1,
        )

        part_obj = list(ss.score.parts)[0]
        wedges = list(part_obj.getElementsByClass(m21dynamics.DynamicWedge))
        end_anchor = wedges[-1].getSpannedElements()[-1]
        end_measure = end_anchor.getContextByClass(m21stream.Measure)

        assert result.success
        assert isinstance(end_anchor, m21note.Note)
        assert end_measure is not None
        assert end_measure.number == 1
        assert end_measure.elementOffset(end_anchor) == 1.0
        assert end_anchor.quarterLength == 1.0

    def test_add_hairpin_same_bar_endpoint_on_note_start_is_noninclusive(
        self,
    ) -> None:
        """A same-measure endpoint on a note should stop before that note."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("E4", "quarter", measure=1, beat=2)
        ss._add_note_one("D4", "quarter", measure=1, beat=3)
        result = ss.add_hairpin(
            "crescendo", start_measure=1, start_beat=1,
            end_measure=1, end_beat=3,
        )
        part_obj = list(ss.score.parts)[0]
        wedges = list(part_obj.getElementsByClass(m21dynamics.DynamicWedge))
        end_anchor = wedges[-1].getSpannedElements()[-1]

        assert result.success
        assert isinstance(end_anchor, m21note.Note)
        assert end_anchor.nameWithOctave == "E4"

    def test_add_hairpin_same_bar_endpoint_inside_rest_uses_exact_anchor(
        self,
    ) -> None:
        """A same-bar endpoint inside a rest should fail without an event."""
        ss = _make_score(measures=1, time_signature="2/4")
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "eighth", measure=1, beat=2)
        measure = _get_measure(ss, 1)
        measure.insert(1.5, m21note.Rest(quarterLength=0.5))

        with pytest.raises(ValueError, match="No note, chord, or visible rest"):
            ss.add_hairpin(
                "crescendo",
                start_measure=1,
                start_beat=1,
                end_measure=1,
                end_beat=2.75,
            )

    def test_add_hairpin_dynamic_at_end_beat_exports_as_arrival(
        self,
    ) -> None:
        """A dynamic at the non-inclusive end beat should follow the wedge."""
        ss = _make_score(measures=1, time_signature="4/4")
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("E4", "quarter", measure=1, beat=2)
        ss._add_note_one("D4", "quarter", measure=1, beat=3)

        ss.add_hairpin(
            "crescendo",
            start_measure=1,
            start_beat=1,
            end_measure=1,
            end_beat=3,
        )
        ss.add_dynamic("f", measure_number=1, beat=3)

        musicxml = ss.to_musicxml_string()
        root = etree.fromstring(musicxml.encode("utf-8"))
        measure = root.xpath(".//*[local-name()='part'][1]/*[local-name()='measure']")[0]
        direction_events = []
        for child in measure:
            if etree.QName(child).localname != "direction":
                continue
            wedge = child.xpath(".//*[local-name()='wedge']")
            dynamics = child.xpath(".//*[local-name()='dynamics']")
            if wedge:
                direction_events.append(("wedge", wedge[0].get("type")))
            elif dynamics:
                direction_events.append(("dynamic", "f"))

        assert ("wedge", "crescendo") in direction_events
        assert ("wedge", "stop") in direction_events
        assert ("dynamic", "f") in direction_events
        assert direction_events.index(("wedge", "stop")) < direction_events.index(
            ("dynamic", "f")
        )

    def test_add_hairpin_rejects_missing_endpoint_event(self) -> None:
        """An endpoint without an exact event should fail before mutation."""
        ss = _make_score()
        with pytest.raises(ValueError, match="No note, chord, or visible rest"):
            ss.add_hairpin(
                "crescendo", start_measure=1, start_beat=1,
                end_measure=1, end_beat=4,
            )
        part_obj = list(ss.score.parts)[0]
        assert list(part_obj.getElementsByClass(m21dynamics.DynamicWedge)) == []

    def test_add_hairpin_to_specific_part(self):
        ss = _make_score(parts=["violin", "cello"])
        result = ss.add_hairpin(
            "crescendo", start_measure=1, start_beat=1,
            end_measure=2, end_beat=1, part=1,
        )
        assert result.success
        assert result.details["part"] == 1
        first_part = list(ss.score.parts)[0]
        second_part = list(ss.score.parts)[1]
        assert list(first_part.getElementsByClass(m21dynamics.DynamicWedge)) == []
        assert len(list(second_part.getElementsByClass(m21dynamics.DynamicWedge))) == 1

    def test_add_hairpin_without_part_targets_all_parts(self) -> None:
        """Omitting part should add a matching hairpin to every part."""
        ss = _make_score(parts=["violin", "viola", "cello"])

        result = ss.add_hairpin(
            "crescendo",
            start_measure=1,
            start_beat=1,
            end_measure=2,
            end_beat=1,
        )

        assert result.success
        assert result.details["parts"] == [0, 1, 2]
        assert result.details["added_parts"] == [0, 1, 2]
        assert result.details["skipped_parts"] == []
        for part_obj in ss.score.parts:
            wedges = list(part_obj.getElementsByClass(m21dynamics.DynamicWedge))
            assert len(wedges) == 1
            assert isinstance(wedges[0], m21dynamics.Crescendo)

    def test_add_hairpin_without_part_is_idempotent_for_matching_wedges(
        self: "TestAddHairpin",
    ) -> None:
        """An all-parts add should skip already matching hairpins."""
        ss = _make_score(parts=["violin", "cello"])
        ss.add_hairpin(
            "crescendo",
            start_measure=1,
            start_beat=1,
            end_measure=2,
            end_beat=1,
        )

        result = ss.add_hairpin(
            "crescendo",
            start_measure=1,
            start_beat=1,
            end_measure=2,
            end_beat=1,
        )

        assert result.success
        assert result.details["added_parts"] == []
        assert result.details["skipped_parts"] == [0, 1]
        for part_obj in ss.score.parts:
            wedges = list(part_obj.getElementsByClass(m21dynamics.DynamicWedge))
            assert len(wedges) == 1

    def test_add_hairpin_without_part_conflict_is_atomic(
        self: "TestAddHairpin",
    ) -> None:
        """A same-start conflicting wedge should prevent all insertions."""
        ss = _make_score(parts=["violin", "cello"])
        ss.add_hairpin(
            "diminuendo",
            start_measure=1,
            start_beat=1,
            end_measure=2,
            end_beat=1,
            part=1,
        )

        with pytest.raises(ValueError, match="part 1"):
            ss.add_hairpin(
                "crescendo",
                start_measure=1,
                start_beat=1,
                end_measure=2,
                end_beat=1,
            )

        first_part = list(ss.score.parts)[0]
        second_part = list(ss.score.parts)[1]
        assert list(first_part.getElementsByClass(m21dynamics.DynamicWedge)) == []
        wedges = list(second_part.getElementsByClass(m21dynamics.DynamicWedge))
        assert len(wedges) == 1
        assert isinstance(wedges[0], m21dynamics.Diminuendo)


class TestRemoveHairpin:
    """Tests for removing hairpins."""

    def test_remove_hairpin_basic(self):
        ss = _make_score()
        ss.add_hairpin(
            "crescendo", start_measure=1, start_beat=1,
            end_measure=2, end_beat=1,
        )
        result = ss.remove_hairpin(start_measure=1, start_beat=1)
        assert result.success
        part_obj = list(ss.score.parts)[0]
        wedges = list(part_obj.getElementsByClass(m21dynamics.DynamicWedge))
        assert len(wedges) == 0

    def test_remove_hairpin_returns_type(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=4)
        ss.add_hairpin(
            "diminuendo", start_measure=1, start_beat=1,
            end_measure=1, end_beat=4,
        )
        result = ss.remove_hairpin(start_measure=1, start_beat=1)
        assert result.details["type"] == "diminuendo"

    def test_remove_hairpin_not_found(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="No hairpin found"):
            ss.remove_hairpin(start_measure=1, start_beat=1)

    def test_remove_hairpin_explicit_part_not_found_is_strict(self) -> None:
        """Explicit part removal should still fail if that part has no wedge."""
        ss = _make_score(parts=["violin", "cello"])
        ss.add_hairpin(
            "crescendo",
            start_measure=1,
            start_beat=1,
            end_measure=2,
            end_beat=1,
            part=0,
        )

        with pytest.raises(ValueError, match="No hairpin found"):
            ss.remove_hairpin(start_measure=1, start_beat=1, part=1)

        first_part = list(ss.score.parts)[0]
        wedges = list(first_part.getElementsByClass(m21dynamics.DynamicWedge))
        assert len(wedges) == 1

    def test_remove_hairpin_without_part_targets_all_parts(self) -> None:
        """Omitting part should remove matching-start hairpins everywhere."""
        ss = _make_score(parts=["violin", "viola", "cello"])
        ss.add_hairpin(
            "crescendo",
            start_measure=1,
            start_beat=1,
            end_measure=2,
            end_beat=1,
        )

        result = ss.remove_hairpin(start_measure=1, start_beat=1)

        assert result.success
        assert result.details["parts"] == [0, 1, 2]
        assert result.details["removed_parts"] == [0, 1, 2]
        assert result.details["type"] == "crescendo"
        for part_obj in ss.score.parts:
            assert list(part_obj.getElementsByClass(m21dynamics.DynamicWedge)) == []

    def test_remove_hairpin_without_part_skips_missing_targets(
        self: "TestRemoveHairpin",
    ) -> None:
        """A missing hairpin in one part should be reported and skipped."""
        ss = _make_score(parts=["violin", "cello"])
        ss.add_hairpin(
            "crescendo",
            start_measure=1,
            start_beat=1,
            end_measure=2,
            end_beat=1,
            part=0,
        )

        result = ss.remove_hairpin(start_measure=1, start_beat=1)

        assert result.success
        assert result.details["parts"] == [0, 1]
        assert result.details["removed_parts"] == [0]
        assert result.details["skipped_parts"] == [1]
        first_part = list(ss.score.parts)[0]
        second_part = list(ss.score.parts)[1]
        assert list(first_part.getElementsByClass(m21dynamics.DynamicWedge)) == []
        assert list(second_part.getElementsByClass(m21dynamics.DynamicWedge)) == []

    def test_remove_hairpin_without_part_all_missing_fails(
        self: "TestRemoveHairpin",
    ) -> None:
        """Blanket hairpin removal should fail when no target part has a wedge."""
        ss = _make_score(parts=["violin", "cello"])

        with pytest.raises(ValueError, match="No hairpin found"):
            ss.remove_hairpin(start_measure=1, start_beat=1)

    def test_remove_hairpin_without_part_does_not_skip_existing_targets_on_fail(
        self: "TestRemoveHairpin",
    ) -> None:
        """Explicit invalid measure validation should still happen before mutation."""
        ss = _make_score(parts=["violin", "cello"], measures=1)
        ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
        ss._add_note_one("D4", "quarter", measure=1, beat=4, part=0)
        ss.add_hairpin(
            "crescendo",
            start_measure=1,
            start_beat=1,
            end_measure=1,
            end_beat=4,
            part=0,
        )

        with pytest.raises(ValueError, match="Measure 3 does not exist"):
            ss.remove_hairpin(start_measure=3, start_beat=1)

        first_part = list(ss.score.parts)[0]
        wedges = list(first_part.getElementsByClass(m21dynamics.DynamicWedge))
        assert len(wedges) == 1


# ==================================================================
# add_articulation / remove_articulation
# ==================================================================

class TestAddArticulation:
    """Tests for adding articulations."""

    def test_add_staccato(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        result = ss.add_articulation("staccato", measure_number=1, beat=1)
        assert result.success
        assert result.details["articulation"] == "staccato"

    def test_add_accent(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        result = ss.add_articulation("accent", measure_number=1, beat=1)
        assert result.success

    def test_add_marcato_alias(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        result = ss.add_articulation("marcato", measure_number=1, beat=1)
        assert result.success
        assert result.details["articulation"] == "marcato"
        note_obj = list(_get_measure(ss, 1).flatten().notes)[0]
        assert any(
            isinstance(articulation, m21articulations.StrongAccent)
            for articulation in note_obj.articulations
        )

    def test_add_tenuto(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        result = ss.add_articulation("tenuto", measure_number=1, beat=1)
        assert result.success

    def test_add_fermata(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        result = ss.add_articulation("fermata", measure_number=1, beat=1)
        assert result.success
        note_obj = list(_get_measure(ss, 1).flatten().notes)[0]
        assert not note_obj.articulations
        assert any(
            isinstance(expression, m21expressions.Fermata)
            for expression in note_obj.expressions
        )
        musicxml = ss.to_musicxml_string()
        assert '<fermata type="upright"' in musicxml
        assert '<fermata type="inverted"' not in musicxml

    def test_add_staccatissimo(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        result = ss.add_articulation("staccatissimo", measure_number=1, beat=1)
        assert result.success

    def test_add_caesura_at_measure_start_targets_requested_note(self):
        ss = _make_score(measures=3)
        ss._add_note_one("G4", "quarter", measure=2, beat=4)
        ss._add_note_one("A4", "quarter", measure=3, beat=1)

        result = ss.add_articulation("caesura", measure_number=3, beat=1)

        assert result.success
        assert result.details["measure"] == 3
        assert result.details["beat"] == 1
        previous_note = list(_get_measure(ss, 2).flatten().notes)[0]
        next_note = list(_get_measure(ss, 3).flatten().notes)[0]
        assert not any(
            isinstance(articulation, m21articulations.Caesura)
            for articulation in previous_note.articulations
        )
        assert any(
            isinstance(articulation, m21articulations.Caesura)
            for articulation in next_note.articulations
        )

    def test_add_articulation_with_enum(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        result = ss.add_articulation(
            ArticulationType.STACCATO, measure_number=1, beat=1,
        )
        assert result.success

    def test_add_articulation_to_chord(self):
        ss = _make_score()
        ss.add_chord(["C4", "E4", "G4"], "quarter", measure=1, beat=1)
        result = ss.add_articulation("accent", measure_number=1, beat=1)
        assert result.success

    def test_add_articulation_on_rest_fails(self):
        ss = _make_score()
        ss.reshape_rests(
            measure=1,
            part=0,
            voice=1,
            start_beat=1,
            total_duration="quarter",
            rests=[{"duration": "quarter"}],
        )
        with pytest.raises(ValueError, match="Cannot add a staccato"):
            ss.add_articulation(
                "staccato",
                measure_number=1,
                beat=1,
                part=0,
            )

    def test_add_articulation_no_note_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="No valid note or chord"):
            ss.add_articulation(
                "staccato",
                measure_number=1,
                beat=1,
            )

    def test_add_articulation_invalid_type_fails(self):
        ss = _make_score()
        ss._add_note_one("C4", measure=1, beat=1)
        with pytest.raises(ValueError, match="Unknown articulation"):
            ss.add_articulation("trill", measure_number=1, beat=1)

    def test_add_multiple_articulations_same_note(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss.add_articulation("staccato", measure_number=1, beat=1)
        ss.add_articulation("accent", measure_number=1, beat=1)
        m = _get_measure(ss, 1)
        notes = list(m.flatten().notes)
        assert len(notes[0].articulations) == 2

    def test_add_articulation_specific_voice(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1, voice=2)
        result = ss.add_articulation(
            "staccato", measure_number=1, beat=1, voice=2,
        )
        assert result.success
        assert result.details["voice"] == 2

    def test_add_articulation_rejects_unsupported_voice(self) -> None:
        """Articulation tools reject voice numbers outside the public range."""
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)

        with pytest.raises(ValueError, match="between 1 and 4"):
            ss.add_articulation("staccato", measure_number=1, beat=1, voice=5)

    def test_add_articulation_specific_part(self):
        ss = _make_score(parts=["violin", "cello"])
        ss._add_note_one("C4", "quarter", measure=1, beat=1, part=1)
        result = ss.add_articulation(
            "staccato", measure_number=1, beat=1, part=1,
        )
        assert result.success
        assert result.details["part"] == 1

    def test_add_articulation_without_part_targets_all_parts(self) -> None:
        """Omitted part applies the articulation to every matching part."""
        ss = _make_score(parts=["violin", "viola", "cello"])
        for part_idx in range(3):
            ss._add_note_one(
                "C4",
                "quarter",
                measure=1,
                beat=1,
                part=part_idx,
            )

        result = ss.add_articulation("staccato", measure_number=1, beat=1)

        assert result.success
        assert result.details["parts"] == [0, 1, 2]
        assert result.details["added_parts"] == [0, 1, 2]
        assert result.details["skipped_parts"] == []
        assert result.details["already_present_parts"] == []
        for part_idx in range(3):
            note_obj = list(
                _get_measure(ss, 1, part=part_idx).flatten().notes
            )[0]
            assert any(
                isinstance(articulation, m21articulations.Staccato)
                for articulation in note_obj.articulations
            )

    def test_add_articulation_without_part_reports_existing_and_skipped(
        self,
    ) -> None:
        """All-parts add reports changed, existing, and untargetable parts."""
        ss = _make_score(parts=["violin", "cello", "bass"])
        ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
        ss._add_note_one("G3", "quarter", measure=1, beat=1, part=1)
        ss.add_articulation("staccato", measure_number=1, beat=1, part=0)

        result = ss.add_articulation("staccato", measure_number=1, beat=1)

        assert result.success
        assert result.details["parts"] == [0, 1, 2]
        assert result.details["added_parts"] == [1]
        assert result.details["already_present_parts"] == [0]
        assert result.details["skipped_parts"] == [2]
        first_note = list(_get_measure(ss, 1, part=0).flatten().notes)[0]
        second_note = list(_get_measure(ss, 1, part=1).flatten().notes)[0]
        first_staccatos = [
            articulation
            for articulation in first_note.articulations
            if isinstance(articulation, m21articulations.Staccato)
        ]
        second_staccatos = [
            articulation
            for articulation in second_note.articulations
            if isinstance(articulation, m21articulations.Staccato)
        ]
        assert len(first_staccatos) == 1
        assert len(second_staccatos) == 1

    def test_add_articulation_without_part_all_missing_targets_fails(
        self,
    ) -> None:
        """All-parts add fails when no part has a valid target."""
        ss = _make_score(parts=["violin", "cello"])

        with pytest.raises(ValueError, match="No valid note or chord"):
            ss.add_articulation("staccato", measure_number=1, beat=1)

    def test_add_fermata_on_rest(self) -> None:
        """Fermatas can be attached to rests."""
        ss = _make_score()
        ss.reshape_rests(
            measure=1,
            part=0,
            voice=1,
            start_beat=1,
            total_duration="quarter",
            rests=[{"duration": "quarter"}],
        )

        result = ss.add_articulation("fermata", measure_number=1, beat=1)

        rests = [
            element
            for element in _get_measure(ss, 1).flatten().notesAndRests
            if isinstance(element, m21note.Rest)
        ]
        assert result.success
        assert result.details["added_parts"] == [0]
        assert any(
            isinstance(expression, m21expressions.Fermata)
            for expression in rests[0].expressions
        )
        musicxml = ss.to_musicxml_string()
        assert '<fermata type="upright"' in musicxml
        assert '<fermata type="inverted"' not in musicxml


class TestRemoveArticulation:
    """Tests for removing articulations."""

    def test_remove_staccato(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss.add_articulation("staccato", measure_number=1, beat=1)
        result = ss.remove_articulation("staccato", measure_number=1, beat=1)
        assert result.success

    def test_remove_articulation_leaves_others(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss.add_articulation("staccato", measure_number=1, beat=1)
        ss.add_articulation("accent", measure_number=1, beat=1)
        ss.remove_articulation("staccato", measure_number=1, beat=1)
        m = _get_measure(ss, 1)
        notes = list(m.flatten().notes)
        remaining = [type(a).__name__ for a in notes[0].articulations]
        assert "Staccato" not in remaining
        assert "Accent" in remaining

    def test_remove_fermata_from_expressions(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss.add_articulation("fermata", measure_number=1, beat=1)
        result = ss.remove_articulation("fermata", measure_number=1, beat=1)
        assert result.success
        note_obj = list(_get_measure(ss, 1).flatten().notes)[0]
        assert not any(
            isinstance(expression, m21expressions.Fermata)
            for expression in note_obj.expressions
        )
        assert "<fermata" not in ss.to_musicxml_string()

    def test_remove_fermata_cleans_legacy_articulation_container(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        note_obj = list(_get_measure(ss, 1).flatten().notes)[0]
        note_obj.articulations.append(m21expressions.Fermata())
        result = ss.remove_articulation("fermata", measure_number=1, beat=1)
        assert result.success
        assert not any(
            isinstance(articulation, m21expressions.Fermata)
            for articulation in note_obj.articulations
        )

    def test_remove_caesura_at_measure_start_targets_requested_note(self):
        ss = _make_score(measures=3)
        ss._add_note_one("G4", "quarter", measure=2, beat=4)
        ss._add_note_one("A4", "quarter", measure=3, beat=1)
        ss.add_articulation("caesura", measure_number=3, beat=1)

        result = ss.remove_articulation("caesura", measure_number=3, beat=1)

        assert result.success
        assert result.details["measure"] == 3
        assert result.details["beat"] == 1
        next_note = list(_get_measure(ss, 3).flatten().notes)[0]
        assert not any(
            isinstance(articulation, m21articulations.Caesura)
            for articulation in next_note.articulations
        )

    def test_remove_articulation_not_present_fails(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        with pytest.raises(ValueError, match="No staccato articulation"):
            ss.remove_articulation("staccato", measure_number=1, beat=1)

    def test_remove_articulation_no_note_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="No note or chord found"):
            ss.remove_articulation(
                "staccato",
                measure_number=1,
                beat=1,
                part=0,
            )

    def test_remove_articulation_without_part_targets_all_parts(self) -> None:
        """Omitted part removes the articulation from every matching part."""
        ss = _make_score(parts=["violin", "viola", "cello"])
        for part_idx in range(3):
            ss._add_note_one(
                "C4",
                "quarter",
                measure=1,
                beat=1,
                part=part_idx,
            )
            ss.add_articulation(
                "staccato",
                measure_number=1,
                beat=1,
                part=part_idx,
            )

        result = ss.remove_articulation("staccato", measure_number=1, beat=1)

        assert result.success
        assert result.details["parts"] == [0, 1, 2]
        assert result.details["removed_parts"] == [0, 1, 2]
        assert result.details["skipped_parts"] == []
        assert result.details["missing_parts"] == []
        for part_idx in range(3):
            note_obj = list(
                _get_measure(ss, 1, part=part_idx).flatten().notes
            )[0]
            assert not any(
                isinstance(articulation, m21articulations.Staccato)
                for articulation in note_obj.articulations
            )

    def test_remove_articulation_without_part_reports_missing_and_skipped(
        self,
    ) -> None:
        """All-parts remove reports changed, missing, and untargetable parts."""
        ss = _make_score(parts=["violin", "cello", "bass"])
        ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
        ss._add_note_one("G3", "quarter", measure=1, beat=1, part=1)
        ss.add_articulation("staccato", measure_number=1, beat=1, part=0)

        result = ss.remove_articulation("staccato", measure_number=1, beat=1)

        assert result.success
        assert result.details["parts"] == [0, 1, 2]
        assert result.details["removed_parts"] == [0]
        assert result.details["missing_parts"] == [1]
        assert result.details["skipped_parts"] == [2]

    def test_remove_articulation_without_part_all_missing_fails(self) -> None:
        """All-parts remove fails when no part has the requested marking."""
        ss = _make_score(parts=["violin", "cello"])
        ss._add_note_one("C4", "quarter", measure=1, beat=1, part=0)
        ss._add_note_one("G3", "quarter", measure=1, beat=1, part=1)

        with pytest.raises(ValueError, match="No staccato articulation"):
            ss.remove_articulation("staccato", measure_number=1, beat=1)

    def test_remove_fermata_from_rest(self) -> None:
        """Fermatas can be removed from rests."""
        ss = _make_score()
        ss.reshape_rests(
            measure=1,
            part=0,
            voice=1,
            start_beat=1,
            total_duration="quarter",
            rests=[{"duration": "quarter"}],
        )
        ss.add_articulation("fermata", measure_number=1, beat=1)

        result = ss.remove_articulation("fermata", measure_number=1, beat=1)

        rests = [
            element
            for element in _get_measure(ss, 1).flatten().notesAndRests
            if isinstance(element, m21note.Rest)
        ]
        assert result.success
        assert result.details["removed_parts"] == [0]
        assert not any(
            isinstance(expression, m21expressions.Fermata)
            for expression in rests[0].expressions
        )
        assert "<fermata" not in ss.to_musicxml_string()


# ==================================================================
# add_slur / remove_slur
# ==================================================================

class TestAddSlur:
    """Tests for adding slurs."""

    def test_add_slur_basic(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=2)
        result = ss.add_slur(
            start_measure=1, start_beat=1,
            end_measure=1, end_beat=2,
        )
        assert result.success
        assert result.details["start_measure"] == 1
        assert result.details["end_measure"] == 1

    def test_add_slur_across_measures(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=2, beat=1)
        result = ss.add_slur(
            start_measure=1, start_beat=1,
            end_measure=2, end_beat=1,
        )
        assert result.success

    def test_add_slur_creates_spanner_in_part(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=2)
        ss.add_slur(
            start_measure=1, start_beat=1,
            end_measure=1, end_beat=2,
        )
        part_obj = list(ss.score.parts)[0]
        slurs = list(part_obj.getElementsByClass(m21spanner.Slur))
        assert len(slurs) == 1

    def test_add_slur_no_start_note_fails(self):
        ss = _make_score()
        ss._add_note_one("D4", "quarter", measure=1, beat=2)
        with pytest.raises(ValueError, match="No note found.*slur start"):
            ss.add_slur(
                start_measure=1, start_beat=1,
                end_measure=1, end_beat=2,
            )

    def test_add_slur_no_end_note_fails(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        with pytest.raises(ValueError, match="No note found.*slur end"):
            ss.add_slur(
                start_measure=1, start_beat=1,
                end_measure=1, end_beat=2,
            )

    def test_add_slur_on_rest_start_fails(self):
        ss = _make_score()
        ss.reshape_rests(
            measure=1,
            part=0,
            voice=1,
            start_beat=1,
            total_duration="quarter",
            rests=[{"duration": "quarter"}],
        )
        ss._add_note_one("D4", "quarter", measure=1, beat=2)
        with pytest.raises(ValueError, match="No note found.*slur start"):
            ss.add_slur(
                start_measure=1, start_beat=1,
                end_measure=1, end_beat=2,
            )

    def test_add_slur_on_rest_end_fails(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss.reshape_rests(
            measure=1,
            part=0,
            voice=1,
            start_beat=2,
            total_duration="quarter",
            rests=[{"duration": "quarter"}],
        )
        with pytest.raises(ValueError, match="No note found.*slur end"):
            ss.add_slur(
                start_measure=1, start_beat=1,
                end_measure=1, end_beat=2,
            )

    def test_add_slur_specific_part(self):
        ss = _make_score(parts=["violin", "cello"])
        ss._add_note_one("C4", "quarter", measure=1, beat=1, part=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=2, part=1)
        result = ss.add_slur(
            start_measure=1, start_beat=1,
            end_measure=1, end_beat=2, part=1,
        )
        assert result.success
        assert result.details["part"] == 1


class TestRemoveSlur:
    """Tests for removing slurs."""

    def test_remove_slur_basic(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=2)
        ss.add_slur(
            start_measure=1, start_beat=1,
            end_measure=1, end_beat=2,
        )
        result = ss.remove_slur(start_measure=1, start_beat=1)
        assert result.success
        part_obj = list(ss.score.parts)[0]
        slurs = list(part_obj.getElementsByClass(m21spanner.Slur))
        assert len(slurs) == 0

    def test_remove_slur_not_found(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="No slur found"):
            ss.remove_slur(start_measure=1, start_beat=1)


# ==================================================================
# add_text_expression
# ==================================================================

class TestAddTextExpression:
    """Tests for adding text expressions."""

    def test_add_text_expression_basic(self):
        ss = _make_score()
        result = ss.add_text_expression("dolce", measure_number=1, beat=1)
        assert result.success
        assert result.details["text"] == "dolce"

    def test_add_text_expression_con_brio(self):
        ss = _make_score()
        result = ss.add_text_expression("con brio", measure_number=2, beat=1)
        assert result.success
        assert result.details["text"] == "con brio"

    def test_add_text_expression_rit(self):
        ss = _make_score()
        result = ss.add_text_expression("rit.", measure_number=3, beat=1)
        assert result.success
        assert result.details["text"] == "rit."

    def test_text_expression_placed_in_measure(self):
        ss = _make_score()
        ss.add_text_expression("dolce", measure_number=1, beat=1)
        m = _get_measure(ss, 1)
        tes = list(m.getElementsByClass(m21expressions.TextExpression))
        assert len(tes) == 1
        assert tes[0].content == "dolce"

    def test_add_text_expression_at_specific_beat(self):
        ss = _make_score()
        result = ss.add_text_expression("cresc.", measure_number=1, beat=3)
        assert result.success
        assert result.details["beat"] == 3.0
        m = _get_measure(ss, 1)
        tes = list(m.getElementsByClass(m21expressions.TextExpression))
        assert abs(m.elementOffset(tes[0]) - 2.0) < 1e-9

    def test_add_text_expression_empty_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="cannot be empty"):
            ss.add_text_expression("", measure_number=1)

    def test_add_text_expression_whitespace_only_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="cannot be empty"):
            ss.add_text_expression("   ", measure_number=1)

    def test_add_text_expression_strips_whitespace(self):
        ss = _make_score()
        result = ss.add_text_expression("  dolce  ", measure_number=1)
        assert result.details["text"] == "dolce"

    def test_add_text_expression_normalizes_doubled_abbreviation_period(
        self: "TestAddTextExpression",
    ) -> None:
        """Common dotted abbreviations should not keep accidental double dots."""
        ss = _make_score()

        result = ss.add_text_expression("cresc..", measure_number=1, beat=1)

        assert result.details["text"] == "cresc."
        measure = _get_measure(ss, 1)
        expressions = list(
            measure.getElementsByClass(m21expressions.TextExpression)
        )
        assert [expression.content for expression in expressions] == ["cresc."]

    def test_add_text_expression_rejects_duplicate_same_position(
        self: "TestAddTextExpression",
    ) -> None:
        """A repeated same-text expression at the same beat should fail."""
        ss = _make_score()
        ss.add_text_expression("cresc.", measure_number=1, beat=1)

        with pytest.raises(ValueError, match="already exists"):
            ss.add_text_expression("cresc..", measure_number=1, beat=1)

        measure = _get_measure(ss, 1)
        expressions = list(
            measure.getElementsByClass(m21expressions.TextExpression)
        )
        assert [expression.content for expression in expressions] == ["cresc."]


class TestRemoveTextExpression:
    """Tests for removing text expressions."""

    def test_remove_text_expression_basic(self):
        ss = _make_score()
        ss.add_text_expression("dolce", measure_number=1, beat=1)
        result = ss.remove_text_expression(measure_number=1, beat=1)
        assert result.success
        assert result.details["text"] == "dolce"
        m = _get_measure(ss, 1)
        tes = list(m.getElementsByClass(m21expressions.TextExpression))
        assert len(tes) == 0

    def test_remove_text_expression_by_text(self):
        ss = _make_score()
        ss.add_text_expression("dolce", measure_number=1, beat=1)
        ss.add_text_expression("cantabile", measure_number=1, beat=1)
        result = ss.remove_text_expression(
            measure_number=1,
            beat=1,
            text="cantabile",
        )
        assert result.success
        m = _get_measure(ss, 1)
        tes = list(m.getElementsByClass(m21expressions.TextExpression))
        assert [te.content for te in tes] == ["dolce"]

    def test_remove_text_expression_wrong_text_fails(self):
        ss = _make_score()
        ss.add_text_expression("dolce", measure_number=1, beat=1)
        with pytest.raises(ValueError, match="No text expression"):
            ss.remove_text_expression(measure_number=1, beat=1, text="rit.")

    def test_remove_text_expression_missing_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="No text expression"):
            ss.remove_text_expression(measure_number=1, beat=1)


# ==================================================================
# set_tempo
# ==================================================================

class TestSetTempo:
    """Tests for setting tempo."""

    def test_set_tempo_basic(self):
        ss = _make_score()
        result = ss.set_tempo(120, measure_number=1, beat=1)
        assert result.success
        assert result.details["bpm"] == 120

    def test_set_tempo_with_text(self):
        ss = _make_score()
        result = ss.set_tempo(132, measure_number=1, text="Allegro")
        assert result.success
        assert result.details["text"] == "Allegro"
        assert "Allegro" in result.description

    def test_set_tempo_default_referent_exports_quarter(self) -> None:
        """Tempo without a referent keeps the existing quarter-note behavior."""
        ss = _make_score()
        result = ss.set_tempo(120, measure_number=1)

        musicxml = ss.to_musicxml_string()

        assert result.details["referent"] == "quarter"
        assert "quarter = 120" in result.description
        assert "<beat-unit>quarter</beat-unit>" in musicxml
        assert "<beat-unit-dot" not in musicxml

    @pytest.mark.parametrize(
        ("referent", "beat_unit", "has_dot", "normalized_referent"),
        [
            ("half", "half", False, "half"),
            ("half note", "half", False, "half"),
            ("dotted half", "half", True, "dotted half"),
            ("dotted half note", "half", True, "dotted half"),
            ("dotted quarter", "quarter", True, "dotted quarter"),
            ("dotted quarter note", "quarter", True, "dotted quarter"),
            ("sixteenth", "16th", False, "16th"),
        ],
    )
    def test_set_tempo_referent_exports_beat_unit(
        self,
        referent: str,
        beat_unit: str,
        has_dot: bool,
        normalized_referent: str,
    ) -> None:
        """Tempo referents export MusicXML beat-unit values and dots."""
        ss = _make_score()
        result = ss.set_tempo(120, measure_number=1, referent=referent)

        musicxml = ss.to_musicxml_string()

        assert result.details["referent"] == normalized_referent
        assert f"<beat-unit>{beat_unit}</beat-unit>" in musicxml
        if has_dot:
            assert "<beat-unit-dot" in musicxml
        else:
            assert "<beat-unit-dot" not in musicxml

    def test_set_tempo_invalid_referent_fails(self) -> None:
        """Unsupported tempo referents fail with a clear validation error."""
        ss = _make_score()

        with pytest.raises(ValueError, match="Unsupported tempo referent"):
            ss.set_tempo(120, measure_number=1, referent="dotted eighth")

    def test_set_tempo_replaces_existing(self):
        ss = _make_score()
        ss.set_tempo(100, measure_number=1, beat=1)
        result = ss.set_tempo(140, measure_number=1, beat=1)
        assert result.success
        assert result.details["replaced"] is True
        m = _get_measure(ss, 1)
        tempos = list(m.getElementsByClass(m21tempo.MetronomeMark))
        bpms = [t.number for t in tempos]
        assert 140 in bpms
        assert 100 not in bpms

    def test_set_tempo_new_position_not_replaced(self):
        ss = _make_score()
        result = ss.set_tempo(120, measure_number=2, beat=1)
        assert result.success
        assert result.details["replaced"] is False

    def test_set_tempo_at_beat(self):
        ss = _make_score()
        result = ss.set_tempo(80, measure_number=1, beat=3)
        assert result.success
        assert result.details["beat"] == 3.0

    def test_set_tempo_zero_bpm_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="positive"):
            ss.set_tempo(0, measure_number=1)

    def test_set_tempo_negative_bpm_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="positive"):
            ss.set_tempo(-60, measure_number=1)

    def test_set_tempo_placed_in_measure(self):
        ss = _make_score()
        ss.set_tempo(96, measure_number=2, beat=1)
        m = _get_measure(ss, 2)
        tempos = list(m.getElementsByClass(m21tempo.MetronomeMark))
        found = [t for t in tempos if t.number == 96]
        assert len(found) >= 1


# ==================================================================
# add_rehearsal_mark
# ==================================================================

class TestAddRehearsalMark:
    """Tests for adding rehearsal marks."""

    def test_add_rehearsal_mark_basic(self):
        ss = _make_score()
        result = ss.add_rehearsal_mark("A", measure_number=1)
        assert result.success
        assert result.details["text"] == "A"

    def test_add_rehearsal_mark_B(self):
        ss = _make_score()
        result = ss.add_rehearsal_mark("B", measure_number=2)
        assert result.success
        assert result.details["measure"] == 2

    def test_rehearsal_mark_placed_in_measure(self):
        ss = _make_score()
        ss.add_rehearsal_mark("A", measure_number=1)
        m = _get_measure(ss, 1)
        rms = list(m.getElementsByClass(m21expressions.RehearsalMark))
        assert len(rms) == 1

    def test_rehearsal_mark_at_offset_zero(self):
        ss = _make_score()
        ss.add_rehearsal_mark("C", measure_number=3)
        m = _get_measure(ss, 3)
        rms = list(m.getElementsByClass(m21expressions.RehearsalMark))
        assert abs(m.elementOffset(rms[0])) < 1e-9

    def test_add_rehearsal_mark_empty_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="cannot be empty"):
            ss.add_rehearsal_mark("", measure_number=1)

    def test_add_rehearsal_mark_whitespace_only_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="cannot be empty"):
            ss.add_rehearsal_mark("   ", measure_number=1)

    def test_add_rehearsal_mark_anchors_in_first_part_only(self):
        ss = _make_score(parts=["violin", "cello"])
        result = ss.add_rehearsal_mark("A", measure_number=1)
        assert result.success
        assert "part" not in result.details
        first_part, _ = ss._resolve_part(0)
        second_part, _ = ss._resolve_part(1)
        first_measure = ss._resolve_measure(first_part, 1)
        second_measure = ss._resolve_measure(second_part, 1)
        assert (
            len(list(first_measure.getElementsByClass(m21expressions.RehearsalMark)))
            == 1
        )
        assert (
            len(list(second_measure.getElementsByClass(m21expressions.RehearsalMark)))
            == 0
        )

    def test_add_rehearsal_mark_no_longer_accepts_part_argument(self):
        ss = _make_score(parts=["violin", "cello"])
        with pytest.raises(TypeError):
            ss.add_rehearsal_mark("A", measure_number=1, part=1)


class TestRemoveRehearsalMark:
    """Tests for removing rehearsal marks."""

    def test_remove_rehearsal_mark_basic(self):
        ss = _make_score()
        ss.add_rehearsal_mark("A", measure_number=1)
        result = ss.remove_rehearsal_mark(measure_number=1)
        assert result.success
        assert result.details["text"] == "A"
        m = _get_measure(ss, 1)
        rms = list(m.getElementsByClass(m21expressions.RehearsalMark))
        assert len(rms) == 0

    def test_remove_rehearsal_mark_by_text(self):
        ss = _make_score()
        ss.add_rehearsal_mark("A", measure_number=1)
        ss.add_rehearsal_mark("B", measure_number=1)
        result = ss.remove_rehearsal_mark(measure_number=1, text="B")
        assert result.success
        m = _get_measure(ss, 1)
        rms = list(m.getElementsByClass(m21expressions.RehearsalMark))
        assert [rm.content for rm in rms] == ["A"]

    def test_remove_rehearsal_mark_cleans_legacy_part_scoped_marks(self):
        ss = _make_score(parts=["violin", "cello"])
        first_part, _ = ss._resolve_part(0)
        second_part, _ = ss._resolve_part(1)
        first_measure = ss._resolve_measure(first_part, 1)
        second_measure = ss._resolve_measure(second_part, 1)
        first_measure.insert(0, m21expressions.RehearsalMark("A"))
        second_measure.insert(0, m21expressions.RehearsalMark("A"))

        result = ss.remove_rehearsal_mark(measure_number=1)

        assert result.success
        assert result.details["removed_count"] == 2
        assert "part" not in result.details
        assert (
            len(list(first_measure.getElementsByClass(m21expressions.RehearsalMark)))
            == 0
        )
        assert (
            len(list(second_measure.getElementsByClass(m21expressions.RehearsalMark)))
            == 0
        )

    def test_remove_rehearsal_mark_no_longer_accepts_part_argument(self):
        ss = _make_score(parts=["violin", "cello"])
        ss.add_rehearsal_mark("A", measure_number=1)
        with pytest.raises(TypeError):
            ss.remove_rehearsal_mark(measure_number=1, part=1)

    def test_remove_rehearsal_mark_missing_fails(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="No rehearsal mark"):
            ss.remove_rehearsal_mark(measure_number=1)


# ==================================================================
# Integration tests
# ==================================================================

class TestExpressionsIntegration:
    """End-to-end scenarios combining multiple expression operations."""

    def test_dynamic_then_hairpin_then_dynamic(self):
        ss = _make_score()
        ss.add_dynamic("p", measure_number=1, beat=1)
        ss.add_hairpin(
            "crescendo", start_measure=1, start_beat=1,
            end_measure=2, end_beat=1,
        )
        ss.add_dynamic("f", measure_number=2, beat=1)

        m1 = _get_measure(ss, 1)
        m2 = _get_measure(ss, 2)
        dyns1 = list(m1.getElementsByClass(m21dynamics.Dynamic))
        dyns2 = list(m2.getElementsByClass(m21dynamics.Dynamic))
        assert any(d.value == "p" for d in dyns1)
        assert any(d.value == "f" for d in dyns2)

    def test_note_with_staccato_and_dynamic(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss.add_articulation("staccato", measure_number=1, beat=1)
        ss.add_dynamic("ff", measure_number=1, beat=1)

        m = _get_measure(ss, 1)
        notes = list(m.flatten().notes)
        assert len(notes) == 1
        assert any(isinstance(a, m21articulations.Staccato) for a in notes[0].articulations)
        dyns = list(m.getElementsByClass(m21dynamics.Dynamic))
        assert len(dyns) == 1

    def test_slur_with_articulations(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss._add_note_one("D4", "quarter", measure=1, beat=2)
        ss._add_note_one("E4", "quarter", measure=1, beat=3)
        ss.add_slur(
            start_measure=1, start_beat=1,
            end_measure=1, end_beat=3,
        )
        ss.add_articulation("staccato", measure_number=1, beat=1)
        ss.add_articulation("accent", measure_number=1, beat=3)

        part_obj = list(ss.score.parts)[0]
        slurs = list(part_obj.getElementsByClass(m21spanner.Slur))
        assert len(slurs) == 1

    def test_tempo_and_rehearsal_mark_at_same_measure(self):
        ss = _make_score()
        ss.set_tempo(144, measure_number=3, text="Vivace")
        ss.add_rehearsal_mark("B", measure_number=3)
        m = _get_measure(ss, 3)
        tempos = list(m.getElementsByClass(m21tempo.MetronomeMark))
        rms = list(m.getElementsByClass(m21expressions.RehearsalMark))
        assert len(tempos) >= 1
        assert len(rms) == 1

    def test_text_expression_and_dynamic_same_beat(self):
        ss = _make_score()
        ss.add_text_expression("dolce", measure_number=1, beat=1)
        ss.add_dynamic("pp", measure_number=1, beat=1)
        m = _get_measure(ss, 1)
        tes = list(m.getElementsByClass(m21expressions.TextExpression))
        dyns = list(m.getElementsByClass(m21dynamics.Dynamic))
        assert len(tes) == 1
        assert len(dyns) == 1

    def test_add_remove_add_dynamic(self):
        ss = _make_score()
        ss.add_dynamic("p", measure_number=1, beat=1)
        ss.remove_dynamic(measure_number=1, beat=1)
        ss.add_dynamic("ff", measure_number=1, beat=1)
        m = _get_measure(ss, 1)
        dyns = list(m.getElementsByClass(m21dynamics.Dynamic))
        assert len(dyns) == 1
        assert dyns[0].value == "ff"

    def test_add_remove_add_articulation(self):
        ss = _make_score()
        ss._add_note_one("C4", "quarter", measure=1, beat=1)
        ss.add_articulation("staccato", measure_number=1, beat=1)
        ss.remove_articulation("staccato", measure_number=1, beat=1)
        ss.add_articulation("tenuto", measure_number=1, beat=1)
        m = _get_measure(ss, 1)
        notes = list(m.flatten().notes)
        art_types = [type(a).__name__ for a in notes[0].articulations]
        assert "Staccato" not in art_types
        assert "Tenuto" in art_types

    def test_all_valid_dynamic_levels(self):
        """Every valid dynamic string should be accepted."""
        from scorespeak.types import VALID_DYNAMICS
        ss = _make_score(measures=len(VALID_DYNAMICS))
        for i, level in enumerate(VALID_DYNAMICS):
            result = ss.add_dynamic(level, measure_number=i + 1, beat=1)
            assert result.success, f"Failed to add dynamic '{level}'"

# ==================================================================
# add_chord_symbol / remove_chord_symbol / get_chord_symbols
# ==================================================================


class TestAddChordSymbol:
    """Tests for ExpressionsMixin.add_chord_symbol (Req 6.1–6.6)."""

    def test_add_chord_symbol_basic(self):
        ss = _make_score()
        result = ss.add_chord_symbol("C", measure_number=1)
        assert result.success
        assert result.details["chord_symbol"] == "C"
        assert result.details["measure"] == 1
        assert result.details["beat"] == 1.0

    def test_add_chord_symbol_minor(self):
        ss = _make_score()
        result = ss.add_chord_symbol("Cm", measure_number=1)
        assert result.success
        assert result.details["chord_symbol"] == "Cm"

    def test_add_chord_symbol_seventh(self):
        ss = _make_score()
        result = ss.add_chord_symbol("C7", measure_number=2)
        assert result.success
        assert result.details["measure"] == 2

    def test_add_chord_symbol_maj7(self):
        ss = _make_score()
        result = ss.add_chord_symbol("Cmaj7", measure_number=1)
        assert result.success

    def test_add_chord_symbol_sharp_minor7(self):
        ss = _make_score()
        result = ss.add_chord_symbol("F#m7", measure_number=1)
        assert result.success
        assert result.details["chord_symbol"] == "F#m7"

    def test_add_chord_symbol_flat_root(self):
        ss = _make_score()
        result = ss.add_chord_symbol("B-", measure_number=1)
        assert result.success
        assert result.details["chord_symbol"] == "B-"

    def test_add_chord_symbol_at_specific_beat(self):
        ss = _make_score()
        result = ss.add_chord_symbol("Am", measure_number=1, beat=3.0)
        assert result.success
        assert result.details["beat"] == 3.0

    def test_add_chord_symbol_invalid_string(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="Cannot parse chord symbol"):
            ss.add_chord_symbol("ZZZNOTACHORD!!!", measure_number=1)

    def test_add_chord_symbol_beat_below_one(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="at least 1.0"):
            ss.add_chord_symbol("C", measure_number=1, beat=0.5)

    def test_add_chord_symbol_beat_beyond_measure(self):
        ss = _make_score()  # 4/4 time
        with pytest.raises(ValueError, match="beyond the end"):
            ss.add_chord_symbol("C", measure_number=1, beat=6.0)

    def test_add_chord_symbol_invalid_measure(self):
        ss = _make_score(measures=4)
        with pytest.raises(ValueError):
            ss.add_chord_symbol("C", measure_number=99)

    def test_add_chord_symbol_specific_part(self):
        ss = _make_score(parts=["Piano", "Bass"])
        result = ss.add_chord_symbol("G7", measure_number=1, part=1)
        assert result.success
        assert result.details["part"] == 1

    def test_add_chord_symbol_placed_in_measure(self):
        """Verify the music21 ChordSymbol object is actually in the measure."""
        from music21 import harmony as m21harmony
        ss = _make_score()
        ss.add_chord_symbol("Dm7", measure_number=2, beat=1.0)
        m = _get_measure(ss, 2)
        chords = list(m.getElementsByClass(m21harmony.ChordSymbol))
        assert len(chords) == 1
        assert chords[0].figure == "Dm7"


class TestRemoveChordSymbol:
    """Tests for ExpressionsMixin.remove_chord_symbol (Req 7.1–7.3)."""

    def test_remove_chord_symbol_basic(self):
        ss = _make_score()
        ss.add_chord_symbol("C", measure_number=1, beat=1.0)
        result = ss.remove_chord_symbol(measure_number=1, beat=1.0)
        assert result.success
        assert result.details["measure"] == 1
        assert result.details["beat"] == 1.0

    def test_remove_chord_symbol_actually_removes(self):
        from music21 import harmony as m21harmony
        ss = _make_score()
        ss.add_chord_symbol("Am", measure_number=1, beat=1.0)
        ss.remove_chord_symbol(measure_number=1, beat=1.0)
        m = _get_measure(ss, 1)
        chords = list(m.getElementsByClass(m21harmony.ChordSymbol))
        assert len(chords) == 0

    def test_remove_chord_symbol_not_found(self):
        ss = _make_score()
        with pytest.raises(ValueError, match="No chord symbol found"):
            ss.remove_chord_symbol(measure_number=1, beat=1.0)

    def test_remove_chord_symbol_wrong_beat(self):
        ss = _make_score()
        ss.add_chord_symbol("C", measure_number=1, beat=1.0)
        with pytest.raises(ValueError, match="No chord symbol found"):
            ss.remove_chord_symbol(measure_number=1, beat=3.0)

    def test_remove_one_of_multiple_chord_symbols(self):
        from music21 import harmony as m21harmony
        ss = _make_score()
        ss.add_chord_symbol("C", measure_number=1, beat=1.0)
        ss.add_chord_symbol("G7", measure_number=1, beat=3.0)
        ss.remove_chord_symbol(measure_number=1, beat=1.0)
        m = _get_measure(ss, 1)
        chords = list(m.getElementsByClass(m21harmony.ChordSymbol))
        assert len(chords) == 1
        assert chords[0].figure == "G7"


class TestGetChordSymbols:
    """Tests for ExpressionsMixin.get_chord_symbols (Req 8.1–8.4)."""

    def test_get_chord_symbols_empty(self):
        ss = _make_score()
        result = ss.get_chord_symbols()
        assert result == []

    def test_get_chord_symbols_single(self):
        ss = _make_score()
        ss.add_chord_symbol("C", measure_number=1, beat=1.0)
        result = ss.get_chord_symbols()
        assert len(result) == 1
        assert result[0]["symbol"] == "C"
        assert result[0]["measure_number"] == 1
        assert result[0]["beat"] == 1.0
        assert result[0]["part_index"] == 0

    def test_get_chord_symbols_multiple_measures(self):
        ss = _make_score()
        ss.add_chord_symbol("C", measure_number=1, beat=1.0)
        ss.add_chord_symbol("F", measure_number=2, beat=1.0)
        ss.add_chord_symbol("G7", measure_number=3, beat=1.0)
        result = ss.get_chord_symbols()
        assert len(result) == 3
        symbols = [r["symbol"] for r in result]
        assert "C" in symbols
        assert "F" in symbols
        assert "G7" in symbols

    def test_get_chord_symbols_filter_by_measure(self):
        ss = _make_score()
        ss.add_chord_symbol("C", measure_number=1, beat=1.0)
        ss.add_chord_symbol("Am", measure_number=2, beat=1.0)
        result = ss.get_chord_symbols(measure_number=1)
        assert len(result) == 1
        assert result[0]["symbol"] == "C"

    def test_get_chord_symbols_filter_by_measure_empty(self):
        ss = _make_score()
        ss.add_chord_symbol("C", measure_number=1, beat=1.0)
        result = ss.get_chord_symbols(measure_number=3)
        assert result == []

    def test_get_chord_symbols_filter_by_part(self):
        ss = _make_score(parts=["Piano", "Bass"])
        ss.add_chord_symbol("C", measure_number=1, part=0)
        ss.add_chord_symbol("G", measure_number=1, part=1)
        result = ss.get_chord_symbols(part=1)
        assert len(result) == 1
        assert result[0]["symbol"] == "G"
        assert result[0]["part_index"] == 1

    def test_get_chord_symbols_multiple_same_measure(self):
        ss = _make_score()
        ss.add_chord_symbol("C", measure_number=1, beat=1.0)
        ss.add_chord_symbol("G7", measure_number=1, beat=3.0)
        result = ss.get_chord_symbols(measure_number=1)
        assert len(result) == 2
        beats = sorted(r["beat"] for r in result)
        assert beats == [1.0, 3.0]
