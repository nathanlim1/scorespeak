"""Property-based tests for the Notation Extensions feature."""

from hypothesis import given, settings
from hypothesis import strategies as st

from music21 import repeat as m21repeat

from scorespeak import ScoreSpeak


# ---------------------------------------------------------------------------
# Property 1: Navigation mark add/remove round-trip
# ---------------------------------------------------------------------------

# Feature: notation-extensions, Property 1: Navigation mark add/remove round-trip
# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 5.1, 5.2, 5.4**

@settings(max_examples=100)
@given(
    mark_type=st.sampled_from(["coda", "segno"]),
    measure=st.integers(min_value=1, max_value=4),
)
def test_nav_mark_add_remove_round_trip(mark_type, measure):
    """Adding then removing a navigation mark leaves no mark in the measure."""
    ss = ScoreSpeak.create(measures=4)
    add_fn = getattr(ss, f"add_{mark_type}")
    result = add_fn(measure)
    assert result.success
    assert result.details["mark_type"] == mark_type
    assert result.details["measure"] == measure

    remove_result = ss.remove_navigation_mark(mark_type, measure)
    assert remove_result.success

    # Verify no mark remains
    part_obj = list(ss.score.parts)[0]
    m = None
    for meas in part_obj.getElementsByClass("Measure"):
        if meas.number == measure:
            m = meas
            break
    target_cls = m21repeat.Coda if mark_type == "coda" else m21repeat.Segno
    remaining = [el for el in m.recurse() if isinstance(el, target_cls)]
    assert len(remaining) == 0


# ---------------------------------------------------------------------------
# Property 2: Da Capo / Dal Segno al-parameter class routing
# ---------------------------------------------------------------------------

# Feature: notation-extensions, Property 2: Da Capo / Dal Segno al-parameter class routing
# **Validates: Requirements 3.2–3.6, 4.2–4.6**

_DC_EXPECTED = {
    None: m21repeat.DaCapo,
    "fine": m21repeat.DaCapoAlFine,
    "coda": m21repeat.DaCapoAlCoda,
}
_DS_EXPECTED = {
    None: m21repeat.DalSegno,
    "fine": m21repeat.DalSegnoAlFine,
    "coda": m21repeat.DalSegnoAlCoda,
}

@settings(max_examples=100)
@given(
    direction=st.sampled_from(["da_capo", "dal_segno"]),
    al=st.sampled_from([None, "fine", "coda"]),
)
def test_dc_ds_al_parameter_class_routing(direction, al):
    """The correct music21 class is inserted based on direction + al combo."""
    ss = ScoreSpeak.create(measures=4)
    add_fn = getattr(ss, f"add_{direction}")
    result = add_fn(measure_number=2, al=al)
    assert result.success
    assert result.details["al"] == al

    expected_map = _DC_EXPECTED if direction == "da_capo" else _DS_EXPECTED
    expected_cls = expected_map[al]

    part_obj = list(ss.score.parts)[0]
    m = None
    for meas in part_obj.getElementsByClass("Measure"):
        if meas.number == 2:
            m = meas
            break
    found = [el for el in m.recurse() if isinstance(el, expected_cls)]
    assert len(found) >= 1, f"Expected {expected_cls.__name__} in measure 2"
