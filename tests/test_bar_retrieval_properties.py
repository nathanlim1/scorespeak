"""Property-based tests for score bar retrieval."""

from hypothesis import given, settings
from hypothesis import strategies as st

from scorespeak import ScoreSpeak


def _make_score(measures: int, parts: int) -> ScoreSpeak:
    """Create a score with deterministic part names."""
    part_specs = [
        {"instrument": "violin", "name": "Violin"},
        {"instrument": "cello", "name": "Cello"},
        {"instrument": "flute", "name": "Flute"},
    ]
    return ScoreSpeak.create(
        measures=measures,
        time_signature="4/4",
        parts=part_specs[:parts],
    )


# Feature: score-bar-retrieval, Property 1: No-query returns all in-scope bars
@settings(max_examples=100)
@given(
    measures=st.integers(min_value=1, max_value=6),
    parts=st.integers(min_value=1, max_value=3),
)
def test_bar_result_set_no_query_returns_sorted_unique_measures(measures, parts):
    """No-query retrieval returns every measure exactly once in order."""
    ss = _make_score(measures, parts)

    result = ss._build_bar_result_set()
    result_measures = [bar["measure_number"] for bar in result["bars"]]

    assert result_measures == list(range(1, measures + 1))


# Feature: score-bar-retrieval, Property 2: Bar range stays within scope
@settings(max_examples=100)
@given(
    measures=st.integers(min_value=1, max_value=6),
    start=st.integers(min_value=1, max_value=8),
    end=st.integers(min_value=1, max_value=8),
)
def test_bar_result_set_bar_range_clamps_to_available_measures(measures, start, end):
    """Returned measures stay within the clamped bar range."""
    ss = _make_score(measures, 2)

    low = min(start, end)
    high = max(start, end)
    result = ss._build_bar_result_set({"scope": {"bar_range": (low, high)}})

    expected_start = max(1, low)
    expected_end = min(measures, high)
    if expected_end < expected_start:
        assert result["bars"] == []
    else:
        assert [bar["measure_number"] for bar in result["bars"]] == list(
            range(expected_start, expected_end + 1)
        )


# Feature: score-bar-retrieval, Property 3: Part entries stay sorted
@settings(max_examples=100)
@given(parts=st.integers(min_value=1, max_value=3))
def test_bar_result_set_part_entries_are_sorted(parts):
    """Returned part payloads are ordered by ascending part index."""
    ss = _make_score(2, parts)
    result = ss._build_bar_result_set()

    for bar in result["bars"]:
        indices = [part["part_index"] for part in bar["parts"]]
        assert indices == sorted(indices)
