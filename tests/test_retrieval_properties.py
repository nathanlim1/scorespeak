"""Property-based tests for the ScoreSpeak Retrieval System."""

import inspect

from hypothesis import given, settings
from hypothesis import strategies as st

from scorespeak import ScoreSpeak
from scorespeak.retrieval import (
    LexicalContextRetriever,
    MethodIndex,
    PUBLIC_TOOL_EXCLUDED_NAMES,
    extract_lexical_context_scope,
)

# Build the index once at module level.
ss = ScoreSpeak.create()
index = MethodIndex(ss)


# Feature: scorespeak-retrieval-system, Property 1: Index contains only public methods
# **Validates: Requirements 1.1**
@settings(max_examples=100)
@given(data=st.data())
def test_index_only_public_methods(data):
    """Every name in the index is public (no leading underscore), and every
    public method on the ScoreSpeak instance appears in the index."""
    index_names = {r.name for r in index.records}

    # No record should start with '_'
    for name in index_names:
        assert not name.startswith("_"), f"Index contains private method: {name}"

    # Every public tool method on the instance must be in the index
    members = inspect.getmembers(ss, predicate=inspect.ismethod)
    public_on_instance = {
        n
        for n, _ in members
        if not n.startswith("_") and n not in PUBLIC_TOOL_EXCLUDED_NAMES
    }
    assert index_names == public_on_instance, (
        f"Mismatch between index and instance public methods.\n"
        f"  In index but not on instance: {index_names - public_on_instance}\n"
        f"  On instance but not in index: {public_on_instance - index_names}"
    )


# Feature: scorespeak-retrieval-system, Property 2: Every MethodRecord has all required fields
# **Validates: Requirements 1.2**
@settings(max_examples=100)
@given(record=st.sampled_from(index.records))
def test_every_record_has_required_fields(record):
    """name, mixin, signature, and docstring are non-None strings;
    tags is a non-empty frozenset."""
    assert isinstance(record.name, str) and record.name is not None
    assert isinstance(record.mixin, str) and record.mixin is not None
    assert isinstance(record.signature, str) and record.signature is not None
    assert isinstance(record.docstring, str) and record.docstring is not None
    assert isinstance(record.tags, frozenset)
    assert len(record.tags) > 0, f"Record '{record.name}' has empty tags"


# Feature: scorespeak-retrieval-system, Property 3: Tags are exactly the underscore-split name tokens
# **Validates: Requirements 1.3**
@settings(max_examples=100)
@given(record=st.sampled_from(index.records))
def test_tags_equal_name_split(record):
    """record.tags must equal frozenset(record.name.split('_'))."""
    expected = frozenset(record.name.split("_"))
    assert record.tags == expected, (
        f"Tags mismatch for '{record.name}': {record.tags} != {expected}"
    )

import re

from scorespeak.retrieval import LexicalRetriever, ResultFormatter, SYNONYM_MAP, _VERB_ORDER

# Module-level retriever for tests that share a single threshold.
retriever = LexicalRetriever(index, threshold=0.5)

# Module-level formatter for Property 12.
formatter = ResultFormatter()

_VERBS = set(_VERB_ORDER)


def _make_context_score(measures: int, parts: int) -> ScoreSpeak:
    """Create a score with deterministic part names for context tests."""
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


def _compute_score(record, query_text):
    """Re-implement the scoring logic independently for property verification.

    Returns 0.0 if no non-verb (domain) token overlaps — matching the
    domain-token requirement in LexicalRetriever.query().
    """
    tokens = re.sub(r"[^\w\s]", "", query_text.lower()).split()
    if not tokens:
        return 0.0
    expanded: set[str] = set()
    for tok in tokens:
        expanded.add(tok)
        if tok in SYNONYM_MAP:
            expanded.update(SYNONYM_MAP[tok])
    tags = record.tags
    overlap = expanded & tags
    # Require at least one non-verb tag to match.
    if not (overlap - _VERBS):
        return 0.0
    return len(overlap) / len(tags)


# ---------------------------------------------------------------------------
# Property 4: Querying does not mutate the index
# ---------------------------------------------------------------------------

# Feature: scorespeak-retrieval-system, Property 4: Querying does not mutate the index
# **Validates: Requirements 1.5**
@settings(max_examples=100)
@given(query=st.text(min_size=1, max_size=50))
def test_query_does_not_mutate_index(query):
    """For any sequence of queries, the list of MethodRecord objects must be
    identical before and after."""
    records_before = list(index.records)
    retriever.query(query)
    records_after = list(index.records)
    assert len(records_before) == len(records_after)
    for rb, ra in zip(records_before, records_after):
        assert rb.name == ra.name
        assert rb.mixin == ra.mixin
        assert rb.tags == ra.tags


# ---------------------------------------------------------------------------
# Property 5+6: Threshold invariant
# ---------------------------------------------------------------------------

# Feature: scorespeak-retrieval-system, Property 5+6: Threshold invariant (no result below threshold; all above-threshold returned)
# **Validates: Requirements 2.7, 5.3, 5.4**
@settings(max_examples=100)
@given(
    query=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Zs", "Nd")),
        min_size=1,
        max_size=30,
    ),
    threshold=st.floats(0.01, 0.99),
)
def test_threshold_invariant(query, threshold):
    """Every result must have score >= threshold, AND every record whose
    independently computed score >= threshold must be in results."""
    ret = LexicalRetriever(index, threshold=threshold)
    results = ret.query(query)

    # Property 5: no result below threshold
    for record, score in results:
        assert score >= threshold, (
            f"{record.name} returned with score {score} < threshold {threshold}"
        )

    # Property 6: all above-threshold records are returned
    expected_names = set()
    for r in index.records:
        if _compute_score(r, query) >= threshold:
            expected_names.add(r.name)
    result_names = {r.name for r, _ in results}
    assert result_names == expected_names


# ---------------------------------------------------------------------------
# Property 7: Results are sorted descending by score
# ---------------------------------------------------------------------------

# Feature: scorespeak-retrieval-system, Property 7: Results are sorted descending by score
# **Validates: Requirements 2.2**
@settings(max_examples=100)
@given(query=st.text(min_size=1, max_size=50))
def test_results_sorted_descending(query):
    """The list returned by query() must be in non-increasing score order."""
    results = retriever.query(query)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Property 8: Direct term round-trip
# ---------------------------------------------------------------------------

# Collect all unique non-verb tokens from all method names for sampling.
# Verb-only tokens won't retrieve anything due to the domain-token requirement.
all_tokens = sorted(
    {tok for r in index.records for tok in r.name.split("_")
     if tok not in _VERBS}
)

# Feature: scorespeak-retrieval-system, Property 8: Direct term round-trip
# **Validates: Requirements 2.3, 5.1**
# Use a threshold low enough that a single-token match always passes.
# The worst case is 1/max_tag_count; we compute it from the index.
_max_tag_count = max(len(r.tags) for r in index.records)
_round_trip_threshold = 1.0 / _max_tag_count - 0.01  # just below the minimum single-token score

@settings(max_examples=100)
@given(token=st.sampled_from(all_tokens))
def test_direct_term_round_trip(token):
    """Querying a single method-name token with a low threshold must include
    every method whose name contains that token."""
    low_ret = LexicalRetriever(index, threshold=_round_trip_threshold)
    results = low_ret.query(token)
    result_names = {r.name for r, _ in results}
    for record in index.records:
        if token in record.tags:
            assert record.name in result_names, (
                f"Token '{token}' should retrieve '{record.name}' but didn't"
            )


# ---------------------------------------------------------------------------
# Property 9: Case normalization is idempotent
# ---------------------------------------------------------------------------

# Feature: scorespeak-retrieval-system, Property 9: Case normalization is idempotent
# **Validates: Requirements 2.4**
@settings(max_examples=100)
@given(
    query=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Zs")),
        min_size=1,
        max_size=30,
    )
)
def test_case_normalization_idempotent(query):
    """query(q) and query(q.upper()) must return the same method names."""
    r1 = {r.name for r, _ in retriever.query(query)}
    r2 = {r.name for r, _ in retriever.query(query.upper())}
    assert r1 == r2


# ---------------------------------------------------------------------------
# Property 10: Synonym expansion is a superset of canonical term results
# ---------------------------------------------------------------------------

# Build (synonym, canonical_token) pairs from SYNONYM_MAP.
_synonym_canonical_pairs = []
for syn, canonical_list in SYNONYM_MAP.items():
    for canon in canonical_list:
        _synonym_canonical_pairs.append((syn, canon))

# Feature: scorespeak-retrieval-system, Property 10: Synonym expansion is a superset of canonical term results
# **Validates: Requirements 2.5, 5.2**
@settings(max_examples=100)
@given(pair=st.sampled_from(_synonym_canonical_pairs))
def test_synonym_superset_of_canonical(pair):
    """Querying a synonym must return a result set that is a superset of
    querying the canonical token directly."""
    synonym, canonical = pair
    synonym_names = {r.name for r, _ in retriever.query(synonym)}
    canonical_names = {r.name for r, _ in retriever.query(canonical)}
    assert canonical_names.issubset(synonym_names), (
        f"Querying synonym '{synonym}' missed methods from canonical '{canonical}': "
        f"{canonical_names - synonym_names}"
    )


# ---------------------------------------------------------------------------
# Property 11: Lower threshold yields more or equal results
# ---------------------------------------------------------------------------

# Feature: scorespeak-retrieval-system, Property 11: Lower threshold yields more or equal results
# **Validates: Requirements 5.7**
@settings(max_examples=100)
@given(
    query=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Zs")),
        min_size=1,
        max_size=30,
    ),
    t1=st.floats(0.01, 0.49),
    t2=st.floats(0.50, 0.99),
)
def test_lower_threshold_more_results(query, t1, t2):
    """With t1 < t2, the number of results at t1 must be >= the number at t2."""
    r1 = LexicalRetriever(index, threshold=t1).query(query)
    r2 = LexicalRetriever(index, threshold=t2).query(query)
    assert len(r1) >= len(r2)


# ---------------------------------------------------------------------------
# Property 13: add/remove discrimination
# ---------------------------------------------------------------------------

# Build (add_X, remove_X) pairs that both exist in the index.
_index_names = {r.name for r in index.records}
_add_remove_pairs = []
for name in sorted(_index_names):
    if name.startswith("add_"):
        domain = name[4:]  # everything after "add_"
        remove_name = f"remove_{domain}"
        if remove_name in _index_names:
            _add_remove_pairs.append((name, remove_name, domain))

# Feature: scorespeak-retrieval-system, Property 13: add/remove discrimination
# **Validates: Requirements 5.5**
@settings(max_examples=100)
@given(pair=st.sampled_from(_add_remove_pairs))
def test_add_remove_discrimination(pair):
    """Querying 'remove X' must score remove_X higher than add_X."""
    add_name, remove_name, domain = pair
    # Use spaces instead of underscores so multi-word domains tokenize correctly
    # (e.g. "ending_bracket" → "ending bracket" → tokens ["ending", "bracket"])
    query_domain = domain.replace("_", " ")
    results = retriever.query(f"remove {query_domain}")
    scores_by_name = {r.name: s for r, s in results}
    # remove_X must be in results
    assert remove_name in scores_by_name, (
        f"'remove {query_domain}' did not return {remove_name}"
    )
    # If add_X is also returned, remove_X must score strictly higher
    if add_name in scores_by_name:
        assert scores_by_name[remove_name] > scores_by_name[add_name], (
            f"remove_X ({scores_by_name[remove_name]}) should score higher "
            f"than add_X ({scores_by_name[add_name]}) for query 'remove {query_domain}'"
        )


# ---------------------------------------------------------------------------
# Property 14: Verb synonym expansion produces equivalent results
# ---------------------------------------------------------------------------

VERB_SYNONYM_PAIRS = [
    ("create", "add"), ("make", "add"), ("put", "add"), ("write", "add"),
    ("attach", "add"), ("apply", "add"), ("append", "add"),
    ("delete", "remove"), ("erase", "remove"), ("clear", "remove"),
    ("take", "remove"), ("strip", "remove"), ("drop", "remove"),
    ("change", "set"), ("update", "set"), ("configure", "set"),
    ("show", "get"), ("fetch", "get"), ("find", "get"), ("display", "list"),
    ("swap", "replace"), ("substitute", "replace"), ("overwrite", "replace"),
]

# Domain tokens drawn from actual method name tokens (non-verb portions).
domain_tokens = sorted(
    {
        tok
        for r in index.records
        for tok in r.name.split("_")
        if tok not in ("add", "insert", "remove", "set", "get", "list", "replace")
    }
)

# Feature: scorespeak-retrieval-system, Property 14: Verb synonym expansion produces equivalent results
# **Validates: Requirements 2.5, 5.2**
@settings(max_examples=100)
@given(
    verb_pair=st.sampled_from(VERB_SYNONYM_PAIRS),
    domain_token=st.sampled_from(domain_tokens),
)
def test_verb_synonym_equivalence(verb_pair, domain_token):
    """Querying with a verb synonym plus a domain token must return a superset
    of the results from querying with the canonical verb plus the same domain
    token."""
    synonym_verb, canonical_verb = verb_pair
    synonym_results = {r.name for r, _ in retriever.query(f"{synonym_verb} {domain_token}")}
    canonical_results = {r.name for r, _ in retriever.query(f"{canonical_verb} {domain_token}")}
    assert canonical_results.issubset(synonym_results), (
        f"'{synonym_verb} {domain_token}' missed methods from "
        f"'{canonical_verb} {domain_token}': {canonical_results - synonym_results}"
    )


# ---------------------------------------------------------------------------
# Property 12: Formatter output contains required fields
# ---------------------------------------------------------------------------

# Feature: scorespeak-retrieval-system, Property 12: Formatter output contains required fields
# **Validates: Requirements 3.1, 3.2**
@settings(max_examples=100)
@given(records=st.lists(st.sampled_from(index.records), min_size=1, max_size=5))
def test_formatter_contains_required_fields(records):
    """For any non-empty list of (MethodRecord, float) pairs, the formatted
    output must contain each record's name, mixin, and signature as substrings,
    and must contain position labels (#1, #2, …) in ascending order."""
    results = [(r, 1.0) for r in records]
    output = formatter.format(results)
    for i, (record, _) in enumerate(results, start=1):
        assert record.name in output, (
            f"Record name '{record.name}' not found in formatter output"
        )
        assert record.mixin in output, (
            f"Record mixin '{record.mixin}' not found in formatter output"
        )
        assert record.signature in output, (
            f"Record signature '{record.signature}' not found in formatter output"
        )
        assert f"#{i}" in output, (
            f"Position label '#{i}' not found in formatter output"
        )


# ---------------------------------------------------------------------------
# Lexical context retrieval properties
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    start=st.integers(min_value=1, max_value=12),
    end=st.integers(min_value=1, max_value=12),
)
def test_context_scope_normalizes_bar_ranges(start, end):
    """Bar-range extraction preserves an inclusive ascending range."""
    ss = _make_context_score(measures=12, parts=2)

    scope = extract_lexical_context_scope(
        ss,
        f"edit bars {start} through {end} in violin",
    )

    assert scope.bar_range == (min(start, end), max(start, end))


@settings(max_examples=100)
@given(
    start_one=st.integers(min_value=1, max_value=4),
    end_one=st.integers(min_value=1, max_value=4),
    start_two=st.integers(min_value=7, max_value=10),
    end_two=st.integers(min_value=7, max_value=10),
)
def test_context_scope_extracts_disconnected_measure_numbers(
    start_one,
    end_one,
    start_two,
    end_two,
):
    """Disconnected mentions become sorted explicit measure numbers."""
    ss = _make_context_score(measures=12, parts=2)

    first_low = min(start_one, end_one)
    first_high = max(start_one, end_one)
    second_low = min(start_two, end_two)
    second_high = max(start_two, end_two)

    scope = extract_lexical_context_scope(
        ss,
        f"edit violin bars {first_low}-{first_high} and {second_low}-{second_high}",
    )

    expected = list(range(first_low, first_high + 1)) + list(
        range(second_low, second_high + 1)
    )
    assert scope.measure_numbers == expected
    assert scope.bar_range is None


@settings(max_examples=100)
@given(
    measures=st.integers(min_value=1, max_value=6),
    parts=st.integers(min_value=1, max_value=3),
)
def test_context_retriever_without_scope_returns_no_bars(measures, parts):
    """Generic queries without explicit scope retrieve no automatic bars."""
    ss = _make_context_score(measures=measures, parts=parts)
    context_retriever = LexicalContextRetriever(ss, threshold=0.5)

    result = context_retriever.query("add a dynamic")

    assert result.scope.bar_context_status == "missing"
    assert result.context_query is None
    assert result.context_bars["bars"] == []


@settings(max_examples=100)
@given(
    measures=st.integers(min_value=1, max_value=6),
    parts=st.integers(min_value=1, max_value=3),
)
def test_context_retriever_end_scope_uses_last_measure(measures, parts):
    """End-oriented missing-bar queries retrieve the score's final bar."""
    ss = _make_context_score(measures=measures, parts=parts)
    context_retriever = LexicalContextRetriever(ss, threshold=0.5)

    result = context_retriever.query("append a dynamic")

    assert result.scope.bar_context_status == "end_fallback"
    assert [bar["measure_number"] for bar in result.context_bars["bars"]] == [measures]
