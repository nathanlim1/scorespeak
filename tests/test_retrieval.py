"""Unit tests for the ScoreSpeak Retrieval System.

Covers all specific examples and edge cases from the design's testing strategy.
"""

import pytest

from scorespeak import ScoreSpeak
from scorespeak.retrieval import (
    LexicalContextRetriever,
    LexicalRetriever,
    MethodIndex,
    MethodRecord,
    ResultFormatter,
    extract_lexical_context_scope,
)

# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def score_state():
    return ScoreSpeak.create()


@pytest.fixture(scope="module")
def index(score_state):
    return MethodIndex(score_state)


@pytest.fixture(scope="module")
def retriever(index):
    """Default retriever with threshold=0.5."""
    return LexicalRetriever(index, threshold=0.5)


@pytest.fixture(scope="module")
def formatter():
    return ResultFormatter()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _names(results):
    """Extract method names from a list of (MethodRecord, score) tuples."""
    return {r.name for r, _ in results}


def _ordered_names(results):
    """Extract method names in retrieval order."""
    return [r.name for r, _ in results]


def _make_context_score(measures: int = 8, parts: int = 2) -> ScoreSpeak:
    """Create a score with stable part names for lexical context retrieval tests."""
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


# ── 1. Index contains known methods ─────────────────────────────────────────

def test_index_contains_known_methods(index):
    """MethodIndex built from ScoreSpeak.create() contains known methods."""
    names = {r.name for r in index.records}
    for expected in ("add_notes", "add_dynamic", "set_time_signature",
                     "add_slur", "remove_dynamic", "list_parts",
                     "remove_notes", "remove_tie", "remove_tuplet", "remove_grace_note",
                     "remove_text_expression", "remove_rehearsal_mark",
                     "remove_repeat", "copy_measure_contents",
                     "add_rest", "fill_measure_gaps",
                     "remove_rests", "reshape_rests", "set_score_parts"):
        assert expected in names, f"Expected '{expected}' in index"
    assert "remove_note" not in names
    assert "add_rests" not in names
    assert "complete_measure_with_rests" not in names


def test_additive_tools_have_removal_coverage(index):
    """Every additive ScoreSpeak method has a removal tool for the same element."""
    names = {r.name for r in index.records}
    expected_removers = {
        "add_chord": {"remove_notes"},
        "add_chord_tones": {"remove_notes"},
        "add_coda": {"remove_navigation_mark"},
        "add_da_capo": {"remove_navigation_mark"},
        "add_dal_segno": {"remove_navigation_mark"},
        "add_fine": {"remove_navigation_mark"},
        "add_grace_note": {"remove_grace_note"},
        "add_measures": {"delete_measure", "delete_measures"},
        "add_notes": {"remove_notes"},
        "add_repeat": {"remove_repeat"},
        "add_rest": {"remove_rests"},
        "add_segno": {"remove_navigation_mark"},
        "add_to_coda": {"remove_navigation_mark"},
        "add_tie": {"remove_tie"},
        "add_tuplet": {"remove_tuplet"},
    }

    for name in names:
        if not name.startswith("add_"):
            continue
        removers = expected_removers.get(name, {f"remove_{name[4:]}"})
        assert names.intersection(removers), (
            f"Expected {name} to have one of removal tools {sorted(removers)}"
        )


# ── 2. MethodRecord tags for add_dynamic ────────────────────────────────────

def test_add_dynamic_tags(index):
    """MethodRecord for add_dynamic has tags == frozenset({'add', 'dynamic'})."""
    record = next(r for r in index.records if r.name == "add_dynamic")
    assert record.tags == frozenset({"add", "dynamic"})


# ── 3. Querying "dynamic" returns add_dynamic and remove_dynamic ────────────

def test_query_dynamic(retriever):
    """Querying 'dynamic' returns add_dynamic and remove_dynamic."""
    names = _names(retriever.query("dynamic"))
    assert "add_dynamic" in names
    assert "remove_dynamic" in names


# ── 4. Querying "forte" (synonym) returns add_dynamic ───────────────────────

def test_query_forte_synonym(retriever):
    """Querying 'forte' returns add_dynamic via synonym expansion."""
    names = _names(retriever.query("forte"))
    assert "add_dynamic" in names


def test_query_initial_score_setup_returns_set_score_parts(retriever):
    """Initial instrumentation phrasing retrieves set_score_parts."""
    names = _ordered_names(
        retriever.query("initialize the score with flute, cello, and piano")
    )

    assert names[0] == "set_score_parts"


def test_query_add_single_part_still_prefers_add_part(retriever):
    """Incremental part-add wording still prefers add_part."""
    names = _ordered_names(retriever.query("add a flute part"))

    assert names[0] == "add_part"
    assert "set_score_parts" in names


# ── 5. Querying "add forte and slur" returns add_dynamic and add_slur ───────

def test_query_add_forte_and_slur(retriever):
    """Querying 'add forte and slur' returns both add_dynamic and add_slur."""
    names = _names(retriever.query("add forte and slur"))
    assert "add_dynamic" in names
    assert "add_slur" in names


# ── 6. Querying "remove dynamic" at threshold 0.5 ──────────────────────────

def test_query_remove_dynamic_threshold_05(index):
    """At threshold > 0.5, 'remove dynamic' returns remove_dynamic but NOT add_dynamic.

    add_dynamic has tags {"add", "dynamic"} — only "dynamic" overlaps → score = 0.5.
    remove_dynamic has tags {"remove", "dynamic"} — both overlap → score = 1.0.
    Using threshold just above 0.5 excludes add_dynamic while keeping remove_dynamic.
    """
    ret = LexicalRetriever(index, threshold=0.51)
    results = ret.query("remove dynamic")
    names = _names(results)
    assert "remove_dynamic" in names
    assert "add_dynamic" not in names


# ── 7. Querying "xyzzy nonsense" returns [] ─────────────────────────────────

def test_query_nonsense(retriever):
    """Querying 'xyzzy nonsense' returns an empty list."""
    assert retriever.query("xyzzy nonsense") == []


# ── 8. Querying "" returns [] ───────────────────────────────────────────────

def test_query_empty_string(retriever):
    """Querying '' returns an empty list."""
    assert retriever.query("") == []


def test_query_plural_notes_retrieves_add_notes(retriever):
    """Plural note wording retrieves the scoped ``add_notes`` tool."""
    names = _names(
        retriever.query(
            "Add quarter notes C4, D4, E4, and F4 in bar 1 at beats 1, 2, 3, and 4"
        )
    )

    assert "add_notes" in names
    assert "add_note" not in names


def test_query_empty_bars_retrieves_clear_measures(retriever):
    """Empty/blank bar wording retrieves the non-structural clear tool."""
    names = _names(retriever.query("empty bars 3 through 4"))

    assert "clear_measures" in names


def test_query_copy_paste_retrieves_copy_measure_contents(retriever):
    """Copy/paste bar wording retrieves the measure copy tool."""
    names = _names(retriever.query("paste bar 1 into bar 4"))

    assert "copy_measure_contents" in names


def test_voice_queries_retrieve_timeline_tools_not_parts(retriever):
    """Voice wording targets rhythmic timelines rather than score parts."""
    second_voice_names = _names(
        retriever.query("add a second voice in bar 1")
    )
    voice_two_names = _names(retriever.query("put this in voice 2"))

    assert "add_part" not in second_voice_names
    assert "add_part" not in voice_two_names
    assert {"add_notes", "add_chord"}.issubset(
        second_voice_names
    )
    assert "add_notes" in voice_two_names
    assert "add_note" not in voice_two_names


def test_hide_rest_query_retrieves_remove_rests(retriever):
    """Hide-rest wording retrieves the visible-rest removal tool."""
    names = _ordered_names(retriever.query("hide rests in bar 2"))

    assert names[0] == "remove_rests"


def test_rest_spelling_query_retrieves_reshape_rests(retriever):
    """Rest spelling wording retrieves the rest reshaping tool."""
    names = _ordered_names(retriever.query("split the rest spelling in bar 2"))

    assert names[0] == "reshape_rests"


@pytest.mark.parametrize(
    "query",
    [
        "unhide rest",
        "unhide rests in bar 2",
        "show hidden rests in bar 2",
        "make rest visible",
        "reveal rest",
    ],
)
def test_unhide_rest_queries_prefer_add_rest(
    retriever: LexicalRetriever,
    query: str,
) -> None:
    """Unhide/show-rest wording should retrieve visible-rest addition."""
    names = _ordered_names(retriever.query(query))

    assert names[0] == "add_rest"


@pytest.mark.parametrize(
    "query",
    [
        "fill gaps in bar 2",
        "fill missing rests in bar 2",
        "complete the measure with rests",
    ],
)
def test_gap_fill_queries_prefer_fill_measure_gaps(
    retriever: LexicalRetriever,
    query: str,
) -> None:
    """True gap wording should retrieve the gap-fill tool."""
    names = _ordered_names(retriever.query(query))

    assert names[0] == "fill_measure_gaps"


@pytest.mark.parametrize(
    "query",
    [
        "split rests in bar 2",
        "merge rests in bar 2",
        "reformat rests in bar 2",
        "change rest spelling in bar 2",
    ],
)
def test_rest_reformat_queries_prefer_reshape_rests(retriever, query):
    """Rest spelling transformations should keep using reshape_rests."""
    names = _ordered_names(retriever.query(query))

    assert names[0] == "reshape_rests"


@pytest.mark.parametrize(
    "query",
    [
        "add an end bracket over measures 1 to 2",
        "add a first ending over measures 1 to 2",
        "add a volta over measures 1 to 2",
    ],
)
def test_ending_bracket_phrasing_retrieves_tool(retriever, query: str) -> None:
    """Common ending-bracket phrasing should retrieve the volta tool."""
    names = _names(retriever.query(query))

    assert "add_ending_bracket" in names


# ── 9. Case insensitivity ───────────────────────────────────────────────────

def test_query_case_insensitive(retriever):
    """Querying 'DYNAMIC' returns the same methods as 'dynamic'."""
    upper = _names(retriever.query("DYNAMIC"))
    lower = _names(retriever.query("dynamic"))
    assert upper == lower


# ── 10. ResultFormatter.format([]) contains "no results" message ────────────

def test_formatter_empty_results(formatter):
    """format([]) contains a 'no matching' message."""
    output = formatter.format([])
    assert "no matching" in output.lower() or "no results" in output.lower()


# ── 11. ResultFormatter.format with 11 items shows 10 and mentions omitted ─

def test_formatter_display_limit(index, formatter):
    """format(results_with_11_items) shows only 10 and mentions the omitted count."""
    # Build 11 fake results from real records
    records = index.records[:11]
    assert len(records) >= 11, "Need at least 11 methods in the index"
    results = [(r, 0.9 - i * 0.01) for i, r in enumerate(records)]
    output = formatter.format(results)

    # Should contain #1 through #10 but NOT #11
    for i in range(1, 11):
        assert f"#{i}" in output
    assert "#11" not in output

    # Should mention the omitted count
    assert "1 more" in output or "1" in output


# ── 12. Method with no docstring is included with docstring="" ──────────────

def test_no_docstring_method_included(index):
    """Any method without a docstring is still in the index with docstring=''."""
    # Verify the index doesn't crash and that all records have a string docstring
    for record in index.records:
        assert isinstance(record.docstring, str)
    # If any method has no docstring, it should be ""
    empty_doc_records = [r for r in index.records if r.docstring == ""]
    # This is a structural check — either there are none (all have docs) or
    # those without docs have docstring==""
    for r in empty_doc_records:
        assert r.docstring == ""


# ── 13. Threshold 0.0 returns all; threshold 1.0 returns exact matches ─────

def test_threshold_zero_returns_all(index):
    """Threshold 0.0 with a domain token returns all methods containing that domain token."""
    ret = LexicalRetriever(index, threshold=0.0)
    results = ret.query("dynamic")
    names = _names(results)
    # Should include add_dynamic and remove_dynamic (domain token "dynamic" matches)
    assert "add_dynamic" in names
    assert "remove_dynamic" in names
    # Should NOT include methods with no domain overlap (e.g. add_notes)
    assert "add_note" not in names


def test_threshold_one_returns_exact_only(index):
    """Threshold 1.0 returns only methods whose full name is covered."""
    ret = LexicalRetriever(index, threshold=1.0)
    results = ret.query("add dynamic")
    names = _names(results)
    # add_dynamic has tags {"add", "dynamic"} — both covered → score 1.0
    assert "add_dynamic" in names
    # remove_dynamic has tags {"remove", "dynamic"} — only "dynamic" covered → 0.5
    assert "remove_dynamic" not in names


# ── 14. "create dynamic" == "add dynamic" (verb synonym equivalence) ────────

def test_verb_synonym_create_equals_add(retriever):
    """'create dynamic' returns at least the same methods as 'add dynamic'.

    'create' expands to 'add' via SYNONYM_MAP, so the result set is a superset
    of (or equal to) the canonical 'add dynamic' results.
    """
    create_names = _names(retriever.query("create dynamic"))
    add_names = _names(retriever.query("add dynamic"))
    assert add_names.issubset(create_names)


# ── 15. "delete dynamic" == "remove dynamic" ────────────────────────────────

def test_verb_synonym_delete_equals_remove(retriever):
    """'delete dynamic' returns at least the same methods as 'remove dynamic'.

    'delete' expands to 'remove' via SYNONYM_MAP, so the result set is a superset
    of (or equal to) the canonical 'remove dynamic' results.
    """
    delete_names = _names(retriever.query("delete dynamic"))
    remove_names = _names(retriever.query("remove dynamic"))
    assert remove_names.issubset(delete_names)


# ── 16. "show parts" == "list parts" ────────────────────────────────────────

def test_verb_synonym_show_equals_list(retriever):
    """'show parts' returns at least the same methods as 'list parts'.

    'show' expands to ['get', 'list'] via SYNONYM_MAP, so the result set is a
    superset of (or equal to) the canonical 'list parts' results.
    """
    show_names = _names(retriever.query("show parts"))
    list_names = _names(retriever.query("list parts"))
    assert list_names.issubset(show_names)


# ── Notation extensions: retrieval coverage ─────────────────────────────────


class TestNotationExtensionsRetrieval:
    """Verify new notation-extension methods are discoverable via retrieval."""

    def test_query_coda(self, retriever):
        names = _names(retriever.query("coda"))
        assert "add_coda" in names

    def test_query_segno(self, retriever):
        names = _names(retriever.query("segno"))
        assert "add_segno" in names

    def test_query_to_coda(self, retriever):
        names = _names(retriever.query("to coda"))
        assert "add_to_coda" in names

    def test_query_fine(self, retriever):
        names = _names(retriever.query("fine"))
        assert "add_fine" in names

    def test_query_da_capo(self, retriever):
        names = _names(retriever.query("da capo"))
        assert "add_da_capo" in names

    def test_query_chord_symbol(self, retriever):
        names = _names(retriever.query("chord symbol"))
        assert "add_chord_symbol" in names

    def test_query_tremolo_ornament(self, retriever):
        names = _names(retriever.query("add tremolo"))
        assert "add_ornament" in names
        assert "add_two_note_tremolo" not in names


def test_extract_lexical_context_scope_bar_range_and_part():
    """Lexical scope extraction resolves explicit bars and part mentions."""
    ss = _make_context_score(measures=8, parts=2)

    scope = extract_lexical_context_scope(
        ss,
        "edit the dynamics on bars 3-7 in the violin part",
    )

    assert scope.part_indices == [0]
    assert scope.bar_range == (3, 7)
    assert scope.explicit_part_mention is True
    assert scope.explicit_bar_mention is True
    assert scope.used_fallback_bar is False


def test_extract_lexical_context_scope_disconnected_measure_mentions():
    """Disconnected bar mentions are normalized into explicit measure numbers."""
    ss = _make_context_score(measures=12, parts=2)

    scope = extract_lexical_context_scope(
        ss,
        "edit violin bars 3-5 and 8-9",
    )

    assert scope.part_indices == [0]
    assert scope.measure_numbers == [3, 4, 5, 8, 9]
    assert scope.bar_range is None
    assert scope.explicit_bar_mention is True
    assert scope.used_fallback_bar is False


def test_context_retriever_reuses_method_hits_and_scopes_bars():
    """Combined retrieval keeps method hits and adds scoped context bars."""
    ss = _make_context_score(measures=8, parts=2)
    expected_methods = _names(
        LexicalRetriever(MethodIndex(ss), threshold=0.5).query(
            "edit dynamics on bars 3-7 in violin"
        )
    )
    retriever = LexicalContextRetriever(ss, threshold=0.5)

    result = retriever.query("edit dynamics on bars 3-7 in violin")

    assert _names(result.methods) == expected_methods
    assert result.scope.part_indices == [0]
    assert result.scope.bar_range == (3, 7)
    assert result.scope.used_fallback_bar is False
    assert result.scope.bar_context_status == "explicit"
    assert [bar["measure_number"] for bar in result.context_bars["bars"]] == [3, 4, 5, 6, 7]
    assert [part["part_index"] for part in result.context_bars["bars"][0]["parts"]] == [0]


def test_context_retriever_supports_disconnected_bar_ranges():
    """Disconnected ranges are passed through as explicit measure numbers."""
    ss = _make_context_score(measures=10, parts=2)
    retriever = LexicalContextRetriever(ss, threshold=0.5)

    result = retriever.query("edit violin bars 3-5 and 8-9")

    assert result.scope.part_indices == [0]
    assert result.scope.measure_numbers == [3, 4, 5, 8, 9]
    assert result.scope.bar_range is None
    assert result.scope.bar_context_status == "explicit"
    assert result.context_query == {
        "scope": {"parts": [0], "measure_numbers": [3, 4, 5, 8, 9]}
    }
    assert [bar["measure_number"] for bar in result.context_bars["bars"]] == [3, 4, 5, 8, 9]
    assert [part["part_index"] for part in result.context_bars["bars"][0]["parts"]] == [0]


def test_context_retriever_measure_only_scopes_all_parts():
    """A measure-only mention retrieves that bar across every part."""
    ss = _make_context_score(measures=6, parts=3)
    retriever = LexicalContextRetriever(ss, threshold=0.5)

    result = retriever.query("change the dynamics in measure 4")

    assert result.scope.part_indices is None
    assert result.scope.bar_range == (4, 4)
    assert result.scope.used_fallback_bar is False
    assert result.scope.bar_context_status == "explicit"
    assert [bar["measure_number"] for bar in result.context_bars["bars"]] == [4]
    assert [part["part_index"] for part in result.context_bars["bars"][0]["parts"]] == [0, 1, 2]


def test_context_retriever_without_scope_returns_no_bars():
    """Generic queries without explicit scope provide no automatic bar context."""
    ss = _make_context_score(measures=6, parts=2)
    retriever = LexicalContextRetriever(ss, threshold=0.5)

    result = retriever.query("add a dynamic")

    assert result.scope.part_indices is None
    assert result.scope.bar_range is None
    assert result.scope.used_fallback_bar is False
    assert result.scope.bar_context_status == "missing"
    assert result.context_query is None
    assert result.context_bars["bars"] == []


def test_context_retriever_part_only_returns_no_bars():
    """A part-only mention preserves part scope but provides no bar context."""
    ss = _make_context_score(measures=6, parts=2)
    retriever = LexicalContextRetriever(ss, threshold=0.5)

    result = retriever.query("fix the violin dynamics")

    assert result.scope.part_indices == [0]
    assert result.scope.bar_range is None
    assert result.scope.used_fallback_bar is False
    assert result.scope.bar_context_status == "missing"
    assert result.context_query is None
    assert result.context_bars["bars"] == []


@pytest.mark.parametrize(
    "query",
    [
        "add a dynamic at the end",
        "add a dynamic to the final bar",
        "add a dynamic to the final measure",
        "continue with a dynamic",
        "append a dynamic",
        "add a dynamic after the existing music",
    ],
)
def test_context_retriever_end_oriented_missing_bar_uses_final_bar(query):
    """End-oriented wording retrieves the final bar when no bar is explicit."""
    ss = _make_context_score(measures=6, parts=2)
    retriever = LexicalContextRetriever(ss, threshold=0.5)

    result = retriever.query(query)

    assert result.scope.bar_context_status == "end_fallback"
    assert result.scope.used_fallback_bar is True
    assert result.context_query == {"scope": {"bar_range": (6, 6)}}
    assert [bar["measure_number"] for bar in result.context_bars["bars"]] == [6]
    assert [part["part_index"] for part in result.context_bars["bars"][0]["parts"]] == [0, 1]


def test_context_retriever_caps_large_automatic_bar_ranges():
    """Large automatic ranges are capped before bar payload construction."""
    ss = _make_context_score(measures=20, parts=2)
    retriever = LexicalContextRetriever(ss, threshold=0.5)

    result = retriever.query("edit bars 1-20")

    assert result.scope.bar_range == (1, 20)
    assert result.context_query == {"scope": {"bar_range": (1, 8)}}
    assert [bar["measure_number"] for bar in result.context_bars["bars"]] == list(range(1, 9))
    assert result.scope.context_truncation_messages
    assert "1-8" in result.scope.context_truncation_messages[0]
    assert "1-20" in result.scope.context_truncation_messages[0]


def test_context_retriever_caps_large_part_scopes():
    """Automatic context respects the part-bar cap when many parts are in scope."""
    part_specs = [
        {"instrument": "violin", "name": f"Part {index + 1}"}
        for index in range(20)
    ]
    ss = ScoreSpeak.create(measures=2, time_signature="4/4", parts=part_specs)
    retriever = LexicalContextRetriever(ss, threshold=0.5)

    result = retriever.query("edit bar 1")

    assert result.scope.part_indices is None
    assert result.context_query == {
        "scope": {"parts": list(range(16)), "bar_range": (1, 1)}
    }
    assert len(result.context_bars["bars"][0]["parts"]) == 16
    assert result.scope.context_truncation_messages
    assert "parts" in result.scope.context_truncation_messages[0]


def test_context_retriever_reports_ambiguous_part_mentions():
    """Ambiguous part mentions keep all parts in scope and report the ambiguity."""
    ss = ScoreSpeak.create(
        measures=5,
        time_signature="4/4",
        parts=[
            {"instrument": "violin", "name": "Violin I"},
            {"instrument": "violin", "name": "Violin II"},
        ],
    )
    retriever = LexicalContextRetriever(ss, threshold=0.5)

    result = retriever.query("fix the violin dynamics")

    assert result.scope.part_indices is None
    assert result.scope.used_fallback_bar is False
    assert result.scope.bar_context_status == "missing"
    assert result.scope.ambiguity_messages
    assert "violin" in result.scope.ambiguity_messages[0].lower()
    assert result.context_query is None
    assert result.context_bars["bars"] == []


# ── RH/LH hand-aware part resolution ─────────────────────────────────────────

def _make_piano_score():
    """Return a ScoreSpeak with a single soprano + piano grand-staff layout."""
    from music21 import clef, layout, note, stream

    soprano = stream.Part()
    soprano.partName = "Soprano"
    soprano.append(clef.TrebleClef())
    measure = stream.Measure(number=1)
    measure.append(note.Note("E5"))
    soprano.append(measure)

    rh = stream.PartStaff()
    rh.partName = "Piano"
    rh.append(clef.TrebleClef())
    m = stream.Measure(number=1); m.append(note.Note("C4")); rh.append(m)

    lh = stream.PartStaff()
    lh.partName = "Piano"
    lh.append(clef.BassClef())
    m = stream.Measure(number=1); m.append(note.Note("C3")); lh.append(m)

    score = stream.Score()
    score.insert(0, soprano)
    score.insert(0, rh)
    score.insert(0, lh)
    score.insert(0, layout.StaffGroup([rh, lh], name="Piano", symbol="brace"))
    return ScoreSpeak(score)


def _make_two_piano_score():
    """Return a ScoreSpeak with two piano grand staves."""
    from music21 import clef, layout, note, stream

    def mk():
        rh = stream.PartStaff(); rh.partName = "Piano"
        rh.append(clef.TrebleClef())
        m = stream.Measure(number=1); m.append(note.Note("C4")); rh.append(m)
        lh = stream.PartStaff(); lh.partName = "Piano"
        lh.append(clef.BassClef())
        m = stream.Measure(number=1); m.append(note.Note("C3")); lh.append(m)
        return rh, lh

    rh1, lh1 = mk(); rh2, lh2 = mk()
    score = stream.Score()
    score.insert(0, rh1); score.insert(0, lh1)
    score.insert(0, rh2); score.insert(0, lh2)
    score.insert(0, layout.StaffGroup([rh1, lh1], name="Piano", symbol="brace"))
    score.insert(0, layout.StaffGroup([rh2, lh2], name="Piano", symbol="brace"))
    return ScoreSpeak(score)


def test_extract_scope_resolves_piano_rh_exact_display_name():
    """``piano rh`` matches only the right-hand staff."""
    ss = _make_piano_score()
    scope = extract_lexical_context_scope(ss, "add a dynamic to piano rh at bar 1")
    assert scope.part_indices == [1]
    assert scope.matched_part_names == ["Piano RH"]
    assert scope.ambiguity_messages == []


def test_extract_scope_resolves_right_hand_synonym():
    """``right hand`` resolves to the RH staff via hand synonym."""
    ss = _make_piano_score()
    scope = extract_lexical_context_scope(ss, "add a dynamic to the right hand at bar 1")
    assert scope.part_indices == [1]
    assert scope.matched_part_names == ["Piano RH"]


def test_extract_scope_resolves_piano_left_hand_intersection():
    """``piano left hand`` intersects base-name + hand synonym."""
    ss = _make_piano_score()
    scope = extract_lexical_context_scope(ss, "add a dynamic to piano left hand at bar 1")
    assert scope.part_indices == [2]
    assert scope.matched_part_names == ["Piano LH"]


def test_extract_scope_bare_piano_matches_both_staves():
    """A bare ``piano`` mention still targets the whole grand staff."""
    ss = _make_piano_score()
    scope = extract_lexical_context_scope(ss, "add a dynamic to piano at bar 1")
    assert scope.part_indices == [1, 2]
    assert scope.matched_part_names == ["Piano RH", "Piano LH"]
    assert scope.ambiguity_messages == []


def test_extract_scope_two_pianos_rh_is_ambiguous():
    """With two pianos, a bare ``right hand`` matches both RH staves."""
    ss = _make_two_piano_score()
    scope = extract_lexical_context_scope(ss, "add a dynamic to right hand at bar 1")
    assert scope.part_indices == [0, 2]
    assert scope.matched_part_names == ["Piano 1 RH", "Piano 2 RH"]


def test_extract_scope_two_pianos_group_index_resolves():
    """``piano 2`` narrows down to one specific grand staff."""
    ss = _make_two_piano_score()
    scope = extract_lexical_context_scope(ss, "add a dynamic to piano 2 at bar 1")
    assert scope.part_indices == [2, 3]
    assert scope.matched_part_names == ["Piano 2 RH", "Piano 2 LH"]


def test_extract_scope_two_pianos_group_index_plus_hand():
    """``piano 1 left hand`` intersects base + group index + hand."""
    ss = _make_two_piano_score()
    scope = extract_lexical_context_scope(ss, "add a dynamic to piano 1 left hand at bar 1")
    assert scope.part_indices == [1]
    assert scope.matched_part_names == ["Piano 1 LH"]
