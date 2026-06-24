"""Utilities for extracting render-sized MusicXML measure windows."""

from __future__ import annotations

from copy import deepcopy
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Iterable

from music21 import base as m21base
from music21 import clef as m21clef
from music21 import dynamics as m21dynamics
from music21 import expressions as m21expressions
from music21 import key as m21key
from music21 import layout as m21layout
from music21 import meter as m21meter
from music21 import spanner as m21spanner
from music21 import stream as m21stream
from music21 import tempo as m21tempo
from lxml import etree

from scorespeak.core import _apply_repeat_bracket_display_labels
from scorespeak.score.musicxml_export import write_musicxml_file

if TYPE_CHECKING:
    from scorespeak import ScoreSpeak


logger = logging.getLogger(__name__)

_ATTRIBUTE_CHILDREN_TO_CARRY = {
    "divisions",
    "key",
    "time",
    "staves",
    "part-symbol",
    "instruments",
    "clef",
    "transpose",
    "staff-details",
}

_WINDOW_SPANNER_CLASSES = (
    m21dynamics.Crescendo,
    m21dynamics.Diminuendo,
    m21expressions.PedalMark,
    m21spanner.Glissando,
    m21spanner.Ottava,
    m21spanner.RepeatBracket,
    m21spanner.Slur,
)

_OCTAVE_SHIFT_START_TYPES = {"down", "up"}
_DIRECTION_CONTEXT_CHILDREN = {
    "footnote",
    "level",
    "offset",
    "staff",
    "voice",
}


def show_rests_for_empty_space(musicxml: str) -> str:
    """Reveal hidden rest notes only for measure staffs with no visible notes."""
    try:
        parser = etree.XMLParser(remove_blank_text=False)
        tree = etree.fromstring(musicxml.encode("utf-8"), parser)
    except etree.XMLSyntaxError:
        logger.warning("Could not parse MusicXML while showing empty-space rests")
        return musicxml

    changed = False
    for measure_element in tree.xpath(".//*[local-name()='measure']"):
        note_elements = _children_by_local_name(measure_element, "note")
        visible_staffs = {
            _note_staff(note_element)
            for note_element in note_elements
            if note_element.get("print-object") != "no"
        }

        for note_element in note_elements:
            if (
                note_element.get("print-object") == "no"
                and _is_rest_note(note_element)
                and _note_staff(note_element) not in visible_staffs
            ):
                note_element.attrib.pop("print-object", None)
                note_element.attrib.pop("print-spacing", None)
                changed = True

    if not changed:
        return musicxml

    doctype = tree.getroottree().docinfo.doctype
    return etree.tostring(
        tree,
        encoding="UTF-8",
        xml_declaration=True,
        doctype=doctype or None,
    ).decode("utf-8")


def _is_rest_note(note_element: etree._Element) -> bool:
    """Return whether ``note_element`` is a MusicXML rest note."""
    return _first_child_by_local_name(note_element, "rest") is not None


def _note_staff(note_element: etree._Element) -> str:
    """Return the note staff text, defaulting to staff 1 when absent."""
    staff_element = _first_child_by_local_name(note_element, "staff")
    if staff_element is None or staff_element.text is None:
        return "1"

    staff_text = staff_element.text.strip()
    if not staff_text:
        return "1"
    return staff_text


def extract_musicxml_window(
    musicxml: str,
    start_measure: int,
    end_measure: int,
) -> str:
    """Return MusicXML containing only the inclusive 1-based measure window.

    The extractor preserves the score-level metadata and part-list, trims each
    ``part`` to the requested measure ordinals, and carries active attributes
    such as key, time, clef, divisions, staves, and transposition into the first
    visible measure so OSMD can render the slice as a standalone score.
    """
    start_measure, end_measure = normalize_measure_window(
        start_measure,
        end_measure,
    )
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(musicxml.encode("utf-8"), parser)
    tree = root.getroottree()

    for part_element in _children_by_local_name(root, "part"):
        _trim_part_to_window(part_element, start_measure, end_measure)

    doctype = tree.docinfo.doctype
    return etree.tostring(
        root,
        encoding="UTF-8",
        xml_declaration=True,
        doctype=doctype or None,
    ).decode("utf-8")


def normalize_measure_window(
    start_measure: int,
    end_measure: int,
) -> tuple[int, int]:
    """Normalize a requested measure window to a valid inclusive range."""
    safe_start = max(1, int(start_measure))
    safe_end = max(safe_start, int(end_measure))
    return safe_start, safe_end


def export_scorespeak_window_musicxml(
    score_state: "ScoreSpeak",
    start_measure: int,
    end_measure: int,
) -> str:
    """Export only a measure window from a live ``ScoreSpeak`` score.

    This avoids music21 serializing a large full score after every edit. The
    resulting MusicXML is intended for display, not archival download; full
    downloads should still use ``ScoreSpeak.to_musicxml_string()``.
    """
    start_measure, end_measure = normalize_measure_window(
        start_measure,
        end_measure,
    )
    window_score = _build_music21_window_score(
        score_state,
        start_measure,
        end_measure,
    )

    fd, tmp_path = tempfile.mkstemp(suffix=".musicxml")
    os.close(fd)
    try:
        write_musicxml_file(window_score, tmp_path)
        musicxml = Path(tmp_path).read_text(encoding="utf-8")
        return _apply_repeat_bracket_display_labels(
            window_score,
            musicxml,
        )
    finally:
        os.unlink(tmp_path)


def _build_music21_window_score(
    score_state: "ScoreSpeak",
    start_measure: int,
    end_measure: int,
) -> m21stream.Score:
    """Build a music21 score containing only the requested measure window."""
    source_score = score_state.score
    window_score = m21stream.Score()
    if source_score.metadata is not None:
        window_score.metadata = deepcopy(source_score.metadata)

    source_to_window_parts: dict[int, m21stream.Part] = {}

    for part_index, source_part in enumerate(source_score.parts):
        window_part = _make_window_part(source_part)
        window_part.partName = source_part.partName
        window_part.partAbbreviation = source_part.partAbbreviation
        _copy_part_preamble(source_part, window_part)

        source_measure_pairs = _copy_measure_window(
            source_part,
            start_measure,
            end_measure,
        )
        copied_element_by_source_id = _copied_element_map(source_measure_pairs)
        kept_measures = [copied_measure for _, copied_measure in source_measure_pairs]
        if kept_measures:
            _ensure_music21_first_measure_context(
                score_state,
                source_part,
                part_index,
                kept_measures[0],
                start_measure,
            )
        for measure in kept_measures:
            window_part.append(measure)
        _copy_supported_spanners(
            source_part,
            window_part,
            copied_element_by_source_id,
        )

        window_score.insert(0, window_part)
        source_to_window_parts[id(source_part)] = window_part

    _copy_staff_groups(source_score, window_score, source_to_window_parts)
    return window_score


def _make_window_part(source_part: m21stream.Part) -> m21stream.Part:
    """Create a window part preserving the source part stream type."""
    if isinstance(source_part, m21stream.PartStaff):
        window_part = m21stream.PartStaff()
    else:
        window_part = m21stream.Part()

    source_id = getattr(source_part, "id", None)
    if isinstance(source_id, str) and not source_id.isdecimal():
        window_part.id = source_id
    return window_part


def _copy_staff_groups(
    source_score: m21stream.Score,
    window_score: m21stream.Score,
    source_to_window_parts: dict[int, m21stream.Part],
) -> None:
    """Copy staff groups whose spanned parts are present in the window."""
    for source_group in source_score.recurse().getElementsByClass(m21layout.StaffGroup):
        spanned_parts = list(source_group.getSpannedElements())
        if len(spanned_parts) < 2:
            continue

        window_parts: list[m21stream.Part] = []
        for spanned_part in spanned_parts:
            window_part = source_to_window_parts.get(id(spanned_part))
            if window_part is None:
                break
            window_parts.append(window_part)
        else:
            group_copy = m21layout.StaffGroup(
                window_parts,
                name=getattr(source_group, "name", None),
                abbreviation=getattr(source_group, "abbreviation", None),
                symbol=getattr(source_group, "symbol", None),
                barTogether=getattr(source_group, "barTogether", True),
            )
            window_score.insert(0, group_copy)


def _copy_part_preamble(
    source_part: m21stream.Part,
    window_part: m21stream.Part,
) -> None:
    """Copy non-measure setup elements, such as instrument declarations."""
    for element in source_part.getElementsByOffset(
        0,
        mustBeginInSpan=False,
        includeEndBoundary=True,
    ):
        if isinstance(element, m21stream.Measure):
            continue
        window_part.insert(element.offset, deepcopy(element))


def _copy_measure_window(
    source_part: m21stream.Part,
    start_measure: int,
    end_measure: int,
) -> list[tuple[m21stream.Measure, m21stream.Measure]]:
    """Return source/deep-copy measure pairs in the inclusive ordinal window."""
    copied_measures: list[tuple[m21stream.Measure, m21stream.Measure]] = []
    for index, measure in enumerate(
        source_part.getElementsByClass(m21stream.Measure),
        start=1,
    ):
        if start_measure <= index <= end_measure:
            copied_measures.append((measure, deepcopy(measure)))
    return copied_measures


def _copied_element_map(
    source_measure_pairs: list[tuple[m21stream.Measure, m21stream.Measure]],
) -> dict[int, m21base.Music21Object]:
    """Return a map from source measure elements to their copied counterparts."""
    copied_by_source_id: dict[int, m21base.Music21Object] = {}
    for source_measure, copied_measure in source_measure_pairs:
        source_elements = [source_measure, *list(source_measure.recurse())]
        copied_elements = [copied_measure, *list(copied_measure.recurse())]
        for source_element, copied_element in zip(source_elements, copied_elements):
            copied_by_source_id[id(source_element)] = copied_element
    return copied_by_source_id


def _copy_supported_spanners(
    source_part: m21stream.Part,
    window_part: m21stream.Part,
    copied_element_by_source_id: dict[int, m21base.Music21Object],
) -> None:
    """Copy supported part-level spanners fully contained in the window."""
    if not copied_element_by_source_id:
        return

    for source_spanner in source_part.getElementsByClass(m21spanner.Spanner):
        copied_spanner = _copy_supported_spanner(
            source_spanner,
            copied_element_by_source_id,
        )
        if copied_spanner is None:
            continue
        window_part.insert(0, copied_spanner)


def _copy_supported_spanner(
    source_spanner: m21spanner.Spanner,
    copied_element_by_source_id: dict[int, m21base.Music21Object],
) -> m21spanner.Spanner | None:
    """Return a copied supported spanner, or None if it cannot be windowed."""
    if not isinstance(source_spanner, _WINDOW_SPANNER_CLASSES):
        return None

    copied_span: list[m21base.Music21Object] = []
    for source_element in source_spanner.getSpannedElements():
        copied_element = copied_element_by_source_id.get(id(source_element))
        if copied_element is None:
            return None
        copied_span.append(copied_element)

    if not copied_span:
        return None

    copied_spanner = _new_spanner_like(source_spanner, copied_span)
    if copied_spanner is None:
        return None
    _copy_spanner_attributes(source_spanner, copied_spanner)
    return copied_spanner


def _new_spanner_like(
    source_spanner: m21spanner.Spanner,
    copied_span: list[m21base.Music21Object],
) -> m21spanner.Spanner | None:
    """Instantiate the same supported spanner type over copied elements."""
    if isinstance(source_spanner, m21spanner.RepeatBracket):
        return m21spanner.RepeatBracket(
            *copied_span,
            number=_repeat_bracket_number(source_spanner),
            overrideDisplay=getattr(source_spanner, "overrideDisplay", None),
        )
    if isinstance(source_spanner, m21spanner.Ottava):
        return m21spanner.Ottava(
            *copied_span,
            type=getattr(source_spanner, "type", "8va"),
            transposing=True,
            placement=getattr(source_spanner, "placement", "above"),
        )
    if isinstance(source_spanner, m21spanner.Glissando):
        return m21spanner.Glissando(
            *copied_span,
            lineType=getattr(source_spanner, "lineType", "wavy"),
            label=getattr(source_spanner, "label", None),
        )
    if isinstance(source_spanner, m21dynamics.Crescendo):
        return m21dynamics.Crescendo(*copied_span)
    if isinstance(source_spanner, m21dynamics.Diminuendo):
        return m21dynamics.Diminuendo(*copied_span)
    if isinstance(source_spanner, m21expressions.PedalMark):
        return m21expressions.PedalMark(*copied_span)
    if isinstance(source_spanner, m21spanner.Slur):
        return m21spanner.Slur(*copied_span)
    return None


def _copy_spanner_attributes(
    source_spanner: m21spanner.Spanner,
    copied_spanner: m21spanner.Spanner,
) -> None:
    """Copy display-relevant attributes after constructing a window spanner."""
    for attribute_name in (
        "abbreviated",
        "idLocal",
        "lineType",
        "measured",
        "niente",
        "pedalForm",
        "pedalType",
        "placement",
        "spread",
    ):
        if not hasattr(source_spanner, attribute_name):
            continue
        try:
            setattr(
                copied_spanner,
                attribute_name,
                deepcopy(getattr(source_spanner, attribute_name)),
            )
        except Exception:
            continue

    copied_spanner.style = deepcopy(source_spanner.style)
    copied_spanner.groups = deepcopy(source_spanner.groups)


def _repeat_bracket_number(
    repeat_bracket: m21spanner.RepeatBracket,
) -> str | list[int]:
    """Return a music21-safe repeat bracket number value."""
    number_range = list(getattr(repeat_bracket, "numberRange", []))
    if number_range:
        return number_range
    return str(getattr(repeat_bracket, "number", ""))


def _ensure_music21_first_measure_context(
    score_state: "ScoreSpeak",
    source_part: m21stream.Part,
    part_index: int,
    measure: m21stream.Measure,
    start_measure: int,
) -> None:
    """Carry active music21 attributes into the first visible measure."""
    if not measure.getElementsByClass(m21meter.TimeSignature):
        measure.insert(
            0,
            deepcopy(score_state._get_active_time_signature_obj(source_part, start_measure)),
        )
    if not _measure_has_key_signature(measure):
        measure.insert(
            0,
            deepcopy(score_state._get_active_key_signature_obj(source_part, start_measure)),
        )
    if not measure.getElementsByClass(m21clef.Clef):
        active_clef = score_state._get_active_clef_obj(source_part, start_measure)
        if active_clef is not None:
            measure.insert(0, deepcopy(active_clef))
    if part_index == 0 and not measure.getElementsByClass(m21tempo.MetronomeMark):
        active_tempo = _get_active_tempo_obj(score_state, source_part, start_measure)
        if active_tempo is not None:
            measure.insert(0, deepcopy(active_tempo))


def _measure_has_key_signature(measure: m21stream.Measure) -> bool:
    """Return whether a measure already contains an explicit key signature."""
    return bool(
        list(measure.getElementsByClass(m21key.KeySignature))
        or list(measure.getElementsByClass(m21key.Key))
    )


def _get_active_tempo_obj(
    score_state: "ScoreSpeak",
    source_part: m21stream.Part,
    start_measure: int,
) -> m21tempo.MetronomeMark | None:
    """Return the active tempo object at a measure, if one exists."""
    measure = score_state._resolve_measure(source_part, start_measure)
    local_tempos = list(measure.getElementsByClass(m21tempo.MetronomeMark))
    if local_tempos:
        return local_tempos[0]
    tempo = measure.getContextByClass(m21tempo.MetronomeMark)
    if tempo is not None:
        return tempo
    return None


def _trim_part_to_window(
    part_element: etree._Element,
    start_measure: int,
    end_measure: int,
) -> None:
    """Trim one MusicXML part element to the requested measure window."""
    measures = _children_by_local_name(part_element, "measure")
    if not measures:
        return

    active_attributes = _active_attributes_before(measures, start_measure)
    kept_measures = [
        measure
        for index, measure in enumerate(measures, start=1)
        if start_measure <= index <= end_measure
    ]

    for measure in measures:
        if measure not in kept_measures:
            part_element.remove(measure)

    if kept_measures:
        _ensure_first_measure_attributes(kept_measures[0], active_attributes)


def _active_attributes_before(
    measures: list[etree._Element],
    start_measure: int,
) -> etree._Element | None:
    """Return the active attributes immediately before a measure ordinal."""
    active_attributes: etree._Element | None = None
    for index, measure in enumerate(measures, start=1):
        if index >= start_measure:
            break
        for attributes in _children_by_local_name(measure, "attributes"):
            active_attributes = _merge_attributes(active_attributes, attributes)
    return active_attributes


def _ensure_first_measure_attributes(
    measure: etree._Element,
    active_attributes: etree._Element | None,
) -> None:
    """Ensure the first rendered measure has enough attributes for OSMD."""
    if active_attributes is None:
        return

    existing_attributes = _children_by_local_name(measure, "attributes")
    if existing_attributes:
        insert_at = _attribute_insert_index(measure)
        leading_attributes = [
            attributes
            for attributes in existing_attributes
            if measure.index(attributes) == insert_at
        ]
        if not leading_attributes:
            measure.insert(insert_at, deepcopy(active_attributes))
            return

        merged = deepcopy(active_attributes)
        for attributes in leading_attributes:
            merged = _merge_attributes(merged, attributes)
        measure.replace(leading_attributes[0], merged)
        for duplicate in leading_attributes[1:]:
            measure.remove(duplicate)
        return

    insert_at = _attribute_insert_index(measure)
    measure.insert(insert_at, deepcopy(active_attributes))


def _merge_attributes(
    base: etree._Element | None,
    override: etree._Element,
) -> etree._Element:
    """Return attributes with override children replacing carried children."""
    merged = deepcopy(override if base is None else base)
    if base is None:
        return merged

    for child in override:
        if not isinstance(child.tag, str):
            continue
        local_name = etree.QName(child).localname
        if local_name not in _ATTRIBUTE_CHILDREN_TO_CARRY:
            continue
        child_identity = _attribute_child_identity(child)
        for existing in list(merged):
            if not isinstance(existing.tag, str):
                continue
            if _attribute_child_identity(existing) == child_identity:
                merged.remove(existing)
        merged.append(deepcopy(child))

    return merged


def _attribute_child_identity(element: etree._Element) -> tuple[str, str | None]:
    """Return the replacement identity for an attributes child element."""
    return (etree.QName(element).localname, element.get("number"))


def _attribute_insert_index(measure: etree._Element) -> int:
    """Return a stable insertion index for synthetic first-measure attributes."""
    for index, child in enumerate(measure):
        if etree.QName(child).localname not in {"print", "bookmark", "link"}:
            return index
    return len(measure)


def _children_by_local_name(
    element: etree._Element,
    local_name: str,
) -> list[etree._Element]:
    """Return direct children whose tag local-name matches ``local_name``."""
    return [
        child
        for child in _element_children(element)
        if etree.QName(child).localname == local_name
    ]


def _element_children(element: etree._Element) -> Iterable[etree._Element]:
    """Yield element children, ignoring comments and processing instructions."""
    for child in element:
        if isinstance(child.tag, str):
            yield child


def _first_child_by_local_name(
    element: etree._Element,
    local_name: str,
) -> etree._Element | None:
    """Return the first direct child with ``local_name``, if present."""
    for child in _element_children(element):
        if etree.QName(child).localname == local_name:
            return child
    return None

