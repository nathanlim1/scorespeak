"""
ScoreSpeak — the central stateful wrapper for music-theory-aware score editing.

Wraps a music21 Score object and exposes all editing operations as methods.
Uses mixin classes from sibling modules for feature-specific operations.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Union

from lxml import etree
from music21 import clef as m21clef
from music21 import converter as m21converter
from music21 import key as m21key
from music21 import metadata as m21metadata
from music21 import meter as m21meter
from music21 import note as m21note
from music21 import spanner as m21spanner
from music21 import stream as m21stream
from music21 import tempo as m21tempo

from .editing.expressions import ExpressionsMixin
from .editing.layout import LayoutMixin
from .editing.markings import MarkingsMixin
from .editing.measures import MeasuresMixin
from .editing.notes import NotesMixin
from .editing.parts import PartsMixin, _make_grand_staff, resolve_instrument
from .editing.signatures import SignaturesMixin
from .music.pitch_space import (
    OPEN_KEY_SIGNATURE_LABEL,
    concert_key_signature_for_stored_key,
    default_stored_pitch_space_for_part,
    is_open_key_signature,
    mark_open_key_signature,
    stored_key_signature_for_concert_key,
)
from .music.validation import (
    default_clef_for_instrument,
    make_clef,
    validate_voice_number,
)
from .query.bar_retrieval import BarRetrievalMixin
from .score.musicxml_export import write_musicxml_file
from .score.staff_groups import build_part_display_labels
from .types import OperationResult


def _repeat_bracket_number_text(repeat_bracket: m21spanner.RepeatBracket) -> str:
    """Return the MusicXML number attribute music21 emits for a repeat bracket."""
    number_range = list(getattr(repeat_bracket, "numberRange", []))
    if not number_range:
        return str(getattr(repeat_bracket, "number", ""))

    if number_range[0] == 0:
        number_text = ""
    else:
        number_text = str(number_range[0])
    for number in number_range[1:]:
        number_text += "," + str(number)
    return number_text


def _repeat_bracket_start_measure(
    repeat_bracket: m21spanner.RepeatBracket,
) -> m21stream.Measure | None:
    """Return the first spanned measure for a repeat bracket, if available."""
    spanned_elements = repeat_bracket.getSpannedElements()
    if not spanned_elements:
        return None

    first_element = spanned_elements[0]
    if isinstance(first_element, m21stream.Measure):
        return first_element
    measure = first_element.getContextByClass(m21stream.Measure)
    if isinstance(measure, m21stream.Measure):
        return measure
    return None


def _repeat_bracket_display_labels(
    score: m21stream.Score,
) -> dict[tuple[int, str, str], str]:
    """Return repeat bracket display labels keyed by part, measure, and number."""
    labels: dict[tuple[int, str, str], str] = {}
    for part_index, part_obj in enumerate(score.parts):
        for repeat_bracket in part_obj.getElementsByClass(m21spanner.RepeatBracket):
            display_label = getattr(repeat_bracket, "overrideDisplay", None)
            if display_label is None:
                continue
            start_measure = _repeat_bracket_start_measure(repeat_bracket)
            if start_measure is None or start_measure.number is None:
                continue
            labels[
                (
                    part_index,
                    str(start_measure.number),
                    _repeat_bracket_number_text(repeat_bracket),
                )
            ] = str(display_label)
    return labels


def _apply_repeat_bracket_display_labels(
    score: m21stream.Score,
    musicxml: str,
) -> str:
    """Inject RepeatBracket display labels that music21 omits during export."""
    labels = _repeat_bracket_display_labels(score)
    if not labels:
        return musicxml

    try:
        parser = etree.XMLParser(remove_blank_text=False)
        root = etree.fromstring(musicxml.encode("utf-8"), parser)
    except etree.XMLSyntaxError:
        return musicxml

    changed = False
    part_elements = root.xpath("./*[local-name()='part']")
    for part_index, part_element in enumerate(part_elements):
        measure_elements = part_element.xpath("./*[local-name()='measure']")
        for measure_element in measure_elements:
            measure_number = measure_element.get("number")
            if measure_number is None:
                continue
            ending_elements = measure_element.xpath(
                "./*[local-name()='barline']/*[local-name()='ending'][@type='start']"
            )
            for ending_element in ending_elements:
                number_text = ending_element.get("number", "")
                display_label = labels.get((part_index, measure_number, number_text))
                if display_label is None:
                    continue
                ending_element.text = display_label
                changed = True

    if not changed:
        return musicxml

    doctype = root.getroottree().docinfo.doctype
    return etree.tostring(
        root,
        encoding="UTF-8",
        xml_declaration=True,
        doctype=doctype or None,
    ).decode("utf-8")


def _namespace_stripped_musicxml(path_or_xml: str) -> str | None:
    """Return MusicXML text with element namespaces removed, if needed."""
    xml_text = _musicxml_text(path_or_xml)
    if xml_text is None:
        return None

    try:
        parser = etree.XMLParser(remove_blank_text=False, recover=False)
        root = etree.fromstring(xml_text.encode("utf-8"), parser)
    except etree.XMLSyntaxError:
        return None

    root_name = etree.QName(root).localname
    root_namespace = etree.QName(root).namespace
    if root_name != "score-partwise" or not root_namespace:
        return None

    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        element.tag = etree.QName(element).localname
    etree.cleanup_namespaces(root)
    doctype = root.getroottree().docinfo.doctype
    return etree.tostring(
        root,
        encoding="UTF-8",
        xml_declaration=True,
        doctype=doctype or None,
    ).decode("utf-8")


def _musicxml_text(path_or_xml: str) -> str | None:
    """Return XML text from a path or XML string."""
    if path_or_xml.lstrip().startswith("<"):
        return path_or_xml
    try:
        return Path(path_or_xml).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _tempo_number(mark: m21tempo.MetronomeMark) -> Optional[float]:
    """Return a numeric BPM value from a tempo mark when one is present."""
    number = getattr(mark, "number", None)
    if number is None:
        return None
    return float(number)


def _measure_tempo_marks(
    measure: m21stream.Measure,
) -> list[m21tempo.MetronomeMark]:
    """Return direct tempo marks in a measure sorted by measure offset."""
    marks = list(measure.getElementsByClass(m21tempo.MetronomeMark))
    marks.sort(key=lambda mark: float(measure.elementOffset(mark)))
    return marks


class ScoreSpeak(
    MeasuresMixin,
    PartsMixin,
    NotesMixin,
    SignaturesMixin,
    ExpressionsMixin,
    MarkingsMixin,
    LayoutMixin,
    BarRetrievalMixin,
):
    """Stateful wrapper around a music21 Score.

    All editing operations are exposed as methods on this class.
    Each method validates inputs against music theory rules and
    returns structured OperationResult objects.

    The score maintains implied continuity: time signatures, key
    signatures, clefs, and tempo markings propagate forward through
    measures until explicitly changed.
    """

    def __init__(self, score: m21stream.Score) -> None:
        self._score = score

    def _commit_shadow_score(self, shadow_state: "ScoreSpeak") -> None:
        """Adopt a transactional shadow score without deepcopy derivation chains."""
        self._score = shadow_state.score
        self._clear_derivation_origins()

    def _clear_derivation_origins(self) -> None:
        """Remove music21 derivation origins that can make export pathologically slow."""
        for element in [self._score, *self._score.recurse()]:
            derivation = getattr(element, "derivation", None)
            if derivation is not None:
                derivation.origin = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        title: str = "Untitled",
        composer: str = "",
        time_signature: str = "4/4",
        key_signature: str = "C",
        tempo: float = 120.0,
        parts: Optional[list[Union[str, dict]]] = None,
        measures: int = 0,
    ) -> "ScoreSpeak":
        """Create a new, empty score.

        Args:
            title: Score title.
            composer: Composer name.
            time_signature: Initial time signature (e.g., "4/4", "3/4", "6/8").
            key_signature: Initial key (e.g., "C", "G", "Bb", "F# minor").
            tempo: Initial tempo in BPM.
            parts: List of part specifications. Each element can be:
                - A string instrument name (e.g., "piano", "violin")
                - A dict with keys: name, instrument, clef, grand_staff
                If None, creates a single Piano part.
            measures: Number of initial empty measures (default 0).

        Returns:
            A new ScoreSpeak instance.
        """
        score = m21stream.Score()

        md = m21metadata.Metadata()
        md.title = title
        md.composer = composer
        score.metadata = md

        ts = m21meter.TimeSignature(time_signature)
        ks = _parse_key_signature(key_signature)
        mm = m21tempo.MetronomeMark(number=tempo)

        if parts is None:
            parts = ["piano"]

        for i, part_spec in enumerate(parts):
            if isinstance(part_spec, str):
                inst_name = part_spec
                part_name = None
                clef_type = None
                grand_staff = False
            elif isinstance(part_spec, dict):
                inst_name = part_spec.get("instrument", "piano")
                part_name = part_spec.get("name")
                clef_type = part_spec.get("clef")
                grand_staff = bool(part_spec.get("grand_staff", False))
            else:
                raise ValueError(
                    f"Part spec must be a string or dict, got "
                    f"{type(part_spec).__name__}."
                )

            if grand_staff:
                grand_staff_parts, staff_group = _make_grand_staff(
                    part_name,
                    inst_name,
                    measures=measures,
                    time_signature=ts,
                    key_signature=ks,
                    tempo=tempo,
                )
                for grand_staff_part in grand_staff_parts:
                    score.insert(0, grand_staff_part)
                score.insert(0, staff_group)
                continue

            inst = resolve_instrument(inst_name)
            part = m21stream.Part(id=f"P{i + 1}")
            part.partName = part_name or inst.partName or inst_name.title()
            part.insert(0, inst)
            default_stored_pitch_space_for_part(part)

            if measures > 0:
                for m_num in range(1, measures + 1):
                    m = m21stream.Measure(number=m_num)
                    if m_num == 1:
                        m.timeSignature = m21meter.TimeSignature(
                            time_signature
                        )
                        m.insert(
                            0,
                            stored_key_signature_for_concert_key(part, ks),
                        )
                        m.insert(0, mm if i == 0 else m21tempo.MetronomeMark(number=tempo))
                        if clef_type:
                            m.insert(0, make_clef(clef_type))
                        else:
                            m.insert(0, default_clef_for_instrument(inst))
                    rest = m21note.Rest(
                        quarterLength=ts.barDuration.quarterLength
                    )
                    m.append(rest)
                    part.append(m)

            score.insert(0, part)

        instance = cls(score)
        return instance

    @classmethod
    def from_musicxml(
        cls,
        path_or_string: Union[str, Path],
    ) -> "ScoreSpeak":
        """Import a score from a MusicXML file or string.

        Args:
            path_or_string: File path to a .musicxml/.xml file,
                or a MusicXML string.

        Returns:
            A new ScoreSpeak wrapping the parsed score.

        Raises:
            FileNotFoundError: If the path doesn't exist.
            ValueError: If the file can't be parsed as MusicXML.
        """
        if isinstance(path_or_string, Path):
            path_or_string = str(path_or_string)

        if (
            isinstance(path_or_string, str)
            and not path_or_string.strip().startswith("<")
            and os.path.exists(path_or_string)
        ):
            path_str = str(path_or_string)
        elif isinstance(path_or_string, str) and not path_or_string.strip().startswith("<"):
            raise FileNotFoundError(
                f"MusicXML file not found: {path_or_string}"
            )
        else:
            path_str = path_or_string

        try:
            parsed = m21converter.parse(path_str, format="musicxml")
        except Exception as exc:
            namespace_stripped = _namespace_stripped_musicxml(path_str)
            if namespace_stripped is None:
                raise ValueError(
                    f"Failed to parse MusicXML: {exc}"
                ) from exc
            try:
                parsed = m21converter.parse(namespace_stripped, format="musicxml")
            except Exception as retry_exc:
                raise ValueError(
                    f"Failed to parse MusicXML: {retry_exc}"
                ) from retry_exc

        if isinstance(parsed, m21stream.Score):
            score = parsed
        elif isinstance(parsed, m21stream.Part):
            score = m21stream.Score()
            score.insert(0, parsed)
        elif isinstance(parsed, m21stream.Opus):
            scores = list(parsed.getElementsByClass(m21stream.Score))
            if scores:
                score = scores[0]
            else:
                raise ValueError(
                    "Parsed an Opus but it contains no scores."
                )
        else:
            score = m21stream.Score()
            score.insert(0, parsed)

        return cls(score)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_musicxml(
        self,
        path: Union[str, Path],
        make_notation: bool = True,
    ) -> OperationResult:
        """Export the score to a MusicXML file.

        Args:
            path: Output file path.
            make_notation: If True, runs music21's makeNotation for
                cleaner output (beaming, etc.). Set False for raw export.

        Returns:
            OperationResult confirming the export.
        """
        path_str = str(path)
        write_musicxml_file(self._score, path_str, make_notation=make_notation)
        path_obj = Path(path_str)
        try:
            musicxml = path_obj.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            musicxml = ""
        if musicxml:
            patched_musicxml = _apply_repeat_bracket_display_labels(
                self._score,
                musicxml,
            )
            if patched_musicxml != musicxml:
                path_obj.write_text(patched_musicxml, encoding="utf-8")

        return OperationResult(
            success=True,
            description=f"Exported score to {path_str}",
            details={"path": path_str},
        )

    def to_musicxml_string(self, make_notation: bool = True) -> str:
        """Export the score as a MusicXML string.

        Args:
            make_notation: If True, runs makeNotation for cleaner output.

        Returns:
            MusicXML content as a string.
        """
        with tempfile.NamedTemporaryFile(
            suffix=".musicxml", delete=False, mode="w"
        ) as f:
            tmp_path = f.name

        try:
            write_musicxml_file(
                self._score,
                tmp_path,
                make_notation=make_notation,
            )
            with open(tmp_path, "r", encoding="utf-8") as f:
                musicxml = f.read()
            return _apply_repeat_bracket_display_labels(self._score, musicxml)
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Score-level queries
    # ------------------------------------------------------------------

    @property
    def score(self) -> m21stream.Score:
        """Direct access to the underlying music21 Score (advanced use)."""
        return self._score

    @property
    def title(self) -> str:
        """The score's title."""
        if self._score.metadata and self._score.metadata.title:
            return self._score.metadata.title
        return ""

    @property
    def composer(self) -> str:
        """The score's composer."""
        if self._score.metadata and self._score.metadata.composer:
            return self._score.metadata.composer
        return ""

    @property
    def part_count(self) -> int:
        """Number of parts in the score."""
        return len(list(self._score.parts))

    @property
    def measure_count(self) -> int:
        """Number of measures in the first part (or 0 if no parts)."""
        parts = list(self._score.parts)
        if not parts:
            return 0
        return self._get_measure_count(parts[0])

    def get_active_time_signature(
        self,
        measure_number: int,
        part: Optional[Union[int, str]] = None,
    ) -> str:
        """Get the active time signature at a given measure (implied continuity).

        Args:
            measure_number: 1-based measure number.
            part: Part index or name (defaults to first part).

        Returns:
            Time signature string like "4/4", "3/4", "6/8".
        """
        part_obj, _ = self._resolve_part(part)
        ts = self._get_active_time_signature_obj(part_obj, measure_number)
        return ts.ratioString

    def get_active_key_signature(
        self,
        measure_number: int,
        part: Optional[Union[int, str]] = None,
    ) -> str:
        """Get the active key signature at a given measure (implied continuity).

        Args:
            measure_number: 1-based measure number.
            part: Part index or name. When omitted, returns the score-level
                concert key. When provided, returns that part's stored/display
                key signature. Transposing parts stored at written pitch may
                therefore return a transposed written key; agent-facing tools
                annotate this result with concert-key context.

        Returns:
            Key signature string like "C major", "G major", "2 sharps".
        """
        if part is None:
            part_obj, _ = self._resolve_part(None)
            stored_ks = self._get_active_key_signature_obj(
                part_obj,
                measure_number,
            )
            ks = concert_key_signature_for_stored_key(part_obj, stored_ks)
            return self._format_key_signature(ks)

        part_obj, _ = self._resolve_part(part)
        ks = self._get_active_key_signature_obj(part_obj, measure_number)
        return self._format_key_signature(ks)

    def get_active_tempo(
        self,
        measure_number: int,
        part: Optional[Union[int, str]] = None,
    ) -> Optional[float]:
        """Get the active tempo (BPM) at a given measure.

        Returns None if no tempo marking is found.
        """
        part_obj, _ = self._resolve_part(part)
        m = self._resolve_measure(part_obj, measure_number)
        current_marks = _measure_tempo_marks(m)
        if current_marks:
            return _tempo_number(current_marks[0])

        measures = sorted(
            part_obj.getElementsByClass(m21stream.Measure),
            key=lambda measure: measure.number,
        )
        for previous_measure in reversed(measures):
            previous_number = getattr(previous_measure, "number", None)
            if previous_number is None or previous_number >= measure_number:
                continue
            previous_marks = _measure_tempo_marks(previous_measure)
            if previous_marks:
                return _tempo_number(previous_marks[-1])

        return None

    @staticmethod
    def _format_key_signature(ks: m21key.KeySignature) -> str:
        """Format a KeySignature object as a human-readable string."""
        if is_open_key_signature(ks):
            return OPEN_KEY_SIGNATURE_LABEL
        if hasattr(ks, "tonic") and ks.tonic is not None:
            return f"{ks.tonic.name} {ks.mode}"
        if ks.sharps >= 0:
            return f"{ks.sharps} sharps"
        return f"{-ks.sharps} flats"

    # ------------------------------------------------------------------
    # Internal helpers used by all mixins
    # ------------------------------------------------------------------

    def _resolve_part(
        self,
        part: Optional[Union[int, str]] = None,
    ) -> tuple[m21stream.Part, int]:
        """Resolve a part identifier to a Part object and its index.

        Args:
            part: 0-based index, raw part name string, display label string
                such as ``"Piano LH"``, or None (first part).

        Returns:
            Tuple of (Part object, 0-based index).

        Raises:
            ValueError: If the part cannot be found.
        """
        parts = list(self._score.parts)
        if not parts:
            raise ValueError("Score has no parts.")

        if part is None:
            return parts[0], 0

        if isinstance(part, int):
            if part < 0 or part >= len(parts):
                raise ValueError(
                    f"Part index {part} is out of range "
                    f"(valid: 0–{len(parts) - 1}). "
                    f"Available parts: {[p.partName for p in parts]}"
                )
            return parts[part], part

        if isinstance(part, str):
            labels = build_part_display_labels(self._score)
            for i, label in labels.items():
                if label.display_name.lower() == part.lower():
                    return parts[i], i
            for i, p in enumerate(parts):
                if p.partName and p.partName.lower() == part.lower():
                    return p, i
            available_parts = [
                labels.get(i).display_name if labels.get(i) else p.partName
                for i, p in enumerate(parts)
            ]
            raise ValueError(
                f"No part named '{part}'. "
                f"Available parts: {available_parts}"
            )

        raise ValueError(
            f"Cannot resolve part identifier: {part} "
            f"(type: {type(part).__name__}). "
            f"Expected an integer index, a part name string, or None."
        )

    def _resolve_parts_or_all(
        self,
        part: Optional[Union[int, str]] = None,
    ) -> list[tuple[m21stream.Part, int]]:
        """Resolve to a specific part or all parts.

        If part is None, returns all parts. Otherwise resolves to one.
        """
        if part is None:
            parts = list(self._score.parts)
            if not parts:
                raise ValueError("Score has no parts.")
            return [(p, i) for i, p in enumerate(parts)]
        else:
            p, i = self._resolve_part(part)
            return [(p, i)]

    def _resolve_measure(
        self,
        part_obj: m21stream.Part,
        measure_number: int,
    ) -> m21stream.Measure:
        """Get a measure by its 1-based number from a part.

        Raises:
            ValueError: If the measure doesn't exist.
        """
        m = part_obj.measure(measure_number)
        if m is None:
            max_m = self._get_measure_count(part_obj)
            if max_m == 0:
                raise ValueError(
                    f"Measure {measure_number} does not exist — "
                    f"this part has no measures yet. "
                    f"Use add_measures() to create measures first."
                )
            raise ValueError(
                f"Measure {measure_number} does not exist. "
                f"This part has {max_m} measure(s) (1–{max_m})."
                f"Use add_measures() to append empty measures first."
            )
        return m

    def _get_measure_count(self, part_obj: m21stream.Part) -> int:
        """Count the number of measures in a part."""
        return len(list(part_obj.getElementsByClass(m21stream.Measure)))

    def _get_active_time_signature_obj(
        self,
        part_obj: m21stream.Part,
        measure_number: int,
    ) -> m21meter.TimeSignature:
        """Get the active TimeSignature object at a measure (implied continuity).

        Checks the measure's own elements first, then searches backward
        through the parent Part for inherited time signatures.
        """
        m = self._resolve_measure(part_obj, measure_number)
        if m.timeSignature is not None:
            return m.timeSignature
        ts = m.getContextByClass(m21meter.TimeSignature)
        if ts is not None:
            return ts
        return m21meter.TimeSignature("4/4")

    def _get_active_key_signature_obj(
        self,
        part_obj: m21stream.Part,
        measure_number: int,
    ) -> m21key.KeySignature:
        """Get the active KeySignature object at a measure.

        Checks within the measure first, then searches backward.
        """
        m = self._resolve_measure(part_obj, measure_number)
        ks_list = list(m.getElementsByClass(m21key.KeySignature))
        if ks_list:
            return ks_list[0]
        ks = m.getContextByClass(m21key.KeySignature)
        if ks is not None:
            return ks
        return m21key.Key("C", "major")

    def _get_active_clef_obj(
        self,
        part_obj: m21stream.Part,
        measure_number: int,
    ) -> m21clef.Clef:
        """Get the active Clef object at a measure.

        Checks within the measure first, then searches backward.
        """
        m = self._resolve_measure(part_obj, measure_number)
        cl_list = list(m.getElementsByClass(m21clef.Clef))
        if cl_list:
            return cl_list[0]
        cl = m.getContextByClass(m21clef.Clef)
        if cl is not None:
            return cl
        return m21clef.TrebleClef()

    def _get_voice_or_measure(
        self,
        measure: m21stream.Measure,
        voice: int = 1,
        create: bool = False,
    ) -> m21stream.Stream:
        """Get a voice within a measure, or the measure itself for voice 1.

        For voice 1, returns the measure directly (standard behavior).
        For other voices, finds or creates a Voice object.
        """
        voice = validate_voice_number(voice)
        if voice == 1:
            voices = list(measure.voices)
            if voices:
                for v in voices:
                    if str(v.id) == "1":
                        return v
                if create:
                    self._promote_direct_voice_one_content(measure)
                    for v in measure.voices:
                        if str(v.id) == "1":
                            return v
                    new_voice = m21stream.Voice(id="1")
                    measure.insert(0, new_voice)
                    return new_voice
                return measure
            return measure

        if create:
            self._promote_direct_voice_one_content(measure)

        voices = list(measure.voices)
        for v in voices:
            if str(v.id) == str(voice):
                return v

        if create:
            new_voice = m21stream.Voice(id=str(voice))
            measure.insert(0, new_voice)
            return new_voice

        voice_ids = [v.id for v in voices] if voices else ["1"]
        raise ValueError(
            f"Voice {voice} not found in this measure. "
            f"Available voices: {voice_ids}. "
            "Inspect the measure to choose an available voice, or use an "
            "appropriate add_* tool to create material in a new voice."
        )

    def _promote_direct_voice_one_content(
        self,
        measure: m21stream.Measure,
    ) -> None:
        """Move direct note-like measure content into an explicit voice 1.

        music21 exports correct MusicXML voice/back-up structure only when
        parallel material is represented as sibling ``Voice`` streams. Surgical
        edits start with voice 1 as direct measure content, so adding another
        voice must first promote those events into ``Voice(id="1")``.
        """
        direct_events = list(measure.getElementsByClass(m21note.GeneralNote))
        if not direct_events:
            return

        voice_one = None
        for existing_voice in measure.voices:
            if str(existing_voice.id) == "1":
                voice_one = existing_voice
                break

        if voice_one is None:
            voice_one = m21stream.Voice(id="1")
            measure.insert(0, voice_one)

        event_offsets = [
            (event, float(measure.elementOffset(event)))
            for event in direct_events
        ]
        for event, offset in sorted(event_offsets, key=lambda item: item[1]):
            measure.remove(event)
            voice_one.insert(offset, event)

    def _get_used_quarter_lengths(
        self,
        container: m21stream.Stream,
    ) -> float:
        """Get the end offset of duration-bearing notes/chords in a stream."""
        used = 0.0
        for element in container.getElementsByClass(m21note.GeneralNote):
            if isinstance(element, m21note.Rest):
                continue
            if getattr(element.duration, "isGrace", False):
                continue
            offset = float(container.elementOffset(element))
            used = max(used, offset + float(element.duration.quarterLength))
        return used

    def __repr__(self) -> str:
        parts = list(self._score.parts)
        part_names = [p.partName or f"Part {i}" for i, p in enumerate(parts)]
        return (
            f"ScoreSpeak(title='{self.title}', "
            f"parts={part_names}, "
            f"measures={self.measure_count})"
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _parse_key_signature(key_str: str) -> m21key.KeySignature:
    """Parse a flexible key signature string into a music21 object.

    Accepts: "C", "C major", "A minor", "Bb", "F#m", "3" (sharps), "-2" (flats).
    """
    text = key_str.strip()
    normalized = text.lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    if normalized in {"open", "atonal", "open atonal", "open/atonal", "none"}:
        return mark_open_key_signature(m21key.KeySignature(0))

    for uc, ac in {"\u266f": "#", "\u266d": "b"}.items():
        text = text.replace(uc, ac)

    try:
        val = int(text)
        return m21key.KeySignature(val)
    except ValueError:
        pass

    if text.endswith("m") and not text.endswith("major") and not text.endswith("minor"):
        tonic = text[:-1].strip()
        return m21key.Key(tonic, "minor")

    parts = text.split()
    if len(parts) == 2:
        tonic, mode = parts
        return m21key.Key(tonic, mode.lower())

    if len(parts) == 1:
        return m21key.Key(parts[0], "major")

    raise ValueError(
        f"Cannot parse key signature '{key_str}'. "
        f"Expected formats: 'C', 'C major', 'A minor', 'Bb', 'F#m', "
        f"or an integer for sharps/flats."
    )
