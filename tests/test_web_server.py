"""Tests for the web app MusicXML serving behavior."""

import io
import shutil
import subprocess
import textwrap
import zipfile
from pathlib import Path
from typing import Any

import pytest
from lxml import etree

import web.server as web_server
from scorespeak import ScoreSpeak
from scorespeak.agent.defaults import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_AGENT_REASONING_EFFORT,
    agent_reasoning_effort_options_payload,
    chat_openai_reasoning_kwargs,
    agent_model_options_payload,
    normalize_agent_reasoning_effort,
    normalize_agent_model,
)
from scorespeak.agent.prompt_split import should_use_prompt_split
from scorespeak.types import OperationResult
from scorespeak.voice import (
    AudioInput,
    SpeechTranscript,
    VoiceProcessingResult,
    VoiceRequest,
    VoiceWarning,
)
from web.musicxml_window import (
    export_scorespeak_window_musicxml,
    extract_musicxml_window,
)
from web.server import (
    AgentSession,
    _extract_changed_measure_range,
    _show_rests_for_empty_space,
    app,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_app_js_assertions(assertion_script: str) -> None:
    """Run browser-renderer assertions against app.js in a minimal Node context."""
    node_path = shutil.which("node")
    if node_path is None:
        pytest.skip("Node.js is required for app.js behavior assertions")

    script = f"""
        const assert = require('assert');
        const fs = require('fs');
        const vm = require('vm');
        const sharedSource = fs.readFileSync('web/vendor/osmd_musicxml_fixes.js', 'utf8');
        const source = fs.readFileSync('web/app.js', 'utf8');
        const context = {{
            console,
            setTimeout,
            clearTimeout,
            performance: {{ now: () => 0 }},
            document: {{ addEventListener: () => {{}} }},
            window: {{
                innerWidth: 1024,
                location: {{ origin: 'http://localhost' }},
                requestIdleCallback: (callback) => callback(),
            }},
        }};
        vm.createContext(context);
        vm.runInContext(sharedSource, context);
        vm.runInContext(
            `${{source}}\\nglobalThis.MusicXMLRenderer = MusicXMLRenderer;`,
            context
        );
        {assertion_script}
    """
    subprocess.run(
        [node_path, "-e", textwrap.dedent(script)],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _fake_dom_parser_js() -> str:
    """Return JS test helpers for minimal MusicXML DOMParser behavior."""
    return """
        let parseCalls = 0;

        function fakeElement(localName, children = [], attrs = {}, textContent = '') {
            return {
                localName,
                nodeName: localName,
                nodeType: 1,
                children,
                childNodes: children,
                textContent,
                getAttribute: (name) => Object.prototype.hasOwnProperty.call(attrs, name)
                    ? attrs[name]
                    : null,
                getElementsByTagName: (name) => descendantsByTag({children}, name),
            };
        }

        function descendantsByTag(node, name) {
            const matches = [];
            const visit = (current) => {
                Array.from(current.children || []).forEach((child) => {
                    if (child.localName === name) {
                        matches.push(child);
                    }
                    visit(child);
                });
            };
            visit(node);
            return matches;
        }

        function directionWith(child) {
            return fakeElement('direction', [
                fakeElement('direction-type', [child]),
            ]);
        }

        function fakeAttrs(attrsText) {
            const attrs = {};
            const attrPattern = /([:\\w.-]+)=["']([^"']*)["']/gi;
            let attrMatch = attrPattern.exec(attrsText || '');
            while (attrMatch) {
                attrs[attrMatch[1]] = attrMatch[2];
                attrMatch = attrPattern.exec(attrsText || '');
            }
            return attrs;
        }

        function fakeNote(content) {
            const children = [];
            if (/<\\s*grace\\b/i.test(content)) {
                children.push(fakeElement('grace'));
            }
            if (/<\\s*chord\\b/i.test(content)) {
                children.push(fakeElement('chord'));
            }
            if (/<\\s*rest\\b/i.test(content)) {
                children.push(fakeElement('rest'));
            }

            const glissandos = [];
            const glissandoPattern = /<\\s*glissando\\b([^>]*)\\/?>/gi;
            let glissandoMatch = glissandoPattern.exec(content);
            while (glissandoMatch) {
                glissandos.push(
                    fakeElement('glissando', [], fakeAttrs(glissandoMatch[1]))
                );
                glissandoMatch = glissandoPattern.exec(content);
            }
            if (glissandos.length) {
                children.push(fakeElement('notations', glissandos));
            }
            const caesuras = [];
            const caesuraPattern = /<\\s*caesura\\b[^>]*\\/?>/gi;
            let caesuraMatch = caesuraPattern.exec(content);
            while (caesuraMatch) {
                caesuras.push(fakeElement('caesura'));
                caesuraMatch = caesuraPattern.exec(content);
            }
            if (caesuras.length) {
                children.push(fakeElement('notations', caesuras));
            }
            return fakeElement('note', children);
        }

        function fakeMeasure(content) {
            const children = [];
            if (/<\\s*coda\\b/i.test(content)) {
                children.push(directionWith(fakeElement('coda')));
            }
            if (/<\\s*segno\\b/i.test(content)) {
                children.push(directionWith(fakeElement('segno')));
            }
            const wordsPattern = /<\\s*words\\b[^>]*>([\\s\\S]*?)<\\/\\s*words\\s*>/gi;
            let wordsMatch = wordsPattern.exec(content);
            while (wordsMatch) {
                children.push(directionWith(fakeElement('words', [], {}, wordsMatch[1])));
                wordsMatch = wordsPattern.exec(content);
            }
            if (/<\\s*repeat\\b[^>]*direction=["']backward["']/i.test(content)) {
                children.push(fakeElement('barline', [
                    fakeElement('repeat', [], {direction: 'backward'}),
                ]));
            }
            const notePattern = /<\\s*note\\b[^>]*>([\\s\\S]*?)<\\/\\s*note\\s*>/gi;
            let noteMatch = notePattern.exec(content);
            while (noteMatch) {
                children.push(fakeNote(noteMatch[1]));
                noteMatch = notePattern.exec(content);
            }
            return fakeElement('measure', children);
        }

        function fakePart(content) {
            const measures = [];
            const measurePattern = /<\\s*measure\\b[^>]*>([\\s\\S]*?)<\\/\\s*measure\\s*>/gi;
            let measureMatch = measurePattern.exec(content);
            while (measureMatch) {
                measures.push(fakeMeasure(measureMatch[1]));
                measureMatch = measurePattern.exec(content);
            }
            return fakeElement('part', measures);
        }

        function fakeDocumentFor(xml) {
            if (xml.includes('MALFORMED_XML')) {
                const parserError = fakeElement('parsererror');
                return {
                    getElementsByTagName: (name) => name === 'parsererror' ? [parserError] : [],
                };
            }

            const parts = [];
            const partPattern = /<\\s*part\\b[^>]*>([\\s\\S]*?)<\\/\\s*part\\s*>/gi;
            let partMatch = partPattern.exec(xml);
            while (partMatch) {
                parts.push(fakePart(partMatch[1]));
                partMatch = partPattern.exec(xml);
            }
            return {
                getElementsByTagName: (name) => {
                    if (name === 'parsererror') return [];
                    if (name === 'part') return parts;
                    return descendantsByTag({children: parts}, name);
                },
            };
        }

        context.DOMParser = class {
            parseFromString(xml) {
                parseCalls += 1;
                return fakeDocumentFor(xml);
            }
        };
    """


def _fake_svg_point_helpers_js() -> str:
    """Return JS helpers for SVG coordinate assertions."""
    return """
        function fakeSvg() {
            return {
                createSVGPoint: () => ({
                    x: 0,
                    y: 0,
                    matrixTransform(matrix) {
                        return {
                            x: this.x * (matrix.a ?? 1) + this.y * (matrix.c ?? 0) + (matrix.e ?? 0),
                            y: this.x * (matrix.b ?? 0) + this.y * (matrix.d ?? 1) + (matrix.f ?? 0),
                        };
                    },
                }),
                getScreenCTM: () => ({
                    inverse: () => ({ a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 }),
                }),
            };
        }

        function fakeStavenote(svg, rect) {
            const notehead = {
                getBoundingClientRect: () => rect,
            };
            return {
                ownerSVGElement: svg,
                querySelector: () => notehead,
            };
        }
    """


def _hidden_rest_notes(musicxml: str) -> list:
    """Return hidden rest notes from a MusicXML string."""
    root = etree.fromstring(musicxml.encode("utf-8"))
    return root.xpath(
        ".//*[local-name()='note'][@print-object='no'][*[local-name()='rest']]"
    )


def _visible_rest_notes(musicxml: str) -> list:
    """Return visible rest notes from a MusicXML string."""
    root = etree.fromstring(musicxml.encode("utf-8"))
    return root.xpath(
        ".//*[local-name()='note'][not(@print-object='no')][*[local-name()='rest']]"
    )


def _musicxml_endings(musicxml: str) -> list:
    """Return repeat ending elements from a MusicXML string."""
    return _musicxml_elements(musicxml, "ending")


def _musicxml_elements(musicxml: str, local_name: str) -> list:
    """Return elements with a given local name from a MusicXML string."""
    root = etree.fromstring(musicxml.encode("utf-8"))
    return root.xpath(f".//*[local-name()='{local_name}']")


def _clef_summaries(attributes: etree._Element) -> list[tuple[str | None, str, str]]:
    """Return clef number, sign, and line triples from an attributes element."""
    return [
        (
            clef.get("number"),
            "".join(clef.xpath("./*[local-name()='sign']/text()")),
            "".join(clef.xpath("./*[local-name()='line']/text()")),
        )
        for clef in attributes.xpath("./*[local-name()='clef']")
    ]


def test_web_musicxml_keeps_empty_measure_rests_visible() -> None:
    """Web MusicXML normalization keeps default full-measure rests visible."""
    ss = ScoreSpeak.create(measures=1)

    raw_xml = ss.to_musicxml_string()
    normalized_xml = _show_rests_for_empty_space(raw_xml)

    assert _visible_rest_notes(raw_xml)
    assert _visible_rest_notes(normalized_xml)
    assert not _hidden_rest_notes(normalized_xml)


def test_web_musicxml_preserves_mid_measure_visible_rests() -> None:
    """Web MusicXML normalization should preserve populated-staff visible rests."""
    ss = ScoreSpeak.create(measures=1)
    ss._add_note_one("C4", "quarter", measure=1, beat=3)

    raw_xml = ss.to_musicxml_string()
    normalized_xml = _show_rests_for_empty_space(raw_xml)

    assert len(_visible_rest_notes(raw_xml)) >= 2
    assert len(_visible_rest_notes(normalized_xml)) == len(
        _visible_rest_notes(raw_xml)
    )


def test_web_musicxml_reveals_only_empty_staff_rests() -> None:
    """Web MusicXML normalization should reveal hidden rests on empty staffs."""
    raw_xml = textwrap.dedent(
        """\
        <?xml version="1.0" encoding="UTF-8"?>
        <score-partwise version="4.0">
          <part-list>
            <score-part id="P1">
              <part-name>Piano</part-name>
            </score-part>
          </part-list>
          <part id="P1">
            <measure number="1">
              <attributes>
                <divisions>1</divisions>
                <key><fifths>0</fifths></key>
                <time><beats>4</beats><beat-type>4</beat-type></time>
                <staves>2</staves>
                <clef number="1"><sign>G</sign><line>2</line></clef>
                <clef number="2"><sign>F</sign><line>4</line></clef>
              </attributes>
              <note>
                <pitch><step>C</step><octave>4</octave></pitch>
                <duration>4</duration>
                <voice>1</voice>
                <type>whole</type>
                <staff>1</staff>
              </note>
              <backup><duration>4</duration></backup>
              <note print-object="no" print-spacing="yes">
                <rest/>
                <duration>4</duration>
                <voice>2</voice>
                <type>whole</type>
                <staff>1</staff>
              </note>
              <backup><duration>4</duration></backup>
              <note print-object="no" print-spacing="yes">
                <rest/>
                <duration>4</duration>
                <voice>5</voice>
                <type>whole</type>
                <staff>2</staff>
              </note>
            </measure>
          </part>
        </score-partwise>
        """
    )

    normalized_xml = _show_rests_for_empty_space(raw_xml)
    root = etree.fromstring(normalized_xml.encode("utf-8"))
    staff_1_hidden_rests = root.xpath(
        ".//*[local-name()='note'][@print-object='no'][*[local-name()='rest']]"
        "[*[local-name()='staff' and normalize-space(.)='1']]"
    )
    staff_2_rests = root.xpath(
        ".//*[local-name()='note'][*[local-name()='rest']]"
        "[*[local-name()='staff' and normalize-space(.)='2']]"
    )

    assert len(staff_1_hidden_rests) == 1
    assert len(staff_2_rests) == 1
    assert staff_2_rests[0].get("print-object") is None
    assert staff_2_rests[0].get("print-spacing") is None


def _make_mxl(musicxml: str) -> bytes:
    """Package a MusicXML string as a compressed MXL archive."""
    buffer = io.BytesIO()
    container_xml = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="score.musicxml" media-type="application/vnd.recordare.musicxml+xml"/>
  </rootfiles>
</container>
"""

    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/vnd.recordare.musicxml",
            compress_type=zipfile.ZIP_STORED,
        )
        archive.writestr("META-INF/container.xml", container_xml)
        archive.writestr("score.musicxml", musicxml)

    return buffer.getvalue()


def test_load_score_accepts_compressed_mxl_upload() -> None:
    """The web upload endpoint should accept compressed .mxl scores."""
    musicxml = ScoreSpeak.create(measures=1).to_musicxml_string()
    mxl_bytes = _make_mxl(musicxml)

    client = app.test_client()
    response = client.post(
        "/api/load",
        data={
            "musicxml": (
                io.BytesIO(mxl_bytes),
                "score.mxl",
            ),
        },
        content_type="multipart/form-data",
    )

    data = response.get_json()
    assert response.status_code == 200
    assert data["success"] is True
    assert data["part_count"] == 1
    assert data["measure_count"] == 1
    assert data["render_range"] == {"start": 1, "end": 1}
    assert data["score_version"] == 0
    assert "<score-partwise" in data["musicxml"]


def test_agent_model_defaults_and_validation() -> None:
    """Shared agent model helpers should expose and validate selectable models."""
    options = agent_model_options_payload()
    ids = [option["id"] for option in options]
    reasoning_options = agent_reasoning_effort_options_payload()
    reasoning_ids = [option["id"] for option in reasoning_options]

    assert DEFAULT_AGENT_MODEL == "gpt-5.4-mini"
    assert DEFAULT_AGENT_REASONING_EFFORT == "low"
    assert ids == ["gpt-5.4-mini", "gpt-5.4", "gpt-5.4-nano"]
    assert reasoning_ids == ["none", "minimal", "low", "medium", "high"]
    assert normalize_agent_model(None) == DEFAULT_AGENT_MODEL
    assert normalize_agent_model("") == DEFAULT_AGENT_MODEL
    assert normalize_agent_model("gpt-5.4") == "gpt-5.4"
    assert normalize_agent_reasoning_effort(None) == DEFAULT_AGENT_REASONING_EFFORT
    assert normalize_agent_reasoning_effort("") == DEFAULT_AGENT_REASONING_EFFORT
    assert normalize_agent_reasoning_effort("api_default") == "api_default"
    assert normalize_agent_reasoning_effort("high") == "high"
    assert chat_openai_reasoning_kwargs("api_default") == {
        "use_responses_api": True,
        "output_version": "responses/v1",
    }
    assert chat_openai_reasoning_kwargs("low") == {
        "use_responses_api": True,
        "output_version": "responses/v1",
        "reasoning": {"effort": "low"},
    }
    with pytest.raises(ValueError):
        normalize_agent_model("gpt-4o")
    with pytest.raises(ValueError):
        normalize_agent_reasoning_effort("xhigh")


def test_chat_openai_reasoning_kwargs_route_tool_calls_to_responses_api() -> None:
    """Reasoning tool calls should use the Responses payload shape."""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=DEFAULT_AGENT_MODEL,
        api_key="test-key",
        **chat_openai_reasoning_kwargs("low"),
    )
    tool = {
        "type": "function",
        "function": {
            "name": "find_bars",
            "description": "Find score bars.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }

    payload = llm._get_request_payload(  # noqa: SLF001
        [HumanMessage(content="Find bar 1.")],
        tools=[tool],
    )

    assert "input" in payload
    assert "messages" not in payload
    assert payload["reasoning"] == {"effort": "low"}
    assert "reasoning_effort" not in payload
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "find_bars",
            "description": "Find score bars.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
    ]


def test_musicxml_window_trims_measures_and_carries_attributes() -> None:
    """Windowed MusicXML should keep active attributes at the first measure."""
    ss = ScoreSpeak.create(
        parts=["Piano", "Violin"],
        measures=4,
        time_signature="4/4",
        key_signature="C",
    )
    ss.set_time_signature("3/4", measure_number=3)
    ss.set_key_signature("G", measure_number=3)

    window = extract_musicxml_window(ss.to_musicxml_string(), 3, 4)
    root = etree.fromstring(window.encode("utf-8"))
    parts = root.xpath(".//*[local-name()='part']")

    assert len(parts) == 2
    for part in parts:
        measures = part.xpath("./*[local-name()='measure']")
        assert len(measures) == 2
        assert measures[0].get("number") == "3"
        first_attributes = measures[0].xpath("./*[local-name()='attributes']")[0]
        time = first_attributes.xpath("./*[local-name()='time']")
        key = first_attributes.xpath("./*[local-name()='key']")
        assert time
        assert key


def test_musicxml_window_preserves_numbered_clefs_when_carrying_attributes() -> None:
    """Windowed MusicXML should merge active clefs independently per staff."""
    musicxml = textwrap.dedent(
        """\
        <?xml version="1.0" encoding="UTF-8"?>
        <score-partwise version="4.0">
          <part-list>
            <score-part id="P1">
              <part-name>Piano</part-name>
            </score-part>
          </part-list>
          <part id="P1">
            <measure number="1">
              <attributes>
                <divisions>1</divisions>
                <key><fifths>0</fifths></key>
                <time><beats>4</beats><beat-type>4</beat-type></time>
                <staves>2</staves>
                <clef number="1"><sign>G</sign><line>2</line></clef>
                <clef number="2"><sign>G</sign><line>2</line></clef>
              </attributes>
              <note>
                <rest/><duration>4</duration><type>whole</type><staff>1</staff>
              </note>
            </measure>
            <measure number="2">
              <attributes>
                <clef number="2"><sign>F</sign><line>4</line></clef>
              </attributes>
              <note>
                <rest/><duration>4</duration><type>whole</type><staff>2</staff>
              </note>
            </measure>
          </part>
        </score-partwise>
        """
    )

    window = extract_musicxml_window(musicxml, 2, 2)
    root = etree.fromstring(window.encode("utf-8"))
    attributes = root.xpath(
        ".//*[local-name()='part'][1]/*[local-name()='measure'][1]"
        "/*[local-name()='attributes']"
    )[0]
    clefs = _clef_summaries(attributes)

    assert ("1", "G", "2") in clefs
    assert ("2", "F", "4") in clefs
    assert ("2", "G", "2") not in clefs


def test_musicxml_window_preserves_delius_piano_staff_clefs() -> None:
    """The Delius benchmark window should retain both initial piano clefs."""
    source_path = (
        _REPO_ROOT
        / "datasets/scores/delius_it_was_a_lover_and_his_lass_bars001-064.musicxml"
    )
    window = extract_musicxml_window(
        source_path.read_text(encoding="utf-8"),
        23,
        25,
    )
    root = etree.fromstring(window.encode("utf-8"))
    piano = root.xpath("./*[local-name()='part'][@id='P2']")[0]
    measure = piano.xpath("./*[local-name()='measure'][1]")[0]
    attributes = measure.xpath("./*[local-name()='attributes']")

    assert measure.get("number") == "23"
    assert _clef_summaries(attributes[0]) == [("1", "G", "2"), ("2", "F", "4")]
    assert _clef_summaries(attributes[1]) == [("2", "G", "2")]
    assert measure.index(attributes[1]) > measure.index(attributes[0])


def test_musicxml_window_inserts_carried_attributes_before_mid_measure_attributes() -> None:
    """Carried attributes should not be merged into mid-measure attributes."""
    musicxml = textwrap.dedent(
        """\
        <?xml version="1.0" encoding="UTF-8"?>
        <score-partwise version="4.0">
          <part-list>
            <score-part id="P1">
              <part-name>Piano</part-name>
            </score-part>
          </part-list>
          <part id="P1">
            <measure number="1">
              <attributes>
                <divisions>1</divisions>
                <key><fifths>0</fifths></key>
                <time><beats>4</beats><beat-type>4</beat-type></time>
                <clef><sign>G</sign><line>2</line></clef>
              </attributes>
              <note><rest/><duration>4</duration><type>whole</type></note>
            </measure>
            <measure number="2">
              <direction>
                <direction-type><words>cresc.</words></direction-type>
              </direction>
              <note><rest/><duration>1</duration><type>quarter</type></note>
              <attributes><clef><sign>F</sign><line>4</line></clef></attributes>
              <note><rest/><duration>3</duration><type>half</type><dot/></note>
            </measure>
          </part>
        </score-partwise>
        """
    )

    window = extract_musicxml_window(musicxml, 2, 2)
    root = etree.fromstring(window.encode("utf-8"))
    measure = root.xpath(".//*[local-name()='part'][1]/*[local-name()='measure']")[0]
    element_names = [
        etree.QName(child).localname
        for child in measure
        if isinstance(child.tag, str)
    ]
    attributes = measure.xpath("./*[local-name()='attributes']")

    assert element_names[:2] == ["attributes", "direction"]
    assert len(attributes) == 2
    assert attributes[0].xpath("./*[local-name()='divisions']")
    assert attributes[0].xpath("./*[local-name()='time']")
    assert attributes[1].xpath(
        "./*[local-name()='clef']/*[local-name()='sign' and text()='F']"
    )


def test_musicxml_window_preserves_source_layout_breaks() -> None:
    """Windowed MusicXML should preserve explicit source page/system breaks."""
    musicxml = textwrap.dedent(
        """\
        <?xml version="1.0" encoding="UTF-8"?>
        <score-partwise version="4.0">
          <part-list>
            <score-part id="P1">
              <part-name>Piano</part-name>
            </score-part>
          </part-list>
          <part id="P1">
            <measure number="1">
              <attributes>
                <divisions>1</divisions>
                <key><fifths>0</fifths></key>
                <time><beats>4</beats><beat-type>4</beat-type></time>
                <clef><sign>G</sign><line>2</line></clef>
              </attributes>
              <note><rest/><duration>4</duration><type>whole</type></note>
            </measure>
            <measure number="2">
              <print new-page="yes" new-system="yes" page-number="12">
                <staff-layout number="1">
                  <staff-distance>80</staff-distance>
                </staff-layout>
              </print>
              <note><rest/><duration>4</duration><type>whole</type></note>
            </measure>
          </part>
        </score-partwise>
        """
    )

    window = extract_musicxml_window(musicxml, 1, 2)
    root = etree.fromstring(window.encode("utf-8"))
    print_elements = root.xpath(".//*[local-name()='print']")

    assert len(print_elements) == 1
    assert print_elements[0].get("new-page") == "yes"
    assert print_elements[0].get("new-system") == "yes"
    assert print_elements[0].get("page-number") == "12"
    assert print_elements[0].xpath("./*[local-name()='staff-layout']")


def test_musicxml_window_endpoint_returns_requested_range() -> None:
    """The window endpoint returns a standalone render slice."""
    client = app.test_client()
    client.post("/api/new", json={"measures": 12})

    response = client.get("/api/musicxml/window?start=5&end=8")
    data = response.get_json()
    root = etree.fromstring(data["musicxml"].encode("utf-8"))
    measures = root.xpath(".//*[local-name()='part'][1]/*[local-name()='measure']")

    assert response.status_code == 200
    assert data["success"] is True
    assert data["render_range"] == {"start": 5, "end": 8}
    assert [measure.get("number") for measure in measures] == ["5", "6", "7", "8"]


def test_new_score_defaults_to_piano_grand_staff() -> None:
    """The main web app creates a RH/LH piano grand staff by default."""
    client = app.test_client()

    response = client.post("/api/new", json={"measures": 8})
    data = response.get_json()

    assert response.status_code == 200
    assert data["success"] is True
    assert data["part_count"] == 2
    assert data["measure_count"] == 8
    loaded = ScoreSpeak.from_musicxml(data["musicxml"])
    assert [part.display_name for part in loaded.list_parts()] == [
        "Piano RH",
        "Piano LH",
    ]


def test_new_score_at_measure_part_target_renders_full_score() -> None:
    """Scores at exactly the measure-part target should not use paging."""
    client = app.test_client()

    response = client.post("/api/new", json={"measures": 100})
    data = response.get_json()

    assert response.status_code == 200
    assert data["success"] is True
    assert data["render_range"] == {"start": 1, "end": 100}


def test_new_score_above_measure_part_target_renders_window() -> None:
    """Scores above the measure-part target should render one target-sized window."""
    client = app.test_client()

    response = client.post("/api/new", json={"measures": 101})
    data = response.get_json()
    root = etree.fromstring(data["musicxml"].encode("utf-8"))
    measures = root.xpath(".//*[local-name()='part'][1]/*[local-name()='measure']")

    assert response.status_code == 200
    assert data["success"] is True
    assert data["render_range"] == {"start": 1, "end": 100}
    assert len(measures) == 100
    assert measures[0].get("number") == "1"
    assert measures[-1].get("number") == "100"


def test_web_sample_musicxml_is_piano_grand_staff() -> None:
    """The bundled sample score should expose RH/LH piano labels."""
    sample_path = Path("web/sample.musicxml")
    score_state = ScoreSpeak.from_musicxml(sample_path)
    musicxml = sample_path.read_text(encoding="utf-8")

    assert "<staves>2</staves>" in musicxml
    assert "<part-group" not in musicxml
    assert [part.display_name for part in score_state.list_parts()] == [
        "Piano RH",
        "Piano LH",
    ]


def test_window_export_preserves_piano_grand_staff_after_adding_part() -> None:
    """Windowed redraws should keep the normal two-staff piano shape."""
    score_state = ScoreSpeak.create(
        parts=[{"instrument": "piano", "name": "Piano", "grand_staff": True}],
        measures=8,
    )
    score_state.add_part(instrument="violin")

    musicxml = export_scorespeak_window_musicxml(score_state, 1, 8)
    loaded_window = ScoreSpeak.from_musicxml(musicxml)

    assert "<staves>2</staves>" in musicxml
    assert "<part-group" not in musicxml
    assert [part.display_name for part in loaded_window.list_parts()[:2]] == [
        "Piano RH",
        "Piano LH",
    ]


def test_scorespeak_window_export_preserves_ending_brackets() -> None:
    """Live render windows should include ending brackets from score spanners."""
    ss = ScoreSpeak.create(measures=4)
    for measure_number in range(1, 5):
        ss._add_note_one("C4", measure=measure_number)
    ss.add_ending_bracket(1, start_measure=2, end_measure=3)

    window = export_scorespeak_window_musicxml(ss, 2, 3)

    endings = _musicxml_endings(window)
    ending_payload = [
        (ending.get("number"), ending.get("type"), ending.text)
        for ending in endings
    ]
    assert ending_payload == [
        ("1", "start", None),
        ("1", "stop", None),
    ]


@pytest.mark.parametrize(
    ("tool_name", "apply_tool", "local_name"),
    [
        ("slur", lambda ss: ss.add_slur(2, 1, 3, 1), "slur"),
        ("hairpin", lambda ss: ss.add_hairpin("crescendo", 2, 1, 3, 1), "wedge"),
        ("ottava", lambda ss: ss.add_ottava("8va", 2, 1, 3, 1), "octave-shift"),
        ("glissando", lambda ss: ss.add_glissando(2, 1, 3, 1), "glissando"),
        ("pedal", lambda ss: ss.add_pedal(2, 1, 3, 1), "pedal"),
    ],
)
def test_scorespeak_window_export_preserves_supported_spanners(
    tool_name: str,
    apply_tool: Any,
    local_name: str,
) -> None:
    """Live render windows should preserve supported part-level spanners."""
    ss = ScoreSpeak.create(measures=4)
    for measure_number in range(1, 5):
        ss._add_note_one("C4", measure=measure_number, beat=1)
        ss._add_note_one("D4", measure=measure_number, beat=2)
    apply_tool(ss)

    full_musicxml = ss.to_musicxml_string()
    window_musicxml = export_scorespeak_window_musicxml(ss, 2, 3)

    assert len(_musicxml_elements(full_musicxml, local_name)) == 2, tool_name
    assert len(_musicxml_elements(window_musicxml, local_name)) == 2, tool_name


def test_agent_session_window_after_ending_bracket_edit() -> None:
    """Cold live render windows should show ending brackets after mutations."""
    ss = ScoreSpeak.create(measures=4)
    for measure_number in range(1, 5):
        ss._add_note_one("C4", measure=measure_number)
    session = AgentSession(ss)

    ss.add_ending_bracket(1, start_measure=2, end_measure=3)
    session.score_version += 1
    session._clear_render_caches()

    window = session.get_musicxml_window(2, 3)

    endings = _musicxml_endings(window)
    ending_payload = [
        (ending.get("number"), ending.get("type"), ending.text)
        for ending in endings
    ]
    assert ending_payload == [
        ("1", "start", None),
        ("1", "stop", None),
    ]


def test_web_renderer_does_not_double_filter_windowed_musicxml() -> None:
    """The browser should render the server-provided window without OSMD filtering."""
    app_js = Path("web/app.js").read_text(encoding="utf-8")

    assert "drawFromMeasureNumber" not in app_js
    assert "drawUpToMeasureNumber" not in app_js


def test_web_renderer_honors_musicxml_layout_breaks() -> None:
    """OSMD should use system and page break tags emitted by ScoreSpeak."""
    app_js = Path("web/app.js").read_text(encoding="utf-8")

    assert "newSystemFromXML: true" in app_js
    assert "newPageFromXML: true" in app_js


def test_web_renderer_applies_osmd_fallbacks_after_render() -> None:
    """SVG repairs should run after OSMD has emitted noteheads."""
    app_js = Path("web/app.js").read_text(encoding="utf-8")
    index_html = Path("web/index.html").read_text(encoding="utf-8")
    helper_js = Path("web/vendor/osmd_musicxml_fixes.js").read_text(encoding="utf-8")

    assert "this.osmd.render();" in app_js
    assert "applyOsmdPostRenderCorrectionsAfterSettled(xmlString)" in app_js
    assert "applyOsmdPostRenderCorrectionsAfterSettled(this.currentMusicXML)" in app_js
    assert "OSMD_POST_RENDER_FIX_TOKEN_PATTERN.test(xmlString || '')" in app_js
    assert "waitForOsmdDomSettled()" in app_js
    assert "ScoreSpeakOsmdFixes.applyPostRender(this.container, xmlString)" in app_js
    assert "/vendor/osmd_musicxml_fixes.js?v=osmd-fixes-5" in index_html
    assert "app.js?v=post-render-fallback-1" in index_html
    assert "scorespeak-glissando-fallback" in helper_js
    assert "scorespeak-caesura-fallback" in helper_js


def test_web_serves_osmd_musicxml_fixes() -> None:
    """The main web app should serve its OSMD fix helper."""
    client = app.test_client()

    response = client.get("/vendor/osmd_musicxml_fixes.js")

    assert response.status_code == 200
    assert b"ScoreSpeakOsmdFixes" in response.data


def test_web_renderer_pairs_musicxml_glissandos_in_render_order() -> None:
    """The fallback should pair glissando anchors against rendered note order."""
    _run_app_js_assertions(
        _fake_dom_parser_js()
        + """
        const xmlDocument = new context.DOMParser().parseFromString(`
            <score-partwise>
              <part id="P1">
                <measure number="1">
                  <note>
                    <pitch/>
                    <notations><glissando number="4" type="start" line-type="wavy"/></notations>
                  </note>
                </measure>
                <measure number="2">
                  <note>
                    <pitch/>
                    <notations><glissando number="4" type="stop" line-type="wavy"/></notations>
                  </note>
                </measure>
              </part>
              <part id="P2">
                <measure number="1"><note><pitch/></note></measure>
              </part>
            </score-partwise>
        `);

        assert.deepStrictEqual(
            JSON.parse(JSON.stringify(
                context.ScoreSpeakOsmdFixes._internal.musicXmlGlissandoSegments(xmlDocument)
            )),
            [
                { startIndex: 0, endIndex: 2, lineType: 'wavy' },
            ]
        );
        """
    )


def test_web_renderer_keeps_chord_glissandos_on_one_rendered_anchor() -> None:
    """Chord member tags should not advance the SVG stavenote index."""
    _run_app_js_assertions(
        _fake_dom_parser_js()
        + """
        const xmlDocument = new context.DOMParser().parseFromString(`
            <score-partwise>
              <part>
                <measure>
                  <note>
                    <pitch/>
                    <notations><glissando number="1" type="start" line-type="solid"/></notations>
                  </note>
                  <note><chord/><pitch/></note>
                  <note>
                    <pitch/>
                    <notations><glissando number="1" type="stop" line-type="solid"/></notations>
                  </note>
                </measure>
              </part>
            </score-partwise>
        `);

        assert.deepStrictEqual(
            JSON.parse(JSON.stringify(
                context.ScoreSpeakOsmdFixes._internal.musicXmlGlissandoSegments(xmlDocument)
            )),
            [
                { startIndex: 0, endIndex: 1, lineType: 'solid' },
            ]
        );
        """
    )


def test_web_renderer_glissando_points_keep_original_svg_bbox_coordinates() -> None:
    """Glissando fallback coordinates should stay on the original SVG bbox path."""
    _run_app_js_assertions(
        """
        const svg = {};
        const notehead = {
            getBBox: () => ({ x: 20, y: 40, width: 12, height: 10 }),
            getBoundingClientRect: () => ({
                left: 200,
                top: 400,
                right: 212,
                bottom: 410,
                width: 12,
                height: 10,
            }),
        };
        const stavenote = {
            ownerSVGElement: svg,
            querySelector: () => notehead,
        };

        const startPoint = context.ScoreSpeakOsmdFixes._internal.stavenoteGlissandoPoint(
            stavenote,
            'start'
        );
        const endPoint = context.ScoreSpeakOsmdFixes._internal.stavenoteGlissandoPoint(
            stavenote,
            'end'
        );

        assert.strictEqual(startPoint.svg, svg);
        assert.strictEqual(startPoint.x, 36);
        assert.strictEqual(startPoint.y, 45);
        assert.strictEqual(endPoint.svg, svg);
        assert.strictEqual(endPoint.x, 16);
        assert.strictEqual(endPoint.y, 45);
        """
    )


def test_web_renderer_finds_single_musicxml_caesura_anchor() -> None:
    """A caesura on a note should map to that rendered stavenote index."""
    _run_app_js_assertions(
        _fake_dom_parser_js()
        + """
        const xmlDocument = new context.DOMParser().parseFromString(`
            <score-partwise>
              <part>
                <measure>
                  <note>
                    <pitch/>
                    <notations><articulations><caesura/></articulations></notations>
                  </note>
                  <note><pitch/></note>
                </measure>
              </part>
            </score-partwise>
        `);

        assert.deepStrictEqual(
            JSON.parse(JSON.stringify(
                context.ScoreSpeakOsmdFixes._internal.musicXmlCaesuraAnchors(xmlDocument)
            )),
            [
                { index: 0 },
            ]
        );
        """
    )


def test_web_renderer_caesura_anchors_follow_rendered_note_order() -> None:
    """Caesura anchors should ignore grace notes and keep chord members on one anchor."""
    _run_app_js_assertions(
        _fake_dom_parser_js()
        + """
        const xmlDocument = new context.DOMParser().parseFromString(`
            <score-partwise>
              <part id="P1">
                <measure number="1">
                  <note>
                    <grace/>
                    <pitch/>
                    <notations><articulations><caesura/></articulations></notations>
                  </note>
                  <note><pitch/></note>
                  <note>
                    <chord/>
                    <pitch/>
                    <notations><articulations><caesura/></articulations></notations>
                  </note>
                </measure>
                <measure number="2"><note><pitch/></note></measure>
              </part>
              <part id="P2">
                <measure number="1"><note><pitch/></note></measure>
                <measure number="2">
                  <note>
                    <pitch/>
                    <notations><articulations><caesura/></articulations></notations>
                  </note>
                </measure>
              </part>
            </score-partwise>
        `);

        assert.deepStrictEqual(
            JSON.parse(JSON.stringify(
                context.ScoreSpeakOsmdFixes._internal.musicXmlCaesuraAnchors(xmlDocument)
            )),
            [
                { index: 0 },
                { index: 3 },
            ]
        );
        """
    )


def test_web_renderer_places_caesura_between_adjacent_notes() -> None:
    """Caesura fallback marks should appear after the attached note."""
    _run_app_js_assertions(
        _fake_svg_point_helpers_js()
        + """
        const svg = fakeSvg();
        const current = fakeStavenote(svg, {
            left: 100,
            top: 200,
            right: 120,
            bottom: 220,
            width: 20,
            height: 20,
        });
        const next = fakeStavenote(svg, {
            left: 180,
            top: 200,
            right: 200,
            bottom: 220,
            width: 20,
            height: 20,
        });

        const point = context.ScoreSpeakOsmdFixes._internal.stavenoteCaesuraPoint(
            current,
            next
        );

        assert.strictEqual(point.x, 150);
        assert.strictEqual(point.y, 196);
        """
    )


def test_web_renderer_keeps_caesura_near_note_when_next_note_is_not_same_staff() -> None:
    """Caesura placement should not use a different staff as the next anchor."""
    _run_app_js_assertions(
        _fake_svg_point_helpers_js()
        + """
        const svg = fakeSvg();
        const current = fakeStavenote(svg, {
            left: 100,
            top: 200,
            right: 120,
            bottom: 220,
            width: 20,
            height: 20,
        });
        const nextStaffNote = fakeStavenote(svg, {
            left: 180,
            top: 320,
            right: 200,
            bottom: 340,
            width: 20,
            height: 20,
        });

        const point = context.ScoreSpeakOsmdFixes._internal.stavenoteCaesuraPoint(
            current,
            nextStaffNote
        );

        assert.strictEqual(point.x, 130);
        assert.strictEqual(point.y, 196);
        """
    )


def test_web_renderer_post_render_corrections_share_one_parse() -> None:
    """Glissando and caesura fallbacks should share one MusicXML parse."""
    _run_app_js_assertions(
        _fake_dom_parser_js()
        + """
        const container = {
            querySelectorAll: () => [],
        };

        const added = context.ScoreSpeakOsmdFixes.applyPostRender(
            container,
            `<score-partwise>
              <part>
                <measure>
                  <note>
                    <pitch/>
                    <notations>
                      <glissando number="1" type="start"/>
                      <articulations><caesura/></articulations>
                    </notations>
                  </note>
                  <note>
                    <pitch/>
                    <notations><glissando number="1" type="stop"/></notations>
                  </note>
                </measure>
              </part>
            </score-partwise>`
        );

        assert.strictEqual(added, 0);
        assert.strictEqual(parseCalls, 1);
        """
    )


def test_web_renderer_post_load_corrections_skip_common_parse_path() -> None:
    """Common renders should not parse XML when no correction can apply."""
    _run_app_js_assertions(
        _fake_dom_parser_js()
        + """
        const renderer = Object.create(context.MusicXMLRenderer.prototype);
        const nonSyntheticBackJump = {
            type: 2,
            parentRepetition: { FromWords: true, EndingParts: [{}] },
        };
        const sourceMeasures = [
            { LastRepetitionInstructions: [nonSyntheticBackJump] },
            { LastRepetitionInstructions: [] },
        ];

        renderer.applyOsmdPostLoadCorrections(
            { Sheet: { SourceMeasures: sourceMeasures } },
            '<score-partwise><part><measure></measure><measure></measure></part></score-partwise>'
        );

        assert.strictEqual(parseCalls, 0);
        assert.deepStrictEqual(sourceMeasures[0].LastRepetitionInstructions, [nonSyntheticBackJump]);
        """
    )


def test_web_renderer_restores_navigation_display_instructions() -> None:
    """Standalone navigation marks should be restored as display instructions."""
    _run_app_js_assertions(
        _fake_dom_parser_js()
        + """
        const renderer = Object.create(context.MusicXMLRenderer.prototype);
        const sourceMeasures = [{}, {}, {}, {}, {}, {}];

        renderer.applyOsmdPostLoadCorrections(
            { Sheet: { SourceMeasures: sourceMeasures } },
            `
            <score-partwise>
              <part>
                <measure><direction><direction-type><words>  To   Coda </words></direction-type></direction></measure>
                <measure><direction><direction-type><coda/></direction-type></direction></measure>
                <measure><direction><direction-type><words>D. S. al Coda</words></direction-type></direction></measure>
                <measure><direction><direction-type><words>D.C. al Fine</words></direction-type></direction></measure>
                <measure><direction><direction-type><words>Fine</words></direction-type></direction></measure>
                <measure><direction><direction-type><segno/></direction-type></direction></measure>
              </part>
            </score-partwise>
            `
        );

        assert.strictEqual(parseCalls, 1);
        assert.deepStrictEqual(JSON.parse(JSON.stringify(sourceMeasures[0].LastRepetitionInstructions)), [
            { measureIndex: 0, type: 7, alignment: 1 },
        ]);
        assert.deepStrictEqual(JSON.parse(JSON.stringify(sourceMeasures[1].FirstRepetitionInstructions)), [
            { measureIndex: 1, type: 12, alignment: 0 },
        ]);
        assert.deepStrictEqual(JSON.parse(JSON.stringify(sourceMeasures[2].LastRepetitionInstructions)), [
            { measureIndex: 2, type: 10, alignment: 1 },
        ]);
        assert.deepStrictEqual(JSON.parse(JSON.stringify(sourceMeasures[3].LastRepetitionInstructions)), [
            { measureIndex: 3, type: 9, alignment: 1 },
        ]);
        assert.deepStrictEqual(JSON.parse(JSON.stringify(sourceMeasures[4].LastRepetitionInstructions)), [
            { measureIndex: 4, type: 6, alignment: 1 },
        ]);
        assert.deepStrictEqual(JSON.parse(JSON.stringify(sourceMeasures[5].FirstRepetitionInstructions)), [
            { measureIndex: 5, type: 13, alignment: 0 },
        ]);
        """
    )


def test_web_renderer_navigation_restoration_is_duplicate_safe() -> None:
    """Existing semantic navigation instructions should not be duplicated."""
    _run_app_js_assertions(
        _fake_dom_parser_js()
        + """
        const renderer = Object.create(context.MusicXMLRenderer.prototype);
        const existingCoda = { measureIndex: 0, type: 12, alignment: 0 };
        const existingToCoda = { measureIndex: 1, type: 7, alignment: 1 };
        const sourceMeasures = [
            { FirstRepetitionInstructions: [existingCoda] },
            { LastRepetitionInstructions: [existingToCoda] },
        ];

        renderer.applyOsmdPostLoadCorrections(
            { Sheet: { SourceMeasures: sourceMeasures } },
            `
            <score-partwise>
              <part>
                <measure><direction><direction-type><coda/></direction-type></direction></measure>
                <measure><direction><direction-type><words>To Coda</words></direction-type></direction></measure>
              </part>
            </score-partwise>
            `
        );

        assert.strictEqual(parseCalls, 1);
        assert.deepStrictEqual(sourceMeasures[0].FirstRepetitionInstructions, [existingCoda]);
        assert.deepStrictEqual(sourceMeasures[1].LastRepetitionInstructions, [existingToCoda]);
        """
    )


def test_web_renderer_navigation_restoration_upgrades_plain_repeat_words() -> None:
    """Qualified navigation words should replace OSMD's plain repeat parse."""
    _run_app_js_assertions(
        _fake_dom_parser_js()
        + """
        const renderer = Object.create(context.MusicXMLRenderer.prototype);
        const sourceMeasures = [
            {
                LastRepetitionInstructions: [
                    { measureIndex: 0, type: 4, alignment: 1 },
                ],
            },
            {
                LastRepetitionInstructions: [
                    { measureIndex: 1, type: 5, alignment: 1 },
                ],
            },
        ];

        renderer.applyOsmdPostLoadCorrections(
            { Sheet: { SourceMeasures: sourceMeasures } },
            `
            <score-partwise>
              <part>
                <measure><direction><direction-type><words>D.C. al Coda</words></direction-type></direction></measure>
                <measure><direction><direction-type><words>D.S. al Coda</words></direction-type></direction></measure>
              </part>
            </score-partwise>
            `
        );

        assert.strictEqual(parseCalls, 1);
        assert.deepStrictEqual(JSON.parse(JSON.stringify(sourceMeasures[0].LastRepetitionInstructions)), [
            { measureIndex: 0, type: 11, alignment: 1 },
        ]);
        assert.deepStrictEqual(JSON.parse(JSON.stringify(sourceMeasures[1].LastRepetitionInstructions)), [
            { measureIndex: 1, type: 10, alignment: 1 },
        ]);
        """
    )


def test_web_renderer_removes_osmd_synthetic_ending_repeat_barlines() -> None:
    """Standalone ending brackets should not make OSMD draw implicit repeats."""
    _run_app_js_assertions(
        _fake_dom_parser_js()
        + """
        const renderer = Object.create(context.MusicXMLRenderer.prototype);
        const syntheticEndingBackJump = {
            type: 2,
            parentRepetition: { FromWords: false, EndingParts: [{}] },
        };
        const sourceMeasures = [
            { LastRepetitionInstructions: [] },
            { LastRepetitionInstructions: [syntheticEndingBackJump] },
        ];

        renderer.applyOsmdPostLoadCorrections(
            { Sheet: { SourceMeasures: sourceMeasures } },
            '<score-partwise><part><measure></measure><measure></measure></part></score-partwise>'
        );

        assert.strictEqual(parseCalls, 1);
        assert.deepStrictEqual(sourceMeasures[1].LastRepetitionInstructions, []);

        parseCalls = 0;
        const realRepeatBackJump = {
            type: 2,
            parentRepetition: { FromWords: false, EndingParts: [{}] },
        };
        const realRepeatMeasures = [
            { LastRepetitionInstructions: [] },
            { LastRepetitionInstructions: [realRepeatBackJump] },
        ];

        renderer.applyOsmdPostLoadCorrections(
            { Sheet: { SourceMeasures: realRepeatMeasures } },
            `
            <score-partwise>
              <part>
                <measure></measure>
                <measure><barline><repeat direction="backward"/></barline></measure>
              </part>
            </score-partwise>
            `
        );

        assert.strictEqual(parseCalls, 1);
        assert.deepStrictEqual(
            realRepeatMeasures[1].LastRepetitionInstructions,
            [realRepeatBackJump]
        );
        """
    )


def test_web_renderer_post_load_corrections_share_one_parse() -> None:
    """Navigation restoration and ending cleanup should share one XML parse."""
    _run_app_js_assertions(
        _fake_dom_parser_js()
        + """
        const renderer = Object.create(context.MusicXMLRenderer.prototype);
        const syntheticEndingBackJump = {
            type: 2,
            parentRepetition: { FromWords: false, EndingParts: [{}] },
        };
        const sourceMeasures = [
            {},
            { LastRepetitionInstructions: [syntheticEndingBackJump] },
        ];

        renderer.applyOsmdPostLoadCorrections(
            { Sheet: { SourceMeasures: sourceMeasures } },
            `
            <score-partwise>
              <part>
                <measure><direction><direction-type><segno/></direction-type></direction></measure>
                <measure></measure>
              </part>
            </score-partwise>
            `
        );

        assert.strictEqual(parseCalls, 1);
        assert.deepStrictEqual(JSON.parse(JSON.stringify(sourceMeasures[0].FirstRepetitionInstructions)), [
            { measureIndex: 0, type: 13, alignment: 0 },
        ]);
        assert.deepStrictEqual(sourceMeasures[1].LastRepetitionInstructions, []);
        """
    )


def test_web_renderer_post_load_corrections_ignore_malformed_xml() -> None:
    """Malformed MusicXML should make the shim no-op instead of failing render."""
    _run_app_js_assertions(
        _fake_dom_parser_js()
        + """
        const renderer = Object.create(context.MusicXMLRenderer.prototype);
        const sourceMeasures = [{}];

        renderer.applyOsmdPostLoadCorrections(
            { Sheet: { SourceMeasures: sourceMeasures } },
            '<score-partwise><part><measure><coda/>MALFORMED_XML'
        );

        assert.strictEqual(parseCalls, 1);
        assert.deepStrictEqual(sourceMeasures, [{}]);
        """
    )


def test_web_renderer_refreshes_after_chat_score_version_changes() -> None:
    """Chat final handlers should render metadata-only score updates."""
    app_js = Path("web/app.js").read_text(encoding="utf-8")

    assert "shouldRefreshScore(event)" in app_js
    assert "shouldRefreshScore(data)" in app_js
    assert "data.score_version !== this.currentScoreVersion" in app_js


def test_web_renderer_measure_window_targets_measure_parts() -> None:
    """Render windows should target 200 measure-parts without a 32-measure cap."""
    _run_app_js_assertions(
        """
        function rendererFor(measureCount, partCount) {
            const renderer = Object.create(context.MusicXMLRenderer.prototype);
            renderer.currentMeasureCount = measureCount;
            renderer.currentPartCount = partCount;
            renderer.currentMeasureStart = 1;
            return renderer;
        }

        let renderer = rendererFor(200, 1);
        assert.strictEqual(renderer.measureWindowSize(), 200);
        assert.strictEqual(renderer.usesMeasurePaging(), false);
        let renderRange = renderer.currentRenderRange();
        assert.strictEqual(renderRange.start, 1);
        assert.strictEqual(renderRange.end, 200);

        renderer = rendererFor(201, 1);
        assert.strictEqual(renderer.measureWindowSize(), 200);
        assert.strictEqual(renderer.usesMeasurePaging(), true);
        renderRange = renderer.currentRenderRange();
        assert.strictEqual(renderRange.start, 1);
        assert.strictEqual(renderRange.end, 200);

        renderer = rendererFor(100, 20);
        assert.strictEqual(renderer.measureWindowSize(), 10);
        assert.strictEqual(renderer.usesMeasurePaging(), true);

        renderer = rendererFor(9, 50);
        assert.strictEqual(renderer.measureWindowSize(), 8);
        assert.strictEqual(renderer.usesMeasurePaging(), true);
        renderRange = renderer.currentRenderRange();
        assert.strictEqual(renderRange.start, 1);
        assert.strictEqual(renderRange.end, 8);
        """
    )


def test_web_renderer_keeps_current_view_when_changed_range_is_visible() -> None:
    """Changed ranges that intersect the current page should not move pagination."""
    _run_app_js_assertions(
        """
        const renderer = Object.create(context.MusicXMLRenderer.prototype);
        renderer.currentMeasureCount = 100;
        renderer.currentPartCount = 20;
        renderer.currentMeasureStart = 1;

        assert.strictEqual(renderer.measureWindowSize(), 10);
        assert.strictEqual(renderer.usesMeasurePaging(), true);

        renderer.updateMeasureStartForChangedRange({ start: 5, end: 13 });
        assert.strictEqual(renderer.currentMeasureStart, 1);

        renderer.currentMeasureStart = 21;
        renderer.updateMeasureStartForChangedRange({ start: 20, end: 22 });
        assert.strictEqual(renderer.currentMeasureStart, 21);
        """
    )


def test_web_renderer_pages_to_aligned_window_for_external_changed_range() -> None:
    """Changed ranges outside the current page should use normal page starts."""
    _run_app_js_assertions(
        """
        const renderer = Object.create(context.MusicXMLRenderer.prototype);
        renderer.currentMeasureCount = 100;
        renderer.currentPartCount = 20;
        renderer.currentMeasureStart = 1;

        renderer.updateMeasureStartForChangedRange({ start: 13, end: 13 });
        assert.strictEqual(renderer.currentMeasureStart, 11);
        let renderRange = renderer.currentRenderRange();
        assert.strictEqual(renderRange.start, 11);
        assert.strictEqual(renderRange.end, 20);

        renderer.updateMeasureStartForChangedRange({ start: 95, end: 95 });
        assert.strictEqual(renderer.currentMeasureStart, 91);
        renderRange = renderer.currentRenderRange();
        assert.strictEqual(renderRange.start, 91);
        assert.strictEqual(renderRange.end, 100);
        """
    )


def test_web_renderer_keeps_unpaged_scores_at_first_measure() -> None:
    """Unpaged scores should re-render the full score after changed ranges."""
    _run_app_js_assertions(
        """
        const renderer = Object.create(context.MusicXMLRenderer.prototype);
        renderer.currentMeasureCount = 8;
        renderer.currentPartCount = 1;
        renderer.currentMeasureStart = 1;

        assert.strictEqual(renderer.usesMeasurePaging(), false);
        renderer.updateMeasureStartForChangedRange({ start: 4, end: 4 });
        assert.strictEqual(renderer.currentMeasureStart, 1);
        const renderRange = renderer.currentRenderRange();
        assert.strictEqual(renderRange.start, 1);
        assert.strictEqual(renderRange.end, 8);
        """
    )


def test_web_renderer_has_visible_render_loading_state() -> None:
    """Rendering should expose an in-score loading state before OSMD blocks."""
    app_js = Path("web/app.js").read_text(encoding="utf-8")
    index_html = Path("web/index.html").read_text(encoding="utf-8")
    styles_css = Path("web/styles.css").read_text(encoding="utf-8")

    assert 'id="loading-title"' in index_html
    assert 'id="loading-detail"' in index_html
    assert 'id="sheet-music-container" aria-live="polite"' in index_html
    assert "this.sheetMusicContainer.classList.add('rendering')" in app_js
    assert "this.loadingStateId += 1" in app_js
    assert "this.hideLoading(loadingStateId)" in app_js
    assert "await this.waitForNextPaint()" in app_js
    assert "#sheet-music-container.rendering::after" in styles_css
    assert "@keyframes render-spin" in styles_css


def test_web_renderer_has_voice_recording_controls() -> None:
    """The browser UI should expose tap-to-record voice input."""
    app_js = Path("web/app.js").read_text(encoding="utf-8")
    index_html = Path("web/index.html").read_text(encoding="utf-8")
    styles_css = Path("web/styles.css").read_text(encoding="utf-8")

    assert 'id="voice-btn"' in index_html
    assert "MediaRecorder" in app_js
    assert "/api/voice" in app_js
    assert "this.streamChatMessage(voiceData.agent_message" in app_js
    assert ".btn-voice.recording" in styles_css


def test_web_renderer_exposes_agent_settings_dialog() -> None:
    """The browser UI should expose and submit centralized agent settings."""
    app_js = Path("web/app.js").read_text(encoding="utf-8")
    index_html = Path("web/index.html").read_text(encoding="utf-8")
    styles_css = Path("web/styles.css").read_text(encoding="utf-8")

    assert 'id="open-agent-settings"' in index_html
    assert 'id="agent-settings-summary"' in index_html
    assert 'id="agent-settings-dialog"' in index_html
    assert 'class="agent-model-row"' not in index_html
    assert 'id="agent-model-select"' in index_html
    assert 'id="agent-reasoning-select"' in index_html
    assert 'id="prompt-split-enabled"' in index_html
    assert 'id="prompt-split-min-sentences"' in index_html
    assert "/api/agent/settings" in app_js
    assert "model: this.selectedAgentModel()" in app_js
    assert "reasoning_effort: this.selectedAgentReasoningEffort()" in app_js
    assert "prompt_split_enabled: this.selectedPromptSplitEnabled()" in app_js
    assert "prompt_split_min_sentences: this.selectedPromptSplitMinSentences()" in app_js
    assert "this.updateAgentSettingsSummary()" in app_js
    assert "this.setAgentSettingsDisabled(true)" in app_js
    assert ".agent-settings-summary" in styles_css
    assert ".agent-model-select" in styles_css
    assert ".agent-reasoning-select" in styles_css
    assert ".settings-dialog" in styles_css
    assert ".settings-toggle" in styles_css
    assert ".settings-number-input" in styles_css


def test_web_upload_shows_loading_before_server_response() -> None:
    """Score uploads should show progress before parsing and initial render finish."""
    app_js = Path("web/app.js").read_text(encoding="utf-8")

    assert "'Uploading score'" in app_js
    assert "${file.name}" in app_js
    assert "'Preparing score'" in app_js
    assert "this.updateLoadingText(" in app_js


def test_agent_session_caches_musicxml_windows() -> None:
    """Repeated render-window requests should reuse exported MusicXML."""
    ss = ScoreSpeak.create(measures=12)
    session = AgentSession(ss, ss.to_musicxml_string())

    first_window = session.get_musicxml_window(5, 8)
    second_window = session.get_musicxml_window(5, 8)

    assert first_window == second_window
    assert len(session._musicxml_window_cache) == 1


def test_agent_session_model_switch_clears_llm_cache() -> None:
    """Changing model or reasoning should rebuild the cached LLM on next use."""
    test_session = AgentSession(ScoreSpeak.create(measures=1))
    cached_llm = object()
    cached_splitter_llm = object()

    test_session.llm = cached_llm
    test_session.splitter_llm = cached_splitter_llm
    selected_model = test_session.set_model("gpt-5.4")

    assert selected_model == "gpt-5.4"
    assert test_session.model == "gpt-5.4"
    assert test_session.llm is None
    assert test_session.splitter_llm is None

    test_session.llm = cached_llm
    test_session.splitter_llm = cached_splitter_llm
    test_session.set_model("gpt-5.4")

    assert test_session.llm is cached_llm
    assert test_session.splitter_llm is cached_splitter_llm

    selected_reasoning = test_session.set_reasoning_effort("high")

    assert selected_reasoning == "high"
    assert test_session.reasoning_effort == "high"
    assert test_session.llm is None
    assert test_session.splitter_llm is None


def test_agent_session_prompt_split_settings_control_activation(
    monkeypatch: Any,
) -> None:
    """Prompt split settings should disable or activate splitting by sentence count."""
    test_session = AgentSession(
        ScoreSpeak.create(measures=1),
        prompt_split_min_sentences=3,
    )

    assert not should_use_prompt_split(
        "One. Two.",
        test_session.prompt_split_config(),
    )
    assert should_use_prompt_split(
        "One. Two. Three.",
        test_session.prompt_split_config(),
    )

    test_session.set_prompt_split_enabled(False)
    captured: dict[str, Any] = {}

    def fake_run_prompt(
        score_state: ScoreSpeak,
        retriever: Any,
        llm: Any,
        user_text: str,
        memory_store: Any,
        **kwargs: Any,
    ) -> str:
        """Capture prompt split kwargs without calling an LLM."""
        del score_state, retriever, llm, user_text, memory_store
        captured.update(kwargs)
        return "ok"

    def fail_get_splitter_llm() -> object:
        """Fail if disabled prompt split still asks for a splitter model."""
        raise AssertionError("splitter LLM should not be built")

    monkeypatch.setattr(test_session, "get_llm", lambda: object())
    monkeypatch.setattr(test_session, "get_splitter_llm", fail_get_splitter_llm)
    monkeypatch.setattr("web.server.run_prompt", fake_run_prompt)

    assert test_session.run_turn("One. Two. Three. Four. Five.") == "ok"
    assert captured["splitter_llm"] is None
    assert not should_use_prompt_split(
        "One. Two. Three. Four. Five.",
        captured["prompt_split_config"],
    )


def test_agent_model_api_returns_options_and_switches(
    monkeypatch: Any,
) -> None:
    """The main web API should expose and update the session model."""
    test_session = AgentSession(ScoreSpeak.create(measures=1))
    monkeypatch.setattr("web.server.session", test_session)
    client = app.test_client()

    options_response = client.get("/api/agent/models")
    switch_response = client.patch(
        "/api/agent/model",
        json={"model": "gpt-5.4", "reasoning_effort": "high"},
    )
    invalid_response = client.patch(
        "/api/agent/model",
        json={"model": "gpt-4o"},
    )
    invalid_reasoning_response = client.patch(
        "/api/agent/model",
        json={"reasoning_effort": "xhigh"},
    )

    assert options_response.status_code == 200
    assert options_response.json["current_model"] == DEFAULT_AGENT_MODEL
    assert options_response.json["current_reasoning_effort"] == "low"
    assert options_response.json["default_reasoning_effort"] == "low"
    assert [effort["id"] for effort in options_response.json["reasoning_efforts"]] == [
        "none",
        "minimal",
        "low",
        "medium",
        "high",
    ]
    assert [model["id"] for model in options_response.json["models"]] == [
        "gpt-5.4-mini",
        "gpt-5.4",
        "gpt-5.4-nano",
    ]
    assert switch_response.status_code == 200
    assert switch_response.json["current_model"] == "gpt-5.4"
    assert switch_response.json["current_reasoning_effort"] == "high"
    assert switch_response.json["prompt_split_enabled"] is True
    assert switch_response.json["prompt_split_min_sentences"] == 7
    assert test_session.model == "gpt-5.4"
    assert test_session.reasoning_effort == "high"
    assert invalid_response.status_code == 400
    assert invalid_reasoning_response.status_code == 400
    assert test_session.model == "gpt-5.4"
    assert test_session.reasoning_effort == "high"


def test_agent_settings_api_returns_options_and_switches(
    monkeypatch: Any,
) -> None:
    """The main web API should expose and update centralized agent settings."""
    test_session = AgentSession(ScoreSpeak.create(measures=1))
    monkeypatch.setattr("web.server.session", test_session)
    client = app.test_client()

    options_response = client.get("/api/agent/settings")
    switch_response = client.patch(
        "/api/agent/settings",
        json={
            "model": "gpt-5.4",
            "reasoning_effort": "medium",
            "prompt_split_enabled": False,
            "prompt_split_min_sentences": 12,
        },
    )
    invalid_min_response = client.patch(
        "/api/agent/settings",
        json={"prompt_split_min_sentences": 0},
    )
    invalid_enabled_response = client.patch(
        "/api/agent/settings",
        json={"prompt_split_enabled": "sometimes"},
    )

    assert options_response.status_code == 200
    assert options_response.json["model"] == DEFAULT_AGENT_MODEL
    assert options_response.json["current_model"] == DEFAULT_AGENT_MODEL
    assert options_response.json["reasoning_effort"] == "low"
    assert options_response.json["current_reasoning_effort"] == "low"
    assert options_response.json["prompt_split_enabled"] is True
    assert options_response.json["default_prompt_split_enabled"] is True
    assert options_response.json["prompt_split_min_sentences"] == 7
    assert options_response.json["default_prompt_split_min_sentences"] == 7
    assert [model["id"] for model in options_response.json["models"]] == [
        "gpt-5.4-mini",
        "gpt-5.4",
        "gpt-5.4-nano",
    ]
    assert [effort["id"] for effort in options_response.json["reasoning_efforts"]] == [
        "none",
        "minimal",
        "low",
        "medium",
        "high",
    ]

    assert switch_response.status_code == 200
    assert switch_response.json["model"] == "gpt-5.4"
    assert switch_response.json["reasoning_effort"] == "medium"
    assert switch_response.json["prompt_split_enabled"] is False
    assert switch_response.json["prompt_split_min_sentences"] == 12
    assert test_session.model == "gpt-5.4"
    assert test_session.reasoning_effort == "medium"
    assert test_session.prompt_split_enabled is False
    assert test_session.prompt_split_min_sentences == 12

    assert invalid_min_response.status_code == 400
    assert invalid_enabled_response.status_code == 400
    assert test_session.prompt_split_enabled is False
    assert test_session.prompt_split_min_sentences == 12


def test_agent_settings_survive_score_replacement(
    monkeypatch: Any,
) -> None:
    """Creating or loading scores should preserve session-scoped agent settings."""
    test_session = AgentSession(
        ScoreSpeak.create(measures=1),
        model="gpt-5.4-nano",
        reasoning_effort="high",
        prompt_split_enabled=False,
        prompt_split_min_sentences=11,
    )
    monkeypatch.setattr("web.server.session", test_session)
    client = app.test_client()

    new_response = client.post("/api/new", json={"measures": 2})
    assert new_response.status_code == 200
    assert web_server.session is not None
    assert web_server.session.model == "gpt-5.4-nano"
    assert web_server.session.reasoning_effort == "high"
    assert web_server.session.prompt_split_enabled is False
    assert web_server.session.prompt_split_min_sentences == 11

    musicxml = ScoreSpeak.create(measures=1).to_musicxml_string()
    load_response = client.post(
        "/api/load",
        data={
            "musicxml": (
                io.BytesIO(musicxml.encode("utf-8")),
                "score.musicxml",
            ),
        },
        content_type="multipart/form-data",
    )

    assert load_response.status_code == 200
    assert web_server.session is not None
    assert web_server.session.model == "gpt-5.4-nano"
    assert web_server.session.reasoning_effort == "high"
    assert web_server.session.prompt_split_enabled is False
    assert web_server.session.prompt_split_min_sentences == 11


def test_changed_measure_range_uses_tool_kwargs_and_details() -> None:
    """Changed range inference uses both tool args and result details."""
    result = OperationResult(
        success=True,
        description="Inserted measure",
        details={
            "measures_inserted": [{"part": 0, "measure": 4}],
            "measures_renumbered_from": 5,
        },
    )

    changed = _extract_changed_measure_range(
        [("insert_measure", {"before": 4}, result)],
        {"start": 1, "end": 8},
    )

    assert changed == {"start": 4, "end": 5}


def test_agent_session_ignores_changed_false_tool_results() -> None:
    """Successful no-op tool results should not mark the score changed."""
    session = AgentSession(ScoreSpeak.create(measures=4))
    result = OperationResult(
        success=True,
        description="No score change",
        details={"changed": False, "measure": 2},
    )

    session.record_tool_result("set_time_signature", {"measure_number": 2}, result)

    assert session.changed_range({"start": 1, "end": 4}) is None


def test_chat_global_edit_changed_range_uses_client_render_range(
    monkeypatch: Any,
) -> None:
    """Global edits should report the browser's current view as changed."""
    score_state = ScoreSpeak.create(measures=100)
    test_session = AgentSession(score_state)

    def fake_run_turn(user_text: str) -> str:
        """Record a deterministic global edit without calling an LLM."""
        test_session._tool_results = [
            (
                "transpose",
                {},
                OperationResult(success=True, description="Transposed score"),
            )
        ]
        test_session.score_version += 1
        return f"Handled: {user_text}"

    monkeypatch.setattr(test_session, "run_turn", fake_run_turn)
    monkeypatch.setattr("web.server.session", test_session)

    client = app.test_client()
    response = client.post(
        "/api/chat",
        json={
            "message": "Transpose everything up a whole step",
            "render_range": {"start": 51, "end": 60},
        },
    )
    data = response.get_json()

    assert response.status_code == 200
    assert data["success"] is True
    assert data["changed_range"] == {"start": 51, "end": 60}


def test_chat_payload_model_selection_updates_session(
    monkeypatch: Any,
) -> None:
    """Chat payloads should switch the model before running the turn."""
    test_session = AgentSession(ScoreSpeak.create(measures=1))
    captured: dict[str, str] = {}

    def fake_run_turn(user_text: str) -> str:
        """Capture the agent options active during the fake turn."""
        captured["model"] = test_session.model
        captured["reasoning_effort"] = test_session.reasoning_effort
        captured["prompt_split_enabled"] = test_session.prompt_split_enabled
        captured["prompt_split_min_sentences"] = (
            test_session.prompt_split_min_sentences
        )
        return f"Handled: {user_text}"

    monkeypatch.setattr(test_session, "run_turn", fake_run_turn)
    monkeypatch.setattr("web.server.session", test_session)
    client = app.test_client()

    response = client.post(
        "/api/chat",
        json={
            "message": "Add C4",
            "model": "gpt-5.4-nano",
            "reasoning_effort": "medium",
            "prompt_split_enabled": False,
            "prompt_split_min_sentences": 14,
        },
    )
    invalid_response = client.post(
        "/api/chat",
        json={"message": "Add C4", "model": "gpt-4o"},
    )

    assert response.status_code == 200
    assert response.json["model"] == "gpt-5.4-nano"
    assert response.json["reasoning_effort"] == "medium"
    assert response.json["prompt_split_enabled"] is False
    assert response.json["prompt_split_min_sentences"] == 14
    assert captured["model"] == "gpt-5.4-nano"
    assert captured["reasoning_effort"] == "medium"
    assert captured["prompt_split_enabled"] is False
    assert captured["prompt_split_min_sentences"] == 14
    assert invalid_response.status_code == 400


def _voice_request(audio: AudioInput) -> VoiceRequest:
    """Build a normalized fake voice request for endpoint tests."""
    return VoiceRequest(audio=audio)


def _speech_result(
    audio: AudioInput,
    text: str,
    warnings: list[VoiceWarning] | None = None,
) -> VoiceProcessingResult:
    """Build a fake successful speech voice result."""
    return VoiceProcessingResult(
        success=True,
        request=_voice_request(audio),
        speech=SpeechTranscript(
            text=text,
            raw_text=text,
            model="fake-stt",
            language="en",
        ),
        warnings=list(warnings or []),
    )


def test_voice_endpoint_requires_audio() -> None:
    """The voice endpoint should reject requests without an audio upload."""
    client = app.test_client()

    response = client.post("/api/voice", data={})
    data = response.get_json()

    assert response.status_code == 400
    assert data["success"] is False
    assert data["error"] == "No audio file provided"


def test_voice_endpoint_returns_agent_message_for_speech(monkeypatch: Any) -> None:
    """The voice endpoint should convert speech results into agent prompts."""
    captured = {}

    class FakeProcessor:
        """Fake voice processor for speech-only endpoint testing."""

        def process(
            self,
            audio: AudioInput,
            *,
            speech_prompt: str | None = None,
            language: str | None = "en",
        ) -> VoiceProcessingResult:
            """Return a deterministic speech result."""
            captured["audio_exists"] = audio.path.exists() if audio.path else False
            captured["language"] = language
            captured["speech_prompt"] = speech_prompt
            return _speech_result(audio, "Add a quarter note in measure 2")

    monkeypatch.setattr("web.server.VoiceInputProcessor", FakeProcessor)
    client = app.test_client()

    response = client.post(
        "/api/voice",
        data={
            "audio": (io.BytesIO(b"fake audio"), "voice.webm"),
            "mode": "speech",
            "language": "en",
            "render_start": "2",
            "render_end": "5",
        },
        content_type="multipart/form-data",
    )
    data = response.get_json()

    assert response.status_code == 200
    assert data["success"] is True
    assert data["speech_text"] == "Add a quarter note in measure 2"
    assert "detected_mode" not in data
    assert "melody_summary" not in data
    assert "Spoken transcript" in data["agent_message"]
    assert "Current rendered measure range: 2-5" in data["agent_message"]
    assert "Sung/hummed melody evidence" not in data["agent_message"]
    assert captured["audio_exists"] is True
    assert captured["language"] == "en"
    assert captured["speech_prompt"]


def test_voice_endpoint_rejects_non_speech_mode() -> None:
    """The voice endpoint should reject removed voice modes."""
    client = app.test_client()

    response = client.post(
        "/api/voice",
        data={
            "audio": (io.BytesIO(b"fake audio"), "voice.webm"),
            "mode": "auto",
        },
        content_type="multipart/form-data",
    )
    data = response.get_json()

    assert response.status_code == 400
    assert data["success"] is False
    assert "Only 'speech' is accepted" in data["error"]


def test_voice_endpoint_reports_speech_warnings(monkeypatch: Any) -> None:
    """The voice endpoint should serialize non-fatal speech warnings."""

    class FakeProcessor:
        """Fake voice processor for warning serialization testing."""

        def process(
            self,
            audio: AudioInput,
            *,
            speech_prompt: str | None = None,
            language: str | None = "en",
        ) -> VoiceProcessingResult:
            """Return speech with a non-fatal STT warning."""
            warning = VoiceWarning(
                code="stt_low_confidence",
                message="Transcript confidence was low.",
            )
            return _speech_result(audio, "Set tempo to 120", [warning])

    monkeypatch.setattr("web.server.VoiceInputProcessor", FakeProcessor)
    client = app.test_client()

    response = client.post(
        "/api/voice",
        data={"audio": (io.BytesIO(b"fake audio"), "voice.webm")},
        content_type="multipart/form-data",
    )
    data = response.get_json()

    assert response.status_code == 200
    assert data["speech_text"] == "Set tempo to 120"
    assert data["warnings"][0]["code"] == "stt_low_confidence"
    assert "melody_summary" not in data
