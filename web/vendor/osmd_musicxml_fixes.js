(function (root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) {
        module.exports = api;
    }
    root.ScoreSpeakOsmdFixes = api;
})(typeof globalThis !== 'undefined' ? globalThis : window, function () {
    const SVG_NS = 'http://www.w3.org/2000/svg';
    const REPETITION_INSTRUCTION = Object.freeze({
        DA_CAPO: 4,
        DAL_SEGNO: 5,
        FINE: 6,
        TO_CODA: 7,
        DAL_SEGNO_AL_FINE: 8,
        DA_CAPO_AL_FINE: 9,
        DAL_SEGNO_AL_CODA: 10,
        DA_CAPO_AL_CODA: 11,
        CODA: 12,
        SEGNO: 13,
    });
    const REPETITION_ALIGNMENT = Object.freeze({
        BEGIN: 0,
        END: 1,
    });

    function applyPostLoad(osmd, xmlString) {
        const sourceMeasures = osmd?.Sheet?.SourceMeasures ?? osmd?.sheet?.SourceMeasures;
        if (!sourceMeasures || typeof sourceMeasures.length !== 'number') {
            return;
        }

        let hasParsedXml = false;
        let xmlDocument = null;
        const getXmlDocument = () => {
            if (!hasParsedXml) {
                hasParsedXml = true;
                xmlDocument = parseMusicXmlDocument(xmlString);
            }
            return xmlDocument;
        };

        if (hasNavigationDisplayTokens(xmlString)) {
            const document = getXmlDocument();
            if (document) {
                restoreNavigationDisplayInstructions(sourceMeasures, document);
            }
        }

        if (!hasSyntheticEndingBackJumpCandidate(sourceMeasures)) {
            return;
        }

        const document = getXmlDocument();
        if (document) {
            removeSyntheticEndingRepeatBarlines(sourceMeasures, document);
        }
    }

    function applyPostRender(container, xmlString) {
        removeGlissandoFallbacks(container);
        removeCaesuraFallbacks(container);

        const hasGlissandos = hasGlissandoNotationTokens(xmlString);
        const hasCaesuras = hasCaesuraNotationTokens(xmlString);
        if (!hasGlissandos && !hasCaesuras) {
            return 0;
        }

        const xmlDocument = parseMusicXmlDocument(xmlString);
        if (!xmlDocument) {
            return 0;
        }

        let added = 0;
        if (hasGlissandos) {
            added += drawGlissandoFallbacks(container, xmlDocument);
        }
        if (hasCaesuras) {
            added += drawCaesuraFallbacks(container, xmlDocument);
        }
        return added;
    }

    function parseMusicXmlDocument(xmlString) {
        if (typeof DOMParser === 'undefined') {
            return null;
        }

        let xmlDocument = null;
        try {
            xmlDocument = new DOMParser().parseFromString(
                xmlString,
                'application/xml'
            );
        } catch {
            return null;
        }
        if (xmlDocument.getElementsByTagName('parsererror').length > 0) {
            return null;
        }
        return xmlDocument;
    }

    function hasGlissandoNotationTokens(xmlString) {
        return (
            typeof xmlString === 'string'
            && /<\s*(?:[\w.-]+:)?glissando\b/i.test(xmlString)
        );
    }

    function hasCaesuraNotationTokens(xmlString) {
        return (
            typeof xmlString === 'string'
            && /<\s*(?:[\w.-]+:)?caesura\b/i.test(xmlString)
        );
    }

    function drawGlissandoFallbacks(container, xmlDocument) {
        const segments = musicXmlGlissandoSegments(xmlDocument);
        if (!segments.length) {
            return 0;
        }

        const stavenotes = Array.from(
            container?.querySelectorAll('svg .vf-stavenote') ?? []
        );
        if (!stavenotes.length) {
            return 0;
        }

        let added = 0;
        segments.forEach((segment) => {
            const startStavenote = stavenotes[segment.startIndex];
            const endStavenote = stavenotes[segment.endIndex];
            if (!startStavenote || !endStavenote) {
                return;
            }

            const startPoint = stavenoteGlissandoPoint(startStavenote, 'start');
            const endPoint = stavenoteGlissandoPoint(endStavenote, 'end');
            if (!startPoint || !endPoint || startPoint.svg !== endPoint.svg) {
                return;
            }
            appendGlissandoFallbackPath(
                startPoint.svg,
                startPoint,
                endPoint,
                segment.lineType
            );
            added += 1;
        });
        return added;
    }

    function musicXmlGlissandoSegments(xmlDocument) {
        const segments = [];
        const openGlissandos = new Map();
        let stavenoteIndex = -1;

        musicXmlMeasuresInRenderOrder(xmlDocument).forEach((measure) => {
            const notes = elementChildren(measure).filter(
                (child) => nodeLocalName(child) === 'note'
            );
            notes.forEach((noteElement) => {
                if (noteHasChild(noteElement, 'grace')) {
                    return;
                }
                if (!noteHasChild(noteElement, 'chord')) {
                    stavenoteIndex += 1;
                }

                const glissandos = descendantsByLocalName(noteElement, 'glissando');
                glissandos.forEach((glissando) => {
                    const number = glissando.getAttribute('number') || '1';
                    const type = (glissando.getAttribute('type') || '').toLowerCase();
                    const lineType = (
                        glissando.getAttribute('line-type')
                        || glissando.getAttribute('lineType')
                        || 'solid'
                    ).toLowerCase();

                    if (type === 'start') {
                        openGlissandos.set(number, {
                            index: stavenoteIndex,
                            lineType,
                        });
                        return;
                    }
                    if (type !== 'stop') {
                        return;
                    }

                    const start = openGlissandos.get(number);
                    if (!start || start.index < 0 || stavenoteIndex < 0) {
                        return;
                    }
                    segments.push({
                        startIndex: start.index,
                        endIndex: stavenoteIndex,
                        lineType: start.lineType || lineType,
                    });
                    openGlissandos.delete(number);
                });
            });
        });

        return segments;
    }

    function drawCaesuraFallbacks(container, xmlDocument) {
        const anchors = musicXmlCaesuraAnchors(xmlDocument);
        if (!anchors.length) {
            return 0;
        }

        const stavenotes = Array.from(
            container?.querySelectorAll('svg .vf-stavenote') ?? []
        );
        if (!stavenotes.length) {
            return 0;
        }

        let added = 0;
        anchors.forEach((anchor) => {
            const stavenote = stavenotes[anchor.index];
            if (!stavenote) {
                return;
            }

            const point = stavenoteCaesuraPoint(
                stavenote,
                stavenotes[anchor.index + 1] || null
            );
            if (!point) {
                return;
            }
            appendCaesuraFallbackPaths(point.svg, point);
            added += 1;
        });
        return added;
    }

    function musicXmlCaesuraAnchors(xmlDocument) {
        const anchors = [];
        let stavenoteIndex = -1;

        musicXmlMeasuresInRenderOrder(xmlDocument).forEach((measure) => {
            const notes = elementChildren(measure).filter(
                (child) => nodeLocalName(child) === 'note'
            );
            notes.forEach((noteElement) => {
                if (noteHasChild(noteElement, 'grace')) {
                    return;
                }
                if (!noteHasChild(noteElement, 'chord')) {
                    stavenoteIndex += 1;
                }
                if (
                    stavenoteIndex >= 0
                    && descendantsByLocalName(noteElement, 'caesura').length > 0
                ) {
                    anchors.push({ index: stavenoteIndex });
                }
            });
        });

        return anchors;
    }

    function musicXmlMeasuresInRenderOrder(xmlDocument) {
        const parts = Array.from(xmlDocument.getElementsByTagName('part'));
        const measuresByPart = parts.map((part) => elementChildren(part).filter(
            (child) => nodeLocalName(child) === 'measure'
        ));
        const measureCount = Math.max(0, ...measuresByPart.map((measures) => measures.length));
        const measures = [];

        for (let measureIndex = 0; measureIndex < measureCount; measureIndex += 1) {
            measuresByPart.forEach((partMeasures) => {
                if (partMeasures[measureIndex]) {
                    measures.push(partMeasures[measureIndex]);
                }
            });
        }
        return measures;
    }

    function noteHasChild(noteElement, localName) {
        return elementChildren(noteElement).some(
            (child) => nodeLocalName(child) === localName
        );
    }

    function stavenoteGlissandoPoint(stavenoteElement, side) {
        const notehead = stavenoteElement.querySelector('.vf-notehead')
            || stavenoteElement.querySelector('.vf-note');
        const svg = stavenoteElement.ownerSVGElement;
        if (!notehead || !svg || typeof notehead.getBBox !== 'function') {
            return null;
        }

        try {
            const bbox = notehead.getBBox();
            const x = side === 'start' ? bbox.x + bbox.width + 4 : bbox.x - 4;
            return {
                svg,
                x,
                y: bbox.y + bbox.height / 2,
            };
        } catch {
            return null;
        }
    }

    function stavenoteCaesuraPoint(stavenoteElement, nextStavenoteElement = null) {
        const box = stavenoteNoteheadBox(stavenoteElement);
        if (!box) {
            return null;
        }

        const nextBox = stavenoteNoteheadBox(nextStavenoteElement);
        let x = box.right + 10;
        if (
            nextBox
            && nextBox.svg === box.svg
            && nextBox.left > box.right + 14
            && Math.abs(nextBox.centerY - box.centerY) < 60
        ) {
            x = box.right + (nextBox.left - box.right) * 0.5;
        }

        return {
            svg: box.svg,
            x,
            y: box.top - 4,
        };
    }

    function stavenoteNoteheadBox(stavenoteElement) {
        if (!stavenoteElement) {
            return null;
        }

        const notehead = stavenoteElement.querySelector('.vf-notehead')
            || stavenoteElement.querySelector('.vf-note');
        const svg = stavenoteElement.ownerSVGElement;
        if (!notehead || !svg) {
            return null;
        }

        return elementBoxInSvg(svg, notehead);
    }

    function elementBoxInSvg(svg, element) {
        if (!svg || !element) {
            return null;
        }

        if (
            typeof element.getBoundingClientRect === 'function'
            && typeof svg.createSVGPoint === 'function'
            && typeof svg.getScreenCTM === 'function'
        ) {
            try {
                const rect = element.getBoundingClientRect();
                const screenMatrix = svg.getScreenCTM();
                if (
                    rect
                    && screenMatrix
                    && typeof screenMatrix.inverse === 'function'
                    && (rect.width || rect.height)
                ) {
                    const inverse = screenMatrix.inverse();
                    const topLeft = clientPointToSvgPoint(svg, inverse, rect.left, rect.top);
                    const bottomRight = clientPointToSvgPoint(svg, inverse, rect.right, rect.bottom);
                    if (topLeft && bottomRight) {
                        return normalizedSvgBox(
                            svg,
                            topLeft.x,
                            topLeft.y,
                            bottomRight.x,
                            bottomRight.y
                        );
                    }
                }
            } catch {
                // Fall back to SVG-local geometry below.
            }
        }

        if (typeof element.getBBox !== 'function') {
            return null;
        }

        try {
            const bbox = element.getBBox();
            if (
                typeof element.getCTM === 'function'
                && typeof svg.createSVGPoint === 'function'
            ) {
                const matrix = element.getCTM();
                if (matrix) {
                    const topLeft = svgLocalPointToSvgPoint(svg, matrix, bbox.x, bbox.y);
                    const bottomRight = svgLocalPointToSvgPoint(
                        svg,
                        matrix,
                        bbox.x + bbox.width,
                        bbox.y + bbox.height
                    );
                    if (topLeft && bottomRight) {
                        return normalizedSvgBox(
                            svg,
                            topLeft.x,
                            topLeft.y,
                            bottomRight.x,
                            bottomRight.y
                        );
                    }
                }
            }

            return normalizedSvgBox(
                svg,
                bbox.x,
                bbox.y,
                bbox.x + bbox.width,
                bbox.y + bbox.height
            );
        } catch {
            return null;
        }
    }

    function clientPointToSvgPoint(svg, inverseMatrix, x, y) {
        const point = svg.createSVGPoint();
        point.x = x;
        point.y = y;
        return point.matrixTransform(inverseMatrix);
    }

    function svgLocalPointToSvgPoint(svg, matrix, x, y) {
        const point = svg.createSVGPoint();
        point.x = x;
        point.y = y;
        return point.matrixTransform(matrix);
    }

    function normalizedSvgBox(svg, x1, y1, x2, y2) {
        const left = Math.min(x1, x2);
        const right = Math.max(x1, x2);
        const top = Math.min(y1, y2);
        const bottom = Math.max(y1, y2);
        return {
            svg,
            left,
            right,
            top,
            bottom,
            centerX: left + (right - left) / 2,
            centerY: top + (bottom - top) / 2,
        };
    }

    function appendGlissandoFallbackPath(svg, startPoint, endPoint, lineType) {
        const overlay = glissandoFallbackOverlay(svg);
        const path = document.createElementNS(SVG_NS, 'path');
        path.setAttribute(
            'd',
            lineType === 'wavy'
                ? wavyGlissandoPath(startPoint, endPoint)
                : solidGlissandoPath(startPoint, endPoint)
        );
        path.setAttribute('class', 'scorespeak-glissando-fallback');
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', '#000000');
        path.setAttribute('stroke-width', '1.4');
        path.setAttribute('stroke-linecap', 'round');
        path.setAttribute('stroke-linejoin', 'round');
        if (lineType === 'dashed') {
            path.setAttribute('stroke-dasharray', '5 3');
        } else if (lineType === 'dotted') {
            path.setAttribute('stroke-dasharray', '1 3');
        }
        overlay.appendChild(path);
    }

    function glissandoFallbackOverlay(svg) {
        let overlay = svg.querySelector('g.scorespeak-glissando-fallback-layer');
        if (!overlay) {
            overlay = document.createElementNS(SVG_NS, 'g');
            overlay.setAttribute(
                'class',
                'scorespeak-glissando-fallback-layer scorespeak-glissando-fallback'
            );
            overlay.setAttribute('pointer-events', 'none');
            svg.appendChild(overlay);
        }
        return overlay;
    }

    function appendCaesuraFallbackPaths(svg, point) {
        const overlay = caesuraFallbackOverlay(svg);
        [
            { xOffset: 0, yOffset: 0 },
            { xOffset: 6, yOffset: 0 },
        ].forEach((stroke) => {
            const path = document.createElementNS(SVG_NS, 'path');
            path.setAttribute(
                'd',
                `M ${svgNumber(point.x + stroke.xOffset)} `
                    + `${svgNumber(point.y + stroke.yOffset)} `
                    + `L ${svgNumber(point.x + stroke.xOffset + 4)} `
                    + `${svgNumber(point.y + stroke.yOffset - 12)}`
            );
            path.setAttribute('class', 'scorespeak-caesura-fallback');
            path.setAttribute('fill', 'none');
            path.setAttribute('stroke', '#000000');
            path.setAttribute('stroke-width', '1.6');
            path.setAttribute('stroke-linecap', 'round');
            overlay.appendChild(path);
        });
    }

    function caesuraFallbackOverlay(svg) {
        let overlay = svg.querySelector('g.scorespeak-caesura-fallback-layer');
        if (!overlay) {
            overlay = document.createElementNS(SVG_NS, 'g');
            overlay.setAttribute(
                'class',
                'scorespeak-caesura-fallback-layer scorespeak-caesura-fallback'
            );
            overlay.setAttribute('pointer-events', 'none');
            svg.appendChild(overlay);
        }
        return overlay;
    }

    function solidGlissandoPath(startPoint, endPoint) {
        return `M ${svgNumber(startPoint.x)} ${svgNumber(startPoint.y)} `
            + `L ${svgNumber(endPoint.x)} ${svgNumber(endPoint.y)}`;
    }

    function wavyGlissandoPath(startPoint, endPoint) {
        const dx = endPoint.x - startPoint.x;
        const dy = endPoint.y - startPoint.y;
        const length = Math.hypot(dx, dy);
        if (length < 1) {
            return solidGlissandoPath(startPoint, endPoint);
        }

        const amplitude = 2.8;
        const wavelength = 8;
        const steps = Math.max(8, Math.ceil(length / 3));
        const perpendicularX = -dy / length;
        const perpendicularY = dx / length;
        const points = [];

        for (let index = 0; index <= steps; index += 1) {
            const t = index / steps;
            const wave = Math.sin((t * length / wavelength) * Math.PI * 2) * amplitude;
            points.push({
                x: startPoint.x + dx * t + perpendicularX * wave,
                y: startPoint.y + dy * t + perpendicularY * wave,
            });
        }

        return points.map((point, index) => (
            `${index === 0 ? 'M' : 'L'} ${svgNumber(point.x)} ${svgNumber(point.y)}`
        )).join(' ');
    }

    function svgNumber(value) {
        return Number(value).toFixed(2).replace(/\.?0+$/, '');
    }

    function removeGlissandoFallbacks(container) {
        container?.querySelectorAll('.scorespeak-glissando-fallback').forEach(
            (element) => element.remove()
        );
    }

    function removeCaesuraFallbacks(container) {
        container?.querySelectorAll('.scorespeak-caesura-fallback').forEach(
            (element) => element.remove()
        );
    }

    function hasNavigationDisplayTokens(xmlString) {
        if (typeof xmlString !== 'string') {
            return false;
        }
        return /<\s*(?:[\w.-]+:)?(?:coda|segno)\b|to\s+coda|fine|d\s*\.?\s*c\s*\.?|d\s*\.?\s*s\s*\.?|da\s+capo|dal\s+segno/i.test(xmlString);
    }

    function hasSyntheticEndingBackJumpCandidate(sourceMeasures) {
        for (let measureIndex = 0; measureIndex < sourceMeasures.length; measureIndex += 1) {
            const instructions = sourceMeasures[measureIndex]?.LastRepetitionInstructions;
            if (!Array.isArray(instructions)) {
                continue;
            }
            if (instructions.some((instruction) => isSyntheticEndingBackJump(instruction))) {
                return true;
            }
        }
        return false;
    }

    function removeSyntheticEndingRepeatBarlines(sourceMeasures, xmlDocument) {
        const realBackwardRepeatMeasureIndexes = backwardRepeatMeasureIndexes(xmlDocument);
        for (let measureIndex = 0; measureIndex < sourceMeasures.length; measureIndex += 1) {
            if (realBackwardRepeatMeasureIndexes.has(measureIndex)) {
                continue;
            }

            const instructions = sourceMeasures[measureIndex]?.LastRepetitionInstructions;
            if (!Array.isArray(instructions)) {
                continue;
            }

            for (let instructionIndex = instructions.length - 1; instructionIndex >= 0; instructionIndex -= 1) {
                if (isSyntheticEndingBackJump(instructions[instructionIndex])) {
                    instructions.splice(instructionIndex, 1);
                }
            }
        }
    }

    function backwardRepeatMeasureIndexes(xmlDocument) {
        const repeatMeasureIndexes = new Set();
        const parts = Array.from(xmlDocument.getElementsByTagName('part'));
        parts.forEach((part) => {
            const measures = elementChildren(part).filter(
                (child) => nodeLocalName(child) === 'measure'
            );
            measures.forEach((measure, measureIndex) => {
                if (measureHasBackwardRepeat(measure)) {
                    repeatMeasureIndexes.add(measureIndex);
                }
            });
        });
        return repeatMeasureIndexes;
    }

    function restoreNavigationDisplayInstructions(sourceMeasures, xmlDocument) {
        const parts = Array.from(xmlDocument.getElementsByTagName('part'));
        parts.forEach((part) => {
            const measures = elementChildren(part).filter(
                (child) => nodeLocalName(child) === 'measure'
            );
            measures.forEach((measure, measureIndex) => {
                const sourceMeasure = sourceMeasures[measureIndex];
                if (!sourceMeasure) {
                    return;
                }
                const instructions = navigationDisplayInstructionsForMeasure(measure);
                instructions.forEach((instruction) => {
                    appendDisplayRepetitionInstruction(
                        sourceMeasure,
                        instruction.fieldName,
                        measureIndex,
                        instruction.type,
                        instruction.alignment
                    );
                });
            });
        });
    }

    function navigationDisplayInstructionsForMeasure(measureElement) {
        const instructions = [];
        descendantsByLocalName(measureElement, 'coda').forEach(() => {
            instructions.push({
                fieldName: 'FirstRepetitionInstructions',
                type: REPETITION_INSTRUCTION.CODA,
                alignment: REPETITION_ALIGNMENT.BEGIN,
            });
        });
        descendantsByLocalName(measureElement, 'segno').forEach(() => {
            instructions.push({
                fieldName: 'FirstRepetitionInstructions',
                type: REPETITION_INSTRUCTION.SEGNO,
                alignment: REPETITION_ALIGNMENT.BEGIN,
            });
        });
        descendantsByLocalName(measureElement, 'words').forEach((wordsElement) => {
            const type = navigationInstructionTypeForWords(wordsElement.textContent);
            if (type === null) {
                return;
            }
            instructions.push({
                fieldName: 'LastRepetitionInstructions',
                type,
                alignment: REPETITION_ALIGNMENT.END,
            });
        });
        return instructions;
    }

    function appendDisplayRepetitionInstruction(sourceMeasure, fieldName, measureIndex, type, alignment) {
        if (!Array.isArray(sourceMeasure[fieldName])) {
            sourceMeasure[fieldName] = [];
        }
        const instructions = sourceMeasure[fieldName];
        for (let index = instructions.length - 1; index >= 0; index -= 1) {
            if (isSupersededNavigationInstruction(instructions[index]?.type, type)) {
                instructions.splice(index, 1);
            }
        }
        if (instructions.some((instruction) => instruction?.type === type)) {
            return;
        }
        instructions.push({ measureIndex, type, alignment });
    }

    function isSupersededNavigationInstruction(existingType, newType) {
        if (
            existingType === REPETITION_INSTRUCTION.DA_CAPO
            && (
                newType === REPETITION_INSTRUCTION.DA_CAPO_AL_FINE
                || newType === REPETITION_INSTRUCTION.DA_CAPO_AL_CODA
            )
        ) {
            return true;
        }
        if (
            existingType === REPETITION_INSTRUCTION.DAL_SEGNO
            && (
                newType === REPETITION_INSTRUCTION.DAL_SEGNO_AL_FINE
                || newType === REPETITION_INSTRUCTION.DAL_SEGNO_AL_CODA
            )
        ) {
            return true;
        }
        return false;
    }

    function navigationInstructionTypeForWords(text) {
        const normalized = normalizeNavigationWords(text);
        const compact = normalized.replace(/[\s.]+/g, '');
        if (compact === 'tocoda') {
            return REPETITION_INSTRUCTION.TO_CODA;
        }
        if (compact === 'fine') {
            return REPETITION_INSTRUCTION.FINE;
        }

        const isDaCapo = compact.startsWith('dc') || normalized.startsWith('da capo');
        const isDalSegno = compact.startsWith('ds') || normalized.startsWith('dal segno');
        const hasAlFine = compact.includes('alfine');
        const hasAlCoda = compact.includes('alcoda');
        if (isDaCapo) {
            if (hasAlCoda) {
                return REPETITION_INSTRUCTION.DA_CAPO_AL_CODA;
            }
            if (hasAlFine) {
                return REPETITION_INSTRUCTION.DA_CAPO_AL_FINE;
            }
            return REPETITION_INSTRUCTION.DA_CAPO;
        }
        if (isDalSegno) {
            if (hasAlCoda) {
                return REPETITION_INSTRUCTION.DAL_SEGNO_AL_CODA;
            }
            if (hasAlFine) {
                return REPETITION_INSTRUCTION.DAL_SEGNO_AL_FINE;
            }
            return REPETITION_INSTRUCTION.DAL_SEGNO;
        }
        return null;
    }

    function normalizeNavigationWords(text) {
        return String(text || '').trim().toLowerCase().replace(/\s+/g, ' ');
    }

    function measureHasBackwardRepeat(measureElement) {
        const barlines = elementChildren(measureElement).filter(
            (child) => nodeLocalName(child) === 'barline'
        );
        return barlines.some((barline) => (
            elementChildren(barline).some((child) => (
                nodeLocalName(child) === 'repeat'
                && child.getAttribute('direction') === 'backward'
            ))
        ));
    }

    function isSyntheticEndingBackJump(instruction) {
        const backJumpLineType = 2;
        if (!instruction || instruction.type !== backJumpLineType) {
            return false;
        }

        const repetition = instruction.parentRepetition;
        if (!repetition || repetition.FromWords) {
            return false;
        }

        const endingParts = repetition.EndingParts ?? repetition.endingParts;
        if (endingParts && typeof endingParts.length === 'number') {
            return endingParts.length > 0;
        }

        const endingIndexDict = repetition.EndingIndexDict ?? repetition.endingIndexDict;
        return Boolean(endingIndexDict && Object.keys(endingIndexDict).length > 0);
    }

    function elementChildren(element) {
        return Array.from(element?.children ?? element?.childNodes ?? []).filter(
            (child) => child && (child.nodeType === 1 || nodeLocalName(child))
        );
    }

    function descendantsByLocalName(element, localName) {
        const matches = [];
        const visit = (node) => {
            elementChildren(node).forEach((child) => {
                if (nodeLocalName(child) === localName) {
                    matches.push(child);
                }
                visit(child);
            });
        };
        visit(element);
        return matches;
    }

    function nodeLocalName(node) {
        return node?.localName || node?.nodeName?.split(':').pop() || '';
    }

    return {
        applyPostLoad,
        applyPostRender,
        _internal: {
            REPETITION_ALIGNMENT,
            REPETITION_INSTRUCTION,
            appendDisplayRepetitionInstruction,
            backwardRepeatMeasureIndexes,
            descendantsByLocalName,
            elementChildren,
            drawCaesuraFallbacks,
            hasGlissandoNotationTokens,
            hasCaesuraNotationTokens,
            hasNavigationDisplayTokens,
            isSupersededNavigationInstruction,
            isSyntheticEndingBackJump,
            measureHasBackwardRepeat,
            musicXmlCaesuraAnchors,
            musicXmlGlissandoSegments,
            musicXmlMeasuresInRenderOrder,
            navigationDisplayInstructionsForMeasure,
            navigationInstructionTypeForWords,
            nodeLocalName,
            normalizeNavigationWords,
            parseMusicXmlDocument,
            removeCaesuraFallbacks,
            removeGlissandoFallbacks,
            removeSyntheticEndingRepeatBarlines,
            restoreNavigationDisplayInstructions,
            solidGlissandoPath,
            stavenoteCaesuraPoint,
            stavenoteGlissandoPoint,
            svgNumber,
            wavyGlissandoPath,
        },
    };
});
