/**
 * Right page margin for OSMD layout (engraving units, same coordinate system as zoom).
 * Default is 5; CSS-only padding does not scale with the rendered score, so this keeps the
 * final barline inset at strong zoom-out.
 */
const OSMD_PAGE_RIGHT_MARGIN = 18;
const TARGET_MEASURE_PARTS_PER_RENDER = 200;
const LARGE_SCORE_COMPLEXITY_LIMIT = TARGET_MEASURE_PARTS_PER_RENDER;
const MIN_MEASURE_WINDOW = 8;
const MAX_CACHED_MEASURE_WINDOWS = 6;
const DEFAULT_AGENT_MODEL = 'gpt-5.4-mini';
const DEFAULT_AGENT_REASONING_EFFORT = 'low';
const DEFAULT_PROMPT_SPLIT_ENABLED = true;
const DEFAULT_PROMPT_SPLIT_MIN_SENTENCES = 7;
const OSMD_POST_RENDER_FIX_TOKEN_PATTERN = /<\s*(?:[\w.-]+:)?(?:glissando|caesura)\b/i;

class MusicXMLRenderer {
    constructor() {
        this.osmd = null;
        this.currentZoom = 1.0;
        this.currentMusicXML = null;
        this.currentMeasureCount = 0;
        this.currentPartCount = 1;
        this.currentMeasureStart = 1;
        this.currentScoreVersion = 0;
        this.renderRequestId = 0;
        this.loadingStateId = 0;
        this.musicxmlWindowCache = new Map();
        this.prefetchRequests = new Set();
        this.apiBaseUrl = window.location.origin;
        this.mediaRecorder = null;
        this.voiceStream = null;
        this.voiceChunks = [];
        this.voiceRecordingStartedAt = null;
        this.isRecordingVoice = false;
        this.isProcessingVoice = false;
        this.isProcessingChat = false;
        this.agentModelOptions = [];
        this.currentAgentModel = DEFAULT_AGENT_MODEL;
        this.reasoningEffortOptions = [];
        this.currentAgentReasoningEffort = DEFAULT_AGENT_REASONING_EFFORT;
        this.currentPromptSplitEnabled = DEFAULT_PROMPT_SPLIT_ENABLED;
        this.currentPromptSplitMinSentences = DEFAULT_PROMPT_SPLIT_MIN_SENTENCES;
        this.initializeElements();
        this.attachEventListeners();
        this.initializeChat();
        this.loadAgentSettings();
        this.loadDefaultScore();
    }

    initializeElements() {
        this.appLayout = document.querySelector('.app-layout');
        this.fileInput = document.getElementById('file-input');
        this.fileName = document.getElementById('file-name');
        this.loadSampleBtn = document.getElementById('load-sample');
        this.newScoreBtn = document.getElementById('new-score');
        this.downloadMusicxmlBtn = document.getElementById('download-musicxml');
        this.zoomInBtn = document.getElementById('zoom-in');
        this.zoomOutBtn = document.getElementById('zoom-out');
        this.zoomResetBtn = document.getElementById('zoom-reset');
        this.rangeControls = document.getElementById('range-controls');
        this.rangePrevBtn = document.getElementById('range-prev');
        this.rangeNextBtn = document.getElementById('range-next');
        this.rangeLabel = document.getElementById('range-label');
        this.errorMessage = document.getElementById('error-message');
        this.loading = document.getElementById('loading');
        this.loadingTitle = document.getElementById('loading-title');
        this.loadingDetail = document.getElementById('loading-detail');
        this.sheetMusicContainer = document.getElementById('sheet-music-container');
        this.container = document.getElementById('osmd-container');
        
        this.chatSidebar = document.getElementById('chat-sidebar');
        this.chatMessages = document.getElementById('chat-messages');
        this.chatForm = document.getElementById('chat-form');
        this.chatInput = document.getElementById('chat-input');
        this.sendBtn = document.getElementById('send-btn');
        this.voiceBtn = document.getElementById('voice-btn');
        this.openAgentSettingsBtn = document.getElementById('open-agent-settings');
        this.agentSettingsDialog = document.getElementById('agent-settings-dialog');
        this.agentSettingsForm = document.getElementById('agent-settings-form');
        this.closeAgentSettingsBtn = document.getElementById('close-agent-settings');
        this.cancelAgentSettingsBtn = document.getElementById('cancel-agent-settings');
        this.saveAgentSettingsBtn = document.getElementById('save-agent-settings');
        this.agentSettingsSummary = document.getElementById('agent-settings-summary');
        this.agentModelSelect = document.getElementById('agent-model-select');
        this.agentReasoningSelect = document.getElementById('agent-reasoning-select');
        this.promptSplitEnabledInput = document.getElementById('prompt-split-enabled');
        this.promptSplitMinSentencesInput = document.getElementById('prompt-split-min-sentences');
        this.toggleChatBtn = document.getElementById('toggle-chat');
        this.chatFab = document.getElementById('toggle-chat-fab');
    }

    attachEventListeners() {
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        this.loadSampleBtn.addEventListener('click', () => this.loadSample());
        this.newScoreBtn.addEventListener('click', () => this.createNewScore());
        this.downloadMusicxmlBtn.addEventListener('click', () => this.downloadMusicXML());
        this.zoomInBtn.addEventListener('click', () => this.zoom(1.2));
        this.zoomOutBtn.addEventListener('click', () => this.zoom(0.8));
        this.zoomResetBtn.addEventListener('click', () => this.resetZoom());
        this.rangePrevBtn.addEventListener('click', () => this.showPreviousMeasureRange());
        this.rangeNextBtn.addEventListener('click', () => this.showNextMeasureRange());
        
        this.chatForm.addEventListener('submit', (e) => this.handleChatSubmit(e));
        this.voiceBtn.addEventListener('click', () => this.handleVoiceToggle());
        this.openAgentSettingsBtn.addEventListener('click', () => this.openAgentSettingsDialog());
        this.agentSettingsForm.addEventListener('submit', (e) => this.handleAgentSettingsSubmit(e));
        this.closeAgentSettingsBtn.addEventListener('click', () => this.closeAgentSettingsDialog());
        this.cancelAgentSettingsBtn.addEventListener('click', () => this.closeAgentSettingsDialog());
        this.agentSettingsDialog.addEventListener('cancel', () => this.renderAgentSettings());
        this.promptSplitEnabledInput.addEventListener('change', () => this.updatePromptSplitInputState());
        this.toggleChatBtn.addEventListener('click', () => this.toggleChat());
        this.chatFab.addEventListener('click', () => this.toggleChat());
    }

    initializeChat() {
        const isMobile = window.innerWidth <= 768;
        if (isMobile) {
            this.chatSidebar.classList.add('hidden');
            this.chatFab.classList.add('visible');
            this.setChatCollapsed(true);
        }
    }

    async loadAgentSettings() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/agent/settings`);
            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Failed to load agent settings');
            }
            this.applyAgentSettingsPayload(data);
            this.renderAgentSettings();
        } catch (error) {
            this.agentModelOptions = [
                { id: DEFAULT_AGENT_MODEL, label: 'GPT-5.4 Mini' },
            ];
            this.currentAgentModel = DEFAULT_AGENT_MODEL;
            this.reasoningEffortOptions = [
                { id: DEFAULT_AGENT_REASONING_EFFORT, label: 'Low' },
            ];
            this.currentAgentReasoningEffort = DEFAULT_AGENT_REASONING_EFFORT;
            this.currentPromptSplitEnabled = DEFAULT_PROMPT_SPLIT_ENABLED;
            this.currentPromptSplitMinSentences = DEFAULT_PROMPT_SPLIT_MIN_SENTENCES;
            this.renderAgentSettings();
            console.error('Agent settings error:', error);
        }
    }

    loadAgentModels() {
        return this.loadAgentSettings();
    }

    applyAgentSettingsPayload(data) {
        this.agentModelOptions = data.models || this.agentModelOptions || [];
        this.currentAgentModel =
            data.model
            || data.current_model
            || data.default_model
            || this.currentAgentModel
            || DEFAULT_AGENT_MODEL;
        this.reasoningEffortOptions =
            data.reasoning_efforts || this.reasoningEffortOptions || [];
        this.currentAgentReasoningEffort =
            data.reasoning_effort
            || data.current_reasoning_effort
            || data.default_reasoning_effort
            || this.currentAgentReasoningEffort
            || DEFAULT_AGENT_REASONING_EFFORT;
        this.currentPromptSplitEnabled =
            typeof data.prompt_split_enabled === 'boolean'
                ? data.prompt_split_enabled
                : this.currentPromptSplitEnabled;
        this.currentPromptSplitMinSentences = Number.isFinite(
            Number(data.prompt_split_min_sentences)
        )
            ? Number(data.prompt_split_min_sentences)
            : this.currentPromptSplitMinSentences;
    }

    renderAgentSettings() {
        this.renderAgentModelOptions();
        this.renderAgentReasoningOptions();
        if (this.promptSplitEnabledInput) {
            this.promptSplitEnabledInput.checked = this.currentPromptSplitEnabled;
        }
        if (this.promptSplitMinSentencesInput) {
            this.promptSplitMinSentencesInput.value = String(
                this.currentPromptSplitMinSentences
            );
        }
        this.updateAgentSettingsSummary();
        this.updatePromptSplitInputState();
    }

    renderAgentModelOptions() {
        if (!this.agentModelSelect) return;
        const options = this.agentModelOptions.length
            ? this.agentModelOptions
            : [{ id: DEFAULT_AGENT_MODEL, label: 'GPT-5.4 Mini' }];

        this.agentModelSelect.innerHTML = '';
        options.forEach((model) => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.label || model.id;
            this.agentModelSelect.appendChild(option);
        });
        this.agentModelSelect.value = this.currentAgentModel;
    }

    renderAgentReasoningOptions() {
        if (!this.agentReasoningSelect) return;
        const options = this.reasoningEffortOptions.length
            ? this.reasoningEffortOptions
            : [{ id: DEFAULT_AGENT_REASONING_EFFORT, label: 'Low' }];

        this.agentReasoningSelect.innerHTML = '';
        options.forEach((effort) => {
            const option = document.createElement('option');
            option.value = effort.id;
            option.textContent = effort.label || effort.id;
            this.agentReasoningSelect.appendChild(option);
        });
        this.agentReasoningSelect.value = this.currentAgentReasoningEffort;
    }

    agentOptionLabel(options, selectedId) {
        const option = options.find((item) => item.id === selectedId);
        return option?.label || selectedId;
    }

    updateAgentSettingsSummary() {
        if (!this.agentSettingsSummary) return;
        const modelLabel = this.agentOptionLabel(
            this.agentModelOptions,
            this.selectedAgentModel()
        );
        const reasoningLabel = this.agentOptionLabel(
            this.reasoningEffortOptions,
            this.selectedAgentReasoningEffort()
        );
        this.agentSettingsSummary.textContent = `${modelLabel} · ${reasoningLabel} reasoning`;
    }

    selectedAgentModel() {
        return this.currentAgentModel || DEFAULT_AGENT_MODEL;
    }

    selectedAgentReasoningEffort() {
        return this.currentAgentReasoningEffort || DEFAULT_AGENT_REASONING_EFFORT;
    }

    selectedPromptSplitEnabled() {
        return this.currentPromptSplitEnabled;
    }

    selectedPromptSplitMinSentences() {
        return this.currentPromptSplitMinSentences || DEFAULT_PROMPT_SPLIT_MIN_SENTENCES;
    }

    draftAgentModel() {
        return this.agentModelSelect?.value || this.selectedAgentModel();
    }

    draftAgentReasoningEffort() {
        return (
            this.agentReasoningSelect?.value
            || this.selectedAgentReasoningEffort()
        );
    }

    draftPromptSplitEnabled() {
        return Boolean(this.promptSplitEnabledInput?.checked);
    }

    draftPromptSplitMinSentences() {
        const rawValue = this.promptSplitMinSentencesInput?.value || '';
        const sentenceCount = Number(rawValue);
        if (!Number.isInteger(sentenceCount) || sentenceCount < 1) {
            throw new Error('Minimum sentences must be a positive whole number');
        }
        return sentenceCount;
    }

    setAgentSettingsDisabled(disabled) {
        [
            this.openAgentSettingsBtn,
            this.agentModelSelect,
            this.agentReasoningSelect,
            this.promptSplitEnabledInput,
            this.saveAgentSettingsBtn,
        ].forEach((element) => {
            if (element) {
                element.disabled = disabled;
            }
        });
        this.updatePromptSplitInputState(disabled);
    }

    setAgentModelDisabled(disabled) {
        this.setAgentSettingsDisabled(disabled);
    }

    updatePromptSplitInputState(forceDisabled = false) {
        if (!this.promptSplitMinSentencesInput) return;
        const promptSplitEnabled = this.promptSplitEnabledInput?.checked ?? true;
        this.promptSplitMinSentencesInput.disabled = forceDisabled || !promptSplitEnabled;
    }

    openAgentSettingsDialog() {
        if (this.isProcessingChat || this.isProcessingVoice) return;
        this.renderAgentSettings();
        if (typeof this.agentSettingsDialog.showModal === 'function') {
            this.agentSettingsDialog.showModal();
        } else {
            this.agentSettingsDialog.setAttribute('open', '');
        }
    }

    closeAgentSettingsDialog() {
        this.renderAgentSettings();
        if (typeof this.agentSettingsDialog.close === 'function') {
            this.agentSettingsDialog.close();
        } else {
            this.agentSettingsDialog.removeAttribute('open');
        }
    }

    async handleAgentSettingsSubmit(event) {
        event.preventDefault();
        try {
            const payload = {
                model: this.draftAgentModel(),
                reasoning_effort: this.draftAgentReasoningEffort(),
                prompt_split_enabled: this.draftPromptSplitEnabled(),
                prompt_split_min_sentences: this.draftPromptSplitMinSentences(),
            };
            this.setAgentSettingsDisabled(true);
            const response = await fetch(`${this.apiBaseUrl}/api/agent/settings`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Failed to update agent settings');
            }
            this.applyAgentSettingsPayload(data);
            this.renderAgentSettings();
            this.closeAgentSettingsDialog();
        } catch (error) {
            this.renderAgentSettings();
            this.showError(`Failed to update settings: ${error.message}`);
            console.error('Agent settings update error:', error);
        } finally {
            this.setAgentSettingsDisabled(this.isProcessingChat || this.isProcessingVoice);
        }
    }

    setChatCollapsed(isCollapsed) {
        this.appLayout.classList.toggle('chat-collapsed', isCollapsed);
    }

    toggleChat() {
        const isHidden = this.chatSidebar.classList.toggle('hidden');
        this.setChatCollapsed(isHidden);
        if (isHidden) {
            this.chatFab.classList.add('visible');
        } else {
            this.chatFab.classList.remove('visible');
        }
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorMessage.classList.add('show');
        setTimeout(() => {
            this.errorMessage.classList.remove('show');
        }, 5000);
    }

    hideError() {
        this.errorMessage.classList.remove('show');
    }

    showLoading(title = 'Rendering sheet music', detail = 'Preparing notation...') {
        this.loadingStateId += 1;
        this.updateLoadingText(title, detail);
        this.loading.classList.add('show');
        this.loading.setAttribute('aria-hidden', 'false');
        this.sheetMusicContainer.classList.add('rendering');
        this.sheetMusicContainer.setAttribute('aria-busy', 'true');
        this.rangePrevBtn.disabled = true;
        this.rangeNextBtn.disabled = true;
        return this.loadingStateId;
    }

    updateLoadingText(title, detail) {
        this.loadingTitle.textContent = title;
        this.loadingDetail.textContent = detail;
    }

    hideLoading(loadingStateId = null) {
        if (loadingStateId !== null && loadingStateId !== this.loadingStateId) {
            return;
        }
        this.loading.classList.remove('show');
        this.loading.setAttribute('aria-hidden', 'true');
        this.sheetMusicContainer.classList.remove('rendering');
        this.sheetMusicContainer.setAttribute('aria-busy', 'false');
        this.updateRangeControls();
    }

    async waitForNextPaint() {
        await new Promise((resolve) => {
            if (typeof requestAnimationFrame === 'function') {
                requestAnimationFrame(() => resolve());
                return;
            }
            setTimeout(resolve, 0);
        });
    }

    async waitForOsmdDomSettled() {
        await this.waitForNextPaint();
        await new Promise((resolve) => setTimeout(resolve, 0));
        await this.waitForNextPaint();
    }

    addChatMessage(text, type = 'agent') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        if (type === 'agent' && text.startsWith('ERROR:')) {
            messageDiv.classList.add('error');
        }
        messageDiv.textContent = text;
        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    addProgressMessage(initialLabel = 'Working...') {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message agent-message progress-message';

        const stepsDiv = document.createElement('div');
        stepsDiv.className = 'progress-steps';

        const finalDiv = document.createElement('div');
        finalDiv.className = 'progress-final';
        finalDiv.hidden = true;

        messageDiv.appendChild(stepsDiv);
        messageDiv.appendChild(finalDiv);
        this.chatMessages.appendChild(messageDiv);

        const progress = {
            message: messageDiv,
            steps: stepsDiv,
            final: finalDiv,
            currentStep: null,
            currentTool: null,
        };
        this.addProgressStep(progress, initialLabel);
        return progress;
    }

    addProgressStep(progress, label, tool = null) {
        if (!label) return;
        if (progress.currentStep?.dataset.label === label) {
            return;
        }

        this.completeProgressStep(progress);

        const step = document.createElement('div');
        step.className = 'progress-step active';
        step.dataset.label = label;
        if (tool) {
            step.dataset.tool = tool;
        }

        const indicator = document.createElement('span');
        indicator.className = 'progress-indicator';
        indicator.setAttribute('aria-hidden', 'true');

        const text = document.createElement('span');
        text.className = 'progress-label';
        text.textContent = label;

        step.appendChild(indicator);
        step.appendChild(text);
        progress.steps.appendChild(step);
        progress.currentStep = step;
        progress.currentTool = tool;
        this.scrollChatToBottom();
    }

    completeProgressStep(progress, tool = null) {
        const step = progress.currentStep;
        if (!step) return;
        if (tool && progress.currentTool && progress.currentTool !== tool) return;

        step.classList.remove('active');
        step.classList.add('complete');
        progress.currentStep = null;
        progress.currentTool = null;
    }

    showProgressFinal(progress, text) {
        this.completeProgressStep(progress);
        progress.steps.hidden = true;
        progress.final.hidden = false;
        progress.final.textContent = text || '(no response produced)';
        progress.message.classList.remove('progress-message');
        this.scrollChatToBottom();
    }

    showProgressError(progress, text) {
        this.completeProgressStep(progress);
        progress.message.classList.add('error');
        progress.steps.hidden = true;
        progress.final.hidden = false;
        progress.final.textContent = text;
        this.scrollChatToBottom();
    }

    scrollChatToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    setScoreMetadata(data, resetRange = true) {
        if (Number.isFinite(data?.measure_count)) {
            this.currentMeasureCount = data.measure_count;
        }
        if (Number.isFinite(data?.part_count)) {
            this.currentPartCount = Math.max(1, data.part_count);
        }
        if (resetRange) {
            this.currentMeasureStart = 1;
        } else if (!this.usesMeasurePaging()) {
            this.currentMeasureStart = 1;
        } else {
            this.currentMeasureStart = Math.min(
                Math.max(1, this.currentMeasureStart),
                Math.max(1, this.currentMeasureCount)
            );
        }
    }

    measureWindowSize() {
        const dynamicWindow = Math.floor(
            TARGET_MEASURE_PARTS_PER_RENDER / this.currentPartCount
        );
        return Math.max(MIN_MEASURE_WINDOW, dynamicWindow);
    }

    usesMeasurePaging() {
        const complexity = this.currentMeasureCount * this.currentPartCount;
        return (
            this.currentMeasureCount > this.measureWindowSize() &&
            complexity > LARGE_SCORE_COMPLEXITY_LIMIT
        );
    }

    currentMeasureEnd() {
        if (!this.usesMeasurePaging()) {
            return this.currentMeasureCount;
        }
        return Math.min(
            this.currentMeasureCount,
            this.currentMeasureStart + this.measureWindowSize() - 1
        );
    }

    updateRangeControls() {
        if (!this.usesMeasurePaging()) {
            this.rangeControls.hidden = true;
            return;
        }

        const endMeasure = this.currentMeasureEnd();
        this.rangeControls.hidden = false;
        this.rangeLabel.textContent = `Measures ${this.currentMeasureStart}-${endMeasure} of ${this.currentMeasureCount}`;
        this.rangePrevBtn.disabled = this.currentMeasureStart <= 1;
        this.rangeNextBtn.disabled = endMeasure >= this.currentMeasureCount;
    }

    async displayScoreResponse(data, resetRange = true) {
        if (Number.isFinite(data?.score_version)) {
            if (data.score_version !== this.currentScoreVersion) {
                this.musicxmlWindowCache.clear();
                this.prefetchRequests.clear();
            }
            this.currentScoreVersion = data.score_version;
        }
        this.setScoreMetadata(data, resetRange);

        if (!resetRange && data?.changed_range) {
            this.updateMeasureStartForChangedRange(data.changed_range);
        } else if (data?.render_range?.start) {
            this.currentMeasureStart = data.render_range.start;
        }

        if (data.musicxml) {
            this.currentMusicXML = data.musicxml;
            if (data.render_range?.start && data.render_range?.end) {
                this.cacheMeasureWindow(
                    data.render_range.start,
                    data.render_range.end,
                    data.musicxml
                );
            }
            await this.renderMusicXML(data.musicxml);
            this.prefetchAdjacentMeasureRanges();
            return;
        }

        await this.fetchAndRenderCurrentMeasureRange();
    }

    shouldRefreshScore(data) {
        if (data?.musicxml) {
            return true;
        }
        if (!Number.isFinite(data?.score_version)) {
            return false;
        }
        return data.score_version !== this.currentScoreVersion;
    }

    async showPreviousMeasureRange() {
        if (!this.currentMeasureCount) return;
        this.currentMeasureStart = Math.max(
            1,
            this.currentMeasureStart - this.measureWindowSize()
        );
        await this.fetchAndRenderCurrentMeasureRange();
    }

    async showNextMeasureRange() {
        if (!this.currentMeasureCount) return;
        const nextStart = this.currentMeasureStart + this.measureWindowSize();
        if (nextStart > this.currentMeasureCount) return;

        this.currentMeasureStart = nextStart;
        await this.fetchAndRenderCurrentMeasureRange();
    }

    normalizeMeasureRange(range) {
        const start = Number(range?.start);
        const end = Number(range?.end ?? start);
        if (
            !Number.isFinite(start) ||
            !Number.isFinite(end) ||
            this.currentMeasureCount < 1
        ) {
            return null;
        }

        const firstMeasure = Math.min(Math.floor(start), Math.floor(end));
        const lastMeasure = Math.max(Math.floor(start), Math.floor(end));
        return {
            start: Math.max(1, Math.min(firstMeasure, this.currentMeasureCount)),
            end: Math.max(1, Math.min(lastMeasure, this.currentMeasureCount)),
        };
    }

    currentRenderRange() {
        if (this.currentMeasureCount < 1) {
            return null;
        }
        return {
            start: this.currentMeasureStart,
            end: this.currentMeasureEnd(),
        };
    }

    measurePageStartForMeasure(measure) {
        const windowSize = this.measureWindowSize();
        const safeMeasure = Math.max(
            1,
            Math.min(Math.floor(Number(measure)), this.currentMeasureCount)
        );
        return Math.floor((safeMeasure - 1) / windowSize) * windowSize + 1;
    }

    measureRangeIntersectsCurrentView(range) {
        const visibleStart = this.currentMeasureStart;
        const visibleEnd = this.currentMeasureEnd();
        return range.start <= visibleEnd && range.end >= visibleStart;
    }

    updateMeasureStartForChangedRange(range) {
        const changedRange = this.normalizeMeasureRange(range);
        if (!changedRange) {
            return;
        }

        if (!this.usesMeasurePaging()) {
            this.currentMeasureStart = 1;
            return;
        }

        if (this.measureRangeIntersectsCurrentView(changedRange)) {
            return;
        }

        this.currentMeasureStart = this.measurePageStartForMeasure(changedRange.start);
    }

    async fetchAndRenderCurrentMeasureRange() {
        const requestId = ++this.renderRequestId;
        const start = this.currentMeasureStart;
        const end = this.currentMeasureEnd();
        const cachedMusicXML = this.getCachedMeasureWindow(start, end);
        if (cachedMusicXML) {
            this.currentMusicXML = cachedMusicXML;
            await this.renderMusicXML(cachedMusicXML);
            this.prefetchAdjacentMeasureRanges();
            return;
        }

        const params = new URLSearchParams({
            start: String(start),
            end: String(end),
        });

        let rendered = false;
        const loadingStateId = this.showLoading(
            'Loading score window',
            `Fetching measures ${start}-${end}...`
        );

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/musicxml/window?${params}`);
            const data = await response.json();
            if (!data.success) {
                throw new Error(data.error || 'Failed to load score window');
            }
            if (requestId !== this.renderRequestId) {
                return;
            }

            if (Number.isFinite(data.score_version)) {
                if (data.score_version !== this.currentScoreVersion) {
                    this.musicxmlWindowCache.clear();
                    this.prefetchRequests.clear();
                }
                this.currentScoreVersion = data.score_version;
            }
            this.setScoreMetadata(data, false);
            if (data.render_range?.start) {
                this.currentMeasureStart = data.render_range.start;
            }
            this.currentMusicXML = data.musicxml;
            if (data.render_range?.start && data.render_range?.end) {
                this.cacheMeasureWindow(
                    data.render_range.start,
                    data.render_range.end,
                    data.musicxml
                );
            }
            await this.renderMusicXML(data.musicxml);
            rendered = true;
            this.prefetchAdjacentMeasureRanges();
        } finally {
            if (!rendered) {
                this.hideLoading(loadingStateId);
            }
        }
    }

    measureWindowCacheKey(start, end, scoreVersion = this.currentScoreVersion) {
        return `${scoreVersion}:${start}:${end}`;
    }

    cacheMeasureWindow(start, end, musicxml) {
        if (!this.usesMeasurePaging() || !musicxml) {
            return;
        }

        const key = this.measureWindowCacheKey(start, end);
        if (this.musicxmlWindowCache.has(key)) {
            this.musicxmlWindowCache.delete(key);
        }
        this.musicxmlWindowCache.set(key, musicxml);

        while (this.musicxmlWindowCache.size > MAX_CACHED_MEASURE_WINDOWS) {
            const oldestKey = this.musicxmlWindowCache.keys().next().value;
            this.musicxmlWindowCache.delete(oldestKey);
        }
    }

    getCachedMeasureWindow(start, end) {
        const key = this.measureWindowCacheKey(start, end);
        const cached = this.musicxmlWindowCache.get(key);
        if (!cached) {
            return null;
        }

        this.musicxmlWindowCache.delete(key);
        this.musicxmlWindowCache.set(key, cached);
        return cached;
    }

    prefetchAdjacentMeasureRanges() {
        if (!this.usesMeasurePaging() || !this.currentMeasureCount) {
            return;
        }

        const windowSize = this.measureWindowSize();
        const starts = [
            this.currentMeasureStart + windowSize,
            this.currentMeasureStart - windowSize,
        ];
        const scoreVersion = this.currentScoreVersion;
        const schedulePrefetch = window.requestIdleCallback
            || ((callback) => setTimeout(callback, 250));

        schedulePrefetch(() => {
            if (scoreVersion !== this.currentScoreVersion) {
                return;
            }
            starts.forEach((start) => this.prefetchMeasureRange(start));
        });
    }

    async prefetchMeasureRange(start) {
        if (!this.usesMeasurePaging() || start < 1 || start > this.currentMeasureCount) {
            return;
        }

        const windowSize = this.measureWindowSize();
        const safeStart = Math.max(1, Math.min(start, this.currentMeasureCount));
        const safeEnd = Math.min(this.currentMeasureCount, safeStart + windowSize - 1);
        const key = this.measureWindowCacheKey(safeStart, safeEnd);
        if (this.musicxmlWindowCache.has(key) || this.prefetchRequests.has(key)) {
            return;
        }

        this.prefetchRequests.add(key);
        const scoreVersion = this.currentScoreVersion;
        const params = new URLSearchParams({
            start: String(safeStart),
            end: String(safeEnd),
        });

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/musicxml/window?${params}`);
            const data = await response.json();
            if (
                data.success &&
                data.musicxml &&
                data.score_version === scoreVersion &&
                data.render_range?.start &&
                data.render_range?.end
            ) {
                this.cacheMeasureWindow(
                    data.render_range.start,
                    data.render_range.end,
                    data.musicxml
                );
            }
        } catch (error) {
            console.debug('Measure range prefetch failed:', error);
        } finally {
            this.prefetchRequests.delete(key);
        }
    }

    async handleStreamEvent(event, progress) {
        if (event.type === 'phase') {
            this.addProgressStep(progress, event.label || 'Working...');
            return false;
        }
        if (event.type === 'tool_start') {
            this.addProgressStep(progress, event.label || 'Working...', event.tool || null);
            return false;
        }
        if (event.type === 'tool_end') {
            this.completeProgressStep(progress, event.tool || null);
            return false;
        }
        if (event.type === 'final') {
            if (this.shouldRefreshScore(event)) {
                await this.displayScoreResponse(event, false);
            }
            this.showProgressFinal(progress, event.response);
            return true;
        }
        if (event.type === 'error') {
            throw new Error(event.error || 'The chat request failed');
        }
        return false;
    }

    async streamChatMessage(message, progress) {
        const response = await fetch(`${this.apiBaseUrl}/api/chat/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message,
                model: this.selectedAgentModel(),
                reasoning_effort: this.selectedAgentReasoningEffort(),
                prompt_split_enabled: this.selectedPromptSplitEnabled(),
                prompt_split_min_sentences: this.selectedPromptSplitMinSentences(),
                render_range: this.currentRenderRange(),
            }),
        });

        if (!response.ok) {
            let errorMessage = response.statusText;
            try {
                const data = await response.json();
                errorMessage = data.error || errorMessage;
            } catch {
                // Use statusText when the error body is not JSON.
            }
            throw new Error(errorMessage);
        }

        if (!response.body) {
            await this.sendLegacyChatMessage(message, progress);
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let sawFinal = false;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;
                const event = JSON.parse(trimmed);
                sawFinal = await this.handleStreamEvent(event, progress) || sawFinal;
            }
        }

        buffer += decoder.decode();
        const trimmed = buffer.trim();
        if (trimmed) {
            const event = JSON.parse(trimmed);
            sawFinal = await this.handleStreamEvent(event, progress) || sawFinal;
        }

        if (!sawFinal) {
            throw new Error('The chat stream ended before the agent returned a response');
        }
    }

    async sendLegacyChatMessage(message, progress) {
        const response = await fetch(`${this.apiBaseUrl}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message,
                model: this.selectedAgentModel(),
                reasoning_effort: this.selectedAgentReasoningEffort(),
                prompt_split_enabled: this.selectedPromptSplitEnabled(),
                prompt_split_min_sentences: this.selectedPromptSplitMinSentences(),
                render_range: this.currentRenderRange(),
            }),
        });

        const data = await response.json();

        if (data.success) {
            if (this.shouldRefreshScore(data)) {
                await this.displayScoreResponse(data, false);
            }
            this.showProgressFinal(progress, data.response);
        } else {
            throw new Error(data.error);
        }
    }

    preferredVoiceRecorderOptions() {
        if (typeof MediaRecorder === 'undefined' || !MediaRecorder.isTypeSupported) {
            return {};
        }

        const preferredTypes = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4',
            'audio/ogg;codecs=opus',
            'audio/ogg',
        ];
        const mimeType = preferredTypes.find((type) => MediaRecorder.isTypeSupported(type));
        return mimeType ? { mimeType } : {};
    }

    voiceExtensionForMimeType(mimeType) {
        const normalized = (mimeType || '').split(';')[0].trim().toLowerCase();
        if (normalized === 'audio/mp4') return 'mp4';
        if (normalized === 'audio/ogg') return 'ogg';
        if (normalized === 'audio/wav') return 'wav';
        if (normalized === 'audio/mpeg') return 'mp3';
        return 'webm';
    }

    async handleVoiceToggle() {
        if (this.isProcessingVoice) return;
        if (this.isRecordingVoice) {
            this.stopVoiceRecording();
            return;
        }
        await this.startVoiceRecording();
    }

    async startVoiceRecording() {
        if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
            this.showError('Voice recording is not supported by this browser');
            return;
        }

        try {
            this.hideError();
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const options = this.preferredVoiceRecorderOptions();
            const recorder = new MediaRecorder(stream, options);

            this.voiceStream = stream;
            this.voiceChunks = [];
            this.mediaRecorder = recorder;
            this.voiceRecordingStartedAt = performance.now();

            recorder.addEventListener('dataavailable', (event) => {
                if (event.data && event.data.size > 0) {
                    this.voiceChunks.push(event.data);
                }
            });

            recorder.addEventListener('stop', () => {
                const mimeType = recorder.mimeType || options.mimeType || 'audio/webm';
                const durationSeconds = this.voiceRecordingStartedAt
                    ? (performance.now() - this.voiceRecordingStartedAt) / 1000
                    : null;
                const blob = new Blob(this.voiceChunks, { type: mimeType });
                this.cleanupVoiceRecording();
                this.processVoiceRecording(blob, mimeType, durationSeconds);
            });

            recorder.start();
            this.setVoiceRecordingState(true);
        } catch (error) {
            this.cleanupVoiceRecording();
            this.showError(`Could not start voice recording: ${error.message}`);
            console.error('Voice recording start error:', error);
        }
    }

    stopVoiceRecording() {
        if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') {
            return;
        }
        this.mediaRecorder.stop();
    }

    cleanupVoiceRecording() {
        if (this.voiceStream) {
            this.voiceStream.getTracks().forEach((track) => track.stop());
        }
        this.mediaRecorder = null;
        this.voiceStream = null;
        this.voiceRecordingStartedAt = null;
        this.setVoiceRecordingState(false);
    }

    setVoiceRecordingState(isRecording) {
        this.isRecordingVoice = isRecording;
        this.voiceBtn.classList.toggle('recording', isRecording);
        this.voiceBtn.setAttribute('aria-pressed', String(isRecording));
        this.voiceBtn.title = isRecording ? 'Stop voice input' : 'Record voice input';
    }

    setVoiceProcessingState(isProcessing) {
        this.isProcessingVoice = isProcessing;
        this.voiceBtn.disabled = isProcessing;
        this.sendBtn.disabled = isProcessing;
        this.chatInput.disabled = isProcessing;
        this.setAgentModelDisabled(isProcessing || this.isProcessingChat);
    }

    async processVoiceRecording(blob, mimeType, durationSeconds) {
        if (!blob.size) {
            this.showError('No audio was captured');
            return;
        }

        this.setVoiceProcessingState(true);
        const voiceProgress = this.addProgressMessage('Interpreting voice input...');
        let activeProgress = voiceProgress;

        try {
            const voiceData = await this.sendVoiceAudio(blob, mimeType, durationSeconds);
            this.showProgressFinal(
                voiceProgress,
                this.formatVoiceInterpretationMessage(voiceData)
            );
            this.addChatMessage(this.formatVoiceUserMessage(voiceData), 'user');

            const progress = this.addProgressMessage('Sending voice request...');
            activeProgress = progress;
            await this.streamChatMessage(voiceData.agent_message, progress);
        } catch (error) {
            const errorMsg = `Voice input failed: ${error.message}`;
            this.showProgressError(activeProgress, errorMsg);
            this.showError(errorMsg);
            console.error('Voice input error:', error);
        } finally {
            this.setVoiceProcessingState(false);
            this.chatInput.focus();
        }
    }

    async sendVoiceAudio(blob, mimeType, durationSeconds) {
        const formData = new FormData();
        const extension = this.voiceExtensionForMimeType(mimeType);
        formData.append('audio', blob, `voice-input.${extension}`);
        formData.append('mode', 'speech');
        formData.append('language', 'en');
        if (this.currentMeasureCount > 0) {
            formData.append('render_start', String(this.currentMeasureStart));
            formData.append('render_end', String(this.currentMeasureEnd()));
        }
        if (durationSeconds !== null && Number.isFinite(durationSeconds)) {
            formData.append('duration_seconds', String(durationSeconds));
        }

        const response = await fetch(`${this.apiBaseUrl}/api/voice`, {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Voice processing failed');
        }
        if (!data.agent_message) {
            throw new Error('Voice processing did not produce an agent request');
        }
        return data;
    }

    formatVoiceInterpretationMessage(data) {
        if (data.speech_text) {
            return `Voice transcript: ${data.speech_text}`;
        }
        return 'Voice transcript unavailable';
    }

    formatVoiceUserMessage(data) {
        if (data.speech_text) {
            return `Voice input: ${data.speech_text}`;
        }
        return 'Voice input';
    }

    async handleChatSubmit(event) {
        event.preventDefault();
        
        const message = this.chatInput.value.trim();
        if (!message) return;

        this.addChatMessage(message, 'user');
        this.chatInput.value = '';
        this.sendBtn.disabled = true;
        this.chatInput.disabled = true;
        this.isProcessingChat = true;
        this.setAgentModelDisabled(true);
        const progress = this.addProgressMessage('Sending request...');

        try {
            await this.streamChatMessage(message, progress);
        } catch (error) {
            const errorMsg = `Failed to send message: ${error.message}`;
            this.showProgressError(progress, errorMsg);
            this.showError(errorMsg);
            console.error('Chat error:', error);
        } finally {
            this.sendBtn.disabled = false;
            this.chatInput.disabled = false;
            this.isProcessingChat = false;
            this.setAgentModelDisabled(this.isProcessingVoice);
            this.chatInput.focus();
        }
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.fileName.textContent = file.name;
        this.hideError();
        const loadingStateId = this.showLoading(
            'Uploading score',
            `${file.name}`
        );
        await this.waitForNextPaint();

        const formData = new FormData();
        formData.append('musicxml', file);

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/load`, {
                method: 'POST',
                body: formData,
            });

            this.updateLoadingText(
                'Preparing score',
                'Parsing MusicXML and selecting the first render window...'
            );
            const data = await response.json();

            if (data.success) {
                await this.displayScoreResponse(data);
                this.addChatMessage(`Loaded ${file.name}. Ready to edit.`, 'agent');
            } else {
                throw new Error(data.error);
            }
        } catch (error) {
            this.hideLoading(loadingStateId);
            this.showError(`Failed to load file: ${error.message}`);
            console.error('File upload error:', error);
        }
    }

    async loadDefaultScore() {
        await this.createNewScore({ announce: false });
    }

    async createNewScore({ announce = true } = {}) {
        this.hideError();
        this.fileName.textContent = 'new-score.musicxml';

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/new`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ measures: 8 }),
            });

            const data = await response.json();

            if (data.success) {
                await this.displayScoreResponse(data);
                if (announce) {
                    this.addChatMessage('Created new 8-measure piano grand staff. What would you like to add?', 'agent');
                }
            } else {
                throw new Error(data.error);
            }
        } catch (error) {
            this.showError(`Failed to create new score: ${error.message}`);
            console.error('New score error:', error);
        }
    }

    /**
     * Apply OSMD engraving margins so staff/page gaps scale correctly at every zoom level.
     *
     * @param {*} osmd OpenSheetMusicDisplay instance from opensheetmusicdisplay.
     */
    applyOsmdMargins(osmd) {
        const rules = osmd.EngravingRules ?? osmd.rules;
        if (!rules) {
            return;
        }
        rules.PageRightMargin = OSMD_PAGE_RIGHT_MARGIN;
    }

    applyOsmdPostLoadCorrections(osmd, xmlString) {
        ScoreSpeakOsmdFixes.applyPostLoad(osmd, xmlString);
    }

    applyOsmdPostRenderCorrections(xmlString) {
        return ScoreSpeakOsmdFixes.applyPostRender(this.container, xmlString);
    }

    async applyOsmdPostRenderCorrectionsAfterSettled(xmlString) {
        if (!OSMD_POST_RENDER_FIX_TOKEN_PATTERN.test(xmlString || '')) {
            return 0;
        }

        await this.waitForOsmdDomSettled();
        return this.applyOsmdPostRenderCorrections(xmlString);
    }

    async loadSample() {
        this.hideError();
        this.fileName.textContent = 'sample.musicxml';

        try {
            const response = await fetch('sample.musicxml');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const text = await response.text();

            const blob = new Blob([text], {
                type: 'application/vnd.recordare.musicxml+xml',
            });
            const formData = new FormData();
            formData.append('musicxml', blob, 'sample.musicxml');

            const loadResponse = await fetch(`${this.apiBaseUrl}/api/load`, {
                method: 'POST',
                body: formData,
            });
            const loadData = await loadResponse.json();

            if (!loadData.success) {
                throw new Error(loadData.error || 'Server failed to load sample');
            }

            await this.displayScoreResponse(loadData);
            this.addChatMessage('Loaded sample piano grand staff. Try asking me to make changes.', 'agent');
        } catch (error) {
            this.showError(`Failed to load sample: ${error.message}`);
            console.error('Sample load error:', error);
        }
    }

    async renderMusicXML(xmlString) {
        const loadingStateId = this.showLoading(
            'Rendering sheet music',
            this.usesMeasurePaging()
                ? `Drawing measures ${this.currentMeasureStart}-${this.currentMeasureEnd()}...`
                : 'Drawing the current score...'
        );
        await this.waitForNextPaint();

        let rendered = false;
        try {
            const renderStart = performance.now();
            if (this.osmd) {
                this.osmd.clear();
            }

            const osmdOptions = {
                autoResize: true,
                backend: 'svg',
                drawTitle: true,
                drawComposer: true,
                drawCredits: true,
                drawPartNames: true,
                drawingParameters: 'default',
                newSystemFromXML: true,
                newPageFromXML: true
            };

            this.updateRangeControls();

            this.osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(
                this.container,
                osmdOptions
            );

            this.applyOsmdMargins(this.osmd);

            const loadStart = performance.now();
            await this.osmd.load(xmlString);
            const loadEnd = performance.now();
            this.applyOsmdPostLoadCorrections(this.osmd, xmlString);
            this.applyOsmdMargins(this.osmd);

            this.osmd.zoom = this.currentZoom;
            const drawStart = performance.now();
            this.osmd.render();
            await this.waitForNextPaint();
            const drawEnd = performance.now();

            console.info('OSMD render timing', {
                scoreVersion: this.currentScoreVersion,
                range: `${this.currentMeasureStart}-${this.currentMeasureEnd()}`,
                loadMs: Math.round(loadEnd - loadStart),
                drawMs: Math.round(drawEnd - drawStart),
                totalMs: Math.round(drawEnd - renderStart),
                xmlBytes: xmlString.length,
            });

            this.hideError();
            rendered = true;
        } catch (error) {
            this.showError(`Failed to render music: ${error.message}`);
            console.error('Render error:', error);
        } finally {
            this.hideLoading(loadingStateId);
            if (rendered) {
                await this.applyOsmdPostRenderCorrectionsAfterSettled(xmlString);
            }
        }
    }

    async zoom(factor) {
        if (!this.osmd) {
            this.showError('Please load a music score first');
            return;
        }

        this.currentZoom *= factor;
        this.currentZoom = Math.max(0.3, Math.min(3.0, this.currentZoom));

        const loadingStateId = this.showLoading('Updating zoom', 'Redrawing the current view...');
        let rendered = false;
        try {
            await this.waitForNextPaint();
            this.osmd.zoom = this.currentZoom;
            this.osmd.render();
            await this.waitForNextPaint();
            rendered = true;
        } catch (error) {
            this.showError(`Zoom failed: ${error.message}`);
            console.error('Zoom error:', error);
        } finally {
            this.hideLoading(loadingStateId);
            if (rendered) {
                await this.applyOsmdPostRenderCorrectionsAfterSettled(this.currentMusicXML);
            }
        }
    }

    async resetZoom() {
        if (!this.osmd) {
            this.showError('Please load a music score first');
            return;
        }

        this.currentZoom = 1.0;

        const loadingStateId = this.showLoading('Resetting zoom', 'Redrawing the current view...');
        let rendered = false;
        try {
            await this.waitForNextPaint();
            this.osmd.zoom = this.currentZoom;
            this.osmd.render();
            await this.waitForNextPaint();
            rendered = true;
        } catch (error) {
            this.showError(`Reset zoom failed: ${error.message}`);
            console.error('Reset zoom error:', error);
        } finally {
            this.hideLoading(loadingStateId);
            if (rendered) {
                await this.applyOsmdPostRenderCorrectionsAfterSettled(this.currentMusicXML);
            }
        }
    }

    async downloadMusicXML() {
        this.hideError();

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/musicxml/download`);

            if (!response.ok) {
                let message = response.statusText;
                try {
                    const data = await response.json();
                    message = data.error || message;
                } catch {
                    // Use statusText if body is not JSON
                }
                throw new Error(message);
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = 'score.musicxml';
            anchor.rel = 'noopener';
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            URL.revokeObjectURL(url);
        } catch (error) {
            const msg = `Download failed: ${error.message}`;
            this.showError(msg);
            console.error('Download MusicXML error:', error);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new MusicXMLRenderer();
});
