/**
 * LLM Search Chat — Client Application
 *
 * Handles: chat messaging, SSE streaming, image upload (paste + picker),
 * file upload (text extraction), model selection, markdown rendering.
 */

// ── State ────────────────────────────────────────────────────────

const state = {
    messages: [],            // { role, content } — content is string or vision array
    streaming: false,
    pendingImage: null,      // data: URL string or null
    pendingFileName: null,   // string or null
    pendingFileContent: null,// string or null
    selectedModel: '',
    models: [],
    abortController: null,
    statsVisible: false,
    lastLiveStats: null,    // from SSE event: stats
    statsPollInterval: null,
    reasoningEnabled: true,  // 🧠 reasoning toggle
};

// ── DOM refs ─────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    messages: $('#messages'),
    chatContainer: $('#chat-container'),
    messageInput: $('#message-input'),
    sendBtn: $('#send-btn'),
    sendIcon: $('#send-icon'),
    stopIcon: $('#stop-icon'),
    attachBtn: $('#attach-btn'),
    fileInput: $('#file-input'),
    modelSelect: $('#model-select'),
    clearBtn: $('#clear-btn'),
    reasoningToggleBtn: $('#reasoning-toggle-btn'),
    imagePreview: $('#image-preview'),
    imagePreviewImg: $('#image-preview-img'),
    removeImageBtn: $('#remove-image-btn'),
    filePreview: $('#file-preview'),
    filePreviewName: $('#file-preview-name'),
    filePreviewSize: $('#file-preview-size'),
    removeFileBtn: $('#remove-file-btn'),
    errorBanner: $('#error-banner'),
    // Settings
    settingsBtn: $('#settings-btn'),
    settingsOverlay: $('#settings-overlay'),
    settingsCloseBtn: $('#settings-close-btn'),
    settingsSaveBtn: $('#settings-save-btn'),
    settingsStatus: $('#settings-status'),
    // Stats panel
    statsToggleBtn: $('#stats-toggle-btn'),
    statsPanel: $('#stats-panel'),
    statsCloseBtn: $('#stats-panel-close-btn'),
    statRequests: $('#stat-requests'),
    statUptime: $('#stat-uptime'),
    statLlmCalls: $('#stat-llm-calls'),
    statLlmAvg: $('#stat-llm-avg'),
    statLlmLast: $('#stat-llm-last'),
    statPromptTps: $('#stat-prompt-tps'),
    statGenTps: $('#stat-gen-tps'),
    statTokens: $('#stat-tokens'),
    statSearches: $('#stat-searches'),
    statFetches: $('#stat-fetches'),
    statPassthrough: $('#stat-passthrough'),
    statHallucinated: $('#stat-hallucinated'),
    statCacheRate: $('#stat-cache-rate'),
    statCacheDetail: $('#stat-cache-detail'),
    statLiveContent: $('#stat-live-content'),
};

// ── Initialization ───────────────────────────────────────────────

async function init() {
    configureMarked();
    initReasoningToggle();
    initEventListeners();
    initSettingsModal();
    initStatsPanel();
    initMobileKeyboardHandler();
    initInstallBanner();
    await loadModels();
    updateSendButton();
    dom.messageInput.focus();
    startStatsPolling();
}

function configureMarked() {
    if (typeof marked === 'undefined') {
        // Fallback: load marked dynamically (shouldn't happen but be safe)
        return;
    }
    marked.setOptions({
        breaks: true,
        gfm: true,
    });
}

function initEventListeners() {
    // Send
    dom.sendBtn.addEventListener('click', () => {
        if (state.streaming) {
            cancelStreaming();
        } else {
            sendMessage();
        }
    });

    dom.messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (state.streaming) {
                cancelStreaming();
            } else {
                sendMessage();
            }
        }
    });

    // Auto-resize textarea
    dom.messageInput.addEventListener('input', () => {
        autoResizeTextarea();
        updateSendButton();
    });

    // Attach button
    dom.attachBtn.addEventListener('click', () => dom.fileInput.click());
    dom.fileInput.addEventListener('change', handleFilePick);

    // Image paste
    document.addEventListener('paste', handleImagePaste);

    // Remove image / file
    dom.removeImageBtn.addEventListener('click', removePendingImage);
    dom.removeFileBtn.addEventListener('click', removePendingFile);

    // Clear conversation
    dom.clearBtn.addEventListener('click', clearConversation);

    // Model selector
    dom.modelSelect.addEventListener('change', () => {
        state.selectedModel = dom.modelSelect.value;
    });
}

// ── Model Loading ────────────────────────────────────────────────

async function loadModels() {
    try {
        const resp = await fetch('/v1/models');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        // OpenAI returns { object: "list", data: [...] }
        // Some backends return an array directly
        const models = data.data || data || [];
        state.models = Array.isArray(models) ? models : [];

        dom.modelSelect.innerHTML = '';

        // Always add a "default" option that uses whatever model is
        // currently loaded in LM Studio — no need to pick a specific one.
        const defaultOption = document.createElement('option');
        defaultOption.value = 'local-model';
        defaultOption.textContent = '🔄 Use currently loaded model';
        dom.modelSelect.appendChild(defaultOption);

        if (state.models.length === 0) {
            // Still show the default option even if model list is empty
            state.selectedModel = 'local-model';
            dom.modelSelect.value = 'local-model';
            return;
        }

        state.models.forEach((model) => {
            const id = model.id || model.name || 'unknown';
            const option = document.createElement('option');
            option.value = id;
            option.textContent = id;
            dom.modelSelect.appendChild(option);
        });

        // Default to "use currently loaded model" rather than a specific one
        state.selectedModel = 'local-model';
        dom.modelSelect.value = 'local-model';
    } catch (error) {
        console.error('Failed to load models:', error);
        // Still provide the default option so the user can chat
        dom.modelSelect.innerHTML = '';
        const defaultOption = document.createElement('option');
        defaultOption.value = 'local-model';
        defaultOption.textContent = '🔄 Use currently loaded model';
        dom.modelSelect.appendChild(defaultOption);
        state.selectedModel = 'local-model';
        dom.modelSelect.value = 'local-model';
        showError('Could not load model list — using default. Is LM Studio running?');
    }
}

// ── Message Sending ──────────────────────────────────────────────

async function sendMessage() {
    const text = dom.messageInput.value.trim();
    const hasImage = state.pendingImage !== null;
    const hasFile = state.pendingFileContent !== null;

    if (!text && !hasImage && !hasFile) return;
    if (state.streaming) return;

    // Dismiss mobile keyboard
    dom.messageInput.blur();

    hideError();

    // Build user message content
    let userContent;

    if (hasImage && text) {
        userContent = [
            { type: 'text', text },
            { type: 'image_url', image_url: { url: state.pendingImage } },
        ];
    } else if (hasImage) {
        userContent = [
            { type: 'image_url', image_url: { url: state.pendingImage } },
        ];
    } else if (hasFile) {
        const ext = (state.pendingFileName || '').split('.').pop() || '';
        const fileBlock = `\n\n\`\`\`${ext}\n${state.pendingFileContent}\n\`\`\``;
        userContent = text ? text + fileBlock : `[File: ${state.pendingFileName}]${fileBlock}`;
    } else {
        userContent = text;
    }

    // Add user message to state and render
    const userMsg = { role: 'user', content: userContent };
    state.messages.push(userMsg);
    renderUserMessage(userMsg);
    scrollToBottom();

    // Clear input
    dom.messageInput.value = '';
    autoResizeTextarea();
    removePendingImage();
    removePendingFile();

    // Start streaming
    state.streaming = true;
    updateSendButton();
    showTypingIndicator();
    scrollToBottom();

    // Build request body
    const body = {
        model: state.selectedModel || 'local-model',
        messages: state.messages.map((m) => ({
            role: m.role,
            content: m.content,
        })),
        stream: true,
        reasoning: state.reasoningEnabled,
    };

    state.abortController = new AbortController();

    try {
        const resp = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: state.abortController.signal,
        });

        if (!resp.ok) {
            const errText = await resp.text().catch(() => 'Unknown error');
            let errMsg = `API error (${resp.status})`;
            try {
                const errJson = JSON.parse(errText);
                errMsg = errJson.error || errJson.detail || errMsg;
            } catch {}
            throw new Error(errMsg);
        }

        hideTypingIndicator();
        await streamResponse(resp);
    } catch (error) {
        hideTypingIndicator();
        if (error.name === 'AbortError') {
            // User cancelled — partial response is already rendered
            state.streaming = false;  // set before renderAllMessages (finally hasn't run yet)
            if (state.messages.length > 0) {
                const last = state.messages[state.messages.length - 1];
                if (last.role === 'assistant') {
                    last.content += '\n\n*[Stopped]*';
                    renderAllMessages();
                }
            }
        } else {
            showError(error.message);
            // Remove the user message if we never got a response
            // (keep it if we got a partial response)
        }
    } finally {
        state.streaming = false;
        state.abortController = null;
        updateSendButton();
        dom.messageInput.focus();
    }
}

function cancelStreaming() {
    if (state.abortController) {
        state.abortController.abort();
        state.abortController = null;
    }
}

// ── SSE Streaming ────────────────────────────────────────────────

async function streamResponse(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    // Reset live stats for this request
    state.lastLiveStats = null;
    updateLiveStats(null);

    // Create assistant message
    const assistantMsg = { role: 'assistant', content: '' };
    state.messages.push(assistantMsg);

    // Create DOM element for streaming
    const msgDiv = createMessageElement(assistantMsg, state.messages.length - 1);
    dom.messages.appendChild(msgDiv);
    const contentDiv = msgDiv.querySelector('.message-content');

    // Throttle: only re-render markdown at most every 50ms
    let lastRender = 0;
    let pendingContent = '';
    let lastEventType = '';

    function flushContent() {
        if (pendingContent !== assistantMsg.content) {
            pendingContent = assistantMsg.content;
            contentDiv.innerHTML = renderMarkdown(assistantMsg.content);
            scrollToBottom();
        }
    }

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                const trimmed = line.trim();

                // Track SSE event type
                if (trimmed.startsWith('event: ')) {
                    lastEventType = trimmed.slice(7);
                    continue;
                }

                if (!trimmed.startsWith('data: ')) continue;

                const data = trimmed.slice(6);
                if (data === '[DONE]') {
                    lastEventType = '';
                    continue;
                }

                // Handle stats events from the tool loop
                if (lastEventType === 'stats') {
                    lastEventType = '';
                    try {
                        const stats = JSON.parse(data);
                        state.lastLiveStats = stats;
                        updateLiveStats(stats);
                    } catch { /* skip */ }
                    continue;
                }

                lastEventType = '';

                try {
                    const parsed = JSON.parse(data);
                    const delta = parsed.choices?.[0]?.delta;
                    // Reasoning models emit reasoning_content instead of content.
                    // Capture whichever the model provides so text renders on-screen.
                    const text = delta?.content || delta?.reasoning_content || '';
                    if (text) {
                        assistantMsg.content += text;
                    }

                    // Throttle markdown rendering
                    const now = performance.now();
                    if (now - lastRender > 50) {
                        flushContent();
                        lastRender = now;
                    }
                } catch {
                    // Skip malformed JSON (edge case)
                }
            }
        }
    } catch (error) {
        if (error.name !== 'AbortError') {
            assistantMsg.content += '\n\n*[Stream interrupted]*';
        }
    } finally {
        // Final render
        flushContent();
        // Refresh stats after stream ends
        fetchStats();
        try { reader.releaseLock(); } catch {}
    }
}

// ── Rendering ────────────────────────────────────────────────────

function renderUserMessage(msg) {
    const msgDiv = createMessageElement(msg, state.messages.length - 1);
    dom.messages.appendChild(msgDiv);
}

function renderAllMessages() {
    dom.messages.innerHTML = '';
    state.messages.forEach((msg, i) => {
        if (i === state.messages.length - 1 && msg.role === 'assistant' && state.streaming) {
            return; // handled by streaming
        }
        dom.messages.appendChild(createMessageElement(msg, i));
    });
    scrollToBottom();
}

function createMessageElement(msg, index) {
    const div = document.createElement('div');
    div.className = `message ${msg.role}`;
    div.id = `msg-${index}`;

    const roleLabel = document.createElement('div');
    roleLabel.className = 'message-role';
    roleLabel.textContent = msg.role === 'user' ? 'You' : 'Assistant';
    div.appendChild(roleLabel);

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (msg.role === 'user' && Array.isArray(msg.content)) {
        // Vision format: array of content parts
        let html = '';
        for (const part of msg.content) {
            if (part.type === 'text') {
                html += escapeHtml(part.text);
            } else if (part.type === 'image_url') {
                html += `<img src="${escapeHtml(part.image_url.url)}" class="attached-image" alt="Attached image">`;
            }
        }
        contentDiv.innerHTML = html;
    } else if (typeof msg.content === 'string') {
        if (msg.role === 'user') {
            contentDiv.textContent = msg.content;
        } else {
            contentDiv.innerHTML = renderMarkdown(msg.content);
        }
    }

    div.appendChild(contentDiv);
    return div;
}

function renderMarkdown(text) {
    if (typeof marked === 'undefined') return escapeHtml(text);
    try {
        const html = marked.parse(text);
        return html;
    } catch {
        return escapeHtml(text);
    }
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ── Typing Indicator ─────────────────────────────────────────────

function showTypingIndicator() {
    hideTypingIndicator();
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.id = 'typing-indicator';
    div.innerHTML = `
        <div class="message-role">Assistant</div>
        <div class="message-content typing-indicator">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
        </div>
    `;
    dom.messages.appendChild(div);
    scrollToBottom();
}

function hideTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

// ── Image Handling ───────────────────────────────────────────────

async function handleImagePaste(e) {
    if (state.streaming) return;

    const items = e.clipboardData?.items;
    if (!items) return;

    for (const item of items) {
        if (item.type.startsWith('image/')) {
            e.preventDefault();
            const file = item.getAsFile();
            if (!file) continue;

            const dataUrl = await fileToDataUrl(file);
            if (dataUrl) {
                state.pendingImage = dataUrl;
                showImagePreview(dataUrl);
                updateSendButton();
            }
            break;
        }
    }
}

async function handleFilePick(e) {
    const file = e.target.files[0];
    if (!file) return;
    e.target.value = ''; // Reset so the same file can be picked again

    const imageTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp', 'image/svg+xml', 'image/bmp'];
    const imageExts = /\.(png|jpe?g|gif|webp|svg|bmp)$/i;

    if (imageTypes.includes(file.type) || imageExts.test(file.name)) {
        // Handle as image
        const dataUrl = await fileToDataUrl(file);
        if (dataUrl) {
            state.pendingImage = dataUrl;
            removePendingFile();
            showImagePreview(dataUrl);
        }
    } else {
        // Handle as text file
        if (file.size > 2 * 1024 * 1024) {
            showError('File is too large. Maximum size is 2 MB.');
            return;
        }
        try {
            const text = await file.text();
            state.pendingFileContent = text;
            state.pendingFileName = file.name;
            removePendingImage();
            showFilePreview(file.name, file.size);
        } catch {
            showError(`Could not read file: ${file.name}. Only text files and images are supported.`);
        }
    }
    updateSendButton();
}

function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
        if (file.size > 10 * 1024 * 1024) {
            reject(new Error('Image too large. Maximum size is 10 MB.'));
            return;
        }
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error('Failed to read file'));
        reader.readAsDataURL(file);
    });
}

function showImagePreview(dataUrl) {
    dom.imagePreviewImg.src = dataUrl;
    dom.imagePreview.classList.remove('hidden');
}

function removePendingImage() {
    state.pendingImage = null;
    dom.imagePreview.classList.add('hidden');
    dom.imagePreviewImg.src = '';
    updateSendButton();
}

// ── File Handling ────────────────────────────────────────────────

function showFilePreview(name, size) {
    dom.filePreviewName.textContent = name;
    dom.filePreviewSize.textContent = formatFileSize(size);
    dom.filePreview.classList.remove('hidden');
}

function removePendingFile() {
    state.pendingFileContent = null;
    state.pendingFileName = null;
    dom.filePreview.classList.add('hidden');
    updateSendButton();
}

function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// ── Reasoning Toggle ────────────────────────────────────────────

function initReasoningToggle() {
    // Load saved preference (default: true / enabled)
    const saved = localStorage.getItem('reasoningEnabled');
    if (saved !== null) {
        state.reasoningEnabled = saved !== 'false';
    }
    updateReasoningButton();

    dom.reasoningToggleBtn.addEventListener('click', () => {
        state.reasoningEnabled = !state.reasoningEnabled;
        localStorage.setItem('reasoningEnabled', state.reasoningEnabled);
        updateReasoningButton();
    });
}

function updateReasoningButton() {
    const btn = dom.reasoningToggleBtn;
    if (state.reasoningEnabled) {
        btn.className = 'active';
        btn.title = 'Reasoning: ON (click to disable chain of thought)';
    } else {
        btn.className = 'inactive';
        btn.title = 'Reasoning: OFF (click to enable chain of thought)';
    }
}

// ── Settings Modal ───────────────────────────────────────────────

function initSettingsModal() {
    // Open
    dom.settingsBtn.addEventListener('click', openSettings);
    // Close
    dom.settingsCloseBtn.addEventListener('click', closeSettings);
    dom.settingsOverlay.addEventListener('click', (e) => {
        if (e.target === dom.settingsOverlay) closeSettings();
    });
    // Save
    dom.settingsSaveBtn.addEventListener('click', saveSettings);

    // Provider toggle: show/hide SearXNG URL
    const providerSelect = $('#cfg-search-provider');
    const searxngGroup = $('#cfg-searxng-group');
    providerSelect.addEventListener('change', () => {
        searxngGroup.style.display =
            providerSelect.value === 'searxng' ? '' : 'none';
    });

    // Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !dom.settingsOverlay.classList.contains('hidden')) {
            closeSettings();
        }
    });
}

async function openSettings() {
    dom.settingsOverlay.classList.remove('hidden');
    dom.settingsStatus.textContent = 'Loading…';

    try {
        const resp = await fetch('/v1/config');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const cfg = data.config || {};

        // Populate form
        $('#cfg-lm-studio-url').value = cfg.lm_studio_url || '';
        $('#cfg-timeout').value = cfg.lm_studio_timeout || 120;
        $('#cfg-search-provider').value = cfg.search_provider || 'searxng';
        $('#cfg-searxng-url').value = cfg.searxng_url || '';
        $('#cfg-search-api-key').value = cfg.search_api_key || '';
        $('#cfg-max-iterations').value = cfg.max_tool_loop_iterations || 10;
        $('#cfg-max-client-tools').value = cfg.max_client_tools || 12;
        $('#cfg-max-results').value = cfg.max_search_results || 5;

        // Show/hide SearXNG URL based on provider
        $('#cfg-searxng-group').style.display =
            (cfg.search_provider || 'searxng') === 'searxng' ? '' : 'none';

        dom.settingsStatus.textContent = '';
    } catch (err) {
        dom.settingsStatus.textContent = `Failed to load: ${err.message}`;
        dom.settingsStatus.style.color = 'var(--danger)';
    }
}

function closeSettings() {
    dom.settingsOverlay.classList.add('hidden');
}

async function saveSettings() {
    dom.settingsStatus.textContent = 'Saving…';
    dom.settingsStatus.style.color = 'var(--accent)';

    const body = {
        lm_studio_url: $('#cfg-lm-studio-url').value.trim() || undefined,
        lm_studio_timeout: parseFloat($('#cfg-timeout').value) || undefined,
        search_provider: $('#cfg-search-provider').value || undefined,
        searxng_url: $('#cfg-searxng-url').value.trim() || undefined,
        search_api_key: $('#cfg-search-api-key').value || undefined,
        max_tool_loop_iterations: parseInt($('#cfg-max-iterations').value) || undefined,
        max_client_tools: parseInt($('#cfg-max-client-tools').value) || undefined,
        max_search_results: parseInt($('#cfg-max-results').value) || undefined,
    };

    // Remove undefined values
    Object.keys(body).forEach(k => { if (body[k] === undefined) delete body[k]; });

    try {
        const resp = await fetch('/v1/config', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const changed = data.changed || {};
        const keys = Object.keys(changed);
        if (keys.length === 0) {
            dom.settingsStatus.textContent = 'No changes.';
        } else {
            dom.settingsStatus.textContent = `Updated: ${keys.join(', ')}`;
            dom.settingsStatus.style.color = '#2e7d32';

            // If LM Studio URL changed, refresh models
            if ('lm_studio_url' in changed) {
                await loadModels();
            }
        }
    } catch (err) {
        dom.settingsStatus.textContent = `Save failed: ${err.message}`;
        dom.settingsStatus.style.color = 'var(--danger)';
    }
}

// ── Performance Stats Panel ─────────────────────────────────────

function initStatsPanel() {
    // Toggle sidebar
    dom.statsToggleBtn.addEventListener('click', toggleStatsPanel);
    dom.statsCloseBtn.addEventListener('click', () => {
        if (state.statsVisible) toggleStatsPanel();
    });
}

function toggleStatsPanel() {
    state.statsVisible = !state.statsVisible;
    if (state.statsVisible) {
        dom.statsPanel.classList.remove('collapsed');
        dom.statsToggleBtn.classList.add('active');
        fetchStats(); // immediate refresh
    } else {
        dom.statsPanel.classList.add('collapsed');
        dom.statsToggleBtn.classList.remove('active');
    }
}

function startStatsPolling() {
    // Poll /stats every 3 seconds
    state.statsPollInterval = setInterval(fetchStats, 3000);
}

async function fetchStats() {
    try {
        const resp = await fetch('/stats');
        if (!resp.ok) return;
        const data = await resp.json();

        dom.statRequests.textContent = data.total_requests || 0;
        dom.statUptime.textContent = formatUptime(data.uptime_seconds || 0);
        dom.statSearches.textContent = data.total_searches || 0;
        dom.statFetches.textContent = data.total_fetch_pages || 0;
        dom.statPassthrough.textContent = data.total_passthrough_tools || 0;
        dom.statHallucinated.textContent = data.total_hallucinated_tools || 0;

        // LLM stats
        dom.statLlmCalls.textContent = data.llm_call_count || 0;
        if (data.llm_avg_ms > 0) {
            dom.statLlmAvg.textContent = formatMs(data.llm_avg_ms);
        }

        // Token stats
        if (data.prompt_tokens_per_second > 0) {
            dom.statPromptTps.textContent = Math.round(data.prompt_tokens_per_second) + ' t/s';
        } else {
            dom.statPromptTps.textContent = '—';
        }
        if (data.tokens_per_second > 0) {
            dom.statGenTps.textContent = Math.round(data.tokens_per_second) + ' t/s';
        } else {
            dom.statGenTps.textContent = '—';
        }
        const pt = data.total_prompt_tokens || 0;
        const ct = data.total_completion_tokens || 0;
        if (pt > 0 || ct > 0) {
            dom.statTokens.textContent = `${pt.toLocaleString()} / ${ct.toLocaleString()}`;
        } else {
            dom.statTokens.textContent = '—';
        }

        // Cache
        if (data.cache_hit_rate !== undefined) {
            dom.statCacheRate.textContent = (data.cache_hit_rate * 100).toFixed(0) + '%';
        }
        dom.statCacheDetail.textContent = `${data.cache_hits || 0} / ${data.cache_misses || 0}`;
    } catch {
        // Silently ignore — server might be busy
    }
}

function updateLiveStats(stats) {
    if (!stats) {
        dom.statLiveContent.innerHTML = '<span class="stat-idle">Waiting for next request…</span>';
        return;
    }

    dom.statLlmLast.textContent = stats.last_llm_ms > 0 ? formatMs(stats.last_llm_ms) : '—';
    if (stats.tokens_per_second > 0) {
        dom.statGenTps.textContent = Math.round(stats.tokens_per_second) + ' t/s';
    }
    if (stats.prompt_tokens_per_second > 0) {
        dom.statPromptTps.textContent = Math.round(stats.prompt_tokens_per_second) + ' t/s';
    }
    if (stats.prompt_tokens > 0 || stats.completion_tokens > 0) {
        dom.statTokens.textContent = `${stats.prompt_tokens} / ${stats.completion_tokens}`;
    }

    let html = '';
    if (stats.llm_call_count > 0) {
        html += `<div class="stat-live-row"><span>LLM Calls</span><span>${stats.llm_call_count}</span></div>`;
        html += `<div class="stat-live-row"><span>Avg Latency</span><span>${formatMs(stats.llm_avg_ms)}</span></div>`;
    }
    if (stats.tokens_per_second > 0) {
        html += `<div class="stat-live-row"><span>Gen Speed</span><span>${Math.round(stats.tokens_per_second)} t/s</span></div>`;
    }
    if (stats.prompt_tokens_per_second > 0) {
        html += `<div class="stat-live-row"><span>Prompt Speed</span><span>${Math.round(stats.prompt_tokens_per_second)} t/s</span></div>`;
    }
    if (stats.web_search_count > 0) {
        html += `<div class="stat-live-row"><span>Searches</span><span>${stats.web_search_count}</span></div>`;
    }
    if (stats.fetch_page_count > 0) {
        html += `<div class="stat-live-row"><span>Pages Fetched</span><span>${stats.fetch_page_count}</span></div>`;
    }
    if (stats.hallucinated_tool_count > 0) {
        html += `<div class="stat-live-row"><span>Blocked</span><span>${stats.hallucinated_tool_count}</span></div>`;
    }
    if (stats.total_iterations > 0) {
        html += `<div class="stat-live-row"><span>Iterations</span><span>${stats.total_iterations}</span></div>`;
    }
    dom.statLiveContent.innerHTML = html || '<span class="stat-idle">Streaming…</span>';
}

function formatMs(ms) {
    if (ms < 1000) return Math.round(ms) + 'ms';
    return (ms / 1000).toFixed(1) + 's';
}

function formatUptime(seconds) {
    if (seconds < 60) return Math.round(seconds) + 's';
    if (seconds < 3600) return Math.round(seconds / 60) + 'm';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
}

// ── UI Helpers ───────────────────────────────────────────────────

function updateSendButton() {
    const hasContent = dom.messageInput.value.trim().length > 0
        || state.pendingImage !== null
        || state.pendingFileContent !== null;

    if (state.streaming) {
        dom.sendBtn.classList.add('streaming');
        dom.sendIcon.classList.add('hidden');
        dom.stopIcon.classList.remove('hidden');
        dom.sendBtn.disabled = false;
        dom.messageInput.disabled = true;
    } else {
        dom.sendBtn.classList.remove('streaming');
        dom.sendIcon.classList.remove('hidden');
        dom.stopIcon.classList.add('hidden');
        dom.sendBtn.disabled = !hasContent;
        dom.messageInput.disabled = false;
    }
}

function autoResizeTextarea() {
    const el = dom.messageInput;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 150) + 'px';
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        dom.chatContainer.scrollTop = dom.chatContainer.scrollHeight;
    });
}

/**
 * Handle mobile virtual keyboard and Safari bottom URL bar.
 *
 * On iOS Safari, the bottom URL bar overlays page content — it sits *on top*
 * of the layout viewport rather than shrinking it.  `safe-area-inset-bottom`
 * only covers the hardware safe area (notch / home indicator), not the bar.
 *
 * We use the Visual Viewport API to measure how much of the page is covered
 * from the bottom and pad the input area accordingly.
 */
function initMobileKeyboardHandler() {
    if (!window.visualViewport) return;

    const inputArea = document.getElementById('input-area');
    let lastHeight = window.visualViewport.height;

    function adjustForViewport() {
        const vh = window.visualViewport.height;
        const offsetTop = window.visualViewport.offsetTop;
        const layoutHeight = window.innerHeight;

        // How many px are covered at the bottom (Safari toolbar / keyboard accessory bar)
        const coveredBottom = layoutHeight - (vh + offsetTop);

        if (coveredBottom > 0) {
            // Pad the input area so it sits above whatever is covering the bottom
            inputArea.style.paddingBottom = (coveredBottom + 12) + 'px';
        } else {
            inputArea.style.paddingBottom = '';
        }

        // Keyboard opened: viewport shrunk by > 150px → scroll chat to bottom
        if (lastHeight - vh > 150) {
            scrollToBottom();
        }

        lastHeight = vh;
    }

    window.visualViewport.addEventListener('resize', adjustForViewport);
    window.visualViewport.addEventListener('scroll', adjustForViewport);

    // Run once on init so the correct padding is applied immediately
    adjustForViewport();
}

function showError(message) {
    dom.errorBanner.innerHTML = `
        <span>${escapeHtml(message)}</span>
        <button class="error-close" aria-label="Dismiss">&times;</button>
    `;
    dom.errorBanner.classList.remove('hidden');
    dom.errorBanner.querySelector('.error-close').addEventListener('click', hideError);

    // Auto-hide after 10 seconds
    clearTimeout(dom.errorBanner._timeout);
    dom.errorBanner._timeout = setTimeout(hideError, 10000);
}

function hideError() {
    dom.errorBanner.classList.add('hidden');
}

function clearConversation() {
    if (state.streaming) {
        cancelStreaming();
    }
    state.messages = [];
    dom.messages.innerHTML = `
        <div class="welcome">
            <p><strong>LLM Search Chat</strong></p>
            <p>Your local LLM with internet search. Try asking a question about current events!</p>
            <p class="welcome-hints">
                <span>🖼️ <strong>Paste</strong> an image from clipboard</span>
                <span>📎 <strong>Attach</strong> a text file or image</span>
                <span>⚡ <strong>Streaming</strong> responses</span>
            </p>
        </div>
    `;
    removePendingImage();
    removePendingFile();
    hideError();
    dom.messageInput.value = '';
    dom.messageInput.style.height = 'auto';
    updateSendButton();
    dom.messageInput.focus();
}

// ── PWA Install Banner ───────────────────────────────────────────

/**
 * Show a banner with instructions on how to add the site to the phone's
 * home screen.  Different instructions for iOS vs Android.
 *
 * Suppressed when:
 *  - Already running in standalone mode (installed)
 *  - On desktop (no touch support)
 *  - User dismissed it this session
 */
function initInstallBanner() {
    const banner = document.getElementById('install-banner');
    const instruction = document.getElementById('install-banner-instruction');
    const dismissBtn = document.getElementById('install-banner-dismiss');

    if (!banner || !instruction || !dismissBtn) return;

    // Don't show if already installed (standalone mode)
    if (window.matchMedia('(display-mode: standalone)').matches) return;

    // Only show on touch devices (phones / tablets)
    const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    if (!isTouchDevice) return;

    // Don't show if user dismissed it this session
    if (sessionStorage.getItem('install-banner-dismissed')) return;

    // Detect platform and set appropriate instructions
    const ua = navigator.userAgent || '';
    const isIOS = /iphone|ipad|ipod/i.test(ua);

    if (isIOS) {
        instruction.textContent =
            'Tap ↑ Share → "Add to Home Screen" to install this app.';
    } else {
        instruction.textContent =
            'Tap ⋮ → "Add to Home Screen" to install this app.';
    }

    banner.classList.remove('hidden');

    // Dismiss button
    dismissBtn.addEventListener('click', () => {
        banner.classList.add('hidden');
        sessionStorage.setItem('install-banner-dismissed', '1');
    });
}

// ── Startup ──────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', init);
