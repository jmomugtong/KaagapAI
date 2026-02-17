const API_BASE_URL = 'http://localhost:8000/api/v1';

// Current active tab
let activeTab = 'classical';

// ============================================
// Service Health Check
// ============================================
async function checkServiceHealth() {
    const statusBar = document.getElementById('status-bar');
    const statusText = document.getElementById('status-text');

    try {
        const res = await fetch('http://localhost:8000/ready');
        const data = await res.json();
        const checks = data.checks || {};
        const allOk = Object.values(checks).every(v => v === 'ok');

        if (allOk) {
            statusBar.className = 'status-bar online';
            statusText.textContent = 'All services operational';
        } else {
            statusBar.className = 'status-bar degraded';
            const degraded = Object.entries(checks)
                .filter(([, v]) => v !== 'ok')
                .map(([k]) => k);
            statusText.textContent = `Degraded: ${degraded.join(', ')}`;
        }
    } catch {
        statusBar.className = 'status-bar offline';
        statusText.textContent = 'API unreachable';
    }
}

checkServiceHealth();
setInterval(checkServiceHealth, 30000);

// ============================================
// Tab Switching
// ============================================
document.querySelectorAll('.tab-btn').forEach(button => {
    button.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        button.classList.add('active');
        activeTab = button.dataset.tab;
        document.getElementById(activeTab).classList.add('active');

        // Show/hide query form based on tab
        const formSection = document.getElementById('query-form-section');
        if (activeTab === 'upload') {
            formSection.classList.add('hidden');
        } else {
            formSection.classList.remove('hidden');
        }
    });
});

// ============================================
// Batch Upload State
// ============================================
let selectedFiles = []; // Array of File objects
let uploadMode = 'files'; // 'files' or 'folder'

// ============================================
// Upload Mode Toggle
// ============================================
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        uploadMode = btn.dataset.mode;

        const filesInput = document.getElementById('file-input-files');
        const folderInput = document.getElementById('file-input-folder');
        const label = document.getElementById('drop-zone-label');

        if (uploadMode === 'folder') {
            filesInput.classList.add('hidden');
            folderInput.classList.remove('hidden');
            label.textContent = 'Click to select a folder or drag files here';
        } else {
            folderInput.classList.add('hidden');
            filesInput.classList.remove('hidden');
            label.textContent = 'Click to select or drag PDF files here';
        }
        // Clear selection on mode switch
        clearFileSelection();
    });
});

// ============================================
// File Selection Handlers
// ============================================
function handleFileSelection(files) {
    const pdfFiles = Array.from(files).filter(f => f.name.toLowerCase().endsWith('.pdf'));
    if (pdfFiles.length === 0) return;

    // Append to current selection (dedup by name+size)
    for (const file of pdfFiles) {
        const exists = selectedFiles.some(f => f.name === file.name && f.size === file.size);
        if (!exists) selectedFiles.push(file);
    }
    renderFileList();
}

document.getElementById('file-input-files').addEventListener('change', (e) => {
    handleFileSelection(e.target.files);
    e.target.value = '';
});

document.getElementById('file-input-folder').addEventListener('change', (e) => {
    handleFileSelection(e.target.files);
    e.target.value = '';
});

// ============================================
// Drag-and-Drop
// ============================================
const dropZone = document.getElementById('file-drop-zone');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');

    const items = e.dataTransfer.items;
    if (items) {
        const files = [];
        const promises = [];
        for (const item of items) {
            const entry = item.webkitGetAsEntry && item.webkitGetAsEntry();
            if (entry) {
                promises.push(collectPdfFiles(entry, files));
            } else if (item.kind === 'file') {
                const file = item.getAsFile();
                if (file && file.name.toLowerCase().endsWith('.pdf')) files.push(file);
            }
        }
        await Promise.all(promises);
        handleFileSelection(files);
    } else if (e.dataTransfer.files.length) {
        handleFileSelection(e.dataTransfer.files);
    }
});

function collectPdfFiles(entry, results) {
    return new Promise((resolve) => {
        if (entry.isFile) {
            entry.file((file) => {
                if (file.name.toLowerCase().endsWith('.pdf')) results.push(file);
                resolve();
            }, () => resolve());
        } else if (entry.isDirectory) {
            const reader = entry.createReader();
            reader.readEntries(async (entries) => {
                await Promise.all(entries.map(e => collectPdfFiles(e, results)));
                resolve();
            }, () => resolve());
        } else {
            resolve();
        }
    });
}

// ============================================
// File List Rendering
// ============================================
function renderFileList() {
    const listEl = document.getElementById('file-list');
    const itemsEl = document.getElementById('file-list-items');
    const countEl = document.getElementById('file-list-count');

    if (selectedFiles.length === 0) {
        listEl.classList.add('hidden');
        return;
    }

    listEl.classList.remove('hidden');
    countEl.textContent = `${selectedFiles.length} file${selectedFiles.length !== 1 ? 's' : ''} selected`;
    itemsEl.innerHTML = selectedFiles.map((file, i) => `
        <div class="file-item">
            <span class="file-item-icon">&#128196;</span>
            <span class="file-item-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>
            <span class="file-item-size">${formatFileSize(file.size)}</span>
            <button type="button" class="file-item-remove" data-index="${i}" title="Remove">&times;</button>
        </div>
    `).join('');

    itemsEl.querySelectorAll('.file-item-remove').forEach(btn => {
        btn.addEventListener('click', () => {
            selectedFiles.splice(parseInt(btn.dataset.index), 1);
            renderFileList();
        });
    });
}

function clearFileSelection() {
    selectedFiles = [];
    document.getElementById('file-input-files').value = '';
    document.getElementById('file-input-folder').value = '';
    renderFileList();
}

document.getElementById('clear-files-btn').addEventListener('click', clearFileSelection);

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ============================================
// Query Form
// ============================================
document.getElementById('query-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('query-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnSpinner = btn.querySelector('.btn-spinner');

    // Show loading state
    btnText.textContent = 'Processing...';
    btnSpinner.classList.remove('hidden');
    btn.disabled = true;

    const formData = {
        question: document.getElementById('question').value,
        max_results: parseInt(document.getElementById('max_results').value),
        confidence_threshold: parseFloat(document.getElementById('confidence_threshold').value)
    };

    try {
        if (activeTab === 'compare') {
            await runCompare(formData);
        } else if (activeTab === 'agentic') {
            await runAgentic(formData);
        } else {
            await runClassical(formData);
        }
    } catch (error) {
        if (activeTab === 'compare') {
            document.getElementById('compare-results').classList.add('hidden');
        } else {
            const container = document.getElementById(`${activeTab}-results`);
            container.innerHTML = renderError(error.message);
            container.classList.remove('hidden');
        }
    } finally {
        btnText.textContent = 'Submit Query';
        btnSpinner.classList.add('hidden');
        btn.disabled = false;
    }
});

// ============================================
// Classical Pipeline
// ============================================
async function runClassical(formData) {
    const container = document.getElementById('classical-results');
    container.classList.add('hidden');

    const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);

    container.innerHTML = renderPipelineResult(data, 'classical');
    container.classList.remove('hidden');
}

// ============================================
// Agentic Pipeline
// ============================================
async function runAgentic(formData) {
    const container = document.getElementById('agentic-results');
    container.classList.add('hidden');

    const response = await fetch(`${API_BASE_URL}/agent/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);

    container.innerHTML = renderPipelineResult(data, 'agentic');
    container.classList.remove('hidden');
}

// ============================================
// Compare Mode
// ============================================
async function runCompare(formData) {
    const container = document.getElementById('compare-results');
    container.classList.add('hidden');

    const response = await fetch(`${API_BASE_URL}/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);

    // Render comparison summary
    const summary = document.getElementById('comparison-summary');
    summary.innerHTML = renderComparisonSummary(data.comparison, data.classical, data.agentic);

    // Render side-by-side panels
    document.getElementById('compare-classical').innerHTML =
        renderComparePanel(data.classical, 'classical');
    document.getElementById('compare-agentic').innerHTML =
        renderComparePanel(data.agentic, 'agentic');

    container.classList.remove('hidden');

    // Animate step timeline items
    animateSteps();
    animateCountUp();
}

// ============================================
// Render: Pipeline Result (single panel)
// ============================================
function renderPipelineResult(data, pipelineType) {
    const confidence = data.confidence || 0;
    const pct = Math.round(confidence * 100);
    const { level, levelClass, barColor } = getConfidenceInfo(confidence);
    const citations = data.citations || [];
    const chunks = data.retrieved_chunks || [];
    const steps = data.steps || [];
    const hasSteps = steps.length > 0;

    return `
        <div class="result-section">
            <div class="result-header">
                <h3>Answer</h3>
                <div class="meta-pills">
                    <span class="pill">${data.processing_time_ms || 0}ms</span>
                    <span class="pill">ID: ${data.query_id || '--'}</span>
                    <span class="pill pipeline-pill ${pipelineType}">${pipelineType}</span>
                </div>
            </div>
            <div class="answer-box">${escapeHtml(data.answer || 'No answer available.')}</div>
            ${data.hallucination_flagged ? '<div class="warning-box">Hallucination detected: some citations may reference sources not found in the retrieval set.</div>' : ''}
        </div>

        ${hasSteps ? renderStepTimeline(steps, pipelineType) : ''}

        <div class="result-section">
            <h3>Confidence</h3>
            <div class="confidence-bar-container">
                <div class="confidence-bar" style="width: ${Math.max(pct, 5)}%; background-color: ${barColor};">
                    <span class="confidence-label">${pct}%</span>
                </div>
            </div>
            <span class="confidence-level ${levelClass}">${level}</span>
        </div>

        <div class="result-section">
            <h3>Citations <span class="count-badge">${citations.length}</span></h3>
            <div class="citations-grid">
                ${citations.length === 0
                    ? '<p style="color: var(--text-muted); font-size: 0.85rem; margin: 0;">No structured citations available.</p>'
                    : citations.map((c, i) => renderCitation(c, i)).join('')}
            </div>
        </div>

        <div class="result-section">
            <h3>Retrieved Chunks <span class="count-badge">${chunks.length}</span></h3>
            <div class="chunks-list">
                ${chunks.map((chunk, i) => renderChunk(chunk, i)).join('')}
            </div>
        </div>

        <div class="result-section">
            <button class="btn-link" onclick="toggleRawJson(this)">Show Raw JSON</button>
            <div class="json-output hidden">${escapeHtml(JSON.stringify(data, null, 2))}</div>
        </div>
    `;
}

// ============================================
// Render: Compare Panel (compact)
// ============================================
function renderComparePanel(data, pipelineType) {
    const confidence = data.confidence || 0;
    const pct = Math.round(confidence * 100);
    const { level, levelClass, barColor } = getConfidenceInfo(confidence);
    const steps = data.steps || [];
    const chunks = data.retrieved_chunks || [];
    const citations = data.citations || [];
    const retrievalPasses = steps.filter(s => s.name === 'retrieve').length || 1;

    const label = pipelineType === 'classical' ? 'Classical' : 'Agentic';
    const timeStr = `${(data.processing_time_ms || 0).toFixed(0)}ms`;

    return `
        <div class="panel-header ${pipelineType}">
            <span class="panel-label">${label}</span>
            <span class="panel-time">${timeStr}</span>
        </div>

        ${steps.length > 0 ? renderStepTimeline(steps, pipelineType) : ''}

        <div class="panel-answer">
            <div class="answer-box compact">${escapeHtml(data.answer || 'No answer available.')}</div>
            ${data.hallucination_flagged ? '<div class="warning-box compact">Hallucination detected</div>' : ''}
        </div>

        <div class="panel-confidence">
            <div class="confidence-bar-container compact">
                <div class="confidence-bar" style="width: ${Math.max(pct, 5)}%; background-color: ${barColor};">
                    <span class="confidence-label">${pct}%</span>
                </div>
            </div>
            <span class="confidence-level ${levelClass}">${level}</span>
        </div>

        <div class="panel-stats">
            <div class="stat"><span class="stat-val">${retrievalPasses}</span><span class="stat-label">retrieval pass${retrievalPasses !== 1 ? 'es' : ''}</span></div>
            <div class="stat"><span class="stat-val">${chunks.length}</span><span class="stat-label">chunks</span></div>
            <div class="stat"><span class="stat-val">${citations.length}</span><span class="stat-label">citations</span></div>
        </div>
    `;
}

// ============================================
// Render: Comparison Summary Bar
// ============================================
function renderComparisonSummary(comp, classical, agentic) {
    const latencyWinner = comp.latency_ratio <= 1 ? 'agentic' : 'classical';
    const confWinner = comp.confidence_delta > 0 ? 'agentic' : comp.confidence_delta < 0 ? 'classical' : 'tie';

    const cTime = (classical.processing_time_ms || 0).toFixed(0);
    const aTime = (agentic.processing_time_ms || 0).toFixed(0);
    const confDelta = (comp.confidence_delta * 100).toFixed(0);
    const confSign = comp.confidence_delta >= 0 ? '+' : '';

    return `
        <div class="summary-grid">
            <div class="summary-item">
                <span class="summary-label">Latency</span>
                <span class="summary-value">${cTime}ms vs ${aTime}ms</span>
                <span class="summary-note ${latencyWinner === 'classical' ? 'winner-classical' : 'winner-agentic'}">${comp.latency_ratio}x</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Confidence</span>
                <span class="summary-value">${Math.round((classical.confidence || 0) * 100)}% vs ${Math.round((agentic.confidence || 0) * 100)}%</span>
                <span class="summary-note ${confWinner === 'agentic' ? 'winner-agentic' : confWinner === 'classical' ? 'winner-classical' : ''}">${confSign}${confDelta}%</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Retrievals</span>
                <span class="summary-value">${comp.classical_retrieval_passes} vs ${comp.agentic_retrieval_passes}</span>
                <span class="summary-note">passes</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Chunks</span>
                <span class="summary-value">${comp.classical_chunks_used} vs ${comp.agentic_chunks_used}</span>
                <span class="summary-note">used</span>
            </div>
        </div>
    `;
}

// ============================================
// Render: Step Timeline
// ============================================
function renderStepTimeline(steps, pipelineType) {
    const items = steps.map((step, i) => {
        const icon = step.name === 'complete' ? '&#10003;' : '&#10003;';
        const detail = Array.isArray(step.detail) ? step.detail.join(', ') : (step.detail || '');
        const duration = step.duration_ms ? `${step.duration_ms}ms` : '';

        return `
            <div class="step-item step-animate" style="animation-delay: ${i * 150}ms;">
                <div class="step-icon ${pipelineType}">${icon}</div>
                <div class="step-content">
                    <span class="step-name">${escapeHtml(step.name)}</span>
                    <span class="step-detail">${escapeHtml(detail)}</span>
                </div>
                ${duration ? `<span class="step-duration">${duration}</span>` : ''}
            </div>
        `;
    }).join('');

    return `
        <div class="result-section step-timeline">
            <h3>Pipeline Steps <span class="count-badge">${steps.length}</span></h3>
            <div class="step-list">${items}</div>
        </div>
    `;
}

// ============================================
// Render helpers
// ============================================
function renderCitation(c, i) {
    const doc = c.document || c.document_id || 'Unknown';
    const section = c.section ? `, ${c.section}` : '';
    const page = c.page ? `, p. ${c.page}` : '';
    const chunkIdx = c.chunk_index !== undefined ? `, Chunk ${c.chunk_index}` : '';
    const score = c.relevance_score !== undefined ? c.relevance_score.toFixed(2) : '';

    return `
        <div class="citation-card">
            <div class="citation-number">${i + 1}</div>
            <div class="citation-details">
                <div class="citation-doc">${escapeHtml(String(doc))}</div>
                <div class="citation-meta">${escapeHtml(section + page + chunkIdx)}</div>
            </div>
            ${score ? `<div class="citation-score">${score}</div>` : ''}
        </div>
    `;
}

function renderChunk(chunk, i) {
    const score = chunk.relevance_score || 0;
    const scoreClass = score >= 0.7 ? 'score-high' : score >= 0.4 ? 'score-medium' : 'score-low';
    const source = chunk.source || 'unknown';
    const expanded = i === 0 ? 'expanded' : '';

    return `
        <div class="chunk-card ${expanded}">
            <div class="chunk-header" onclick="this.parentElement.classList.toggle('expanded')">
                <span class="chunk-label">Chunk #${chunk.chunk_index !== undefined ? chunk.chunk_index : i} &middot; Doc ${chunk.document_id} &middot; ${escapeHtml(source)}</span>
                <div style="display:flex;align-items:center;gap:0.5rem;">
                    <span class="chunk-score ${scoreClass}">${score.toFixed(2)}</span>
                    <span class="chunk-toggle">&#9660;</span>
                </div>
            </div>
            <div class="chunk-body">${escapeHtml(chunk.text || '')}</div>
        </div>
    `;
}

function renderError(message) {
    return `
        <div class="result-section">
            <div class="result-header"><h3>Error</h3></div>
            <div class="answer-box" style="color: var(--red);">${escapeHtml(message)}</div>
        </div>
    `;
}

function getConfidenceInfo(confidence) {
    if (confidence >= 0.85) {
        return { level: 'High Confidence', levelClass: 'confidence-high', barColor: '#22c55e' };
    } else if (confidence >= 0.70) {
        return { level: 'Medium Confidence', levelClass: 'confidence-medium', barColor: '#f59e0b' };
    }
    return { level: 'Low Confidence', levelClass: 'confidence-low', barColor: '#ef4444' };
}

function toggleRawJson(btn) {
    const rawEl = btn.nextElementSibling;
    if (rawEl.classList.contains('hidden')) {
        rawEl.classList.remove('hidden');
        btn.textContent = 'Hide Raw JSON';
    } else {
        rawEl.classList.add('hidden');
        btn.textContent = 'Show Raw JSON';
    }
}

// ============================================
// Animations
// ============================================
function animateSteps() {
    document.querySelectorAll('.step-animate').forEach(el => {
        el.classList.remove('step-animate');
        void el.offsetWidth; // trigger reflow
        el.classList.add('step-animate');
    });
}

function animateCountUp() {
    document.querySelectorAll('.stat-val[data-target]').forEach(el => {
        const target = parseInt(el.dataset.target);
        let current = 0;
        const step = Math.ceil(target / 20);
        const interval = setInterval(() => {
            current += step;
            if (current >= target) {
                current = target;
                clearInterval(interval);
            }
            el.textContent = current;
        }, 30);
    });
}

// ============================================
// Upload Form — Batch Upload
// ============================================
document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    if (selectedFiles.length === 0) {
        alert('Please select at least one PDF file.');
        return;
    }

    const btn = document.getElementById('upload-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnSpinner = btn.querySelector('.btn-spinner');
    const progressEl = document.getElementById('batch-progress');
    const progressLabel = document.getElementById('batch-progress-label');
    const progressPct = document.getElementById('batch-progress-pct');
    const progressBar = document.getElementById('batch-progress-bar');
    const statusesEl = document.getElementById('batch-file-statuses');
    const resultsContainer = document.getElementById('upload-results');

    // Disable form
    btnText.textContent = 'Uploading...';
    btnSpinner.classList.remove('hidden');
    btn.disabled = true;
    resultsContainer.classList.add('hidden');

    // Show progress section with per-file statuses
    progressEl.classList.remove('hidden');
    progressBar.style.width = '0%';
    statusesEl.innerHTML = selectedFiles.map((file, i) => `
        <div class="file-item status-pending" id="batch-file-${i}">
            <span class="file-item-status" title="Pending">&#9679;</span>
            <span class="file-item-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>
            <span class="file-item-size">${formatFileSize(file.size)}</span>
        </div>
    `).join('');

    const documentType = document.getElementById('document_type').value;
    const department = document.getElementById('department').value || 'General';
    const total = selectedFiles.length;
    let completed = 0;
    let failed = 0;
    let totalChunks = 0;
    const startTime = Date.now();
    const CONCURRENCY = 3;

    // Upload a single file and update its row status
    async function uploadOne(i) {
        const file = selectedFiles[i];
        const rowEl = document.getElementById(`batch-file-${i}`);

        rowEl.className = 'file-item status-uploading';
        rowEl.querySelector('.file-item-status').innerHTML = '';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('document_type', documentType);
        formData.append('metadata', JSON.stringify({ department, timestamp: new Date().toISOString() }));

        try {
            const response = await fetch(`${API_BASE_URL}/upload`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);

            rowEl.className = 'file-item status-success';
            rowEl.querySelector('.file-item-status').innerHTML = '&#10003;';
            totalChunks += data.chunks_created || 0;
        } catch (err) {
            rowEl.className = 'file-item status-failed';
            rowEl.querySelector('.file-item-status').innerHTML = '&#10007;';
            rowEl.querySelector('.file-item-name').title = err.message;
            failed++;
        }

        completed++;
        const pct = Math.round((completed / total) * 100);
        progressBar.style.width = `${pct}%`;
        progressPct.textContent = `${pct}%`;
        progressLabel.textContent = `Uploading... ${completed} of ${total} done`;
    }

    // Process files in batches of CONCURRENCY
    for (let i = 0; i < total; i += CONCURRENCY) {
        const batch = [];
        for (let j = i; j < Math.min(i + CONCURRENCY, total); j++) {
            batch.push(uploadOne(j));
        }
        await Promise.all(batch);
    }

    const elapsed = Date.now() - startTime;

    // Show summary
    progressLabel.textContent = `Done — ${completed - failed} of ${total} uploaded`;
    document.getElementById('upload-file-count').textContent = `${completed - failed} of ${total}`;
    document.getElementById('upload-chunks').textContent = `${totalChunks} chunks`;
    document.getElementById('upload-time').textContent = `${(elapsed / 1000).toFixed(1)}s`;

    const failuresRow = document.getElementById('upload-failures-row');
    if (failed > 0) {
        failuresRow.classList.remove('hidden');
        document.getElementById('upload-failures').textContent = `${failed} file${failed !== 1 ? 's' : ''}`;
    } else {
        failuresRow.classList.add('hidden');
    }

    resultsContainer.classList.remove('hidden');

    // Reset
    btnText.textContent = 'Upload Documents';
    btnSpinner.classList.add('hidden');
    btn.disabled = false;
    clearFileSelection();
});

// ============================================
// Utility
// ============================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
