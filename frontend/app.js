const API_BASE_URL = 'http://localhost:8000/api/v1';

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
        document.getElementById(button.dataset.tab).classList.add('active');
    });
});

// ============================================
// File Input Display
// ============================================
document.getElementById('file').addEventListener('change', (e) => {
    const fileInput = e.target;
    const dropText = document.querySelector('.file-drop-text');
    const fileSelected = document.getElementById('file-selected');
    const fileNameDisplay = document.getElementById('file-name-display');

    if (fileInput.files.length > 0) {
        dropText.classList.add('hidden');
        fileSelected.classList.remove('hidden');
        fileNameDisplay.textContent = fileInput.files[0].name;
    } else {
        dropText.classList.remove('hidden');
        fileSelected.classList.add('hidden');
    }
});

// ============================================
// Query Form
// ============================================
document.getElementById('query-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('query-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnSpinner = btn.querySelector('.btn-spinner');
    const resultsContainer = document.getElementById('query-results');

    // Show loading state
    btnText.textContent = 'Processing...';
    btnSpinner.classList.remove('hidden');
    btn.disabled = true;
    resultsContainer.classList.add('hidden');

    const formData = {
        question: document.getElementById('question').value,
        max_results: parseInt(document.getElementById('max_results').value),
        confidence_threshold: parseFloat(document.getElementById('confidence_threshold').value)
    };

    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || `HTTP ${response.status}`);
        }

        renderQueryResults(data);
        resultsContainer.classList.remove('hidden');
    } catch (error) {
        renderQueryError(error.message);
        resultsContainer.classList.remove('hidden');
    } finally {
        btnText.textContent = 'Submit Query';
        btnSpinner.classList.add('hidden');
        btn.disabled = false;
    }
});

function renderQueryResults(data) {
    // Answer
    document.getElementById('answer-text').textContent = data.answer || 'No answer available.';

    // Meta pills
    document.getElementById('query-time-pill').textContent = `${data.processing_time_ms}ms`;
    document.getElementById('query-id-pill').textContent = `ID: ${data.query_id}`;

    // Hallucination warning
    const warningEl = document.getElementById('hallucination-warning');
    if (data.hallucination_flagged) {
        warningEl.classList.remove('hidden');
    } else {
        warningEl.classList.add('hidden');
    }

    // Confidence bar
    const confidence = data.confidence || 0;
    const pct = Math.round(confidence * 100);
    const bar = document.getElementById('confidence-bar');
    const label = document.getElementById('confidence-label');
    const levelEl = document.getElementById('confidence-level');

    bar.style.width = `${Math.max(pct, 5)}%`;
    label.textContent = `${pct}%`;

    let level, levelClass, barColor;
    if (confidence >= 0.85) {
        level = 'High Confidence';
        levelClass = 'confidence-high';
        barColor = '#22c55e';
    } else if (confidence >= 0.70) {
        level = 'Medium Confidence';
        levelClass = 'confidence-medium';
        barColor = '#f59e0b';
    } else {
        level = 'Low Confidence';
        levelClass = 'confidence-low';
        barColor = '#ef4444';
    }

    bar.style.backgroundColor = barColor;
    levelEl.textContent = level;
    levelEl.className = `confidence-level ${levelClass}`;

    // Citations
    const citationsList = document.getElementById('citations-list');
    const citationsCount = document.getElementById('citations-count');
    const citations = data.citations || [];
    citationsCount.textContent = citations.length;

    if (citations.length === 0) {
        citationsList.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem; margin: 0;">No structured citations available.</p>';
    } else {
        citationsList.innerHTML = citations.map((c, i) => {
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
        }).join('');
    }

    // Retrieved Chunks
    const chunksList = document.getElementById('chunks-list');
    const chunksCount = document.getElementById('chunks-count');
    const chunks = data.retrieved_chunks || [];
    chunksCount.textContent = chunks.length;

    chunksList.innerHTML = chunks.map((chunk, i) => {
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
    }).join('');

    // Raw JSON
    document.getElementById('raw-json').textContent = JSON.stringify(data, null, 2);
}

function renderQueryError(message) {
    document.getElementById('answer-text').textContent = `Error: ${message}`;
    document.getElementById('hallucination-warning').classList.add('hidden');
    document.getElementById('confidence-bar').style.width = '0%';
    document.getElementById('confidence-label').textContent = '0%';
    document.getElementById('confidence-level').textContent = '';
    document.getElementById('confidence-level').className = 'confidence-level';
    document.getElementById('citations-list').innerHTML = '';
    document.getElementById('citations-count').textContent = '0';
    document.getElementById('chunks-list').innerHTML = '';
    document.getElementById('chunks-count').textContent = '0';
    document.getElementById('query-time-pill').textContent = '--';
    document.getElementById('query-id-pill').textContent = 'Error';
    document.getElementById('raw-json').textContent = JSON.stringify({ error: message }, null, 2);
}

// ============================================
// Upload Form
// ============================================
document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('upload-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnSpinner = btn.querySelector('.btn-spinner');
    const resultsContainer = document.getElementById('upload-results');
    const successEl = document.getElementById('upload-success');
    const errorEl = document.getElementById('upload-error');

    const fileInput = document.getElementById('file');
    if (fileInput.files.length === 0) {
        alert('Please select a PDF file.');
        return;
    }

    // Show loading state
    btnText.textContent = 'Uploading...';
    btnSpinner.classList.remove('hidden');
    btn.disabled = true;
    resultsContainer.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('document_type', document.getElementById('document_type').value);

    const metadata = {
        department: document.getElementById('department').value || 'General',
        timestamp: new Date().toISOString()
    };
    formData.append('metadata', JSON.stringify(metadata));

    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || `HTTP ${response.status}`);
        }

        // Show success
        successEl.classList.remove('hidden');
        errorEl.classList.add('hidden');
        document.getElementById('upload-doc-id').textContent = data.document_id;
        document.getElementById('upload-filename').textContent = data.filename;
        document.getElementById('upload-chunks').textContent = `${data.chunks_created} chunks`;
        document.getElementById('upload-time').textContent = `${data.processing_time_ms}ms`;
        resultsContainer.classList.remove('hidden');

        // Reset file input
        fileInput.value = '';
        document.querySelector('.file-drop-text').classList.remove('hidden');
        document.getElementById('file-selected').classList.add('hidden');
    } catch (error) {
        // Show error
        successEl.classList.add('hidden');
        errorEl.classList.remove('hidden');
        document.getElementById('upload-error-msg').textContent = error.message;
        resultsContainer.classList.remove('hidden');
    } finally {
        btnText.textContent = 'Upload Document';
        btnSpinner.classList.add('hidden');
        btn.disabled = false;
    }
});

// ============================================
// Toggle Raw JSON
// ============================================
document.getElementById('toggle-raw').addEventListener('click', () => {
    const rawEl = document.getElementById('raw-json');
    const toggleBtn = document.getElementById('toggle-raw');
    if (rawEl.classList.contains('hidden')) {
        rawEl.classList.remove('hidden');
        toggleBtn.textContent = 'Hide Raw JSON';
    } else {
        rawEl.classList.add('hidden');
        toggleBtn.textContent = 'Show Raw JSON';
    }
});

// ============================================
// Utility
// ============================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
