/* KaagapAI Frontend — Tailwind CDN dark theme, v7 */

const API = 'http://localhost:8000';

// ============================================================
// PDF Viewer
// ============================================================

function openPdfViewer(source) {
  const url = `${API}/api/v1/documents/${encodeURIComponent(source)}`;
  const displayName = source.replace(/^PH_/, '').replace(/_/g, ' ').replace(/\.pdf$/i, '');
  document.getElementById('pdf-modal-title').textContent = displayName;
  document.getElementById('pdf-modal-frame').src = url;
  document.getElementById('pdf-modal').classList.remove('hidden');
}

function closePdfViewer() {
  document.getElementById('pdf-modal').classList.add('hidden');
  document.getElementById('pdf-modal-frame').src = '';
}

// Close on Escape key or clicking backdrop
document.addEventListener('keydown', e => { if (e.key === 'Escape') closePdfViewer(); });
document.getElementById('pdf-modal').addEventListener('click', e => {
  if (e.target.id === 'pdf-modal') closePdfViewer();
});

// ============================================================
// Utilities
// ============================================================

function confidencePill(conf) {
  const pct = Math.round((conf ?? 0) * 100);
  let cls = 'bg-red-900 text-red-300';
  if (pct >= 85) cls = 'bg-green-900 text-green-300';
  else if (pct >= 70) cls = 'bg-yellow-900 text-yellow-300';
  return `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${cls}">Confidence ${pct}%</span>`;
}

function citationBadges(citations) {
  if (!citations || citations.length === 0) return '<span class="text-gray-600 text-xs">No citations</span>';
  return citations.map(c => {
    const label = [c.document, c.section && `§${c.section}`, c.page && `p.${c.page}`].filter(Boolean).join(', ');
    return `<span class="inline-block bg-gray-800 border border-gray-700 text-blue-300 text-xs rounded px-2 py-0.5 mr-1 mb-1">[${label}]</span>`;
  }).join('');
}

function processingTime(ms) {
  if (ms == null) return '';
  return `<span class="text-xs text-gray-500">${Math.round(ms)} ms</span>`;
}

function setLoading(btnId, spinnerId, textId, loading, label) {
  const btn = document.getElementById(btnId);
  const spinner = document.getElementById(spinnerId);
  const text = document.getElementById(textId);
  btn.disabled = loading;
  spinner.classList.toggle('hidden', !loading);
  if (label) text.textContent = loading ? 'Running…' : label;
}

function resultCard(data) {
  const chunks = data.retrieved_chunks || [];
  const chunkList = chunks.length
    ? chunks.map(c => {
        const text = c.text || c.content || '';
        const score = c.relevance_score != null ? `<span class="text-gray-500 text-xs">${(c.relevance_score * 100).toFixed(0)}%</span>` : '';
        const source = c.source || 'Unknown';
        const displayName = source.replace(/^PH_/, '').replace(/_/g, ' ').replace(/\.pdf$/i, '');
        return `<li class="text-xs bg-gray-800 rounded p-2 space-y-1">
          <div class="flex items-center justify-between">
            <button onclick="openPdfViewer('${source.replace(/'/g, "\\'")}')" class="text-blue-400 font-medium truncate hover:underline text-left cursor-pointer" title="View ${source}">${displayName}</button>
            ${score}
          </div>
          <p class="text-gray-400">${text.substring(0, 200)}…</p>
        </li>`;
      }).join('')
    : '<li class="text-xs text-gray-600">No chunks</li>';

  const hallTag = data.hallucination_flagged
    ? '<span class="inline-block bg-red-900 text-red-300 text-xs rounded px-2 py-0.5 ml-2">⚠ Hallucination flagged</span>' : '';
  const cachedTag = data.cached
    ? '<span class="inline-block bg-gray-800 text-gray-400 text-xs rounded px-2 py-0.5 ml-2">Cached</span>' : '';

  return `
    <div class="flex items-start gap-2 flex-wrap">
      ${confidencePill(data.confidence)}
      ${processingTime(data.processing_time_ms)}
      ${hallTag}${cachedTag}
    </div>
    <div class="text-sm text-gray-100 leading-relaxed whitespace-pre-line">${(data.answer || '<em class="text-gray-500">No answer</em>').replace(/\*\*(.+?)\*\*/g, '<strong class="text-yellow-300">$1</strong>')}</div>
    ${data.citations && data.citations.length ? `<div><span class="text-xs text-gray-500 block mb-1">Citations</span>${citationBadges(data.citations)}</div>` : ''}
    <details class="group">
      <summary class="text-xs text-gray-500 cursor-pointer hover:text-gray-300 select-none">
        Retrieved chunks (${chunks.length})
      </summary>
      <ul class="mt-2 space-y-1 list-none">${chunkList}</ul>
    </details>
  `;
}

// ============================================================
// Tab routing
// ============================================================

const TABS = ['query', 'agentic', 'compare', 'upload', 'monitor'];
let activeTab = 'query';

function activateTab(name) {
  activeTab = name;
  TABS.forEach(t => {
    document.getElementById(`tab-${t}`).classList.toggle('hidden', t !== name);
  });
  document.querySelectorAll('.tab-btn').forEach(btn => {
    const active = btn.dataset.tab === name;
    btn.classList.toggle('border-blue-500', active);
    btn.classList.toggle('text-blue-400', active);
    btn.classList.toggle('border-transparent', !active);
    btn.classList.toggle('text-gray-400', !active);
    btn.classList.toggle('hover:text-gray-200', !active);
  });
  if (name === 'monitor') refreshMonitor();
}

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.classList.add('border-transparent', 'text-gray-400', 'hover:text-gray-200');
  btn.addEventListener('click', () => activateTab(btn.dataset.tab));
});
activateTab('query');

// ============================================================
// Status badge
// ============================================================

async function checkStatus() {
  const dot = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  try {
    const res = await fetch(`${API}/ready`);
    const data = await res.json();
    const checks = data.checks || {};
    const allOk = Object.values(checks).every(v => v === 'ok');
    if (allOk) {
      dot.className = 'w-2 h-2 rounded-full bg-green-500';
      text.textContent = 'All services online';
      text.className = 'text-green-400 text-sm';
    } else {
      const bad = Object.entries(checks).filter(([, v]) => v !== 'ok').map(([k]) => k);
      dot.className = 'w-2 h-2 rounded-full bg-yellow-500';
      text.textContent = `Degraded: ${bad.join(', ')}`;
      text.className = 'text-yellow-400 text-sm';
    }
  } catch {
    dot.className = 'w-2 h-2 rounded-full bg-red-500';
    text.textContent = 'API unreachable';
    text.className = 'text-red-400 text-sm';
  }
}

checkStatus();
setInterval(checkStatus, 30000);

// ============================================================
// Sample question chips
// ============================================================

document.querySelectorAll('.sample-q').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = document.getElementById(btn.dataset.target);
    if (target) {
      target.value = btn.textContent.trim();
      target.focus();
    }
  });
});

// ============================================================
// Query tab (Classical RAG)
// ============================================================

document.getElementById('query-form').addEventListener('submit', async e => {
  e.preventDefault();
  const question = document.getElementById('q-question').value.trim();
  if (!question) return;
  setLoading('q-btn', 'q-spinner', 'q-btn-text', true, 'Submit Query');

  const result = document.getElementById('query-result');
  const placeholder = document.getElementById('query-placeholder');
  result.classList.add('hidden');
  placeholder.classList.add('hidden');

  try {
    const res = await fetch(`${API}/api/v1/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        max_results: parseInt(document.getElementById('q-max').value),
        confidence_threshold: parseFloat(document.getElementById('q-conf').value),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    result.innerHTML = resultCard(data);
    result.classList.remove('hidden');
  } catch (err) {
    result.innerHTML = `<div class="text-red-400 text-sm">${err.message}</div>`;
    result.classList.remove('hidden');
  } finally {
    setLoading('q-btn', 'q-spinner', 'q-btn-text', false, 'Submit Query');
  }
});

// ============================================================
// Agentic tab
// ============================================================

function stepsTimeline(steps) {
  if (!steps || steps.length === 0) return '<p class="text-gray-600 text-xs">No steps recorded.</p>';
  return steps.map((s, i) => {
    const icon = {
      pii_redact_input: '🔒', cache_check: '💾', cache_hit: '⚡',
      classify: '🏷', decompose: '✂', embed: '🔢', retrieve: '🔍',
      multi_query: '🔀', rerank: '📊', synthesize: '✍', reflect: '🤔',
      retry: '🔄', retry_retrieve: '🔄', deduplicate: '🧹',
      web_search_fallback: '🌐', direct_answer: '💡', complete: '✅',
    }[s.name] || '⚙';
    return `
      <div class="timeline-item relative pl-8 pb-3" style="position:relative">
        <div class="absolute left-0 top-1 w-6 h-6 rounded-full bg-gray-800 border border-gray-700 flex items-center justify-center text-xs">${icon}</div>
        <div class="flex items-baseline gap-2">
          <span class="text-xs font-medium text-gray-300">${s.name}</span>
          <span class="text-xs text-gray-600">${s.duration_ms != null ? s.duration_ms + ' ms' : ''}</span>
        </div>
        ${s.detail ? `<p class="text-xs text-gray-500 mt-0.5">${s.detail}</p>` : ''}
      </div>`;
  }).join('');
}

document.getElementById('agentic-form').addEventListener('submit', async e => {
  e.preventDefault();
  const question = document.getElementById('a-question').value.trim();
  if (!question) return;
  setLoading('a-btn', 'a-spinner', 'a-btn-text', true, 'Submit Agentic Query');

  const result = document.getElementById('agentic-result');
  const stepsPanel = document.getElementById('agentic-steps-panel');
  const stepsEl = document.getElementById('agentic-steps');
  const placeholder = document.getElementById('agentic-placeholder');
  result.classList.add('hidden');
  stepsPanel.classList.add('hidden');
  placeholder.classList.add('hidden');

  try {
    const res = await fetch(`${API}/api/v1/agent/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        max_results: parseInt(document.getElementById('a-max').value),
        confidence_threshold: parseFloat(document.getElementById('a-conf').value),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    result.innerHTML = resultCard(data);
    result.classList.remove('hidden');
    if (data.steps && data.steps.length > 0) {
      stepsEl.innerHTML = stepsTimeline(data.steps);
      stepsPanel.classList.remove('hidden');
    }
  } catch (err) {
    result.innerHTML = `<div class="text-red-400 text-sm">${err.message}</div>`;
    result.classList.remove('hidden');
  } finally {
    setLoading('a-btn', 'a-spinner', 'a-btn-text', false, 'Submit Agentic Query');
  }
});

// ============================================================
// Compare tab
// ============================================================

function metricTile(label, value, note) {
  return `
    <div class="bg-gray-800 rounded-lg p-3">
      <div class="text-xs text-gray-500 mb-1">${label}</div>
      <div class="text-lg font-semibold text-white">${value}</div>
      ${note ? `<div class="text-xs text-gray-600">${note}</div>` : ''}
    </div>`;
}

function pipelinePanel(data, accent) {
  return `
    <div class="flex items-center gap-2 flex-wrap mt-2">
      ${confidencePill(data.confidence)}
      ${processingTime(data.processing_time_ms)}
    </div>
    <div class="text-sm text-gray-100 leading-relaxed mt-2">${data.answer || '<em class="text-gray-500">No answer</em>'}</div>
    ${data.citations && data.citations.length ? `<div class="mt-2">${citationBadges(data.citations)}</div>` : ''}
    <div class="text-xs text-gray-600 mt-2">${(data.retrieved_chunks || []).length} chunks used</div>
  `;
}

document.getElementById('compare-form').addEventListener('submit', async e => {
  e.preventDefault();
  const question = document.getElementById('c-question').value.trim();
  if (!question) return;
  setLoading('c-btn', 'c-spinner', 'c-btn-text', true, 'Compare');

  const metrics = document.getElementById('compare-metrics');
  const panels = document.getElementById('compare-panels');
  const placeholder = document.getElementById('compare-placeholder');
  metrics.classList.add('hidden');
  panels.classList.add('hidden');
  placeholder.classList.add('hidden');

  try {
    const res = await fetch(`${API}/api/v1/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, max_results: 5 }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);

    const comp = data.comparison || {};
    metrics.innerHTML =
      metricTile('Latency Ratio', `${comp.latency_ratio ?? '—'}×`, 'agentic / classical') +
      metricTile('Confidence Δ', (comp.confidence_delta != null ? (comp.confidence_delta >= 0 ? '+' : '') + comp.confidence_delta.toFixed(3) : '—'), 'agentic − classical') +
      metricTile('Classical Chunks', comp.classical_chunks_used ?? '—', '') +
      metricTile('Agentic Chunks', comp.agentic_chunks_used ?? '—', '');
    metrics.classList.remove('hidden');

    document.getElementById('compare-classical').innerHTML =
      `<div class="flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-blue-500"></span><span class="text-sm font-semibold text-blue-400">Classical RAG</span></div>` +
      pipelinePanel(data.classical);
    document.getElementById('compare-agentic').innerHTML =
      `<div class="flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-purple-500"></span><span class="text-sm font-semibold text-purple-400">Agentic RAG</span></div>` +
      pipelinePanel(data.agentic);
    panels.classList.remove('hidden');
  } catch (err) {
    placeholder.textContent = `Error: ${err.message}`;
    placeholder.classList.remove('hidden');
  } finally {
    setLoading('c-btn', 'c-spinner', 'c-btn-text', false, 'Compare');
  }
});

// ============================================================
// Upload tab
// ============================================================

let selectedFiles = [];

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const dropLabel = document.getElementById('drop-label');
const fileListEl = document.getElementById('file-list');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('border-gray-400'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('border-gray-400'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('border-gray-400');
  addFiles([...e.dataTransfer.files]);
});
fileInput.addEventListener('change', () => addFiles([...fileInput.files]));

function addFiles(files) {
  const pdfs = files.filter(f => f.name.endsWith('.pdf'));
  selectedFiles = [...selectedFiles, ...pdfs.filter(f => !selectedFiles.find(s => s.name === f.name))];
  renderFileList();
}

function renderFileList() {
  if (selectedFiles.length === 0) {
    dropLabel.textContent = 'Click or drag PDF files here';
    fileListEl.classList.add('hidden');
    fileListEl.innerHTML = '';
    return;
  }
  dropLabel.textContent = `${selectedFiles.length} file(s) selected`;
  fileListEl.innerHTML = selectedFiles.map((f, i) =>
    `<div class="flex items-center justify-between bg-gray-800 rounded px-3 py-1.5 text-xs" id="file-item-${i}">
      <span class="text-gray-300 truncate max-w-xs">${f.name}</span>
      <button type="button" onclick="removeFile(${i})" class="text-gray-600 hover:text-red-400 ml-2">✕</button>
    </div>`
  ).join('');
  fileListEl.classList.remove('hidden');
}

window.removeFile = function(i) {
  selectedFiles.splice(i, 1);
  renderFileList();
};

async function uploadSingle(file, docType, meta) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('document_type', docType);
  fd.append('metadata', meta);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300000); // 5 min
  try {
    const res = await fetch(`${API}/api/v1/upload`, { method: 'POST', body: fd, signal: controller.signal });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    return res.json();
  } finally {
    clearTimeout(timeout);
  }
}

document.getElementById('upload-form').addEventListener('submit', async e => {
  e.preventDefault();
  if (selectedFiles.length === 0) { alert('Select at least one PDF.'); return; }

  const docType = document.getElementById('upload-type').value;
  const meta = document.getElementById('upload-meta').value || '{}';
  const progressEl = document.getElementById('upload-progress');
  const progressBar = document.getElementById('progress-bar');
  const progressLabel = document.getElementById('progress-label');
  const progressPct = document.getElementById('progress-pct');
  const fileStatuses = document.getElementById('upload-file-statuses');
  const summary = document.getElementById('upload-summary');

  setLoading('upload-btn', 'upload-spinner', 'upload-btn-text', true, 'Upload');
  progressBar.style.width = '0%';
  progressPct.textContent = '0%';
  progressLabel.textContent = `Uploading 0 of ${selectedFiles.length}…`;
  progressEl.classList.remove('hidden');
  summary.classList.add('hidden');
  fileStatuses.innerHTML = selectedFiles.map(f =>
    `<div id="fstatus-${f.name.replace(/\W/g,'_')}" class="flex items-center gap-2 text-xs text-gray-400">
      <span class="w-3 h-3 rounded-full bg-gray-700 flex-shrink-0"></span> ${f.name}
    </div>`
  ).join('');

  const CONCURRENCY = 3;
  let done = 0, succeeded = 0, totalChunks = 0;
  const n = selectedFiles.length;

  const queue = [...selectedFiles];
  const results = [];

  async function worker() {
    while (queue.length > 0) {
      const file = queue.shift();
      const key = file.name.replace(/\W/g,'_');
      const el = document.getElementById(`fstatus-${key}`);
      if (el) el.querySelector('span').className = 'w-3 h-3 rounded-full bg-yellow-500 flex-shrink-0 animate-pulse';
      try {
        const r = await uploadSingle(file, docType, meta);
        results.push({ file: file.name, ok: true, chunks: r.chunks_created });
        if (el) {
          el.querySelector('span').className = 'w-3 h-3 rounded-full bg-green-500 flex-shrink-0';
          el.innerHTML += ` <span class="text-green-600 ml-auto">${r.chunks_created} chunks</span>`;
        }
        succeeded++;
        totalChunks += r.chunks_created || 0;
      } catch (err) {
        results.push({ file: file.name, ok: false, error: err.message });
        if (el) {
          el.querySelector('span').className = 'w-3 h-3 rounded-full bg-red-500 flex-shrink-0';
          el.innerHTML += ` <span class="text-red-500 ml-auto truncate">${err.message}</span>`;
        }
      }
      done++;
      const pct = Math.round((done / n) * 100);
      progressBar.style.width = `${pct}%`;
      progressPct.textContent = `${pct}%`;
      progressLabel.textContent = `Uploading ${done} of ${n}…`;
    }
  }

  const workers = Array.from({ length: Math.min(CONCURRENCY, n) }, worker);
  await Promise.all(workers);

  setLoading('upload-btn', 'upload-spinner', 'upload-btn-text', false, 'Upload');
  progressBar.style.width = '100%';
  progressLabel.textContent = 'Complete';

  const failed = n - succeeded;
  summary.innerHTML = `
    <div class="flex items-center gap-2 mb-2"><span class="text-${failed ? 'yellow' : 'green'}-400 font-medium">${failed ? '⚠' : '✓'} Upload ${failed ? 'partial' : 'complete'}</span></div>
    <div class="text-gray-300">Files: <strong>${succeeded}</strong> succeeded${failed ? `, <span class="text-red-400">${failed} failed</span>` : ''}</div>
    <div class="text-gray-300">Total chunks: <strong>${totalChunks}</strong></div>
  `;
  summary.classList.remove('hidden');
  selectedFiles = [];
  renderFileList();
});

// ============================================================
// Monitor tab
// ============================================================

const SERVICE_LABELS = {
  database: 'Database',
  redis: 'Redis',
  ollama: 'Ollama LLM',
  embedding_model: 'Embeddings',
};

async function refreshMonitor() {
  const grid = document.getElementById('service-grid');
  const updated = document.getElementById('monitor-updated');
  grid.innerHTML = '<div class="col-span-4 text-gray-600 text-sm text-center py-4">Loading…</div>';
  try {
    const res = await fetch(`${API}/ready`);
    const data = await res.json();
    const checks = data.checks || {};
    grid.innerHTML = Object.entries(checks).map(([key, val]) => {
      const label = SERVICE_LABELS[key] || key;
      let dotCls = 'bg-gray-600';
      let valCls = 'text-gray-400';
      let valLabel = val;
      if (val === 'ok') { dotCls = 'bg-green-500'; valCls = 'text-green-400'; valLabel = 'Online'; }
      else if (val === 'degraded') { dotCls = 'bg-yellow-500'; valCls = 'text-yellow-400'; valLabel = 'Degraded'; }
      else if (val === 'error') { dotCls = 'bg-red-500'; valCls = 'text-red-400'; valLabel = 'Error'; }
      else if (val === 'unavailable') { dotCls = 'bg-red-700'; valCls = 'text-red-500'; valLabel = 'Unavailable'; }
      return `
        <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
          <div class="w-3 h-3 rounded-full ${dotCls} mx-auto mb-2"></div>
          <div class="text-xs text-gray-400">${label}</div>
          <div class="text-sm font-medium ${valCls} mt-1">${valLabel}</div>
        </div>`;
    }).join('') || '<div class="col-span-4 text-gray-600 text-sm text-center">No checks returned.</div>';
    updated.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
  } catch {
    grid.innerHTML = '<div class="col-span-4 text-red-500 text-sm text-center py-4">Failed to reach API at ' + API + '</div>';
  }
}

document.getElementById('refresh-btn').addEventListener('click', refreshMonitor);
