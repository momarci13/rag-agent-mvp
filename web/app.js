// ROG-Agent — Chat UI + comprehensive report output

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  currentSection: 'dashboard',
  theme: localStorage.getItem('theme') || 'light',
  sidebarCollapsed: false,
  systemHealth: null,
  // Chat state
  currentTaskId: null,
  isThinking: false,
};

// ── Utility ───────────────────────────────────────────────────────────────────
function $(id) { return document.getElementById(id); }

async function fetchJson(url, options = {}) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
  }
  return resp.json();
}

function setTheme(theme) {
  document.documentElement.className = theme;
  state.theme = theme;
  localStorage.setItem('theme', theme);
  $('themeToggle').textContent = theme === 'dark' ? '☀️' : '🌙';
  const prismTheme = $('prism-theme');
  if (prismTheme) {
    prismTheme.href = theme === 'dark'
      ? 'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-dark.min.css'
      : 'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css';
  }
}

function toggleSidebar() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  $('sidebar').classList.toggle('collapsed', state.sidebarCollapsed);
  document.querySelector('.main-content').classList.toggle('sidebar-collapsed', state.sidebarCollapsed);
  localStorage.setItem('sidebarCollapsed', state.sidebarCollapsed);
}

function showSection(sectionId) {
  document.querySelectorAll('.nav-item').forEach(item =>
    item.classList.toggle('active', item.dataset.section === sectionId)
  );
  document.querySelectorAll('.section').forEach(sec =>
    sec.classList.toggle('active', sec.id === `${sectionId}-section`)
  );
  const titles = { dashboard: 'Dashboard', tasks: 'Chat', reports: 'Report Viewer', data: 'Data Management', settings: 'Settings' };
  $('pageTitle').textContent = titles[sectionId] || 'Finance Assistant.ai';
  state.currentSection = sectionId;
  localStorage.setItem('currentSection', sectionId);
}

function addActivityItem(icon, title) {
  const feed = $('activityFeed');
  if (!feed) return;
  const item = document.createElement('div');
  item.className = 'activity-item';
  item.innerHTML = `<div class="activity-icon">${icon}</div><div class="activity-content"><div class="activity-title">${title}</div><div class="activity-time">Just now</div></div>`;
  feed.insertBefore(item, feed.firstChild);
  while (feed.children.length > 10) feed.removeChild(feed.lastChild);
}

function updateSystemStatus(health) {
  state.systemHealth = health;
  const ind = $('statusIndicator');
  const sh = $('systemHealth');
  if (health?.status === 'ok') {
    ind && ind.classList.add('active');
    if (sh) { sh.textContent = 'Healthy'; sh.style.color = 'var(--text-accent)'; }
  } else {
    ind && ind.classList.remove('active');
    if (sh) { sh.textContent = health?.status || 'Unknown'; sh.style.color = '#ef4444'; }
  }
}

// ── Chat rendering ─────────────────────────────────────────────────────────────

function setInput(text) {
  const inp = $('chatInput');
  if (inp) { inp.value = text; inp.focus(); }
}

function scrollToBottom() {
  const thread = $('chatThread');
  if (thread) thread.scrollTop = thread.scrollHeight;
}

function hideWelcome() {
  const w = $('chatWelcome');
  if (w) w.style.display = 'none';
}

function appendUserBubble(text) {
  hideWelcome();
  const thread = $('chatThread');
  const div = document.createElement('div');
  div.className = 'chat-message user-message';
  div.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
  thread.appendChild(div);
  scrollToBottom();
}

function appendThinkingIndicator() {
  const thread = $('chatThread');
  const div = document.createElement('div');
  div.className = 'chat-message agent-message';
  div.id = 'thinkingIndicator';
  div.innerHTML = `<div class="agent-bubble thinking"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>`;
  thread.appendChild(div);
  scrollToBottom();
  return div;
}

function removeThinkingIndicator() {
  const ind = $('thinkingIndicator');
  if (ind) ind.remove();
}

function appendAgentCard(data) {
  removeThinkingIndicator();
  const thread = $('chatThread');
  const narrative = data.narrative || {};
  const code = data.code || '';
  const pdfUrl = data.pdf_url || null;
  const taskId = data.task_id || '';

  const keyResultsHtml = Array.isArray(narrative.key_results) && narrative.key_results.length
    ? `<ul class="key-results">${narrative.key_results.map(r => `<li>${escapeHtml(r)}</li>`).join('')}</ul>`
    : '';

  const limitationsHtml = narrative.limitations
    ? `<div class="narrative-section"><strong>Limitations</strong><p>${escapeHtml(narrative.limitations)}</p></div>`
    : '';

  const codeHtml = code
    ? `<details class="code-details">
        <summary>Show Generated Code</summary>
        <pre><code class="language-python">${escapeHtml(code.slice(0, 6000))}</code></pre>
      </details>`
    : '';

  const pdfHtml = pdfUrl
    ? `<a class="pdf-btn" href="${pdfUrl}" target="_blank" download>Download PDF Report</a>`
    : '';

  const noNarrative = !narrative.objective;

  const div = document.createElement('div');
  div.className = 'chat-message agent-message';
  div.dataset.taskId = taskId;

  div.innerHTML = `
    <div class="agent-card">
      ${noNarrative ? '<div class="narrative-section"><p>Task completed. Results saved.</p></div>' : `
      <div class="narrative-section">
        <strong>Objective</strong>
        <p>${escapeHtml(narrative.objective || '')}</p>
      </div>
      <div class="narrative-section">
        <strong>Methodology</strong>
        <p>${escapeHtml(narrative.methodology || '')}</p>
      </div>
      ${keyResultsHtml ? `<div class="narrative-section"><strong>Key Results</strong>${keyResultsHtml}</div>` : ''}
      <div class="narrative-section">
        <strong>Analysis</strong>
        <p>${escapeHtml(narrative.analysis || '')}</p>
      </div>
      <div class="narrative-section">
        <strong>Conclusions</strong>
        <p>${escapeHtml(narrative.conclusions || '')}</p>
      </div>
      ${limitationsHtml}
      `}
      ${codeHtml}
      <div class="card-actions">
        ${pdfHtml}
        <span class="task-id-label">ID: ${taskId.slice(0, 8)}</span>
      </div>
    </div>
  `;

  thread.appendChild(div);
  if (data.discovered_sources && data.discovered_sources.length > 0) {
    const tid = data.task_id || taskId;
    if (tid) thread.appendChild(renderSourcePanel(data.discovered_sources, tid));
  }
  if (window.Prism) setTimeout(() => Prism.highlightAll(), 50);
  scrollToBottom();
}

function appendFollowUpBubble(text, sources, taskId) {
  removeThinkingIndicator();
  const thread = $('chatThread');
  const div = document.createElement('div');
  div.className = 'chat-message agent-message';
  div.innerHTML = `<div class="agent-bubble follow-up">${escapeHtml(text)}</div>`;
  thread.appendChild(div);
  if (sources && sources.length > 0 && taskId) {
    thread.appendChild(renderSourcePanel(sources, taskId));
  }
  scrollToBottom();
}

function appendErrorBubble(text) {
  removeThinkingIndicator();
  const thread = $('chatThread');
  const div = document.createElement('div');
  div.className = 'chat-message agent-message';
  div.innerHTML = `<div class="agent-bubble error-bubble">Error: ${escapeHtml(text)}</div>`;
  thread.appendChild(div);
  scrollToBottom();
}

// ── Background task progress (SSE) ────────────────────────────────────────────

function appendProgressCard(taskId) {
  removeThinkingIndicator();
  const thread = $('chatThread');
  const div = document.createElement('div');
  div.className = 'chat-message agent-message';
  div.id = `progress-card-${taskId}`;
  div.innerHTML = `
    <div class="agent-card progress-card">
      <div class="progress-label">Working on your request…</div>
      <div class="progress-bar-track">
        <div class="progress-bar-fill" id="progress-fill-${taskId}"></div>
      </div>
      <div class="progress-status" id="progress-status-${taskId}">Initializing pipeline…</div>
    </div>`;
  thread.appendChild(div);
  scrollToBottom();
}

function subscribeToTaskProgress(taskId) {
  const es = new EventSource(`/api/tasks/${taskId}/stream`);
  let percent = 10;

  es.addEventListener('progress', (e) => {
    const meta = JSON.parse(e.data);
    const statusEl = $(`progress-status-${taskId}`);
    const fillEl = $(`progress-fill-${taskId}`);
    if (statusEl && meta.progress) statusEl.textContent = meta.progress;
    percent = Math.min(percent + 15, 85);
    if (fillEl) fillEl.style.width = `${percent}%`;
  });

  es.addEventListener('completed', (e) => {
    es.close();
    const result = JSON.parse(e.data);
    const progressCard = $(`progress-card-${taskId}`);
    if (progressCard) progressCard.remove();
    appendAgentCard(result);
    if (result.task_id && result.task_id !== taskId) {
      state.currentTaskId = result.task_id;
    }
    state.isThinking = false;
    setSendBtnState(false);
    addActivityItem('⚡', `Completed task`);
    loadConversationList();
    loadDashboardStats();
  });

  es.addEventListener('failed', (e) => {
    es.close();
    const data = JSON.parse(e.data);
    const progressCard = $(`progress-card-${taskId}`);
    if (progressCard) progressCard.remove();
    appendErrorBubble(data.error || 'Task failed');
    state.isThinking = false;
    setSendBtnState(false);
  });

  es.onerror = () => {
    es.close();
    const progressCard = $(`progress-card-${taskId}`);
    if (progressCard) progressCard.remove();
    appendErrorBubble('Lost connection to task stream. The task may still be running — refresh to check.');
    state.isThinking = false;
    setSendBtnState(false);
  };
}

// ── Source approval panel ─────────────────────────────────────────────────────

function renderSourcePanel(sources, taskId) {
  const srcLabels = { arxiv: 'arXiv', openalex: 'OpenAlex', semantic_scholar: 'Semantic Scholar' };
  const itemsHtml = sources.map(s => {
    const label = srcLabels[s.source] || s.source;
    const hopBadge = s.hop > 0 ? '<span class="source-badge badge-hop">cited</span> ' : '';
    const srcBadge = `<span class="source-badge badge-${s.source}">${label}</span>`;
    const year = s.year ? ` (${s.year})` : '';
    return `<label class="source-item">
      <input type="checkbox" value="${escapeHtml(s.id)}" checked>
      ${hopBadge}${srcBadge}
      <span class="source-title">${escapeHtml(s.title + year)}</span>
    </label>`;
  }).join('');

  const panel = document.createElement('div');
  panel.className = 'source-panel';
  panel.innerHTML = `
    <div class="source-panel-header">📚 Discovered Sources (${sources.length}) — select to add to Knowledge Base</div>
    <div class="source-list">${itemsHtml}</div>
    <div class="source-panel-footer">
      <button class="source-approve-btn" onclick="approveSelectedSources('${escapeHtml(taskId)}', this.closest('.source-panel'))">Add selected to KB</button>
    </div>`;
  return panel;
}

async function approveSelectedSources(taskId, panelEl) {
  const checked = panelEl.querySelectorAll('input[type="checkbox"]:checked');
  const sourceIds = Array.from(checked).map(cb => cb.value);
  const btn = panelEl.querySelector('.source-approve-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Adding…'; }
  try {
    const result = await fetchJson('/api/kb/approve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: taskId, source_ids: sourceIds }),
    });
    const footer = panelEl.querySelector('.source-panel-footer');
    if (footer) footer.innerHTML = `<span class="source-success">✓ ${result.ingested} source${result.ingested !== 1 ? 's' : ''} added to KB</span>`;
    panelEl.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.disabled = true);
  } catch (err) {
    if (btn) { btn.disabled = false; btn.textContent = 'Add selected to KB'; }
    appendErrorBubble(`Failed to approve sources: ${err.message}`);
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ── Send message ───────────────────────────────────────────────────────────────

async function sendMessage() {
  if (state.isThinking) return;
  const input = $('chatInput');
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  input.style.height = 'auto';
  appendUserBubble(text);
  appendThinkingIndicator();
  state.isThinking = true;
  setSendBtnState(true);

  try {
    if (!state.currentTaskId) {
      // New conversation → run-task (now returns {status: "queued", task_id} immediately)
      const data = await fetchJson('/run-task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task: text }),
      });
      state.currentTaskId = data.task_id;
      $('chatTitle').textContent = text.slice(0, 60) + (text.length > 60 ? '…' : '');
      addActivityItem('⚡', `Task: ${text.slice(0, 40)}`);

      if (data.status === 'queued') {
        // Async path: show progress bar and subscribe to SSE stream
        appendProgressCard(data.task_id);
        subscribeToTaskProgress(data.task_id);
        // isThinking stays true until SSE completes/fails — don't reset here
        return;
      } else {
        // Legacy sync fallback (if server returns full result)
        appendAgentCard(data);
        await loadConversationList();
        await loadDashboardStats();
      }
    } else {
      // Follow-up → conversation endpoint
      const data = await fetchJson(`/api/tasks/${state.currentTaskId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: text, iteration: 0 }),
      });
      appendFollowUpBubble(
        data.assistant_response || 'Done.',
        data.discovered_sources || [],
        data.task_id || state.currentTaskId,
      );
    }
  } catch (err) {
    appendErrorBubble(err.message);
  } finally {
    state.isThinking = false;
    setSendBtnState(false);
  }
}

function setSendBtnState(loading) {
  const btn = $('sendBtn');
  const txt = $('sendBtnText');
  if (!btn) return;
  btn.disabled = loading;
  if (txt) txt.textContent = loading ? '...' : 'Send';
}

function startNewChat() {
  state.currentTaskId = null;
  $('chatTitle').textContent = 'New Conversation';
  const thread = $('chatThread');
  thread.innerHTML = `
    <div class="chat-welcome" id="chatWelcome">
      <div class="welcome-icon">⚡</div>
      <h3>Start a research task</h3>
      <p>Ask anything — financial models, trading strategies, ML experiments, academic writing.</p>
      <div class="welcome-examples">
        <button class="example-chip" onclick="setInput('Backtest a 50/200 SMA crossover on SPY, QQQ, IWM')">SMA crossover backtest</button>
        <button class="example-chip" onclick="setInput('Compute the volatility of SPY over 2020-2024 with bootstrap CIs')">SPY volatility analysis</button>
        <button class="example-chip" onclick="setInput('Build an LSTM model to forecast AAPL weekly returns')">LSTM return forecast</button>
        <button class="example-chip" onclick="setInput('Write a 3-page report on momentum factor anomalies')">Momentum factor report</button>
      </div>
    </div>`;
  $('chatInput').focus();
}

// ── Conversation list ──────────────────────────────────────────────────────────

async function loadConversationList() {
  try {
    const data = await fetchJson('/api/tasks?limit=30&sort_by=-updated_at');
    const list = $('conversationList');
    if (!list) return;
    if (!data.tasks || data.tasks.length === 0) {
      list.innerHTML = '<div class="conv-placeholder">No conversations yet</div>';
      return;
    }
    list.innerHTML = data.tasks.map(t => `
      <div class="conv-item ${t.task_id === state.currentTaskId ? 'active' : ''}"
           onclick="openConversation('${t.task_id}', ${JSON.stringify(t.title).replace(/"/g, '&quot;')})">
        <div class="conv-title">${escapeHtml((t.title || 'Untitled').slice(0, 45))}</div>
        <div class="conv-meta">${t.status} &bull; ${new Date(t.updated_at).toLocaleDateString()}</div>
      </div>
    `).join('');
  } catch (e) {
    console.error('Failed to load conversation list:', e);
  }
}

async function openConversation(taskId, title) {
  state.currentTaskId = taskId;
  $('chatTitle').textContent = (title || taskId).slice(0, 60);

  // Highlight active in sidebar
  document.querySelectorAll('.conv-item').forEach(el =>
    el.classList.toggle('active', el.onclick?.toString().includes(taskId))
  );

  // Load task and re-render thread
  try {
    const data = await fetchJson(`/api/tasks/${taskId}`);
    const task = data.task;
    const thread = $('chatThread');
    thread.innerHTML = '';

    // Render original task as user bubble
    if (task.task) appendUserBubble(task.task);

    // Render artifacts as agent cards (last DS/quant/writing artifact)
    const lastArtifact = (task.artifacts || []).filter(a => a.type !== 'literature').slice(-1)[0];
    if (lastArtifact) {
      const report = lastArtifact.report || {};
      const narrative = report.narrative || {};
      const hasPdf = report.pdf && typeof report.pdf === 'object' && report.pdf.pdf;
      appendAgentCard({
        task_id: taskId,
        narrative,
        code: (lastArtifact.payload || {}).code || '',
        pdf_url: hasPdf ? `/api/reports/${taskId}/pdf` : null,
      });
    }

    // Render subsequent messages
    (task.messages || []).forEach(msg => {
      if (msg.role === 'user') appendUserBubble(msg.content);
      else if (msg.role === 'assistant') appendFollowUpBubble(msg.content);
    });

    showSection('tasks');
    addActivityItem('📋', `Opened: ${(title || taskId).slice(0, 30)}`);
  } catch (e) {
    appendErrorBubble(`Failed to load conversation: ${e.message}`);
  }
}

// ── Dashboard ──────────────────────────────────────────────────────────────────

async function loadDashboardStats() {
  try {
    const data = await fetchJson('/api/tasks?limit=100');
    const el = $('totalRuns');
    if (el) el.textContent = data.total || 0;
    const active = (data.tasks || []).filter(t => t.status === 'in-progress').length;
    const activeEl = $('activeTasks');
    if (activeEl) activeEl.textContent = active;
    const reportsEl = $('totalReports');
    if (reportsEl) reportsEl.textContent = (data.tasks || []).filter(t => t.status === 'completed').length;
  } catch (e) {
    console.error('Dashboard stats failed:', e);
  }
}

async function loadSystemHealth() {
  try {
    const h = await fetchJson('/health');
    updateSystemStatus(h);
  } catch (e) {
    updateSystemStatus({ status: 'error' });
  }
}

// ── Reports (kept from original, simplified) ──────────────────────────────────

async function loadReports() {
  // Placeholder — real reports come from /api/reports/{task_id}/pdf
  const list = $('reportsList');
  if (!list) return;
  try {
    const data = await fetchJson('/api/tasks?limit=50&sort_by=-updated_at');
    const withPdf = (data.tasks || []).filter(t => t.status === 'completed');
    if (withPdf.length === 0) {
      list.innerHTML = '<div class="no-reports">No completed tasks with reports yet</div>';
      return;
    }
    list.innerHTML = withPdf.map(t => `
      <div class="file-item" onclick="openConversation('${t.task_id}', ${JSON.stringify(t.title).replace(/"/g, '&quot;')})">
        <span class="file-icon">📄</span>
        <span class="file-name">${escapeHtml((t.title || 'Untitled').slice(0, 40))}</span>
        <a class="pdf-link" href="/api/reports/${t.task_id}/pdf" target="_blank" download onclick="event.stopPropagation()">PDF</a>
      </div>
    `).join('');
  } catch (e) {
    list.innerHTML = `<div class="no-reports">Error: ${escapeHtml(e.message)}</div>`;
  }
}

// ── Data ingestion ─────────────────────────────────────────────────────────────

async function ingestDocuments() {
  const path = $('ingestPath')?.value.trim();
  if (!path) { alert('Please enter a document path'); return; }
  const out = $('ingestOutput');
  if (out) out.textContent = 'Ingesting documents...';
  try {
    const result = await fetchJson('/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    if (out) out.textContent = `Done!\nTotal chunks: ${result.chunks}\nAdded: ${result.added}`;
    addActivityItem('📚', `Ingested ${result.added} chunks from ${path}`);
  } catch (e) {
    if (out) out.textContent = `Failed: ${e.message}`;
  }
}

// ── Event listeners ────────────────────────────────────────────────────────────

function setupEventListeners() {
  $('sidebarToggle')?.addEventListener('click', toggleSidebar);
  $('mobileMenuBtn')?.addEventListener('click', () => $('sidebar').classList.toggle('mobile-open'));

  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      showSection(item.dataset.section);
      $('sidebar').classList.remove('mobile-open');
    });
  });

  $('themeToggle')?.addEventListener('click', () => setTheme(state.theme === 'dark' ? 'light' : 'dark'));
  $('themeSelect')?.addEventListener('change', e => setTheme(e.target.value));

  // Chat
  $('newChatBtn')?.addEventListener('click', startNewChat);

  const chatInput = $('chatInput');
  if (chatInput) {
    chatInput.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
    chatInput.addEventListener('input', () => {
      chatInput.style.height = 'auto';
      chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
    });
  }

  // Data
  $('ingestBtn')?.addEventListener('click', ingestDocuments);

  // Reports
  $('refreshReportsBtn')?.addEventListener('click', loadReports);

  // Settings
  $('healthCheckBtn')?.addEventListener('click', async () => {
    const hs = $('healthStatus');
    if (hs) hs.textContent = 'Checking...';
    await loadSystemHealth();
    if (hs) hs.textContent = state.systemHealth?.status === 'ok' ? 'System is healthy' : 'System has issues';
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', e => {
    if (e.ctrlKey || e.metaKey) {
      const map = { '1': 'dashboard', '2': 'tasks', '3': 'reports', '4': 'data', '5': 'settings' };
      if (map[e.key]) { e.preventDefault(); showSection(map[e.key]); }
    }
  });
}

// ── Init ───────────────────────────────────────────────────────────────────────

async function init() {
  const savedTheme = localStorage.getItem('theme') || 'light';
  const savedSection = localStorage.getItem('currentSection') || 'dashboard';
  const savedCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';

  setTheme(savedTheme);
  const ts = $('themeSelect');
  if (ts) ts.value = savedTheme;
  if (savedCollapsed) toggleSidebar();

  setupEventListeners();

  await Promise.all([loadSystemHealth(), loadConversationList(), loadDashboardStats(), loadReports()]);

  showSection(savedSection);
  addActivityItem('🚀', 'Finance Assistant.ai ready');
}

// Global helpers for inline onclick attributes
window.setInput = setInput;
window.sendMessage = sendMessage;
window.openConversation = openConversation;
window.approveSelectedSources = approveSelectedSources;

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
