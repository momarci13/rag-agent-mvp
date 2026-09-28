// Model Studio: Modeler and Validator agents

const SECTIONS = ['projects', 'new', 'library', 'settings', 'about'];
const TITLES = { projects: 'Projects', new: 'New project', library: 'Library', settings: 'Settings', about: 'Privacy and terms', project: 'Project' };
const OUTCOME = {
  approved: ['Approved', 'ok'],
  approved_with_conditions: ['Approved with conditions', 'warn'],
  remediation_required: ['Remediation required', 'bad'],
  rejected: ['Rejected', 'bad'],
};
const STATUS = {
  queued: ['Queued', 'neutral'], running: ['Running', 'info'], awaiting_signoff: ['Awaiting sign-off', 'warn'],
  signed_off: ['Signed off', 'ok'], failed: ['Failed', 'bad'], cancelled: ['Cancelled', 'neutral'],
};
const SEV_ORDER = { critical: 4, high: 3, medium: 2, low: 1, observation: 0 };

const state = { theme: 'light', sidebarCollapsed: false, project: null, stream: null, findingFilter: 'open', refreshTimer: null };

// ── Storage (preferences in localStorage; token in sessionStorage) ──────────
const prefs = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, String(v)); } catch { /* unavailable */ } },
};
const token = {
  get() { try { return sessionStorage.getItem('studioToken') || ''; } catch { return ''; } },
  set(v) { try { v ? sessionStorage.setItem('studioToken', v) : sessionStorage.removeItem('studioToken'); } catch { /* unavailable */ } },
};

// ── Utilities ────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
function esc(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
async function api(url, opts = {}) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let d = await r.text();
    try { d = JSON.parse(d).detail || d; } catch { /* not JSON */ }
    throw new Error(`${r.status}: ${typeof d === 'string' ? d : JSON.stringify(d)}`);
  }
  return r.json();
}
function when(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleDateString([], { day: 'numeric', month: 'short' }) + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
function tag(label, kind) { return `<span class="tag tag--${kind}">${esc(label)}</span>`; }
function sevTag(s) { return tag(s, { critical: 'bad', high: 'bad', medium: 'warn', low: 'neutral', observation: 'neutral' }[s] || 'neutral'); }
function fileUrl(pid, rel) { return `/api/projects/${encodeURIComponent(pid)}/files/${rel.split('/').map(encodeURIComponent).join('/')}`; }
function setError(id, msg) {
  const el = $(id);
  if (!el) return;
  let e = $(`${id}Error`);
  if (!e) {
    e = document.createElement('p'); e.id = `${id}Error`; e.className = 'field-error';
    el.insertAdjacentElement('afterend', e);
    el.setAttribute('aria-describedby', [el.getAttribute('aria-describedby'), e.id].filter(Boolean).join(' '));
  }
  e.textContent = msg || ''; e.hidden = !msg;
  if (msg) el.setAttribute('aria-invalid', 'true'); else el.removeAttribute('aria-invalid');
}
function requireFields(list) {
  let first = null;
  list.forEach(([id, msg, check]) => {
    const el = $(id);
    const ok = check ? check(el) : Boolean(el && String(el.value || '').trim());
    setError(id, ok ? '' : msg);
    if (!ok && !first) first = el;
  });
  if (first) first.focus();
  return !first;
}

// ── Theme and layout ─────────────────────────────────────────────────────────
function prefersDark() { return window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches; }
function setTheme(t) {
  t = ['light', 'dark', 'auto'].includes(t) ? t : 'light';
  document.documentElement.classList.remove('light', 'dark', 'auto');
  document.documentElement.classList.add(t);
  state.theme = t; prefs.set('theme', t);
  const dark = t === 'dark' || (t === 'auto' && prefersDark());
  $('themeToggle').textContent = dark ? 'Light mode' : 'Dark mode';
  $('themeToggle').setAttribute('aria-pressed', String(dark));
  $('themeSelect').value = t;
}
function toggleSidebar() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  $('sidebar').classList.toggle('collapsed', state.sidebarCollapsed);
  document.querySelector('.main-content').classList.toggle('sidebar-collapsed', state.sidebarCollapsed);
  const b = $('sidebarToggle');
  b.setAttribute('aria-expanded', String(!state.sidebarCollapsed));
  b.setAttribute('aria-label', state.sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar');
  b.innerHTML = `<span aria-hidden="true">${state.sidebarCollapsed ? '&raquo;' : '&laquo;'}</span>`;
  prefs.set('sidebarCollapsed', state.sidebarCollapsed);
}
function setMobileMenu(open) {
  $('sidebar').classList.toggle('mobile-open', open);
  $('mobileMenuBtn').setAttribute('aria-expanded', String(open));
}

// ── Routing ──────────────────────────────────────────────────────────────────
function route() {
  const hash = location.hash.replace('#', '') || prefs.get('lastSection') || 'projects';
  const [name, id] = hash.split('/');
  const section = name === 'project' && id ? 'project' : (SECTIONS.includes(name) ? name : 'projects');
  document.querySelectorAll('.section').forEach(s => s.classList.toggle('active', s.id === `${section}-section`));
  document.querySelectorAll('.nav-item').forEach(n => {
    const active = n.dataset.section === section || (section === 'project' && n.dataset.section === 'projects');
    n.classList.toggle('active', active);
    active ? n.setAttribute('aria-current', 'page') : n.removeAttribute('aria-current');
  });
  $('pageTitle').textContent = TITLES[section];
  document.title = `${TITLES[section]} · Model Studio`;
  if (section !== 'project') prefs.set('lastSection', section);
  if (state.stream && section !== 'project') { state.stream.close(); state.stream = null; }
  if (section === 'projects') loadProjects();
  if (section === 'library') loadLibrary();
  if (section === 'project') openProject(id);
  setMobileMenu(false);
}

// ── Projects list ────────────────────────────────────────────────────────────
async function loadProjects() {
  const body = $('projectRows');
  try {
    const { projects } = await api('/api/projects');
    if (!projects.length) {
      body.innerHTML = '<tr><td colspan="7" class="empty">No projects yet. <a href="#new">Start one</a>.</td></tr>';
      return;
    }
    body.innerHTML = projects.map(p => {
      const [sl, sk] = STATUS[p.status] || [p.status, 'neutral'];
      const [ol, ok] = OUTCOME[p.outcome] || ['', 'neutral'];
      const f = p.open_findings;
      const counts = ['critical', 'high', 'medium', 'low'].filter(s => f[s]).map(s => `${f[s]} ${s}`).join(', ') || 'None';
      return `<tr>
        <td><a href="#project/${esc(p.project_id)}">${esc(p.title)}</a></td>
        <td>${p.mode === 'develop_and_validate' ? 'Develop + validate' : 'External validation'}</td>
        <td>${tag(sl, sk)}</td>
        <td>${ol ? tag(ol, ok) : '<span class="muted">Pending</span>'}</td>
        <td class="num">${p.rounds} / ${p.max_rounds}</td>
        <td>${esc(counts)}</td>
        <td class="num">${esc(when(p.updated_at))}</td>
      </tr>`;
    }).join('');
  } catch (e) {
    body.innerHTML = `<tr><td colspan="7" class="empty">Could not load projects: ${esc(e.message)}</td></tr>`;
  }
}

// ── New project ──────────────────────────────────────────────────────────────
function currentMode() { return document.querySelector('input[name="mode"]:checked').value; }
function updateModeFields() {
  const ext = currentMode() === 'validate_external';
  $('packageField').hidden = !ext;
  $('roundsField').hidden = ext;
  $('pPackage').required = ext;
}
async function loadFrameworks() {
  try {
    const { frameworks } = await api('/api/frameworks');
    $('frameworkChoices').innerHTML = frameworks.map(f => `
      <label class="check"><input type="checkbox" name="frameworks" value="${esc(f.key)}" checked> ${esc(f.label)}</label>`).join('');
  } catch (e) {
    $('frameworkChoices').textContent = 'Could not load frameworks.';
  }
}
async function createProject() {
  if ($('pToken').value) token.set($('pToken').value);
  const ext = currentMode() === 'validate_external';
  const checks = [
    ['pTitle', 'Enter a title.'],
    ['pBrief', 'Describe what the model must do.'],
    ['pData', 'Add at least one data file.', el => el.files.length > 0],
    ['pToken', 'Enter the STUDIO_API_TOKEN set on the server.', el => Boolean(el.value || token.get())],
  ];
  if (ext) checks.splice(3, 0, ['pPackage', 'Add the model code and documentation.', el => el.files.length > 0]);
  if (!requireFields(checks)) return;

  const fd = new FormData();
  fd.append('mode', currentMode());
  fd.append('title', $('pTitle').value.trim());
  fd.append('brief', $('pBrief').value.trim());
  fd.append('max_rounds', ext ? '1' : $('pRounds').value);
  fd.append('challenger', $('pChallenger').value);
  document.querySelectorAll('input[name="frameworks"]:checked').forEach(c => fd.append('frameworks', c.value));
  const add = (id, field) => Array.from($(id).files).forEach(f => fd.append(field, f));
  add('pData', 'data'); add('pConcept', 'concept_papers'); add('pRegs', 'regulations');
  if (ext) add('pPackage', 'package');

  const btn = $('createBtn');
  btn.disabled = true; btn.textContent = 'Uploading';
  $('createStatus').textContent = 'Uploading files and starting the agents.';
  try {
    const r = await api('/api/projects', { method: 'POST', body: fd, headers: { 'X-Studio-Token': $('pToken').value || token.get() } });
    $('projectForm').reset(); updateModeFields(); loadFrameworks();
    $('createStatus').textContent = '';
    location.hash = `#project/${r.project_id}`;
  } catch (e) {
    $('createStatus').textContent = `Could not start: ${e.message}`;
  } finally {
    btn.disabled = false; btn.textContent = 'Start project';
  }
}

// ── Project detail ───────────────────────────────────────────────────────────
async function openProject(id) {
  if (state.stream) { state.stream.close(); state.stream = null; }
  $('traceLog').innerHTML = '';
  $('projectTitle').textContent = 'Loading';
  try {
    await refreshProject(id);
  } catch (e) {
    $('projectTitle').textContent = 'Project not found';
    $('projectMeta').textContent = e.message;
    return;
  }
  const p = state.project;
  renderTrace(p.trace, true);
  if (['queued', 'running'].includes(p.status)) {
    const es = new EventSource(`/api/projects/${encodeURIComponent(id)}/stream?after=${p.trace.length}`);
    state.stream = es;
    es.addEventListener('trace', ev => {
      renderTrace([JSON.parse(ev.data)], false);
      clearTimeout(state.refreshTimer);
      state.refreshTimer = setTimeout(() => refreshProject(id).catch(() => {}), 800);
    });
    es.addEventListener('done', () => { es.close(); state.stream = null; refreshProject(id); });
    es.onerror = () => { es.close(); state.stream = null; };
  }
}

async function refreshProject(id) {
  const { project } = await api(`/api/projects/${encodeURIComponent(id)}`);
  state.project = project;
  renderProject(project);
  loadFiles(id);
}

function renderTrace(events, reset) {
  const log = $('traceLog');
  if (reset) log.innerHTML = '';
  events.forEach(e => {
    const li = document.createElement('li');
    li.className = `trace trace--${e.status}`;
    li.innerHTML = `<span class="trace-agent">${esc(e.agent)}</span>
      <span class="trace-step">${esc(e.step.replace(/-/g, ' '))}</span>
      <time class="trace-time">${esc(new Date(e.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))}</time>
      <span class="trace-msg">${esc((e.message || '').split('\n')[0])}</span>`;
    log.appendChild(li);
  });
  log.scrollTop = log.scrollHeight;
}

function renderProject(p) {
  const inp = p.input;
  $('projectTitle').textContent = inp.title;
  $('pageTitle').textContent = inp.title;
  $('projectMeta').textContent = `${inp.mode === 'develop_and_validate' ? 'Develop and validate' : 'Validation of an external model'} · created ${when(p.created_at)} · frameworks: ${inp.frameworks.join(', ') || 'generic practice only'}`;
  const [sl, sk] = STATUS[p.status] || [p.status, 'neutral'];
  const oc = OUTCOME[p.final_outcome || (p.rounds.at(-1)?.validator?.outcome)];
  $('projectTags').innerHTML = tag(sl, sk) + (oc ? tag(oc[0], oc[1]) : '');
  const err = $('projectError');
  err.hidden = !p.error; err.textContent = p.error ? `The run stopped: ${p.error}` : '';

  renderRounds(p);
  renderFindings(p);
  renderRequirements(p);
  renderDocs(p);

  $('signoffPanel').hidden = !['awaiting_signoff', 'signed_off'].includes(p.status);
  $('signoffList').innerHTML = (p.signoffs || []).map(s =>
    `<li><strong>${esc(s.name)}</strong>${s.role ? `, ${esc(s.role)}` : ''}: ${esc(s.decision)} on ${esc(when(s.at))}${s.comment ? `<br><span class="muted">${esc(s.comment)}</span>` : ''}</li>`).join('');
}

function countsText(run) {
  if (!run) return 'not run';
  const c = { passed: 0, failed: 0, error: 0, skipped: 0 };
  (run.results || []).forEach(r => { c[r.outcome] += 1; });
  if (run.collection_error) return 'could not run';
  return `${c.passed} passed, ${c.failed} failed${c.error ? `, ${c.error} errors` : ''}`;
}

function renderRounds(p) {
  const list = $('roundList');
  if (!p.rounds.length) { list.innerHTML = '<li class="muted">Waiting for the first round.</li>'; return; }
  list.innerHTML = p.rounds.map(r => {
    const m = r.modeler, v = r.validator;
    const oc = OUTCOME[v.outcome];
    const primary = m.spec?.acceptance_criteria?.find(a => a.primary);
    const pm = primary && m.pipeline?.metrics ? m.pipeline.metrics[primary.metric] : undefined;
    const rep = v.replication;
    const ch = v.challenger;
    const modelerLine = p.input.mode === 'validate_external'
      ? `<p><strong>Subject:</strong> ${esc(m.spec?.title || 'reading package')}</p>`
      : `<p><strong>Modeler:</strong> ${esc(m.spec ? `${m.spec.title} (${m.spec.category.replace('_', ' ')})` : 'designing')}${m.tests ? ` · tests ${countsText(m.tests)}` : ''}${pm !== undefined ? ` · ${esc(primary.metric)} ${Number(pm).toPrecision(4)}` : ''}${m.responses?.length ? ` · answered ${m.responses.length} findings` : ''}</p>`;
    const validatorLine = `<p><strong>Validator:</strong> ${rep ? (rep.reproduced ? 'replicated' : rep.pipeline_ok ? `not reproduced (${esc(rep.mismatches.join(', '))})` : 'pipeline failed') : 'pending'}${v.independent_tests ? ` · independent tests ${countsText(v.independent_tests)}` : ''}${ch && ch.primary_metric ? ` · challenger ${esc(ch.primary_metric)} ${ch.challenger_metrics[ch.primary_metric] !== undefined ? Number(ch.challenger_metrics[ch.primary_metric]).toPrecision(4) : 'n/a'}` : ''}</p>`;
    return `<li class="round">
      <div class="round-head"><span class="round-no">Round ${r.number}</span>${oc ? tag(oc[0], oc[1]) : '<span class="muted">in progress</span>'}</div>
      ${modelerLine}${validatorLine}
      ${v.conclusion ? `<details><summary>Validator conclusion</summary><div class="prose-block">${esc(v.conclusion).replace(/\n\n/g, '</p><p>').replace(/^/, '<p>')}</p></div></details>` : ''}
    </li>`;
  }).join('');
}

function renderFindings(p) {
  const box = $('findingList');
  const last = p.rounds.at(-1);
  const all = last ? last.validator.findings : [];
  const rows = all.filter(f => state.findingFilter === 'all' || f.status === 'open')
    .sort((a, b) => (a.status === 'open' ? 0 : 1) - (b.status === 'open' ? 0 : 1) || SEV_ORDER[b.severity] - SEV_ORDER[a.severity]);
  if (!all.length) { box.innerHTML = `<p class="muted">${last ? 'No findings.' : 'Findings appear after the first validation.'}</p>`; return; }
  if (!rows.length) { box.innerHTML = '<p class="muted">No open findings.</p>'; return; }
  box.innerHTML = rows.map(f => `
    <details class="finding">
      <summary>
        ${sevTag(f.severity)}
        <span class="finding-title">${esc(f.title)}</span>
        <span class="finding-id">${esc(f.finding_id)}</span>
        ${f.status !== 'open' ? tag(f.status.replace('_', ' '), 'ok') : ''}
      </summary>
      <div class="finding-body">
        <p class="muted">${esc(f.area)} · ${esc(f.source.replace('_', ' '))} · raised in round ${f.raised_round}${f.closed_round ? `, closed in round ${f.closed_round}` : ''}</p>
        <p>${esc(f.description)}</p>
        ${f.evidence?.length ? `<h4>Evidence</h4><ul>${f.evidence.slice(0, 6).map(e => `<li><pre>${esc(e)}</pre></li>`).join('')}</ul>` : ''}
        ${f.citations?.length ? `<p><strong>References:</strong> ${f.citations.map(c => esc(c.locator ? `${c.source}, ${c.locator}` : c.source)).join('; ')}</p>` : ''}
        ${f.recommendation ? `<p><strong>Recommendation:</strong> ${esc(f.recommendation)}</p>` : ''}
        ${f.modeler_response ? `<p><strong>Modeler response (${esc(f.modeler_response.action.replace('_', ' '))}):</strong> ${esc(f.modeler_response.explanation)}</p>` : ''}
      </div>
    </details>`).join('');
}

function renderRequirements(p) {
  const last = p.rounds.at(-1);
  const v = last?.validator;
  const box = $('requirementTable');
  if (!v || !v.requirements.length) { box.innerHTML = '<p class="muted">Available after the validator has run.</p>'; $('requirementSummary').textContent = ''; return; }
  const assess = Object.fromEntries((v.review?.requirement_assessments || []).map(a => [a.req_id, a]));
  const counts = {};
  v.requirements.forEach(r => { const st = assess[r.req_id]?.status || 'not_assessed'; counts[st] = (counts[st] || 0) + 1; });
  $('requirementSummary').textContent = `${v.requirements.length} items: ` + Object.entries(counts).map(([k, n]) => `${n} ${k.replace(/_/g, ' ')}`).join(', ');
  const kind = { met: 'ok', partially_met: 'warn', not_met: 'bad', not_applicable: 'neutral', not_assessed: 'neutral' };
  box.innerHTML = `<table class="data-table"><thead><tr><th scope="col">ID</th><th scope="col">Requirement</th><th scope="col">Source</th><th scope="col">Status</th></tr></thead><tbody>
    ${v.requirements.map(r => {
      const a = assess[r.req_id]; const st = a ? a.status : 'not_assessed';
      const src = r.citation ? (r.citation.locator ? `${r.citation.source}, ${r.citation.locator}` : r.citation.source) : r.framework;
      return `<tr><td class="req-id">${esc(r.req_id)}</td><td>${esc(r.text)}${a?.evidence ? `<br><span class="muted">${esc(a.evidence)}</span>` : ''}</td><td>${esc(src)}</td><td>${tag(st.replace(/_/g, ' '), kind[st])}</td></tr>`;
    }).join('')}</tbody></table>`;
}

function renderDocs(p) {
  const items = [];
  p.rounds.forEach(r => {
    const m = r.modeler, v = r.validator;
    if (m.document_docx) items.push([`Modelling document v${r.number}`, m.document_docx, m.document_pdf]);
    if (v.report_docx) items.push([`Validation report, round ${r.number}`, v.report_docx, v.report_pdf]);
  });
  $('docList').innerHTML = items.length
    ? items.reverse().map(([label, docx, pdf]) => `<li><span>${esc(label)}</span>
        <span class="doc-links"><a href="${fileUrl(p.project_id, docx)}">DOCX</a>${pdf ? ` <a href="${fileUrl(p.project_id, pdf)}">PDF</a>` : ''}</span></li>`).join('')
    : '<li class="muted">Documents appear as each agent finishes.</li>';
}

async function loadFiles(id) {
  try {
    const { files } = await api(`/api/projects/${encodeURIComponent(id)}/tree`);
    $('fileList').innerHTML = files.map(f => `<li><a href="${fileUrl(id, f.path)}" target="_blank" rel="noopener">${esc(f.path)}</a></li>`).join('') || '<li class="muted">No files yet.</li>';
  } catch { /* project may be gone */ }
}

async function submitSignoff() {
  const p = state.project;
  if (!p) return;
  if (!requireFields([['soName', 'Enter your name.']])) return;
  if (!token.get()) { $('signoffStatus').textContent = 'Enter the Studio API token in Settings first.'; return; }
  if (!confirm(`Record "${$('soDecision').value}" by ${$('soName').value.trim()} for this model?`)) return;
  const btn = $('signoffBtn'); btn.disabled = true;
  try {
    await api(`/api/projects/${encodeURIComponent(p.project_id)}/signoff`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Studio-Token': token.get() },
      body: JSON.stringify({ name: $('soName').value.trim(), role: $('soRole').value.trim(), decision: $('soDecision').value, comment: $('soComment').value.trim() }),
    });
    $('signoffStatus').textContent = 'Recorded. The validation report now shows the sign-off.';
    $('signoffForm').reset();
    refreshProject(p.project_id);
  } catch (e) {
    $('signoffStatus').textContent = `Not recorded: ${e.message}`;
  } finally { btn.disabled = false; }
}

// ── Library ──────────────────────────────────────────────────────────────────
async function loadLibrary() {
  const body = $('libraryRows');
  try {
    const { documents } = await api('/api/library');
    body.innerHTML = documents.map(d => `<tr>
      <td>${esc(d.title)}${d.builtin ? ' <span class="muted">(built-in summary)</span>' : ''}</td>
      <td>${esc(d.kind.replace('_', ' '))}</td><td>${esc(d.framework || '')}</td>
      <td class="num">${d.chunks}</td><td class="num">${d.projects} project${d.projects === 1 ? '' : 's'}</td></tr>`).join('')
      || '<tr><td colspan="5" class="empty">The library is empty.</td></tr>';
  } catch (e) {
    body.innerHTML = `<tr><td colspan="5" class="empty">Could not load the library: ${esc(e.message)}</td></tr>`;
  }
}
async function addToLibrary() {
  if (!requireFields([['libFiles', 'Choose at least one file.', el => el.files.length > 0]])) return;
  if (!token.get()) { $('libStatus').textContent = 'Enter the Studio API token in Settings first.'; return; }
  const fd = new FormData();
  fd.append('kind', $('libKind').value);
  Array.from($('libFiles').files).forEach(f => fd.append('files', f));
  $('libBtn').disabled = true; $('libStatus').textContent = 'Indexing documents.';
  try {
    const r = await api('/api/library', { method: 'POST', body: fd, headers: { 'X-Studio-Token': token.get() } });
    $('libStatus').textContent = `Added ${r.added.length} document${r.added.length === 1 ? '' : 's'}.`;
    $('libraryForm').reset(); loadLibrary();
  } catch (e) {
    $('libStatus').textContent = `Could not add: ${e.message}`;
  } finally { $('libBtn').disabled = false; }
}

// ── Health ───────────────────────────────────────────────────────────────────
async function checkHealth(verbose) {
  try {
    const h = await api('/health');
    const ok = h.status === 'ok';
    $('statusIndicator').classList.toggle('active', ok);
    $('statusIndicator').classList.toggle('error', !ok);
    $('statusText').textContent = ok ? 'Agents ready' : 'Model provider unavailable';
    if (verbose) $('healthStatus').textContent = ok
      ? `Healthy. ${h.provider} model ${h.model}. ${h.token_configured ? 'API token is configured.' : 'STUDIO_API_TOKEN is not set on the server, so projects cannot be started.'}`
      : `Not healthy: ${h.detail || 'the OpenAI API did not respond or the key is missing'}.`;
  } catch (e) {
    $('statusIndicator').classList.add('error');
    $('statusText').textContent = 'Server unreachable';
    if (verbose) $('healthStatus').textContent = e.message;
  }
}

// ── Wiring ───────────────────────────────────────────────────────────────────
function init() {
  setTheme(prefs.get('theme') || 'light');
  if (prefs.get('sidebarCollapsed') === 'true') toggleSidebar();
  $('pToken').value = token.get();
  $('settingsToken').value = token.get();

  $('sidebarToggle').addEventListener('click', toggleSidebar);
  $('mobileMenuBtn').addEventListener('click', () => setMobileMenu(!$('sidebar').classList.contains('mobile-open')));
  document.addEventListener('click', e => {
    if ($('sidebar').classList.contains('mobile-open') && !$('sidebar').contains(e.target) && !$('mobileMenuBtn').contains(e.target)) setMobileMenu(false);
  });
  $('themeToggle').addEventListener('click', () => setTheme((state.theme === 'dark' || (state.theme === 'auto' && prefersDark())) ? 'light' : 'dark'));
  $('themeSelect').addEventListener('change', e => setTheme(e.target.value));
  $('settingsToken').addEventListener('change', e => { token.set(e.target.value.trim()); $('pToken').value = token.get(); });
  $('forgetToken').addEventListener('click', () => { token.set(''); $('settingsToken').value = ''; $('pToken').value = ''; });
  $('healthCheckBtn').addEventListener('click', () => { $('healthStatus').textContent = 'Checking.'; checkHealth(true); });

  document.querySelectorAll('input[name="mode"]').forEach(r => r.addEventListener('change', updateModeFields));
  $('projectForm').addEventListener('submit', e => { e.preventDefault(); createProject(); });
  $('signoffForm').addEventListener('submit', e => { e.preventDefault(); submitSignoff(); });
  $('libraryForm').addEventListener('submit', e => { e.preventDefault(); addToLibrary(); });
  document.querySelectorAll('.seg-btn').forEach(b => b.addEventListener('click', () => {
    state.findingFilter = b.dataset.filter;
    document.querySelectorAll('.seg-btn').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    if (state.project) renderFindings(state.project);
  }));
  document.querySelectorAll('input[required], textarea[required]').forEach(el =>
    el.addEventListener('input', () => setError(el.id, '')));

  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && /^[1-5]$/.test(e.key)) {
      e.preventDefault(); location.hash = `#${SECTIONS[Number(e.key) - 1]}`;
    }
    if (e.key === 'Escape') setMobileMenu(false);
  });
  window.addEventListener('hashchange', () => { route(); $('main').focus({ preventScroll: true }); });

  updateModeFields();
  loadFrameworks();
  route();
  checkHealth(false);
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
