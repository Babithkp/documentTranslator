/* ── State ──────────────────────────────────────────────────────────────────── */
let selectedFile = null;
let selectedDpi  = 150;
let currentJobId = null;
let evtSource    = null;

/* ── Step map: SSE step names → DOM IDs ────────────────────────────────────── */
const STEP_ORDER = ['render', 'ocr', 'translate', 'draw', 'assemble'];
const STEP_MAP   = {
  render:    'step-render',
  ocr:       'step-ocr',
  layout:    'step-ocr',     // sub-step of OCR
  translate: 'step-translate',
  draw:      'step-draw',
  assemble:  'step-assemble',
};

/* ── File selection ─────────────────────────────────────────────────────────── */
function onFileSelected(input) {
  if (input.files && input.files[0]) setFile(input.files[0]);
}

function setFile(file) {
  selectedFile = file;

  document.getElementById('dz-idle').style.display  = 'none';
  document.getElementById('dz-file').style.display  = '';
  document.getElementById('file-name').textContent  = file.name;
  document.getElementById('file-size').textContent  = formatBytes(file.size);
  document.getElementById('translate-btn').disabled = false;
}

function clearFile(event) {
  event.stopPropagation();
  selectedFile = null;
  document.getElementById('file-input').value        = '';
  document.getElementById('dz-idle').style.display   = '';
  document.getElementById('dz-file').style.display   = 'none';
  document.getElementById('translate-btn').disabled  = true;
}

/* ── Drag & drop ────────────────────────────────────────────────────────────── */
function onDragOver(e) {
  e.preventDefault();
  e.currentTarget.classList.add('drag-over');
}
function onDragLeave(e) {
  e.currentTarget.classList.remove('drag-over');
}
function onDrop(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
}

/* ── DPI / quality ──────────────────────────────────────────────────────────── */
function setQuality(btn) {
  document.querySelectorAll('.quality-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  selectedDpi = parseInt(btn.dataset.dpi, 10);
}

/* ── Translation ────────────────────────────────────────────────────────────── */
async function startTranslation() {
  if (!selectedFile) return;

  const language = document.getElementById('language-select').value;

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('language', language);
  formData.append('dpi', selectedDpi);

  showSection('progress');
  resetSteps();
  setProgress(2, 'Uploading file…');

  let response;
  try {
    response = await fetch('/translate', { method: 'POST', body: formData });
  } catch {
    showError('Network error — could not reach the server.');
    return;
  }

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    showError(data.error || `Server error ${response.status}.`);
    return;
  }

  const { job_id } = await response.json();
  currentJobId = job_id;
  connectSSE(job_id);
}

/* ── SSE progress stream ────────────────────────────────────────────────────── */
function connectSSE(jobId) {
  evtSource = new EventSource(`/progress/${jobId}`);

  evtSource.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.heartbeat) return;

    if (ev.error) {
      evtSource.close();
      showError(ev.error);
      return;
    }

    if (ev.step) {
      setProgress(ev.progress ?? 0, ev.message ?? '');
      activateStep(ev.step);

      const pageInfo = document.getElementById('page-info');
      pageInfo.textContent = (ev.page && ev.total)
        ? `Page ${ev.page} of ${ev.total}`
        : '';
    }

    if (ev.done) {
      evtSource.close();
      setProgress(100, 'Done!');
      markAllDone();
      setTimeout(() => showResult(jobId), 600);
    }
  };

  evtSource.onerror = () => {
    evtSource.close();
    showError('Lost connection to the server. Please try again.');
  };
}

/* ── Step UI ────────────────────────────────────────────────────────────────── */
function resetSteps() {
  document.querySelectorAll('.step').forEach(s => s.classList.remove('active', 'done'));
  document.querySelectorAll('.step-line').forEach(l => l.classList.remove('done'));
}

function activateStep(stepName) {
  const targetId = STEP_MAP[stepName];
  if (!targetId) return;

  const targetIdx = STEP_ORDER.indexOf(
    Object.keys(STEP_MAP).find(k => STEP_MAP[k] === targetId && STEP_ORDER.includes(k)) ?? stepName
  );

  STEP_ORDER.forEach((key, idx) => {
    const el = document.getElementById(`step-${key}`);
    if (!el) return;
    el.classList.remove('active', 'done');
    if (idx < targetIdx)  el.classList.add('done');
    if (idx === targetIdx) el.classList.add('active');
  });
}

function markAllDone() {
  STEP_ORDER.forEach(key => {
    const el = document.getElementById(`step-${key}`);
    if (el) { el.classList.remove('active'); el.classList.add('done'); }
  });
}

/* ── Progress bar ───────────────────────────────────────────────────────────── */
function setProgress(pct, message) {
  const bar   = document.getElementById('progress-bar');
  const label = document.getElementById('progress-pct');
  const msg   = document.getElementById('progress-message');
  const track = document.getElementById('progress-track');

  bar.style.width = `${pct}%`;
  label.textContent = `${Math.round(pct)}%`;
  if (message) msg.textContent = message;
  if (track) track.setAttribute('aria-valuenow', Math.round(pct));
}

/* ── Section visibility ─────────────────────────────────────────────────────── */
function showSection(name) {
  ['upload', 'progress', 'result', 'error'].forEach(s => {
    document.getElementById(`${s}-section`).style.display = (s === name) ? '' : 'none';
  });
}

function showResult(jobId) {
  document.getElementById('download-link').href = `/download/${jobId}`;
  showSection('result');
}

function showError(message) {
  document.getElementById('error-message').textContent = message;
  showSection('error');
}

function resetUI() {
  if (evtSource) { evtSource.close(); evtSource = null; }
  currentJobId = null;

  selectedFile = null;
  document.getElementById('file-input').value        = '';
  document.getElementById('dz-idle').style.display   = '';
  document.getElementById('dz-file').style.display   = 'none';
  document.getElementById('translate-btn').disabled  = true;

  setProgress(0, 'Initialising pipeline…');
  resetSteps();
  document.getElementById('page-info').textContent = '';

  showSection('upload');
}

/* ── Helpers ────────────────────────────────────────────────────────────────── */
function formatBytes(bytes) {
  if (bytes < 1024)         return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
