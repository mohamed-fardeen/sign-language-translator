// app.js -- UI glue: button handlers, model load, run prediction.

const MODEL_URL = (window.SIGNLANG_MODEL_URL || '/artifacts/exported/browser/model_int8.onnx');
const VOCAB_URL = (window.SIGNLANG_VOCAB_URL || '/artifacts/exported/browser/vocab.json');
const API_BASE = (window.SIGNLANG_API_BASE || '');

const els = {
  start: document.getElementById('startBtn'),
  record: document.getElementById('recordBtn'),
  predict: document.getElementById('predictBtn'),
  stop: document.getElementById('stopBtn'),
  status: document.getElementById('status'),
  top: document.getElementById('topGloss'),
  topK: document.getElementById('topK'),
  latency: document.getElementById('latency'),
  modelVer: document.getElementById('modelVer'),
  video: document.getElementById('webcam'),
};

let vocab = null;
let lastClip = null;
let recordingPromise = null;

function setStatus(msg) { els.status.textContent = msg; }

async function init() {
  try {
    const res = await fetch(VOCAB_URL);
    if (res.ok) vocab = await res.json();
  } catch (_) { /* vocab may not be served in dev */ }
  try {
    await window.Inference.load(MODEL_URL);
    els.modelVer.textContent = `browser:${window.Inference.backend}`;
  } catch (_) {
    els.modelVer.textContent = 'browser:not-loaded';
  }
}

els.start.onclick = async () => {
  setStatus('Starting camera...');
  await window.Capture.start(els.video);
  els.start.disabled = true;
  els.record.disabled = false;
  els.stop.disabled = false;
  setStatus('Camera ready. Click Record to capture a clip (~2-3 s).');
  await init();
};

els.stop.onclick = () => {
  window.Capture.stop(els.video);
  els.start.disabled = false;
  els.record.disabled = true;
  els.predict.disabled = true;
  els.stop.disabled = true;
  setStatus('Stopped.');
};

els.record.onclick = async () => {
  els.record.disabled = true;
  els.predict.disabled = true;
  setStatus('Recording... (~2-3 s)');
  recordingPromise = window.Capture.recordStart(els.video);
  try {
    lastClip = await recordingPromise;
    setStatus('Clip captured. Click Predict.');
    els.predict.disabled = false;
  } catch (e) {
    setStatus('Recording failed: ' + e);
  } finally {
    els.record.disabled = false;
  }
};

els.predict.onclick = async () => {
  if (!lastClip) {
    setStatus('No clip captured.');
    return;
  }
  if (window.Inference.isLoaded()) {
    setStatus('Running browser inference...');
    const t0 = performance.now();
    const out = await window.Inference.run(lastClip);
    const latency = performance.now() - t0;
    renderResult(out, latency);
  } else if (API_BASE) {
    setStatus('Calling server...');
    const t0 = performance.now();
    const out = await callServer(lastClip);
    const latency = performance.now() - t0;
    renderServer(out, latency);
  } else {
    setStatus('No model available.');
  }
};

function labelFor(id) {
  if (!vocab || !vocab.id_to_gloss) return String(id);
  return vocab.id_to_gloss[String(id)] || String(id);
}

function renderResult(out, latencyMs) {
  const top = out.topK[0];
  els.top.textContent = labelFor(top.id);
  els.topK.innerHTML = '';
  for (const item of out.topK) {
    const li = document.createElement('li');
    li.textContent = `${labelFor(item.id)} -- ${(item.prob * 100).toFixed(1)}%`;
    els.topK.appendChild(li);
  }
  els.latency.textContent = latencyMs.toFixed(1);
  setStatus(`Done in ${latencyMs.toFixed(1)} ms (browser).`);
}

function renderServer(resp, latencyMs) {
  els.top.textContent = resp.gloss_label;
  els.topK.innerHTML = '';
  for (const item of resp.top_k) {
    const li = document.createElement('li');
    li.textContent = `${item.label} -- ${(item.prob * 100).toFixed(1)}%`;
    els.topK.appendChild(li);
  }
  els.latency.textContent = latencyMs.toFixed(1);
  els.modelVer.textContent = `server:${resp.model_version}`;
  setStatus(`Done in ${latencyMs.toFixed(1)} ms (server).`);
}

async function callServer(clip) {
  const tokRes = await fetch(`${API_BASE}/v1/auth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ device_key: 'browser-anon-device-key-1234' }),
  });
  if (!tokRes.ok) throw new Error(`token ${tokRes.status}`);
  const { access_token } = await tokRes.json();
  const res = await fetch(`${API_BASE}/v1/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${access_token}` },
    body: JSON.stringify({ clip, top_k: 5, beam_size: 1 }),
  });
  if (!res.ok) throw new Error(`server ${res.status}`);
  return res.json();
}
