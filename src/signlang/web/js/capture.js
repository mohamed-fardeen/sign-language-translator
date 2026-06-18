// capture.js -- webcam + MediaPipe Holistic landmark extraction.
// MVP scope: pose + 2 hands only. Face extraction intentionally removed
// to speed up CPU extraction; see ARCHITECTURE.md.

const POSE_DIM = 33 * 3;       // 99
const HAND_DIM = 21 * 3;       // 63
const CLIP_FRAMES = 64;
const TARGET_FPS = 30;

const state = {
  stream: null,
  holistic: null,
  recording: false,
  frames: [],
  lastFrameTs: 0,
  resolve: null,
};

window.Capture = {
  start: async function (videoEl) {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false,
    });
    videoEl.srcObject = state.stream;
    await new Promise((res) => { videoEl.onloadedmetadata = res; });

    state.holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`,
    });
    state.holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      refineFaceLandmarks: false,
    });
    state.holistic.onResults(onHolisticResults);

    const overlay = document.getElementById('overlay');
    if (overlay) {
      overlay.width = videoEl.videoWidth || 640;
      overlay.height = videoEl.videoHeight || 480;
    }
  },

  stop: function (videoEl) {
    if (state.stream) {
      state.stream.getTracks().forEach((t) => t.stop());
      state.stream = null;
    }
    videoEl.srcObject = null;
    state.holistic = null;
    state.recording = false;
  },

  // Returns a promise that resolves with a finalised clip when the
  // auto-stop condition is reached (>= CLIP_FRAMES * 2 frames captured).
  recordStart: function (videoEl) {
    state.frames = [];
    state.recording = true;
    state.lastFrameTs = performance.now();
    return new Promise((resolve) => {
      state.resolve = resolve;
      processFrame(videoEl);
    });
  },
};

const videoEl = document.getElementById('webcam');

function onHolisticResults(results) {
  const overlay = document.getElementById('overlay');
  if (overlay) {
    const ctx = overlay.getContext('2d');
    ctx.save();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (results.poseLandmarks) drawLandmarks(ctx, results.poseLandmarks, [0, 255, 255], overlay);
    if (results.leftHandLandmarks) drawLandmarks(ctx, results.leftHandLandmarks, [255, 0, 0], overlay);
    if (results.rightHandLandmarks) drawLandmarks(ctx, results.rightHandLandmarks, [0, 255, 0], overlay);
    ctx.restore();
  }

  if (!state.recording) return;
  const now = performance.now();
  if (now - state.lastFrameTs < 1000 / TARGET_FPS) return;
  state.lastFrameTs = now;
  state.frames.push({
    pose: flatten(results.poseLandmarks, 33),
    lh: flatten(results.leftHandLandmarks, 21),
    rh: flatten(results.rightHandLandmarks, 21),
    mask: 1,
  });
}

function flatten(lms, n) {
  const out = new Float32Array(n * 3);
  if (!lms) return out;
  for (let i = 0; i < Math.min(n, lms.length); i++) {
    out[i * 3 + 0] = lms[i].x;
    out[i * 3 + 1] = lms[i].y;
    out[i * 3 + 2] = lms[i].z;
  }
  return out;
}

function drawLandmarks(ctx, lms, color, canvas) {
  ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
  const w = canvas.width, h = canvas.height;
  for (const lm of lms) {
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

async function processFrame(videoEl) {
  if (!state.recording || !state.holistic) return;
  try {
    await state.holistic.send({ image: videoEl });
  } catch (e) { console.warn('holistic.send failed', e); }
  if (state.frames.length >= CLIP_FRAMES * 2) {
    state.recording = false;
    const clip = finalizeClip();
    if (state.resolve) {
      const r = state.resolve;
      state.resolve = null;
      r(clip);
    }
    return;
  }
  setTimeout(() => processFrame(videoEl), 1000 / TARGET_FPS);
}

function finalizeClip() {
  const frames = state.frames;
  if (frames.length === 0) return null;
  const n = Math.min(frames.length, CLIP_FRAMES);
  const start = Math.max(0, Math.floor((frames.length - n) / 2));
  const sel = frames.slice(start, start + n);
  const pose = [], lh = [], rh = [], mask = [];
  for (let i = 0; i < sel.length; i++) {
    pose.push(Array.from(sel[i].pose));
    lh.push(Array.from(sel[i].lh));
    rh.push(Array.from(sel[i].rh));
    mask.push(true);
  }
  for (let i = sel.length; i < CLIP_FRAMES; i++) {
    pose.push(new Array(POSE_DIM).fill(0));
    lh.push(new Array(HAND_DIM).fill(0));
    rh.push(new Array(HAND_DIM).fill(0));
    mask.push(false);
  }
  return { pose, lh, rh, mask };
}
