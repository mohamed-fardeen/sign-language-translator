// inference.js -- ONNX Runtime Web wrapper for the browser-side model.

const POSE_DIM = 99, HAND_DIM = 63, FACE_DIM = 120, T_FRAMES = 64;

window.Inference = {
  session: null,
  backend: 'wasm',

  configure: function (backend) {
    if (window.ort && window.ort.env) {
      window.ort.env.wasm.numThreads = navigator.hardwareConcurrency || 2;
    }
    this.backend = backend;
  },

  load: async function (url) {
    if (!window.ort) throw new Error('onnxruntime-web not loaded');
    const opts = { executionProviders: [this.backend] };
    this.session = await window.ort.InferenceSession.create(url, opts);
    return this.session;
  },

  isLoaded: function () {
    return this.session !== null;
  },

  run: async function (clip) {
    if (!this.session) throw new Error('Model not loaded');
    const pose = new window.ort.Tensor('float32', flattenClip(clip.pose, POSE_DIM), [1, T_FRAMES, POSE_DIM]);
    const lh   = new window.ort.Tensor('float32', flattenClip(clip.lh,   HAND_DIM), [1, T_FRAMES, HAND_DIM]);
    const rh   = new window.ort.Tensor('float32', flattenClip(clip.rh,   HAND_DIM), [1, T_FRAMES, HAND_DIM]);
    const face = new window.ort.Tensor('float32', flattenClip(clip.face, FACE_DIM), [1, T_FRAMES, FACE_DIM]);
    const feeds = { pose, lh, rh, face };
    const out = await this.session.run(feeds);
    const logits = out.logits || out[Object.keys(out)[0]];
    const data = logits.data;
    const shape = logits.dims;
    const T = shape[1] || T_FRAMES;
    const V = shape[2] || (data.length / T);
    const preds = [];
    const confidences = [];
    for (let t = 0; t < T; t++) {
      let bestIdx = 0, bestVal = -Infinity;
      for (let v = 0; v < V; v++) {
        const val = data[t * V + v];
        if (val > bestVal) { bestVal = val; bestIdx = v; }
      }
      preds.push(bestIdx);
      confidences.push(softmaxVal(data, t, V, bestIdx));
    }
    const collapsed = collapseCTC(preds, 0);
    const tokenMeans = computeTopK(data, T, V, 5);
    return { tokens: collapsed, topK: tokenMeans };
  },
};

function flattenClip(arr2d, dim) {
  const out = new Float32Array(T_FRAMES * dim);
  for (let t = 0; t < T_FRAMES; t++) {
    const row = arr2d[t];
    for (let d = 0; d < dim; d++) out[t * dim + d] = row[d] || 0;
  }
  return out;
}

function collapseCTC(seq, blank) {
  const out = [];
  let prev = -1;
  for (const i of seq) {
    if (i !== blank && i !== prev) out.push(i);
    prev = i;
  }
  return out;
}

function softmaxVal(data, t, V, idx) {
  let max = -Infinity;
  for (let v = 0; v < V; v++) {
    const x = data[t * V + v];
    if (x > max) max = x;
  }
  let sum = 0;
  for (let v = 0; v < V; v++) sum += Math.exp(data[t * V + v] - max);
  return Math.exp(data[t * V + idx] - max) / sum;
}

function computeTopK(data, T, V, k) {
  const mean = new Float32Array(V);
  for (let t = 0; t < T; t++) {
    let max = -Infinity;
    for (let v = 0; v < V; v++) {
      const x = data[t * V + v];
      if (x > max) max = x;
    }
    let sum = 0;
    for (let v = 0; v < V; v++) sum += Math.exp(data[t * V + v] - max);
    for (let v = 0; v < V; v++) {
      mean[v] += Math.exp(data[t * V + v] - max) / sum;
    }
  }
  for (let v = 0; v < V; v++) mean[v] /= T;
  const order = Array.from(mean.keys()).sort((a, b) => mean[b] - mean[a]).slice(0, k);
  return order.map((v) => ({ id: v, prob: mean[v] }));
}
