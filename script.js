// ===== script.js (GH Pages optimized: WebGL/WASM + caching + progress bar) =====

let model = null;
let isTraining = false;
let globalMeanRating = 3.5;

// Reused tiny tensors for fast predict()
let predUserVar = null;
let predMovieVar = null;

// URL flags
const params = new URLSearchParams(location.search);
const FAST_MODE = params.has('fast');   // ?fast=1 -> subsample to ~60k ratings for quicker train
const NO_CACHE  = params.has('nocache'); // ?nocache=1 -> ignore cached model

// ---------- UI ----------
function updateStatus(message, isError = false) {
  const el = document.getElementById('status');
  if (!el) return;
  el.textContent = message;
  el.style.borderLeftColor = isError ? '#e74c3c' : '#3498db';
  el.style.background = isError ? '#fdedec' : '#f8f9fa';
}
function updateResult(message, className = '') {
  const el = document.getElementById('result');
  if (!el) return;
  el.innerHTML = message;
  el.className = `result ${className}`;
}
function populateUserDropdown() {
  const sel = document.getElementById('user-select');
  sel.innerHTML = '';
  for (const uid of userIdsSorted) { // robust to gaps
    const opt = document.createElement('option');
    opt.value = uid;
    opt.textContent = `User ${uid}`;
    sel.appendChild(opt);
  }
  sel.disabled = false;
}
function populateMovieDropdown() {
  const sel = document.getElementById('movie-select');
  sel.innerHTML = '';
  for (const m of movies) {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.year ? `${m.title} (${m.year})` : m.title;
    sel.appendChild(opt);
  }
  sel.disabled = false;
}

// ---------- Progress bar (throttled to keep UI cheap) ----------
function ensureProgressBar() {
  let wrap = document.getElementById('train-progress-wrap');
  if (wrap) return wrap;
  const status = document.getElementById('status');
  if (!status) return null;

  wrap = document.createElement('div');
  wrap.id = 'train-progress-wrap';
  wrap.style.margin = '12px 0 6px';

  const bar = document.createElement('div');
  bar.style.width = '100%';
  bar.style.height = '10px';
  bar.style.background = '#eef3f7';
  bar.style.border = '1px solid #dde5ee';
  bar.style.borderRadius = '8px';
  bar.style.overflow = 'hidden';

  const fill = document.createElement('div');
  fill.id = 'train-progress-fill';
  fill.style.height = '100%';
  fill.style.width = '0%';
  fill.style.background = 'linear-gradient(90deg, #60a5fa, #3b82f6)';
  fill.style.transition = 'width 120ms linear';

  const text = document.createElement('div');
  text.id = 'train-progress-text';
  text.style.fontSize = '12px';
  text.style.color = '#55657a';
  text.style.marginTop = '6px';
  text.textContent = '0%';

  bar.appendChild(fill);
  wrap.appendChild(bar);
  wrap.appendChild(text);
  status.parentElement.insertBefore(wrap, status.nextSibling);
  return wrap;
}
let _lastUiUpdate = 0;
function setProgress(pct, label) {
  const now = performance.now();
  if (now - _lastUiUpdate < 120) return; // throttle to ~8 fps
  _lastUiUpdate = now;

  ensureProgressBar();
  const fill = document.getElementById('train-progress-fill');
  const text = document.getElementById('train-progress-text');
  const p = Math.max(0, Math.min(100, pct));
  if (fill) fill.style.width = `${p}%`;
  if (text) text.textContent = `${p.toFixed(1)}%${label ? ' • ' + label : ''}`;
}

// ---------- Backend selection (WebGL → WASM) ----------
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src; s.async = true;
    s.onload = resolve; s.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(s);
  });
}
async function ensureBackend() {
  try { await tf.setBackend('webgl'); } catch (_) {}
  await tf.ready();

  if (tf.getBackend() === 'webgl') {
    // Mixed precision for throughput
    try { tf.env().set('WEBGL_FORCE_F16_TEXTURES', true); } catch (_) {}
    try { tf.env().set('WEBGL_PACK', true); } catch (_) {}
    updateStatus('TensorFlow.js backend: webgl (GPU)');
    return 'webgl';
  }

  // WASM fallback (fast on many GH Pages clients)
  updateStatus('WebGL not available — loading WASM backend…');
  await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@latest/dist/tf-backend-wasm.js');
  if (tf.wasm && tf.wasm.setWasmPaths) {
    tf.wasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@latest/dist/');
  }
  const threads = Math.min(4, (navigator.hardwareConcurrency || 4));
  try { tf.env().set('WASM_NUM_THREADS', threads); } catch (_) {}
  await tf.setBackend('wasm');
  await tf.ready();
  updateStatus(`TensorFlow.js backend: wasm (${threads} threads)`);
  return 'wasm';
}

// ---------- Model ----------
function createModel(userDim, movieDim, latentDim = 16) {
  const l2 = tf.regularizers.l2({ l2: 1e-6 });
  const glorot = tf.initializers.glorotUniform({ seed: 1337 });

  const userInput  = tf.input({ shape: [1], dtype: 'int32', name: 'userInput' });
  const movieInput = tf.input({ shape: [1], dtype: 'int32', name: 'movieInput' });

  const userEmb = tf.layers.embedding({
    inputDim: userDim + 1, outputDim: latentDim,
    embeddingsInitializer: glorot, embeddingsRegularizer: l2, name: 'userEmbedding'
  }).apply(userInput);

  const movieEmb = tf.layers.embedding({
    inputDim: movieDim + 1, outputDim: latentDim,
    embeddingsInitializer: glorot, embeddingsRegularizer: l2, name: 'movieEmbedding'
  }).apply(movieInput);

  const u = tf.layers.flatten().apply(userEmb);
  const v = tf.layers.flatten().apply(movieEmb);

  const ub = tf.layers.flatten().apply(
    tf.layers.embedding({
      inputDim: userDim + 1, outputDim: 1,
      embeddingsInitializer: 'zeros', embeddingsRegularizer: l2, name: 'userBias'
    }).apply(userInput)
  );
  const vb = tf.layers.flatten().apply(
    tf.layers.embedding({
      inputDim: movieDim + 1, outputDim: 1,
      embeddingsInitializer: 'zeros', embeddingsRegularizer: l2, name: 'movieBias'
    }).apply(movieInput)
  );

  const dot = tf.layers.dot({ axes: 1 }).apply([u, v]);
  const sum = tf.layers.add().apply([dot, ub, vb]);
  const out = tf.layers.activation({ activation: 'linear' }).apply(sum);

  return tf.model({ inputs: [userInput, movieInput], outputs: out });
}

// ---------- Cache helpers (skip retraining on GH Pages) ----------
function dataSignature() {
  // changes if dataset size or id ranges change
  return `v3-u${maxUserId}-m${maxMovieId}-n${ratings.length}`;
}
async function tryLoadCachedModel(sig) {
  if (NO_CACHE) return false;
  try {
    model = await tf.loadLayersModel(`indexeddb://mf-${sig}`);
    const mean = localStorage.getItem(`mf-mean-${sig}`);
    if (mean) globalMeanRating = parseFloat(mean);
    updateStatus('Loaded trained model from cache.');
    // Pre-alloc predict vars after loading
    predUserVar  = tf.variable(tf.tensor2d([[0]], [1, 1], 'int32'));
    predMovieVar = tf.variable(tf.tensor2d([[0]], [1, 1], 'int32'));
    document.getElementById('predict-btn').disabled = false;
    return true;
  } catch (_) {
    return false;
  }
}
async function saveCachedModel(sig) {
  try {
    await model.save(`indexeddb://mf-${sig}`);
    localStorage.setItem(`mf-mean-${sig}`, String(globalMeanRating));
  } catch (e) {
    // Ignore cache errors
    console.warn('Cache save failed:', e);
  }
}

// Optional quick mode: reservoir sample to ~60k ratings
function maybeSubsample(arr, target = 60000) {
  if (!FAST_MODE || arr.length <= target) return arr;
  const res = new Array(Math.min(target, arr.length));
  for (let i = 0; i < res.length; i++) res[i] = arr[i];
  for (let i = res.length; i < arr.length; i++) {
    const j = Math.floor(Math.random() * (i + 1));
    if (j < res.length) res[j] = arr[i];
  }
  return res;
}

// ---------- Training ----------
function earlyStopping(pat = 1) {
  let best = Infinity, wait = 0;
  return { onEpochEnd: (_, logs) => {
    const cur = Number.isFinite(logs.val_loss) ? logs.val_loss : logs.loss;
    if (cur < best - 1e-4) { best = cur; wait = 0; }
    else if (++wait >= pat) { model.stopTraining = true; }
  }};
}

async function trainModel() {
  const btn = document.getElementById('predict-btn');
  try {
    isTraining = true;
    if (btn) btn.disabled = true;

    ensureProgressBar();
    setProgress(0, 'Preparing…');

    const backend = tf.getBackend();
    const BATCH  = backend === 'webgl' ? 4096 : backend === 'wasm' ? 1024 : 256;
    const EPOCHS = backend === 'webgl' ? 6    : backend === 'wasm' ? 8    : 4;
    const VAL_SPLIT = 0.0; // fastest; rely on early stopping with training loss

    model = createModel(maxUserId, maxMovieId, 16);
    model.compile({ optimizer: tf.train.adam(0.003), loss: 'meanSquaredError', metrics: ['mae'] });

    // Subsample if FAST mode
    const trainRatings = maybeSubsample(ratings);
    const N = trainRatings.length;

    const uids = new Int32Array(N);
    const mids = new Int32Array(N);
    const y    = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const r = trainRatings[i];
      uids[i] = r.userId | 0;
      mids[i] = r.movieId | 0;
      y[i] = r.rating;
    }

    // Center targets
    let s = 0; for (let i = 0; i < N; i++) s += y[i];
    globalMeanRating = s / N;
    for (let i = 0; i < N; i++) y[i] -= globalMeanRating;

    const Xuser  = tf.tensor2d(uids, [N, 1], 'int32');
    const Xmovie = tf.tensor2d(mids, [N, 1], 'int32');
    const Y      = tf.tensor2d(y,    [N, 1], 'float32');

    const stepsPerEpoch = Math.ceil(N / BATCH);
    let epochIdx = 0;

    const progCb = {
      onTrainBegin: () => setProgress(0, 'Starting…'),
      onEpochBegin: (e) => { epochIdx = e; },
      onBatchEnd: (b) => {
        const overall = ((epochIdx + (b + 1) / stepsPerEpoch) / EPOCHS) * 100;
        setProgress(overall, `Epoch ${epochIdx + 1}/${EPOCHS}`);
      },
      onEpochEnd: (e, logs) => {
        const pct = ((e + 1) / EPOCHS) * 100;
        setProgress(pct, `Epoch ${e + 1}/${EPOCHS} — loss ${logs.loss.toFixed(4)}`);
        updateStatus(`Epoch ${e + 1}: loss ${logs.loss.toFixed(4)} • mae ${logs.mae?.toFixed(4)}`);
      },
      onTrainEnd: () => setProgress(100, 'Finalizing…'),
    };

    updateStatus(
      `Training ${N.toLocaleString()} ratings — batch ${BATCH}, up to ${EPOCHS} epochs${FAST_MODE ? ' (FAST mode)' : ''}…`
    );

    await model.fit([Xuser, Xmovie], Y, {
      epochs: EPOCHS,
      batchSize: BATCH,
      shuffle: true,
      validationSplit: VAL_SPLIT,
      callbacks: [progCb, earlyStopping(1)]
    });

    // Pre-alloc predict vars
    predUserVar  = tf.variable(tf.tensor2d([[0]], [1, 1], 'int32'));
    predMovieVar = tf.variable(tf.tensor2d([[0]], [1, 1], 'int32'));

    tf.dispose([Xuser, Xmovie, Y]);

    // Cache for next visits (instant load)
    await saveCachedModel(dataSignature());

    updateStatus('Model ready. Select a user & movie, then click Predict.');
    if (btn) btn.disabled = false;
    isTraining = false;
  } catch (err) {
    console.error('Training error:', err);
    updateStatus('Error training model: ' + err.message, true);
    isTraining = false;
    if (btn) btn.disabled = false;
  }
}

// ---------- Prediction ----------
async function predictRating() {
  if (isTraining) {
    updateResult('Model is still training. Please wait…', 'medium');
    return;
  }
  const userId = parseInt(document.getElementById('user-select').value, 10);
  const movieId = parseInt(document.getElementById('movie-select').value, 10);
  if (!userId || !movieId) {
    updateResult('Please select both a user and a movie.', 'medium');
    return;
  }

  try {
    if (!predUserVar || !predMovieVar) {
      predUserVar  = tf.variable(tf.tensor2d([[userId]], [1, 1], 'int32'));
      predMovieVar = tf.variable(tf.tensor2d([[movieId]], [1, 1], 'int32'));
    } else {
      predUserVar.assign(tf.tensor2d([[userId]], [1, 1], 'int32'));
      predMovieVar.assign(tf.tensor2d([[movieId]], [1, 1], 'int32'));
    }

    const centered = await tf.tidy(() => model.predict([predUserVar, predMovieVar])).data();
    const predicted = Math.min(5, Math.max(1, centered[0] + globalMeanRating));

    const movie = movies.find(x => x.id === movieId);
    const title = movie ? (movie.year ? `${movie.title} (${movie.year})` : movie.title) : `Movie ${movieId}`;

    let cls = 'medium';
    if (predicted >= 4) cls = 'high';
    else if (predicted <= 2) cls = 'low';

    updateResult(
      `Predicted rating for User ${userId} on “<strong>${title}</strong>”: <strong>${predicted.toFixed(2)}</strong>/5`,
      cls
    );
  } catch (err) {
    console.error('Prediction error:', err);
    updateResult('Error making prediction: ' + err.message, 'low');
  }
}

// ---------- App init ----------
window.onload = async () => {
  try {
    updateStatus('Initializing TensorFlow.js…');
    const backend = await ensureBackend(); // webgl → wasm
    updateStatus('Loading MovieLens data…');
    await loadData(); // from data.js
    populateUserDropdown();
    populateMovieDropdown();

    // Try cache first (instant start on repeat visits)
    const sig = dataSignature();
    const cached = await tryLoadCachedModel(sig);

    if (!cached) {
      updateStatus('Data loaded. Starting training…');
      await trainModel();
    } else {
      updateStatus('Model loaded from cache. Ready for predictions.');
    }
  } catch (err) {
    console.error('Initialization error:', err);
    updateStatus('Error initializing application: ' + err.message, true);
  }
};
