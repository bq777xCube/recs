// ===== script.js (CPU-only • 1 epoch • bias pre-initialization to avoid 3.5 clustering) =====

let model = null;
let isTraining = false;
let globalMeanRating = 3.5;

// Reused tiny tensors for predict()
let predUserVar = null;
let predMovieVar = null;

// Dense ID maps (original ID -> dense index starting at 1)
let userIdToIdx = null;
let movieIdToIdx = null;
let userCount = 0;
let movieCount = 0;

// URL flags
const params    = new URLSearchParams(location.search);
const FAST_MODE = params.has('fast');     // ?fast=1 -> subsample ~60k for quicker single-epoch
const NO_CACHE  = params.has('nocache');  // ?nocache=1 -> ignore cache this session

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
  const list = (Array.isArray(userIdsSorted) && userIdsSorted.length)
    ? userIdsSorted
    : Array.from({length: numUsers}, (_,i)=>i+1);
  for (const uid of list) {
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

// ---------- Progress bar (throttled) ----------
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
  fill.style.transition = 'width 200ms linear';

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
  if (now - _lastUiUpdate < 180) return; // ~5-6 fps
  _lastUiUpdate = now;
  ensureProgressBar();
  const fill = document.getElementById('train-progress-fill');
  const text = document.getElementById('train-progress-text');
  const p = Math.max(0, Math.min(100, pct));
  if (fill) fill.style.width = `${p}%`;
  if (text) text.textContent = `${p.toFixed(1)}%${label ? ' • ' + label : ''}`;
}

// ---------- Backend: HARD-LOCK CPU ----------
async function ensureCpuOnly() {
  try { tf.removeBackend('wasm'); } catch {}
  try { tf.removeBackend('webgl'); } catch {}
  await tf.setBackend('cpu');
  await tf.ready();
  const active = tf.getBackend();
  updateStatus(`TensorFlow.js backend: ${active}`);
  if (active !== 'cpu') throw new Error(`Expected CPU backend, got "${active}"`);
}

// ---------- Dense ID mapping ----------
function buildDenseIdMaps() {
  const users = (Array.isArray(userIdsSorted) && userIdsSorted.length)
    ? userIdsSorted.slice()
    : Array.from(new Set(ratings.map(r => r.userId))).sort((a,b)=>a-b);
  userCount = users.length;
  userIdToIdx = new Map();
  users.forEach((uid, i) => userIdToIdx.set(uid, i + 1)); // 1-based

  movieCount = movies.length;
  movieIdToIdx = new Map();
  movies.forEach((m, i) => movieIdToIdx.set(m.id, i + 1)); // 1-based
}

// ---------- Model (MF + biases + L2) ----------
function createModel(uCount, mCount, latentDim = 8) {
  const l2 = tf.regularizers.l2({ l2: 1e-6 });
  const glorot = tf.initializers.glorotUniform({ seed: 1337 });

  const userInput  = tf.input({ shape: [1], dtype: 'int32', name: 'userInput' });
  const movieInput = tf.input({ shape: [1], dtype: 'int32', name: 'movieInput' });

  const userEmb = tf.layers.embedding({
    inputDim: uCount + 1, outputDim: latentDim,
    embeddingsInitializer: glorot, embeddingsRegularizer: l2, name: 'userEmbedding'
  }).apply(userInput);

  const movieEmb = tf.layers.embedding({
    inputDim: mCount + 1, outputDim: latentDim,
    embeddingsInitializer: glorot, embeddingsRegularizer: l2, name: 'movieEmbedding'
  }).apply(movieInput);

  const u = tf.layers.flatten().apply(userEmb);
  const v = tf.layers.flatten().apply(movieEmb);

  const ub = tf.layers.flatten().apply(
    tf.layers.embedding({
      inputDim: uCount + 1, outputDim: 1,
      embeddingsInitializer: 'zeros', embeddingsRegularizer: l2, name: 'userBias'
    }).apply(userInput)
  );
  const vb = tf.layers.flatten().apply(
    tf.layers.embedding({
      inputDim: mCount + 1, outputDim: 1,
      embeddingsInitializer: 'zeros', embeddingsRegularizer: l2, name: 'movieBias'
    }).apply(movieInput)
  );

  const dot = tf.layers.dot({ axes: 1 }).apply([u, v]);
  const sum = tf.layers.add().apply([dot, ub, vb]);
  return tf.model({ inputs: [userInput, movieInput], outputs: sum });
}

// ---------- Cache ----------
function dataSignature() {
  // include shape + 1-epoch marker so cache invalidates when dataset changes
  return `cpu-dense-e1-biasinit-u${userCount}-m${movieCount}-n${ratings.length}-k8`;
}
async function tryLoadCachedModel(sig) {
  if (NO_CACHE) return false;
  try {
    model = await tf.loadLayersModel(`indexeddb://mf-${sig}`);
    const mean = localStorage.getItem(`mf-mean-${sig}`);
    if (mean) globalMeanRating = parseFloat(mean);
    updateStatus('Loaded trained model from cache.');
    predUserVar  = tf.variable(tf.tensor2d([[1]], [1, 1], 'int32'));
    predMovieVar = tf.variable(tf.tensor2d([[1]], [1, 1], 'int32'));
    document.getElementById('predict-btn').disabled = false;
    return true;
  } catch { return false; }
}
async function saveCachedModel(sig) {
  try {
    await model.save(`indexeddb://mf-${sig}`);
    localStorage.setItem(`mf-mean-${sig}`, String(globalMeanRating));
  } catch (e) { console.warn('Cache save failed:', e); }
}

// Optional subsample for quicker single-epoch (first run)
function reservoirSample(arr, target) {
  if (arr.length <= target) return arr;
  const k = target;
  const res = new Array(k);
  for (let i = 0; i < k; i++) res[i] = arr[i];
  for (let i = k; i < arr.length; i++) {
    const j = Math.floor(Math.random() * (i + 1));
    if (j < k) res[j] = arr[i];
  }
  return res;
}
function maybeSubsample(arr) {
  if (FAST_MODE) return reservoirSample(arr, 60000);
  return arr;
}

// ---------- Bias pre-initialization from data ----------
function computeBiasInitializers(trainRatings) {
  // We compute:
  //   globalMean = mean(r)
  //   userBias[u] = mean(r_u) - globalMean
  //   movieBias[i]= mean(r_i) - globalMean
  // and set embedding weights for userBias/movieBias to these values.
  const uSum = new Float32Array(userCount + 1);
  const uCnt = new Uint32Array(userCount + 1);
  const mSum = new Float32Array(movieCount + 1);
  const mCnt = new Uint32Array(movieCount + 1);

  let sum = 0;
  for (const r of trainRatings) {
    const ui = (userIdToIdx.get(r.userId) || 1);
    const mi = (movieIdToIdx.get(r.movieId) || 1);
    const y = r.rating;
    sum += y;
    uSum[ui] += y; uCnt[ui] += 1;
    mSum[mi] += y; mCnt[mi] += 1;
  }
  const N = trainRatings.length;
  const mu = sum / N;

  const uBias = new Float32Array(userCount + 1);
  const mBias = new Float32Array(movieCount + 1);
  for (let u = 1; u <= userCount; u++) {
    uBias[u] = uCnt[u] ? (uSum[u] / uCnt[u]) - mu : 0;
  }
  for (let m = 1; m <= movieCount; m++) {
    mBias[m] = mCnt[m] ? (mSum[m] / mCnt[m]) - mu : 0;
  }
  return { mu, uBias, mBias };
}

// ---------- Training (1 EPOCH ONLY, with bias init) ----------
async function trainModel() {
  const btn = document.getElementById('predict-btn');
  try {
    isTraining = true;
    if (btn) btn.disabled = true;

    ensureProgressBar();
    setProgress(0, 'Preparing…');

    // Build dense maps
    buildDenseIdMaps();

    // Subsample if requested, then compute bias initializers on THAT set
    const trainRatings = maybeSubsample(ratings);
    const { mu, uBias, mBias } = computeBiasInitializers(trainRatings);
    globalMeanRating = mu;

    // Create model
    model = createModel(userCount, movieCount, 8);

    // Load bias initializers into embedding layers BEFORE training
    const userBiasLayer  = model.getLayer('userBias');
    const movieBiasLayer = model.getLayer('movieBias');
    const uBiasTensor = tf.tensor2d(uBias, [userCount + 1, 1], 'float32');
    const mBiasTensor = tf.tensor2d(mBias, [movieCount + 1, 1], 'float32');
    userBiasLayer.setWeights([uBiasTensor]);
    movieBiasLayer.setWeights([mBiasTensor]);
    uBiasTensor.dispose(); mBiasTensor.dispose();

    // Slightly higher LR so a single pass moves embeddings meaningfully
    model.compile({ optimizer: tf.train.adam(0.008), loss: 'meanSquaredError' });

    // Build training tensors (centered around mu)
    const N = trainRatings.length;
    const uids = new Int32Array(N);
    const mids = new Int32Array(N);
    const y    = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const r = trainRatings[i];
      uids[i] = (userIdToIdx.get(r.userId) || 1) | 0;
      mids[i] = (movieIdToIdx.get(r.movieId) || 1) | 0;
      y[i] = r.rating - mu;
    }

    const Xuser  = tf.tensor2d(uids, [N, 1], 'int32');
    const Xmovie = tf.tensor2d(mids, [N, 1], 'int32');
    const Y      = tf.tensor2d(y,    [N, 1], 'float32');

    // Big batches to reduce steps
    const threads = Math.max(2, Math.min(8, (navigator.hardwareConcurrency || 4)));
    const BATCH  = threads >= 8 ? 2048 : 1536;
    const EPOCHS = 1;
    const stepsPerEpoch = Math.ceil(N / BATCH);

    let epochIdx = 0;
    const BATCH_UI_STEP = 4; // update UI every 4 batches to keep DOM cheap
    const progCb = {
      onTrainBegin: () => setProgress(0, 'Starting…'),
      onEpochBegin: (e) => { epochIdx = e; },
      onBatchEnd: (b) => {
        if ((b + 1) % BATCH_UI_STEP !== 0) return;
        const overall = ((epochIdx + (b + 1) / stepsPerEpoch) / EPOCHS) * 100;
        setProgress(overall, `Epoch ${epochIdx + 1}/${EPOCHS}`);
      },
      onEpochEnd: (e, logs) => {
        setProgress(100, `Epoch ${e + 1}/${EPOCHS} — loss ${logs.loss.toFixed(4)}`);
        updateStatus(`Epoch ${e + 1}: loss ${logs.loss.toFixed(4)} (bias-initialized)`);
      },
      onTrainEnd: () => setProgress(100, 'Finalizing…'),
    };

    updateStatus(
      `Training ${N.toLocaleString()} ratings — batch ${BATCH}, 1 epoch (bias-initialized)…`
    );

    await model.fit([Xuser, Xmovie], Y, {
      epochs: EPOCHS,
      batchSize: BATCH,
      shuffle: true,
      validationSplit: 0,
      callbacks: [progCb]
    });

    // Prep predict vars (dense indices)
    predUserVar  = tf.variable(tf.tensor2d([[1]], [1, 1], 'int32'));
    predMovieVar = tf.variable(tf.tensor2d([[1]], [1, 1], 'int32'));

    tf.dispose([Xuser, Xmovie, Y]);
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

  const uIdx = userIdToIdx?.get(userId);
  const mIdx = movieIdToIdx?.get(movieId);
  if (!uIdx || !mIdx) {
    updateResult('Selected user/movie not found in training index.', 'low');
    return;
  }

  try {
    if (!predUserVar || !predMovieVar) {
      predUserVar  = tf.variable(tf.tensor2d([[uIdx]], [1, 1], 'int32'));
      predMovieVar = tf.variable(tf.tensor2d([[mIdx]], [1, 1], 'int32'));
    } else {
      predUserVar.assign(tf.tensor2d([[uIdx]], [1, 1], 'int32'));
      predMovieVar.assign(tf.tensor2d([[mIdx]], [1, 1], 'int32'));
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
    await ensureCpuOnly();               // lock to CPU (no WASM/WebGL)

    updateStatus('Loading MovieLens data…');
    await loadData();                    // from data.js
    populateUserDropdown();
    populateMovieDropdown();

    // Build mappings now (needed for cache + predictions)
    buildDenseIdMaps();

    // Try cache first (instant after one successful training)
    const sig = dataSignature();
    if (await tryLoadCachedModel(sig)) {
      updateStatus('Model loaded from cache. Ready for predictions.');
      return;
    }

    updateStatus('Data loaded. Starting 1-epoch training on CPU (bias-initialized)…');
    await trainModel();
  } catch (err) {
    console.error('Initialization error:', err);
    updateStatus('Error initializing application: ' + err.message, true);
  }
};
