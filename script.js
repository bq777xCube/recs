// ===== script.js (FAST + robust IDs + progress bar) =====

let model = null;
let isTraining = false;
let globalMeanRating = 3.5;

// Reusable tensors to avoid allocs on every predict()
let predUserVar = null;
let predMovieVar = null;

// ---------- UI helpers ----------
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
  // Use actual IDs (works even if not contiguous)
  for (const uid of userIdsSorted) {
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

// ---------- Progress bar (injected) ----------
function ensureProgressBar() {
  let wrap = document.getElementById('train-progress-wrap');
  if (wrap) return wrap;

  const status = document.getElementById('status');
  if (!status) return null;

  wrap = document.createElement('div');
  wrap.id = 'train-progress-wrap';
  wrap.style.margin = '12px 0 6px';

  const bar = document.createElement('div');
  bar.id = 'train-progress';
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
function setProgress(pct, label) {
  const wrap = ensureProgressBar();
  if (!wrap) return;
  const p = Math.max(0, Math.min(100, pct));
  const fill = document.getElementById('train-progress-fill');
  const text = document.getElementById('train-progress-text');
  if (fill) fill.style.width = `${p}%`;
  if (text) text.textContent = `${p.toFixed(1)}%${label ? ' • ' + label : ''}`;
}

// ---------- Backend (fast on GH Pages) ----------
async function ensureBackend() {
  try { await tf.setBackend('webgl'); } catch (_) {}
  await tf.ready();
  updateStatus(`TensorFlow.js backend: ${tf.getBackend()}`);
}

// ---------- Model (MF + biases + L2) ----------
function createModel(userDim, movieDim, latentDim = 24) {
  const l2 = tf.regularizers.l2({ l2: 1e-6 });
  const seed = 1337;
  const glorot = tf.initializers.glorotUniform({ seed });

  // Inputs MUST be int32 for embeddings
  const userInput = tf.input({ shape: [1], dtype: 'int32', name: 'userInput' });
  const movieInput = tf.input({ shape: [1], dtype: 'int32', name: 'movieInput' });

  const userEmb = tf.layers.embedding({
    inputDim: userDim + 1,
    outputDim: latentDim,
    embeddingsInitializer: glorot,
    embeddingsRegularizer: l2,
    name: 'userEmbedding'
  }).apply(userInput);

  const movieEmb = tf.layers.embedding({
    inputDim: movieDim + 1,
    outputDim: latentDim,
    embeddingsInitializer: glorot,
    embeddingsRegularizer: l2,
    name: 'movieEmbedding'
  }).apply(movieInput);

  const u = tf.layers.flatten().apply(userEmb);
  const v = tf.layers.flatten().apply(movieEmb);

  const ub = tf.layers.flatten().apply(
    tf.layers.embedding({
      inputDim: userDim + 1,
      outputDim: 1,
      embeddingsInitializer: 'zeros',
      embeddingsRegularizer: l2,
      name: 'userBias'
    }).apply(userInput)
  );
  const vb = tf.layers.flatten().apply(
    tf.layers.embedding({
      inputDim: movieDim + 1,
      outputDim: 1,
      embeddingsInitializer: 'zeros',
      embeddingsRegularizer: l2,
      name: 'movieBias'
    }).apply(movieInput)
  );

  const dot = tf.layers.dot({ axes: 1, name: 'dotUserMovie' }).apply([u, v]);
  const sum = tf.layers.add({ name: 'addBias' }).apply([dot, ub, vb]);
  const out = tf.layers.activation({ activation: 'linear', name: 'rating' }).apply(sum);

  return tf.model({ inputs: [userInput, movieInput], outputs: out, name: 'mf_recommender' });
}

// ---------- Training (big batch + early stop) ----------
function makeEarlyStopping(patience = 2) {
  let best = Infinity, wait = 0;
  return {
    onEpochEnd: (_, logs) => {
      const cur = Number.isFinite(logs.val_loss) ? logs.val_loss : logs.loss;
      if (cur < best - 1e-4) { best = cur; wait = 0; }
      else if (++wait >= patience) { model.stopTraining = true; }
    }
  };
}

async function trainModel() {
  const btn = document.getElementById('predict-btn');
  try {
    isTraining = true;
    if (btn) btn.disabled = true;

    ensureProgressBar();
    setProgress(0, 'Preparing…');

    // Use MAX IDs for embedding sizes (robust to gaps in IDs)
    model = createModel(maxUserId, maxMovieId, 24);
    model.compile({ optimizer: tf.train.adam(0.002), loss: 'meanSquaredError', metrics: ['mae'] });

    // Build tensors (int32 indices!)
    const N = ratings.length;
    const uids = new Int32Array(N);
    const mids = new Int32Array(N);
    const y = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const r = ratings[i];
      uids[i] = r.userId | 0;
      mids[i] = r.movieId | 0;
      y[i] = r.rating;
    }

    // Center targets around global mean
    let s = 0; for (let i = 0; i < N; i++) s += y[i];
    globalMeanRating = s / N;
    for (let i = 0; i < N; i++) y[i] -= globalMeanRating;

    const Xuser = tf.tensor2d(uids, [N, 1], 'int32');
    const Xmovie = tf.tensor2d(mids, [N, 1], 'int32');
    const Y = tf.tensor2d(y, [N, 1], 'float32');

    const gpu = tf.getBackend() === 'webgl';
    const batchSize = gpu ? 2048 : 256;
    const epochs = 12;
    const valSplit = 0.1;
    const trainSize = Math.floor(N * (1 - valSplit));
    const stepsPerEpoch = Math.ceil(trainSize / batchSize);

    let epochIdx = 0;
    const progCb = {
      onTrainBegin: () => setProgress(0, 'Starting…'),
      onEpochBegin: (e) => { epochIdx = e; },
      onBatchEnd: (b) => {
        const overall = ((epochIdx + (b + 1) / stepsPerEpoch) / epochs) * 100;
        setProgress(overall, `Epoch ${epochIdx + 1}/${epochs}`);
      },
      onEpochEnd: (e, logs) => {
        const pct = ((e + 1) / epochs) * 100;
        setProgress(pct, `Epoch ${e + 1}/${epochs} — val_loss ${logs.val_loss?.toFixed(4)}`);
        updateStatus(`Epoch ${e + 1}: loss ${logs.loss.toFixed(4)} • val_loss ${logs.val_loss?.toFixed(4)} • mae ${logs.mae?.toFixed(4)}`);
      },
      onTrainEnd: () => setProgress(100, 'Finalizing…'),
    };

    updateStatus(`Training ${N.toLocaleString()} ratings — batch ${batchSize}, up to ${epochs} epochs…`);
    await model.fit([Xuser, Xmovie], Y, {
      epochs, batchSize, shuffle: true, validationSplit: valSplit,
      callbacks: [progCb, makeEarlyStopping(2)]
    });

    // Preallocate predict vars
    predUserVar = tf.variable(tf.tensor2d([[0]], [1, 1], 'int32'));
    predMovieVar = tf.variable(tf.tensor2d([[0]], [1, 1], 'int32'));

    tf.dispose([Xuser, Xmovie, Y]);
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
      predUserVar = tf.variable(tf.tensor2d([[userId]], [1, 1], 'int32'));
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
      `Predicted rating for User ${userId} on "<strong>${title}</strong>": <strong>${predicted.toFixed(2)}</strong>/5`,
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
    await ensureBackend();
    updateStatus('Loading MovieLens data…');
    await loadData();
    populateUserDropdown();
    populateMovieDropdown();
    updateStatus('Data loaded. Starting training…');
    await trainModel();
  } catch (err) {
    console.error('Initialization error:', err);
    updateStatus('Error initializing application: ' + err.message, true);
  }
};
