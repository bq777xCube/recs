// ===== script.js (GH Pages optimized) =====
// Fast & precise Matrix Factorization in TF.js
// - WebGL backend (with fallback) and tf.ready() wait
// - User/Movie embeddings + per-entity biases
// - Target centering by global mean
// - L2 regularization + early stopping
// - Large batch on GPU, smaller on CPU

let model = null;
let isTraining = false;
let globalMeanRating = 3.5; // set from data during training

// ---------- UI helpers ----------
function updateStatus(message, isError = false) {
  const el = document.getElementById('status');
  if (!el) return;
  el.textContent = message;
  el.style.borderLeftColor = isError ? '#ef4444' : '#60a5fa';
  el.style.background = isError ? 'rgba(239,68,68,0.08)' : 'rgba(96,165,250,0.08)';
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
  for (let i = 1; i <= numUsers; i++) {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `User ${i}`;
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

// ---------- Backend selection (GH Pages) ----------
async function ensureBackend() {
  try {
    // Prefer WebGL on GitHub Pages for speed
    await tf.setBackend('webgl');
  } catch (_) {
    // Ignore — TF.js will fallback to CPU
  }
  await tf.ready();
  updateStatus(`TensorFlow.js backend: ${tf.getBackend()}`);
}

// ---------- Model ----------
function createModel(userCount, movieCount, latentDim = 24) {
  const l2 = tf.regularizers.l2({ l2: 1e-6 });
  const seed = 1337;
  const glorot = tf.initializers.glorotUniform({ seed });

  const userInput = tf.input({ shape: [1], dtype: 'int32', name: 'userInput' });
  const movieInput = tf.input({ shape: [1], dtype: 'int32', name: 'movieInput' });

  const userEmbedding = tf.layers.embedding({
    inputDim: userCount + 1,
    outputDim: latentDim,
    embeddingsInitializer: glorot,
    embeddingsRegularizer: l2,
    name: 'userEmbedding'
  }).apply(userInput);

  const movieEmbedding = tf.layers.embedding({
    inputDim: movieCount + 1,
    outputDim: latentDim,
    embeddingsInitializer: glorot,
    embeddingsRegularizer: l2,
    name: 'movieEmbedding'
  }).apply(movieInput);

  const userVec = tf.layers.flatten().apply(userEmbedding);
  const movieVec = tf.layers.flatten().apply(movieEmbedding);

  const userBias = tf.layers.flatten().apply(
    tf.layers.embedding({
      inputDim: userCount + 1,
      outputDim: 1,
      embeddingsInitializer: 'zeros',
      embeddingsRegularizer: l2,
      name: 'userBias'
    }).apply(userInput)
  );

  const movieBias = tf.layers.flatten().apply(
    tf.layers.embedding({
      inputDim: movieCount + 1,
      outputDim: 1,
      embeddingsInitializer: 'zeros',
      embeddingsRegularizer: l2,
      name: 'movieBias'
    }).apply(movieInput)
  );

  const dot = tf.layers.dot({ axes: 1, name: 'dotUserMovie' }).apply([userVec, movieVec]);
  const sum = tf.layers.add({ name: 'addBias' }).apply([dot, userBias, movieBias]);
  const output = tf.layers.activation({ activation: 'linear', name: 'rating' }).apply(sum);

  return tf.model({ inputs: [userInput, movieInput], outputs: output, name: 'mf_recommender' });
}

// ---------- Training ----------
function earlyStopping(patience = 2) {
  let best = Infinity;
  let wait = 0;
  return {
    onEpochEnd: async (epoch, logs) => {
      const cur = Number.isFinite(logs.val_loss) ? logs.val_loss : logs.loss;
      if (cur < best - 1e-4) { best = cur; wait = 0; }
      else if (++wait >= patience) {
        model.stopTraining = true;
        updateStatus(`Early stopping at epoch ${epoch + 1} (best val_loss ${best.toFixed(4)}).`);
      }
    }
  };
}

async function trainModel() {
  const btn = document.getElementById('predict-btn');
  try {
    isTraining = true;
    if (btn) btn.disabled = true;

    updateStatus('Building model…');
    model = createModel(numUsers, numMovies, 24);

    updateStatus('Compiling model…');
    model.compile({
      optimizer: tf.train.adam(0.002),     // slightly higher LR for faster convergence
      loss: 'meanSquaredError',
      metrics: ['mae']
    });

    // Prepare data tensors (IMPORTANT: int32 indices for embeddings)
    updateStatus('Preparing training data…');

    const N = ratings.length;
    const userIds = new Int32Array(N);
    const movieIds = new Int32Array(N);
    const y = new Float32Array(N);

    for (let i = 0; i < N; i++) {
      const r = ratings[i];
      userIds[i] = r.userId | 0;
      movieIds[i] = r.movieId | 0;
      y[i] = r.rating;
    }

    // Center targets around global mean (improves MF stability)
    let sum = 0; for (let i = 0; i < N; i++) sum += y[i];
    globalMeanRating = sum / N;
    for (let i = 0; i < N; i++) y[i] = y[i] - globalMeanRating;

    const Xuser = tf.tensor2d(userIds, [N, 1], 'int32');
    const Xmovie = tf.tensor2d(movieIds, [N, 1], 'int32');
    const Y = tf.tensor2d(y, [N, 1], 'float32');

    // Larger batch on GPU, smaller on CPU
    const isGPU = tf.getBackend() === 'webgl';
    const batchSize = isGPU ? 1024 : 256;

    updateStatus(`Training (${N.toLocaleString()} ratings) — batch ${batchSize}, up to 12 epochs…`);

    await model.fit([Xuser, Xmovie], Y, {
      epochs: 12,
      batchSize,
      shuffle: true,
      validationSplit: 0.1,
      callbacks: [
        {
          onEpochEnd: (epoch, logs) => {
            updateStatus(
              `Epoch ${epoch + 1} — loss: ${logs.loss.toFixed(4)} • val_loss: ${logs.val_loss?.toFixed(4)} • mae: ${logs.mae?.toFixed(4)}`
            );
          }
        },
        earlyStopping(2)
      ]
    });

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
    const centered = await tf.tidy(() => {
      const u = tf.tensor2d([[userId]], [1, 1], 'int32');
      const m = tf.tensor2d([[movieId]], [1, 1], 'int32');
      return model.predict([u, m]);
    }).data();

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
    await ensureBackend(); // WebGL on GH Pages
    updateStatus('Loading MovieLens data… (files must be next to index.html)');
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
