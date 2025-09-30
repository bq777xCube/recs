// ======= Speed & scaling knobs =======
const SAMPLE_FRACTION = 0.30;   // train on ~30% of ratings
const MAX_PER_USER    = 20;     // cap ratings per user
const EPOCHS          = 8;      // few epochs (fast) — raise to 10–12 if you want a tad more accuracy
const BATCH_SIZE      = 512;    // larger batch = faster per epoch (GPU permitting)
const LATENT_DIM      = 8;      // embedding size
const LR              = 0.002;  // slightly higher LR for faster convergence
const VAL_SPLIT       = 0.10;   // 10% validation split

// Rating range (MovieLens 1–5)
const Y_MIN = 1;
const Y_MAX = 5;
const Y_SPAN = Y_MAX - Y_MIN;   // 4

let model = null;
let isTraining = false;

// ---------- Init ----------
window.onload = async function () {
  try {
    updateStatus('Initializing TensorFlow.js backend…');
    await tf.setBackend('webgl'); // fastest if available
    await tf.ready();

    updateStatus('Loading MovieLens data...');
    await loadData();

    populateUserDropdown();
    populateMovieDropdown();

    updateStatus('Data loaded. Training model...');
    await trainModel();
  } catch (error) {
    console.error('Initialization error:', error);
    updateStatus('Error initializing application: ' + error.message, true);
  }
};

function populateUserDropdown() {
  const el = document.getElementById('user-select');
  el.innerHTML = '';
  for (const uid of userIdsSorted) {
    const opt = document.createElement('option');
    opt.value = uid;
    opt.textContent = `User ${uid}`;
    el.appendChild(opt);
  }
}

function populateMovieDropdown() {
  const el = document.getElementById('movie-select');
  el.innerHTML = '';
  for (const m of movies) {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.year ? `${m.title} (${m.year})` : m.title;
    el.appendChild(opt);
  }
}

// ---------- Model: embeddings + user/item biases + sigmoid head ----------
function createModel(latentDim = LATENT_DIM) {
  const userInput  = tf.input({ shape: [1], name: 'userInput'  });
  const movieInput = tf.input({ shape: [1], name: 'movieInput' });

  // latent
  const userEmb = tf.layers.embedding({
    inputDim: maxUserId + 1,
    outputDim: latentDim,
    embeddingsInitializer: 'heNormal',
    name: 'userEmbedding'
  }).apply(userInput);

  const movieEmb = tf.layers.embedding({
    inputDim: maxMovieId + 1,
    outputDim: latentDim,
    embeddingsInitializer: 'heNormal',
    name: 'movieEmbedding'
  }).apply(movieInput);

  const uVec = tf.layers.flatten().apply(userEmb);
  const mVec = tf.layers.flatten().apply(movieEmb);

  // interaction
  const dot = tf.layers.dot({ axes: 1 }).apply([uVec, mVec]);

  // biases (key to avoid “always 1”)
  const userBiasEmb = tf.layers.embedding({
    inputDim: maxUserId + 1,
    outputDim: 1,
    embeddingsInitializer: 'zeros',
    name: 'userBias'
  }).apply(userInput);
  const movieBiasEmb = tf.layers.embedding({
    inputDim: maxMovieId + 1,
    outputDim: 1,
    embeddingsInitializer: 'zeros',
    name: 'movieBias'
  }).apply(movieInput);

  const uB = tf.layers.flatten().apply(userBiasEmb);
  const mB = tf.layers.flatten().apply(movieBiasEmb);

  // sum -> scalar logit
  const sum = tf.layers.add().apply([dot, uB, mB]);

  // sigmoid to squash into [0,1] (we'll denormalize back to [1,5])
  const y01 = tf.layers.activation({ activation: 'sigmoid' }).apply(sum);

  // explicit shape [1]
  const out = tf.layers.reshape({ targetShape: [1] }).apply(y01);

  return tf.model({ inputs: [userInput, movieInput], outputs: out });
}

// ---------- Sampling (fast training set) ----------
function sampleRatings(allRatings, fraction = SAMPLE_FRACTION, maxPerUser = MAX_PER_USER) {
  const byUser = new Map();
  for (const r of allRatings) {
    if (!byUser.has(r.userId)) byUser.set(r.userId, []);
    byUser.get(r.userId).push(r);
  }
  const shuffle = (a) => { for (let i=a.length-1;i>0;i--){const j=(Math.random()*(i+1))|0; [a[i],a[j]]=[a[j],a[i]];} };

  const limited = [];
  for (const arr of byUser.values()) {
    shuffle(arr);
    const take = Math.min(arr.length, maxPerUser);
    for (let i = 0; i < take; i++) limited.push(arr[i]);
  }
  shuffle(limited);
  const keep = Math.max(1, Math.floor(limited.length * fraction));
  return limited.slice(0, keep);
}

// ---------- Train ----------
async function trainModel() {
  try {
    isTraining = true;
    document.getElementById('predict-btn').disabled = true;

    const subset = sampleRatings(ratings, SAMPLE_FRACTION, MAX_PER_USER);
    updateStatus(`Training on ${subset.length} ratings (fast, normalized targets)…`);

    // Build/compile
    model = createModel(LATENT_DIM);
    model.compile({
      optimizer: tf.train.adam(LR),
      loss: 'meanSquaredError' // works fine with sigmoid head and normalized labels
    });

    // Prepare tensors (embedding indices must be int32)
    const userIds = subset.map(r => r.userId);
    const movieIds = subset.map(r => r.movieId);
    // normalize ratings to [0,1] to match sigmoid output
    const yNorm   = subset.map(r => (r.rating - Y_MIN) / Y_SPAN);

    const userTensor  = tf.tensor2d(userIds, [userIds.length, 1], 'int32');
    const movieTensor = tf.tensor2d(movieIds,[movieIds.length, 1], 'int32');
    const yTensor     = tf.tensor2d(yNorm,   [yNorm.length,   1], 'float32');

    let epochStart = 0;
    await model.fit([userTensor, movieTensor], yTensor, {
      epochs: EPOCHS,
      batchSize: BATCH_SIZE,
      validationSplit: VAL_SPLIT,
      shuffle: true,
      callbacks: [
        { onEpochBegin: () => { epochStart = performance.now(); } },
        {
          onEpochEnd: (epoch, logs) => {
            const sec = ((performance.now() - epochStart) / 1000).toFixed(1);
            updateStatus(`Epoch ${epoch + 1}/${EPOCHS} — loss ${logs.loss.toFixed(4)} — val ${logs.val_loss?.toFixed(4) ?? '—'} — ${sec}s`);
          }
        }
      ]
    });

    tf.dispose([userTensor, movieTensor, yTensor]);

    updateStatus('Training complete. You can predict now.');
    document.getElementById('predict-btn').disabled = false;
    isTraining = false;
  } catch (error) {
    console.error('Training error:', error);
    updateStatus('Error training model: ' + error.message, true);
    isTraining = false;
  }
}

// ---------- Predict ----------
async function predictRating() {
  if (isTraining) {
    updateResult('Model is still training. Please wait...', 'medium');
    return;
  }

  const userId  = parseInt(document.getElementById('user-select').value, 10);
  const movieId = parseInt(document.getElementById('movie-select').value, 10);
  if (!userId || !movieId) {
    updateResult('Please select both a user and a movie.', 'medium');
    return;
  }

  try {
    const u = tf.tensor2d([[userId]],  [1, 1], 'int32');
    const m = tf.tensor2d([[movieId]], [1, 1], 'int32');

    const pred01 = model.predict([u, m]);         // in [0,1]
    const arr    = await pred01.data();
    let y01      = arr[0];

    // denormalize back to [1,5]
    let y = y01 * Y_SPAN + Y_MIN;

    // Optional safety clamp
    if (Number.isFinite(y)) y = Math.max(Y_MIN, Math.min(Y_MAX, y));

    tf.dispose([u, m, pred01]);

    const mv = movies.find(x => x.id === movieId);
    const title = mv ? (mv.year ? `${mv.title} (${mv.year})` : mv.title) : `Movie ${movieId}`;

    let cls = 'medium';
    if (y >= 4) cls = 'high';
    else if (y <= 2) cls = 'low';

    updateResult(`Predicted rating for User ${userId} on "${title}": <strong>${y.toFixed(2)}/${Y_MAX}</strong>`, cls);
  } catch (error) {
    console.error('Prediction error:', error);
    updateResult('Error making prediction: ' + error.message, 'low');
  }
}

// ---------- UI helpers ----------
function updateStatus(message, isError = false) {
  const el = document.getElementById('status');
  el.textContent = message;
  el.style.borderLeftColor = isError ? '#e74c3c' : '#3498db';
  el.style.background = isError ? '#fdedec' : '#f8f9fa';
}
function updateResult(message, className = '') {
  const el = document.getElementById('result');
  el.innerHTML = message;
  el.className = `result ${className}`;
}
