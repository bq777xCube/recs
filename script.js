// ======= Speed & training knobs (tune as you like) =======
const SAMPLE_FRACTION = 0.50;   // a bit more data for better spread (still fast)
const MAX_PER_USER    = 40;     // cap ratings/user
const EPOCHS          = 10;     // a few more epochs to escape the mean
const BATCH_SIZE      = 512;    // large batch for speed
const LATENT_DIM      = 12;     // a bit more capacity
const LR              = 0.002;  // learning rate
const VAL_SPLIT       = 0.10;

// Rating bounds
const Y_MIN = 1, Y_MAX = 5;

let model = null;
let isTraining = false;

// ---------- Init ----------
window.onload = async function () {
  try {
    updateStatus('Initializing TensorFlow.js backend…');
    await tf.setBackend('webgl');
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

// ---------- Model: embeddings + user/item biases + LINEAR head ----------
function createModel(latentDim = LATENT_DIM) {
  const userInput  = tf.input({ shape: [1], name: 'userInput'  });
  const movieInput = tf.input({ shape: [1], name: 'movieInput' });

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

  const dot = tf.layers.dot({ axes: 1 }).apply([uVec, mVec]);

  // user/movie biases
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

  // sum → [batch,1]
  const sum = tf.layers.add().apply([dot, uB, mB]);

  // Linear head with its own bias (acts as global mean μ and a scale weight)
  const out = tf.layers.dense({ units: 1, useBias: true, activation: 'linear', name: 'linearHead' }).apply(sum);

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

// Globals to store scaling for de-normalization
let yMean = 3.0;
let yStd  = 1.0;

// ---------- Train (z-score targets + linear head) ----------
async function trainModel() {
  try {
    isTraining = true;
    document.getElementById('predict-btn').disabled = true;

    const subset = sampleRatings(ratings, SAMPLE_FRACTION, MAX_PER_USER);
    updateStatus(`Training on ${subset.length} ratings (z-score targets, fast)…`);

    // Compute z-score scaling on the SAME subset to align training/prediction
    const ys = subset.map(r => r.rating);
    yMean = ys.reduce((a,b)=>a+b,0) / ys.length;
    const var_ = ys.reduce((a,b)=>a+(b - yMean)*(b - yMean),0) / Math.max(1, ys.length-1);
    yStd = Math.max(Math.sqrt(var_), 1e-6); // avoid div-by-zero

    // z-score labels
    const yZ = ys.map(v => (v - yMean) / yStd);

    // Build/compile
    model = createModel(LATENT_DIM);
    model.compile({
      optimizer: tf.train.adam(LR),
      loss: 'meanSquaredError'
    });

    // Tensors (embedding indices must be int32)
    const userIds = subset.map(r => r.userId);
    const movieIds = subset.map(r => r.movieId);

    const userTensor  = tf.tensor2d(userIds, [userIds.length, 1], 'int32');
    const movieTensor = tf.tensor2d(movieIds,[movieIds.length, 1], 'int32');
    const yTensor     = tf.tensor2d(yZ,     [yZ.length,     1], 'float32');

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

    updateStatus('Model training completed successfully!');
    document.getElementById('predict-btn').disabled = false;
    isTraining = false;
  } catch (error) {
    console.error('Training error:', error);
    updateStatus('Error training model: ' + error.message, true);
    isTraining = false;
  }
}

// ---------- Predict (de-normalize) ----------
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

    const predZ = model.predict([u, m]);    // z-score output (unbounded)
    const arr   = await predZ.data();
    let yhat    = arr[0] * yStd + yMean;    // de-normalize back to rating scale

    // Clamp to [1,5] for display sanity
    if (Number.isFinite(yhat)) yhat = Math.max(Y_MIN, Math.min(Y_MAX, yhat));

    tf.dispose([u, m, predZ]);

    const mv = movies.find(x => x.id === movieId);
    const title = mv ? (mv.year ? `${mv.title} (${mv.year})` : mv.title) : `Movie ${movieId}`;

    let cls = 'medium';
    if (yhat >= 4) cls = 'high';
    else if (yhat <= 2) cls = 'low';

    updateResult(`Predicted rating for User ${userId} on "${title}": <strong>${yhat.toFixed(2)}/${Y_MAX}</strong>`, cls);
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
