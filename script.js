// ======= Speed presets (Turbo default) =======
const TRAINING_PRESETS = {
  turbo: {
    sampleFraction: 0.30, // 30% of ratings
    maxPerUser: 20,       // cap per user
    epochs: 6,
    batchSize: 512,
    latentDim: 8,
    patience: 2,
    lr: 0.002
  },
  full: {
    sampleFraction: 1.0,
    maxPerUser: Infinity,
    epochs: 10,
    batchSize: 256,
    latentDim: 10,
    patience: 2,
    lr: 0.001
  }
};

let model = null;
let isTraining = false;

// ======= Init =======
window.onload = async function () {
  try {
    updateStatus('Initializing backend…');
    await tf.setBackend('webgl'); // fast path
    await tf.ready();

    updateStatus('Loading MovieLens data…');
    await loadData();

    populateUserDropdown();
    populateMovieDropdown();

    // If the optional toggle exists, retrain when it changes
    const toggle = document.getElementById('turbo-toggle');
    if (toggle) {
      toggle.addEventListener('change', () => trainFromUI());
    }

    // Kick off initial training
    await trainFromUI();
  } catch (error) {
    console.error('Initialization error:', error);
    updateStatus('Error initializing application: ' + error.message, true);
  }
};

function turboEnabled() {
  const el = document.getElementById('turbo-toggle');
  // If toggle not present, default to Turbo mode for speed
  return el ? !!el.checked : true;
}

async function trainFromUI() {
  const preset = turboEnabled() ? TRAINING_PRESETS.turbo : TRAINING_PRESETS.full;
  await trainModel(preset);
}

// ======= UI Fillers =======
function populateUserDropdown() {
  const userSelect = document.getElementById('user-select');
  userSelect.innerHTML = '';
  for (const uid of userIdsSorted) {
    const option = document.createElement('option');
    option.value = uid;
    option.textContent = `User ${uid}`;
    userSelect.appendChild(option);
  }
}

function populateMovieDropdown() {
  const movieSelect = document.getElementById('movie-select');
  movieSelect.innerHTML = '';
  for (const movie of movies) {
    const option = document.createElement('option');
    option.value = movie.id;
    option.textContent = movie.year ? `${movie.title} (${movie.year})` : movie.title;
    movieSelect.appendChild(option);
  }
}

// ======= Sampling & Tensor prep =======
function sampleRatings({ sampleFraction, maxPerUser }) {
  const byUser = new Map();
  for (const r of ratings) {
    if (!byUser.has(r.userId)) byUser.set(r.userId, []);
    byUser.get(r.userId).push(r);
  }

  const limited = [];
  for (const arr of byUser.values()) {
    shuffleInPlace(arr);
    const take = Math.min(arr.length, maxPerUser);
    for (let i = 0; i < take; i++) limited.push(arr[i]);
  }

  shuffleInPlace(limited);
  const keep = Math.max(1, Math.floor(limited.length * sampleFraction));
  return limited.slice(0, keep);
}

function shuffleInPlace(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

function tensorsFromRatings(rArr) {
  const userIds = rArr.map(r => r.userId);
  const movieIds = rArr.map(r => r.movieId);
  const ratingValues = rArr.map(r => r.rating);

  const userTensor = tf.tensor2d(userIds, [userIds.length, 1], 'int32');
  const movieTensor = tf.tensor2d(movieIds, [movieIds.length, 1], 'int32');
  const ratingTensor = tf.tensor2d(ratingValues, [ratingValues.length, 1], 'float32');
  return { userTensor, movieTensor, ratingTensor };
}

// ======= Model =======
function createModel(latentDim = 10) {
  const userInput = tf.input({ shape: [1], name: 'userInput' });
  const movieInput = tf.input({ shape: [1], name: 'movieInput' });

  const userEmbedding = tf.layers.embedding({
    inputDim: maxUserId + 1,
    outputDim: latentDim,
    embeddingsInitializer: 'heNormal',
    name: 'userEmbedding'
  }).apply(userInput);

  const movieEmbedding = tf.layers.embedding({
    inputDim: maxMovieId + 1,
    outputDim: latentDim,
    embeddingsInitializer: 'heNormal',
    name: 'movieEmbedding'
  }).apply(movieInput);

  const userVector = tf.layers.flatten().apply(userEmbedding);
  const movieVector = tf.layers.flatten().apply(movieEmbedding);

  const dotProduct = tf.layers.dot({ axes: 1 }).apply([userVector, movieVector]);
  const prediction = tf.layers.reshape({ targetShape: [1] }).apply(dotProduct);

  return tf.model({ inputs: [userInput, movieInput], outputs: prediction });
}

// ======= Training =======
async function trainModel({
  sampleFraction,
  maxPerUser,
  epochs,
  batchSize,
  latentDim,
  patience,
  lr
}) {
  try {
    isTraining = true;
    document.getElementById('predict-btn').disabled = true;

    const subset = sampleRatings({ sampleFraction, maxPerUser });
    updateStatus(
      `Preparing training set… Using ${subset.length} ratings (fraction=${sampleFraction}, max/user=${maxPerUser}).`
    );

    const { userTensor, movieTensor, ratingTensor } = tensorsFromRatings(subset);

    if (model) {
      model.dispose();
      model = null;
      await tf.nextFrame();
    }
    model = createModel(latentDim);
    model.compile({
      optimizer: tf.train.adam(lr),
      loss: 'meanSquaredError'
    });

    const earlyStop = tf.callbacks.earlyStopping({
      monitor: 'val_loss',
      patience,
      restoreBestWeights: true
    });

    let lastEpochStart = performance.now();
    await model.fit([userTensor, movieTensor], ratingTensor, {
      epochs,
      batchSize,
      validationSplit: 0.1,
      shuffle: true,
      callbacks: [
        { onEpochBegin: () => { lastEpochStart = performance.now(); } },
        {
          onEpochEnd: (epoch, logs) => {
            const ms = performance.now() - lastEpochStart;
            updateStatus(
              `Epoch ${epoch + 1}/${epochs} — loss ${logs.loss.toFixed(4)} — ` +
              `val ${logs.val_loss?.toFixed(4) ?? '—'} — ${(ms/1000).toFixed(1)}s`
            );
          }
        },
        earlyStop
      ]
    });

    tf.dispose([userTensor, movieTensor, ratingTensor]);

    updateStatus('Training complete. You can predict now.');
    document.getElementById('predict-btn').disabled = false;
    isTraining = false;
  } catch (error) {
    console.error('Training error:', error);
    updateStatus('Error training model: ' + error.message, true);
    isTraining = false;
  }
}

// ======= Predict =======
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
    const userTensor = tf.tensor2d([[userId]], [1, 1], 'int32');
    const movieTensor = tf.tensor2d([[movieId]], [1, 1], 'int32');

    const prediction = model.predict([userTensor, movieTensor]);
    const rating = await prediction.data();
    let predictedRating = rating[0];

    if (Number.isFinite(predictedRating)) {
      predictedRating = Math.max(1, Math.min(5, predictedRating));
    }

    tf.dispose([userTensor, movieTensor, prediction]);

    const movie = movies.find((m) => m.id === movieId);
    const movieTitle = movie ? (movie.year ? `${movie.title} (${movie.year})` : movie.title) : `Movie ${movieId}`;

    let ratingClass = 'medium';
    if (predictedRating >= 4) ratingClass = 'high';
    else if (predictedRating <= 2) ratingClass = 'low';

    updateResult(
      `Predicted rating for User ${userId} on "${movieTitle}": <strong>${predictedRating.toFixed(2)}/5</strong>`,
      ratingClass
    );
  } catch (error) {
    console.error('Prediction error:', error);
    updateResult('Error making prediction: ' + error.message, 'low');
  }
}

// ======= UI helpers =======
function updateStatus(message, isError = false) {
  const statusElement = document.getElementById('status');
  statusElement.textContent = message;
  statusElement.style.borderLeftColor = isError ? '#e74c3c' : '#3498db';
  statusElement.style.background = isError ? '#fdedec' : '#f8f9fa';
}

function updateResult(message, className = '') {
  const resultElement = document.getElementById('result');
  resultElement.innerHTML = message;
  resultElement.className = `result ${className}`;
}
