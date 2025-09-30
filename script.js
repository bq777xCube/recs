// ======= Speed knobs (tweak if needed) =======
const SAMPLE_FRACTION = 0.30;   // train on ~30% of ratings
const MAX_PER_USER    = 20;     // cap ratings per user to keep it small/balanced
const EPOCHS          = 6;      // fewer epochs (keeps it fast)
const BATCH_SIZE      = 512;    // larger batch = faster per epoch (GPU permitting)
const LATENT_DIM      = 8;      // smaller embedding
const LR              = 0.002;  // slightly higher LR for faster convergence
const VAL_SPLIT       = 0.10;   // 10% validation split

// Global variables
let model = null;
let isTraining = false;

// Initialize application when window loads
window.onload = async function () {
  try {
    updateStatus('Initializing TensorFlow.js backend…');
    // Fastest in-browser path if available
    await tf.setBackend('webgl');
    await tf.ready();

    updateStatus('Loading MovieLens data...');
    await loadData();

    // Populate dropdowns
    populateUserDropdown();
    populateMovieDropdown();

    // Update status and start training
    updateStatus('Data loaded. Training model...');
    await trainModel();
  } catch (error) {
    console.error('Initialization error:', error);
    updateStatus('Error initializing application: ' + error.message, true);
  }
};

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

  movies.forEach((movie) => {
    const option = document.createElement('option');
    option.value = movie.id;
    option.textContent = movie.year ? `${movie.title} (${movie.year})` : movie.title;
    movieSelect.appendChild(option);
  });
}

/**
 * Create a simple MF model: <u, v> dot product.
 * Embedding dimensions sized by max ID + 1; indices must be int32.
 */
function createModel(latentDim = LATENT_DIM) {
  const userInput = tf.input({ shape: [1], name: 'userInput' });
  const movieInput = tf.input({ shape: [1], name: 'movieInput' });

  const userEmbedding = tf.layers
    .embedding({
      inputDim: maxUserId + 1,
      outputDim: latentDim,
      embeddingsInitializer: 'heNormal',
      name: 'userEmbedding',
    })
    .apply(userInput);

  const movieEmbedding = tf.layers
    .embedding({
      inputDim: maxMovieId + 1,
      outputDim: latentDim,
      embeddingsInitializer: 'heNormal',
      name: 'movieEmbedding',
    })
    .apply(movieInput);

  const userVector = tf.layers.flatten().apply(userEmbedding);
  const movieVector = tf.layers.flatten().apply(movieEmbedding);

  const dotProduct = tf.layers.dot({ axes: 1 }).apply([userVector, movieVector]);
  const prediction = tf.layers.reshape({ targetShape: [1] }).apply(dotProduct);

  return tf.model({ inputs: [userInput, movieInput], outputs: prediction });
}

/** Fast sampling: random subset + per-user cap to shrink training set */
function sampleRatings(allRatings, fraction = SAMPLE_FRACTION, maxPerUser = MAX_PER_USER) {
  const byUser = new Map();
  for (const r of allRatings) {
    if (!byUser.has(r.userId)) byUser.set(r.userId, []);
    byUser.get(r.userId).push(r);
  }

  const shuffle = (arr) => {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  };

  // limit examples per user
  const limited = [];
  for (const arr of byUser.values()) {
    shuffle(arr);
    const take = Math.min(arr.length, maxPerUser);
    for (let i = 0; i < take; i++) limited.push(arr[i]);
  }

  // global shuffle + fraction
  shuffle(limited);
  const keep = Math.max(1, Math.floor(limited.length * fraction));
  return limited.slice(0, keep);
}

async function trainModel() {
  try {
    isTraining = true;
    document.getElementById('predict-btn').disabled = true;

    // ===== Speed: sample and shrink =====
    const subset = sampleRatings(ratings, SAMPLE_FRACTION, MAX_PER_USER);
    updateStatus(`Training on ${subset.length} ratings (fast mode)…`);

    // Create & compile model
    model = createModel(LATENT_DIM);
    model.compile({
      optimizer: tf.train.adam(LR),
      loss: 'meanSquaredError',
    });

    // Tensors (embedding indices must be int32)
    const userIds = subset.map((r) => r.userId);
    const movieIds = subset.map((r) => r.movieId);
    const ratingValues = subset.map((r) => r.rating);

    const userTensor = tf.tensor2d(userIds, [userIds.length, 1], 'int32');
    const movieTensor = tf.tensor2d(movieIds, [movieIds.length, 1], 'int32');
    const ratingTensor = tf.tensor2d(ratingValues, [ratingValues.length, 1], 'float32');

    let epochStart = 0;
    await model.fit([userTensor, movieTensor], ratingTensor, {
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
      ],
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

async function predictRating() {
  if (isTraining) {
    updateResult('Model is still training. Please wait...', 'medium');
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

    // Clip to 1..5 to keep outputs reasonable
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

// UI helpers
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
