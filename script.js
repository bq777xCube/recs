// Global variables
let model = null;
let isTraining = false;

// Initialize application when window loads
window.onload = async function () {
  try {
    updateStatus('Loading MovieLens data...');

    // Load data first
    await loadData();

    // Populate dropdowns
    populateUserDropdown();
    populateMovieDropdown();

    // Update status and start training
    updateStatus('Data loaded. Training model...');

    // Train the model
    await trainModel();
  } catch (error) {
    console.error('Initialization error:', error);
    updateStatus('Error initializing application: ' + error.message, true);
  }
};

function populateUserDropdown() {
  const userSelect = document.getElementById('user-select');
  userSelect.innerHTML = '';

  // Use actual user IDs from the dataset (robust if not strictly 1..N)
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

  // Add movies
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
function createModel(latentDim = 10) {
  // Inputs (shape [1]); embedding will receive int32 tensors at runtime
  const userInput = tf.input({ shape: [1], name: 'userInput' });
  const movieInput = tf.input({ shape: [1], name: 'movieInput' });

  // Embeddings
  const userEmbedding = tf.layers
    .embedding({
      inputDim: maxUserId + 1,
      outputDim: latentDim,
      name: 'userEmbedding',
    })
    .apply(userInput);

  const movieEmbedding = tf.layers
    .embedding({
      inputDim: maxMovieId + 1,
      outputDim: latentDim,
      name: 'movieEmbedding',
    })
    .apply(movieInput);

  // Flatten embeddings to latent vectors
  const userVector = tf.layers.flatten().apply(userEmbedding);
  const movieVector = tf.layers.flatten().apply(movieEmbedding);

  // Dot product -> predicted rating
  const dotProduct = tf.layers.dot({ axes: 1 }).apply([userVector, movieVector]);

  // Reshape to [1] output
  const prediction = tf.layers.reshape({ targetShape: [1] }).apply(dotProduct);

  // Create model
  const model = tf.model({
    inputs: [userInput, movieInput],
    outputs: prediction,
  });

  return model;
}

async function trainModel() {
  try {
    isTraining = true;
    document.getElementById('predict-btn').disabled = true;

    // Create model
    model = createModel(10);

    // Compile model
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
    });

    // Prepare training data (indices must be int32 for embedding layers)
    const userIds = ratings.map((r) => r.userId);
    const movieIds = ratings.map((r) => r.movieId);
    const ratingValues = ratings.map((r) => r.rating);

    const userTensor = tf.tensor2d(userIds, [userIds.length, 1], 'int32');
    const movieTensor = tf.tensor2d(movieIds, [movieIds.length, 1], 'int32');
    const ratingTensor = tf.tensor2d(ratingValues, [ratingValues.length, 1], 'float32');

    updateStatus('Training model... (This may take a moment)');

    await model.fit([userTensor, movieTensor], ratingTensor, {
      epochs: 10,
      batchSize: 64,
      validationSplit: 0.1,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          updateStatus(`Training epoch ${epoch + 1}/10 - loss: ${logs.loss.toFixed(4)}`);
        },
      },
    });

    // Clean up tensors
    tf.dispose([userTensor, movieTensor, ratingTensor]);

    // Update UI
    updateStatus('Model training completed successfully!');
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
    // Create input tensors as int32 indices
    const userTensor = tf.tensor2d([[userId]], [1, 1], 'int32');
    const movieTensor = tf.tensor2d([[movieId]], [1, 1], 'int32');

    // Predict
    const prediction = model.predict([userTensor, movieTensor]);
    const rating = await prediction.data();
    let predictedRating = rating[0];

    // Optional: clip to 1..5 to keep outputs reasonable
    if (Number.isFinite(predictedRating)) {
      predictedRating = Math.max(1, Math.min(5, predictedRating));
    }

    // Clean up
    tf.dispose([userTensor, movieTensor, prediction]);

    // Display result
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
