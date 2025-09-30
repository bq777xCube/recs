// ======= Fast defaults (no checkbox required) =======
const PRESET_TURBO = {
  sampleFraction: 0.30,  // train on ~30% of ratings
  maxPerUser: 20,        // cap per user
  epochs: 6,
  batchSize: 512,
  latentDim: 8,
  patience: 2,
  lr: 0.002
};
const PRESET_FULL = {
  sampleFraction: 1.0,
  maxPerUser: Infinity,
  epochs: 10,
  batchSize: 256,
  latentDim: 10,
  patience: 2,
  lr: 0.001
};

// If no toggle is present, default to Turbo for speed
const USE_TURBO_DEFAULT = true;

let model = null;
let isTraining = false;

window.onload = async () => {
  try {
    updateStatus('Initializing…');
    await tf.setBackend('webgl'); // GPU if available
    await tf.ready();

    updateStatus('Loading data…');
    await loadData();

    populateUserDropdown();
    populateMovieDropdown();

    // If (and only if) the toggle exists, hook it up
    const toggle = document.getElementById('turbo-toggle');
    if (toggle) {
      toggle.addEventListener('change', () => trainFromUI());
    }

    await trainFromUI();
  } catch (e) {
    console.error(e);
    updateStatus('Error initializing application: ' + e.message, true);
  }
};

// Decide whether Turbo is enabled (safe even if toggle is missing)
function turboEnabled() {
  const el = document.getElementById('turbo-toggle');
  return el ? !!el.checked : USE_TURBO_DEFAULT;
}

async function trainFromUI() {
  const preset = turboEnabled() ? PRESET_TURBO : PRESET_FULL;
  await trainModel(preset);
}

// ======= UI fillers =======
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

// ======= Sampling for speed =======
function sampleRatings({ sampleFraction, maxPerUser }) {
  const byUser = new Map();
  for (const r of ratings) {
    if (!byUser.has(r.userId)) byUser.set(r.userId, []);
    byUser.get(r.userId).push(r);
  }
  const limited = [];
  for (const arr of byUser.values()) {
    shuffle(arr);
    const take = Math.min(arr.length, maxPerUser);
    for (let i = 0; i < take; i++) limited.push(arr[i]);
  }
  shuffle(limited);
  const keep = Math.max(1, Math.floor(limited.length * sampleFraction));
  return limited.slice(0, keep);
}
function shuffle(a){ for(let i=a.length-1;i>0;i--){const j=(Math.random()*(i+1))|0; [a[i],a[j]]=[a[j],a[i]];} }

function tensorsFromRatings(arr) {
  const users  = tf.tensor2d(arr.map(r=>r.userId),  [arr.length,1], 'int32');
  const movies = tf.tensor2d(arr.map(r=>r.movieId), [arr.length,1], 'int32');
  const y      = tf.tensor2d(arr.map(r=>r.rating),  [arr.length,1], 'float32');
  return { users, movies, y };
}

// ======= Model =======
function createModel(latentDim=8) {
  const uIn = tf.input({ shape:[1], name:'user' });
  const mIn = tf.input({ shape:[1], name:'movie' });

  const uEmb = tf.layers.embedding({
    inputDim: maxUserId+1,
    outputDim: latentDim,
    embeddingsInitializer:'heNormal'
  }).apply(uIn);

  const mEmb = tf.layers.embedding({
    inputDim: maxMovieId+1,
    outputDim: latentDim,
    embeddingsInitializer:'heNormal'
  }).apply(mIn);

  const uVec = tf.layers.flatten().apply(uEmb);
  const mVec = tf.layers.flatten().apply(mEmb);

  const dot  = tf.layers.dot({ axes:1 }).apply([uVec, mVec]);
  const out  = tf.layers.reshape({ targetShape:[1] }).apply(dot);

  return tf.model({ inputs:[uIn, mIn], outputs:out });
}

// ======= Train (fixed EarlyStopping) =======
async function trainModel({ sampleFraction, maxPerUser, epochs, batchSize, latentDim, patience, lr }) {
  isTraining = true;
  document.getElementById('predict-btn').disabled = true;

  const subset = sampleRatings({ sampleFraction, maxPerUser });
  updateStatus(`Training on ${subset.length} ratings…`);

  const { users, movies:moviesT, y } = tensorsFromRatings(subset);

  if (model) { model.dispose(); model = null; await tf.nextFrame(); }
  model = createModel(latentDim);
  model.compile({ optimizer: tf.train.adam(lr), loss: 'meanSquaredError' });

  // NOTE: restoreBestWeights is NOT implemented in TFJS
  const earlyStop = tf.callbacks.earlyStopping({
    monitor: 'val_loss',
    patience
  });

  let start = 0;
  await model.fit([users, moviesT], y, {
    epochs, batchSize, validationSplit:0.1, shuffle:true,
    callbacks: [
      { onEpochBegin: ()=>{ start = performance.now(); } },
      { onEpochEnd: (epoch, logs)=>{
          const sec = ((performance.now()-start)/1000).toFixed(1);
          updateStatus(`Epoch ${epoch+1}/${epochs} — loss ${logs.loss.toFixed(4)} — val ${logs.val_loss?.toFixed(4) ?? '—'} — ${sec}s`);
        } },
      earlyStop
    ]
  });

  tf.dispose([users, moviesT, y]);
  updateStatus('Training complete. You can predict now.');
  document.getElementById('predict-btn').disabled = false;
  isTraining = false;
}

// ======= Predict =======
async function predictRating() {
  if (isTraining) { updateResult('Model is still training…','medium'); return; }
  const userId  = parseInt(document.getElementById('user-select').value, 10);
  const movieId = parseInt(document.getElementById('movie-select').value, 10);
  if (!userId || !movieId) { updateResult('Please select both a user and a movie.','medium'); return; }

  const u = tf.tensor2d([[userId]], [1,1], 'int32');
  const m = tf.tensor2d([[movieId]], [1,1], 'int32');
  const pred = model.predict([u,m]);
  const val = (await pred.data())[0];
  tf.dispose([u,m,pred]);

  const clipped = Number.isFinite(val) ? Math.max(1, Math.min(5, val)) : NaN;
  const mv = movies.find(x=>x.id===movieId);
  const title = mv ? (mv.year ? `${mv.title} (${mv.year})` : mv.title) : `Movie ${movieId}`;

  let cls = 'medium'; if (clipped>=4) cls='high'; else if (clipped<=2) cls='low';
  updateResult(`Predicted rating for User ${userId} on "${title}": <strong>${clipped.toFixed(2)}/5</strong>`, cls);
}

// ======= UI helpers =======
function updateStatus(msg, isErr=false){
  const el = document.getElementById('status');
  el.textContent = msg;
  el.style.borderLeftColor = isErr ? '#e74c3c' : '#3498db';
  el.style.background = isErr ? '#fdedec' : '#f8f9fa';
}
function updateResult(msg, cls=''){
  const el = document.getElementById('result');
  el.innerHTML = msg;
  el.className = `result ${cls}`;
}
