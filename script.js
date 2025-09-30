// ============================================================
//  Matrix Factorization (MF) with biases, trained in-browser.
//  Rating ~ μ + b_u + b_i + <P_u, Q_i>
//  - μ: global mean
//  - b_u: user bias
//  - b_i: item bias
//  - P_u, Q_i: user/item embeddings (latent factors)
//  Optimization: MSE over observed ratings with L2 regularization.
// ============================================================

// --- Mappings & tensors ---
let userIds = [];              // all unique user IDs from ratings (original IDs)
let movieIds = [];             // all movie IDs from movies (original IDs)
let userIdToIdx = new Map();   // original userId -> 0..U-1
let movieIdToIdx = new Map();  // original movieId -> 0..M-1
let idxToUserId = [];          // reverse maps (for dropdowns)
let idxToMovieId = [];

let globalMean = 0;            // μ

// --- TF Variables (learned parameters) ---
let P = null;  // Users latent matrix: [U, K]
let Q = null;  // Items latent matrix: [M, K]
let bu = null; // User bias: [U]
let bi = null; // Item bias: [M]

// --- UI helpers ---
const $ = (id) => document.getElementById(id);
const setStatus = (msg, cls='muted') => { const el=$('status-line'); el.textContent=msg; el.className=cls; };
const setBar = (pct) => { $('bar').style.width = `${Math.max(0, Math.min(100, pct))}%`; };

// Initialize on load
window.onload = async function () {
  try {
    setStatus('Loading data…');
    await loadData();
    prepareMappings();
    populateDropdowns();
    setStatus(`Loaded ${userIds.length} users, ${movieIds.length} movies, ${ratings.length} ratings. Click “Train Model”.`, 'ok');
  } catch (e) {
    console.error(e);
  }
};

// ------------------------------------------------------------
// Build mappings (sparse original IDs -> dense indices 0..U-1 / 0..M-1)
// Also compute global mean rating μ.
// ------------------------------------------------------------
function prepareMappings() {
  const uSet = new Set();
  const mSet = new Set();
  for (const r of ratings) {
    uSet.add(r.userId);
    mSet.add(r.itemId);
  }

  userIds = [...uSet].sort((a,b)=>a-b);
  movieIds = movies.map(m => m.id); // keep all parsed movies (even if unrated)
  // If you prefer only rated movies, replace with [...mSet].sort((a,b)=>a-b);

  userIdToIdx.clear(); movieIdToIdx.clear();
  idxToUserId = []; idxToMovieId = [];

  userIds.forEach((uid, i) => { userIdToIdx.set(uid, i); idxToUserId[i]=uid; });
  movieIds.forEach((mid, i) => { movieIdToIdx.set(mid, i); idxToMovieId[i]=mid; });

  // Global mean over observed ratings
  if (ratings.length > 0) {
    const s = ratings.reduce((acc, r) => acc + r.rating, 0);
    globalMean = s / ratings.length;
  } else {
    globalMean = 3.5; // fallback
  }
}

// ------------------------------------------------------------
// Fill user/movie selects
// ------------------------------------------------------------
function populateDropdowns() {
  const userSel = $('user-select');
  userSel.innerHTML = '<option value="" disabled selected>Select user</option>';
  idxToUserId.forEach((uid, idx) => {
    const opt = document.createElement('option');
    opt.value = String(uid);
    opt.textContent = `User ${uid}`;
    userSel.appendChild(opt);
  });

  const movieSel = $('movie-select');
  movieSel.innerHTML = '<option value="" disabled selected>Select movie</option>';
  // sort movies by title for nicer UX
  const sorted = [...movies].sort((a,b)=>a.title.localeCompare(b.title));
  for (const m of sorted) {
    const opt = document.createElement('option');
    opt.value = String(m.id);
    opt.textContent = m.title;
    movieSel.appendChild(opt);
  }
}

// ------------------------------------------------------------
// Build a mini-dataset as dense index arrays for TF (Int32) + ratings (Float32)
// Optionally shuffle.
// ------------------------------------------------------------
function buildDataset() {
  const uIdx = new Int32Array(ratings.length);
  const iIdx = new Int32Array(ratings.length);
  const y    = new Float32Array(ratings.length);

  for (let k = 0; k < ratings.length; k++) {
    const r = ratings[k];
    const ui = userIdToIdx.get(r.userId);
    const ii = movieIdToIdx.get(r.itemId);
    // Skip ratings for items not in our movie list (rare, but safe-guard)
    uIdx[k] = ui ?? 0;
    iIdx[k] = ii ?? 0;
    y[k]    = r.rating;
  }

  // Shuffle in-place (Fisher–Yates)
  for (let k = y.length - 1; k > 0; k--) {
    const j = (Math.random()* (k+1))|0;
    [uIdx[k], uIdx[j]] = [uIdx[j], uIdx[k]];
    [iIdx[k], iIdx[j]] = [iIdx[j], iIdx[k]];
    [y[k],    y[j]]    = [y[j],    y[k]];
  }

  // simple train/val split
  const valFrac = 0.1;
  const nVal = Math.max(1, Math.floor(y.length * valFrac));
  const nTrain = y.length - nVal;

  const ds = {
    train: {
      users: tf.tensor1d(uIdx.subarray(0, nTrain), 'int32'),
      items: tf.tensor1d(iIdx.subarray(0, nTrain), 'int32'),
      ratings: tf.tensor1d(y.subarray(0, nTrain), 'float32'),
    },
    val: {
      users: tf.tensor1d(uIdx.subarray(nTrain), 'int32'),
      items: tf.tensor1d(iIdx.subarray(nTrain), 'int32'),
      ratings: tf.tensor1d(y.subarray(nTrain), 'float32'),
    }
  };
  return ds;
}

// ------------------------------------------------------------
// Create learnable variables for MF with biases
// Shapes:
//  P: [U, K], Q: [M, K], bu: [U], bi: [M]
// Small random init helps break symmetry.
// ------------------------------------------------------------
function initVariables(U, M, K) {
  const rand = (shape, scale=0.05) => tf.randomNormal(shape, 0, scale, 'float32');
  if (P) { P.dispose(); Q.dispose(); bu.dispose(); bi.dispose(); }
  P  = tf.variable(rand([U, K]));
  Q  = tf.variable(rand([M, K]));
  bu = tf.variable(tf.zeros([U]));
  bi = tf.variable(tf.zeros([M]));
}

// ------------------------------------------------------------
// Prediction for batches: y_hat = μ + b_u + b_i + <P_u, Q_i>
// Using embedding lookup via gather()
// ------------------------------------------------------------
function predictBatch(uIdx, iIdx) {
  return tf.tidy(() => {
    const Pu = tf.gather(P, uIdx);      // [B, K]
    const Qi = tf.gather(Q, iIdx);      // [B, K]
    const bu_b = tf.gather(bu, uIdx);   // [B]
    const bi_b = tf.gather(bi, iIdx);   // [B]
    const dot = tf.sum(tf.mul(Pu, Qi), -1); // [B]
    return dot.add(bu_b).add(bi_b).add(globalMean);
  });
}

// ------------------------------------------------------------
// Train loop (Adam)
// Loss = MSE + λ (||P||^2 + ||Q||^2 + ||bu||^2 + ||bi||^2)
// ------------------------------------------------------------
async function trainModel() {
  try {
    const K = Math.max(4, Math.min(128, parseInt($('k-input').value, 10) || 20));
    const epochs = Math.max(1, Math.min(200, parseInt($('epochs-input').value, 10) || 15));
    const lr = Math.max(1e-3, Math.min(5e-1, parseFloat($('lr-input').value) || 0.05));

    if (ratings.length === 0) {
      setStatus('No ratings found. Check u.data.', 'err');
      return;
    }

    setStatus(`Preparing dataset…`);
    const U = userIds.length;
    const M = movieIds.length;
    initVariables(U, M, K);

    const ds = buildDataset();
    const optimizer = tf.train.adam(lr);
    const lambda = 0.0005; // L2 regularization weight
    const clamp = (t) => t.clipByValue(1, 5); // ratings are 1..5

    $('predict-btn').disabled = true;
    setBar(0);

    // simple early stopping
    let bestVal = Infinity;
    let patience = 5, patienceLeft = patience;
    let bestSnapshot = null;

    for (let ep = 1; ep <= epochs; ep++) {
      // ----- TRAIN STEP -----
      const trainLoss = optimizer.minimize(() => {
        const yhat = predictBatch(ds.train.users, ds.train.items);
        const mse = tf.losses.meanSquaredError(ds.train.ratings, clamp(yhat));
        // L2 regularization
        const reg = tf.addN([
          tf.sum(tf.square(P)),
          tf.sum(tf.square(Q)),
          tf.sum(tf.square(bu)),
          tf.sum(tf.square(bi)),
        ]).mul(lambda);
        return mse.add(reg);
      }, true);

      // ----- EVAL STEP -----
      const yhatVal = predictBatch(ds.val.users, ds.val.items);
      const valMSE = tf.losses.meanSquaredError(ds.val.ratings, clamp(yhatVal));
      const [train, v] = await Promise.all([trainLoss.data(), valMSE.data()]);

      const trainMSE = train[0];
      const valMse = v[0];
      setStatus(`Epoch ${ep}/${epochs} — train MSE: ${trainMSE.toFixed(4)} | val MSE: ${valMse.toFixed(4)}`, 'warn');
      setBar((ep/epochs)*100);

      // Early stopping on val
      if (valMse + 1e-6 < bestVal) {
        bestVal = valMse;
        patienceLeft = patience;
        // snapshot variables
        bestSnapshot?.forEach(t => t.dispose());
        bestSnapshot = [P.clone(), Q.clone(), bu.clone(), bi.clone()];
      } else {
        patienceLeft--;
        if (patienceLeft <= 0) {
          setStatus(`Early stopped at epoch ${ep}. Best val MSE: ${bestVal.toFixed(4)}`, 'ok');
          break;
        }
      }

      // small pause to keep UI responsive
      await tf.nextFrame();
    }

    // Restore best snapshot if available
    if (bestSnapshot) {
      P.assign(bestSnapshot[0]); Q.assign(bestSnapshot[1]);
      bu.assign(bestSnapshot[2]); bi.assign(bestSnapshot[3]);
      bestSnapshot.forEach(t => t.dispose());
    }

    ds.train.users.dispose(); ds.train.items.dispose(); ds.train.ratings.dispose();
    ds.val.users.dispose(); ds.val.items.dispose(); ds.val.ratings.dispose();

    $('predict-btn').disabled = false;
    setStatus('Model training completed successfully!', 'ok');
    $('result').textContent = 'Pick a user and a movie, then click “Predict Rating”.';
  } catch (err) {
    console.error(err);
    setStatus('Training failed: ' + err.message, 'err');
  }
}

// ------------------------------------------------------------
// Predict a single (user, movie) rating using learned variables
// ------------------------------------------------------------
function predictRating() {
  const userId = parseInt($('user-select').value, 10);
  const movieId = parseInt($('movie-select').value, 10);
  const res = $('result');

  if (!Number.isFinite(userId) || !Number.isFinite(movieId)) {
    res.textContent = 'Please pick both user and movie.';
    return;
  }
  if (!P || !Q) {
    res.textContent = 'Model is not trained yet.';
    return;
  }

  const ui = userIdToIdx.get(userId);
  const mi = movieIdToIdx.get(movieId);
  if (ui == null || mi == null) {
    res.textContent = 'Unknown user or movie id.';
    return;
  }

  // Single-item tensors for gather()
  const u = tf.tensor1d([ui], 'int32');
  const i = tf.tensor1d([mi], 'int32');
  const yhat = predictBatch(u, i).clipByValue(1, 5);

  yhat.data().then(arr => {
    const pred = arr[0];
    const movie = movies.find(m => m.id === movieId);
    res.innerHTML = `Predicted rating for <b>User ${userId}</b> on <b>“${escapeHtml(movie?.title || movieId)}”</b>: <b>${pred.toFixed(2)}/5</b>`;
  }).finally(() => {
    u.dispose(); i.dispose(); yhat.dispose();
  });
}

// ------------------------------------------------------------
// Utility
// ------------------------------------------------------------
function escapeHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;')
    .replace(/'/g,'&#039;');
}
