// --- Fast lookup + pipeline state ---
let movieById = new Map();

// One-hot encoding over canonical sub-genres (Stage 3 / 4)
let masterKeywords = {
  subGenres: [], // canonical list
  themes: []     // left empty in this MovieLens demo
};

let genreIndex = null;   // Map sub-genre -> index in vector
let movieVectors = [];   // Float32Array aligned with movies[]

const USE_LLM = true;    // use Gemini for Stage 2 in the browser

// ---------------------- Initialization ----------------------

window.addEventListener("load", () => {
  initApp().catch(err => {
    console.error(err);
    const resultElement = document.getElementById("result");
    resultElement.textContent = "Error while loading data.";
  });
});

async function initApp() {
  const resultElement = document.getElementById("result");
  const statusElement = document.getElementById("status");
  const selectElement = document.getElementById("movie-select");

  setupGeminiKeyUI();

  resultElement.textContent = "Loading MovieLens data...";
  statusElement.textContent = "Stage 1: Reading raw CSV files (u.item, u.data)...";

  // Stage 1: Read raw data
  await loadData();
  movies.forEach(m => movieById.set(m.id, m));

  // Stage 3 + 4: Build master lists & one-hot vectors
  prepareIndexes();

  // Populate dropdown
  populateMoviesDropdown();

  resultElement.textContent = "Pick a movie and click 'Get recommendations'.";
  statusElement.textContent = "Ready. Pipeline: Read → Extract → Consolidate → Encode.";

  // Button handler
  document
    .getElementById("recommend-btn")
    .addEventListener("click", async () => {
      const val = selectElement.value;
      if (!val) {
        statusElement.textContent = "Please select a movie first.";
        return;
      }
      const movieId = parseInt(val, 10);
      await handleRecommend(movieId);
    });
}

// ---------------------- Gemini key helpers ----------------------

function getGeminiKey() {
  return localStorage.getItem("gemini_api_key") || "";
}

function setGeminiKey(key) {
  if (key && key.trim()) {
    localStorage.setItem("gemini_api_key", key.trim());
  }
}

function setupGeminiKeyUI() {
  const input = document.getElementById("gemini-key-input");
  const btn = document.getElementById("save-key-btn");
  const status = document.getElementById("key-status");

  const stored = getGeminiKey();
  if (stored) {
    input.value = stored.slice(0, 4) + "*****"; // masked
    status.textContent = "Key is stored locally.";
  }

  btn.addEventListener("click", () => {
    const raw = prompt("Paste your Gemini API key (it will be saved in this browser only):");
    if (!raw) {
      status.textContent = "Key not changed.";
      return;
    }
    setGeminiKey(raw);
    input.value = raw.slice(0, 4) + "*****";
    status.textContent = "Key saved locally.";
  });
}

// ---------------------- Stage 2: Extract Features ----------------------

/**
 * Calls Gemini 2.0 Flash to extract features
 * from movie description.
 *
 * Expected JSON structure from LLM:
 * { "sub-genre": [...], "themes": [...] }
 */
async function extractFeaturesLLM(movie) {
  const description = movie.description || movie.title || "";
  const prompt =
    `Extract the following features from the movie description below. ` +
    `Return the answer as a JSON object.\n\n` +
    `- Sub-genre: (e.g., Space Opera, Heist, Romantic Comedy)\n` +
    `- Themes: (e.g., Good vs Evil, Coming of Age, Redemption)\n\n` +
    `Description: ${description}`;

  const raw = await callGeminiModel(prompt);

  // be defensive: if parsing fails, fall back to known genres
  try {
    const parsed = JSON.parse(raw);
    const rawSub = parsed["sub-genre"] || parsed["sub_genre"] || [];
    const rawThemes = parsed["themes"] || [];
    return {
      sub_genre: Array.isArray(rawSub) ? rawSub : [String(rawSub)],
      themes: Array.isArray(rawThemes) ? rawThemes : [String(rawThemes)]
    };
  } catch (e) {
    console.warn("Failed to parse LLM output, falling back to MovieLens genres", e);
    return {
      sub_genre: movie.genres || [],
      themes: []
    };
  }
}

/**
 * Fallback extractor without LLM (uses MovieLens genres as sub-genres).
 */
function extractFeaturesRuleBased(movie) {
  return {
    sub_genre: movie.genres || [],
    themes: []
  };
}

/**
 * Stage 2 main entry: choose between Gemini and fallback.
 */
async function extractFeatures(movie) {
  if (USE_LLM) {
    try {
      return await extractFeaturesLLM(movie);
    } catch (e) {
      console.warn("Gemini call failed, using fallback extractor", e);
      return extractFeaturesRuleBased(movie);
    }
  }
  return extractFeaturesRuleBased(movie);
}

// ---------------------- Stage 3 & 4: Master Lists + Encoding ----------------------

function prepareIndexes() {
  // Stage 3: master keyword lists
  const subGenresSet = new Set();
  for (const m of movies) {
    (m.genres || []).forEach(g => subGenresSet.add(g));
  }

  masterKeywords.subGenres = Array.from(subGenresSet).sort();
  masterKeywords.themes = []; // no themes in MovieLens 100k demo

  // Build index for subGenres
  genreIndex = new Map();
  masterKeywords.subGenres.forEach((g, idx) => {
    genreIndex.set(g, idx);
  });

  // Stage 4: create one-hot encoded vectors for all movies
  movieVectors = movies.map(m => {
    const features = extractFeaturesRuleBased(m); // dataset side uses genres
    return encodeFeatures(features);
  });
}

/**
 * Encode features into one-hot vector aligned with masterKeywords.subGenres.
 */
function encodeFeatures(features) {
  const dim = masterKeywords.subGenres.length;
  const vec = new Float32Array(dim);

  (features.sub_genre || []).forEach(g => {
    const idx = genreIndex.get(g);
    if (idx !== undefined) {
      vec[idx] = 1;
    }
  });

  return vec;
}

// ---------------------- Similarity & Recommendation ----------------------

function cosineSimilarity(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;

  const len = a.length;
  for (let i = 0; i < len; i++) {
    const x = a[i];
    const y = b[i];
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function getSimilarMovies(baseMovieId, topK = 10) {
  const baseIndex = movies.findIndex(m => m.id === baseMovieId);
  if (baseIndex === -1) return [];

  const baseVec = movieVectors[baseIndex];

  const scores = [];
  for (let i = 0; i < movies.length; i++) {
    if (i === baseIndex) continue;
    const sim = cosineSimilarity(baseVec, movieVectors[i]);
    if (sim <= 0) continue;
    scores.push({ movie: movies[i], score: sim });
  }

  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, topK);
}

// ---------------------- UI Handlers ----------------------

async function handleRecommend(movieId) {
  const statusElement = document.getElementById("status");
  const resultElement = document.getElementById("result");
  const cardsElement = document.getElementById("cards");

  const movie = movieById.get(movieId);
  if (!movie) {
    statusElement.textContent = "Movie not found.";
    return;
  }

  statusElement.textContent = "Running pipeline: Extract (Gemini) → Encode → Cosine similarity...";
  resultElement.textContent = "";

  // Stage 2: extract features for this specific movie (Gemini or fallback)
  const extracted = await extractFeatures(movie);

  // Stage 4: encode features (same encoder as dataset)
  const encoded = encodeFeatures(extracted);

  // Recommendations (content-based, cosine similarity)
  const recs = getSimilarMovies(movieId, 10);

  // Render recommendations
  cardsElement.innerHTML = "";
  if (recs.length === 0) {
    resultElement.textContent = "No similar movies found.";
  } else {
    resultElement.textContent = `Top ${recs.length} similar movies to "${movie.title}":`;
    for (const { movie: m, score } of recs) {
      const li = document.createElement("li");
      const titleSpan = document.createElement("span");
      titleSpan.className = "movie-title";
      titleSpan.textContent = m.title;

      const scoreSpan = document.createElement("span");
      scoreSpan.className = "score";
      scoreSpan.textContent = score.toFixed(3);

      li.appendChild(titleSpan);
      li.appendChild(scoreSpan);
      cardsElement.appendChild(li);
    }
  }

  // Update the 4-stage visualization
  updatePipelineView(movie, extracted, encoded);

  statusElement.textContent = "Done. You can try another movie.";
}

function populateMoviesDropdown() {
  const selectElement = document.getElementById("movie-select");

  // Clear existing options
  selectElement.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select a movie…";
  selectElement.appendChild(placeholder);

  const sortedMovies = [...movies].sort((a, b) =>
    a.title.localeCompare(b.title)
  );

  for (const m of sortedMovies) {
    const opt = document.createElement("option");
    opt.value = String(m.id);
    opt.textContent = m.title;
    selectElement.appendChild(opt);
  }
}

// ---------------------- Pipeline UI (Stages 1–4) ----------------------

function updatePipelineView(movie, extractedFeatures, encodedVector) {
  // Stage 1
  const s1In = document.getElementById("stage1-input");
  const s1Out = document.getElementById("stage1-output");

  s1In.textContent = movie.rawLine || "(raw CSV line not available)";
  s1Out.textContent = movie.description || movie.title || "";

  // Stage 2
  const s2In = document.getElementById("stage2-input");
  const s2Out = document.getElementById("stage2-output");

  const descriptionForPrompt = movie.description || movie.title || "";
  s2In.textContent =
`"Extract the following features from the movie description below. Return the answer as a JSON object.

- Sub-genre: (e.g., Space Opera, Heist, Romantic Comedy)
- Themes: (e.g., Good vs Evil, Coming of Age, Redemption)

Description: ${descriptionForPrompt}"`;

  s2Out.textContent = JSON.stringify(
    {
      "sub-genre": extractedFeatures.sub_genre || [],
      themes: extractedFeatures.themes || []
    },
    null,
    2
  );

  // Stage 3
  const s3In = document.getElementById("stage3-input");
  const s3Out = document.getElementById("stage3-output");

  s3In.textContent = "[…raw keywords from many movies…]\n(e.g., Sci-Fi, Science Fiction, Space Opera, Galactic Adventure)";
  s3Out.textContent = JSON.stringify(
    {
      master_sub_genres: masterKeywords.subGenres,
      master_themes: masterKeywords.themes
    },
    null,
    2
  );

  // Stage 4
  const s4Out = document.getElementById("stage4-output");
  const s4Vec = document.getElementById("stage4-vector");

  s4Out.textContent = JSON.stringify(
    {
      sub_genre: extractedFeatures.sub_genre || [],
      themes: extractedFeatures.themes || []
    },
    null,
    2
  );

  const vectorArray = Array.from(encodedVector);
  const preview = vectorArray.slice(0, 20);
  s4Vec.textContent =
    "[ " +
    preview.map(v => v.toFixed(0)).join(", ") +
    (vectorArray.length > 20 ? ", … ]" : " ]");
}

// ---------------------- Gemini Call (direct from browser) ----------------------

/**
 * Frontend → Google Generative Language API (Gemini 2.0 Flash).
 * Works on GitHub Pages, but your API key is used in the browser.
 * Recommend restricting the key by domain in Google Cloud console.
 */
async function callGeminiModel(prompt) {
  const apiKey = getGeminiKey();
  if (!apiKey) {
    throw new Error("Gemini API key is not set. Click 'Save' and paste the key.");
  }

  const url =
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent" +
    "?key=" +
    encodeURIComponent(apiKey);

  const body = {
    contents: [
      {
        parts: [{ text: prompt }]
      }
    ],
    generationConfig: {
      temperature: 0.2,
      maxOutputTokens: 256
    }
  };

  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  if (!resp.ok) {
    const text = await resp.text();
    console.error("Gemini error:", text);
    throw new Error("Gemini API call failed: " + resp.status);
  }

  const data = await resp.json();
  const text =
    data.candidates?.[0]?.content?.parts?.[0]?.text ??
    JSON.stringify(data);

  return text;
}
