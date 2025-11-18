// --- Global state ---

let movieById = new Map();

// LLM features per movie: id -> { sub_genre: [...], themes: [...] }
let llmFeaturesById = new Map();

// Master keyword lists (Stage 3)
let masterKeywords = {
  subGenres: [],
  themes: []
};

// indices for one-hot
let subGenreIndex = new Map();
let themeIndex = new Map();

// one-hot vectors for ALL movies (for similarity)
let movieVectors = []; // Float32Array[]

const USE_LLM = true; // use Gemini for selected movie

// ---------------------- Init ----------------------

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

  resultElement.textContent = "Loading movies_metadata.csv...";
  statusElement.textContent = "Stage 1: Reading raw CSV file...";

  // Stage 1: read data
  await loadData();
  movies.forEach(m => movieById.set(m.id, m));

  // Stage 3 + 4 initial: мастер-лист из жанров датасета
  rebuildMasterAndVectors();

  // UI
  populateMoviesDropdown();

  resultElement.textContent = "Pick a movie and click 'Get recommendations'.";
  statusElement.textContent =
    "Ready. Pipeline: Read → Extract (LLM) → Consolidate → Encode → Recommend.";

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
    input.value = stored.slice(0, 4) + "*****";
    status.textContent = "Key is stored locally.";
  }

  btn.addEventListener("click", () => {
    const raw = prompt(
      "Paste your Gemini API key (it will be saved in this browser only):"
    );
    if (!raw) {
      status.textContent = "Key not changed.";
      return;
    }
    setGeminiKey(raw);
    input.value = raw.slice(0, 4) + "*****";
    status.textContent = "Key saved locally.";
  });
}

// ---------------------- Stage 2: LLM extraction ----------------------

function cleanLLMJson(raw) {
  let txt = (raw || "").trim();

  if (txt.startsWith("```")) {
    txt = txt.replace(/^```[a-zA-Z0-9]*\s*/, "");
    const fenceIndex = txt.lastIndexOf("```");
    if (fenceIndex !== -1) {
      txt = txt.slice(0, fenceIndex);
    }
    txt = txt.trim();
  }

  const first = txt.indexOf("{");
  const last = txt.lastIndexOf("}");
  if (first !== -1 && last !== -1 && last > first) {
    txt = txt.slice(first, last + 1);
  }

  return txt.trim();
}

function normalizeList(value) {
  if (!value) return [];
  if (Array.isArray(value)) {
    return value.map(String).map(s => s.trim()).filter(Boolean);
  }
  return [String(value).trim()].filter(Boolean);
}

/**
 * Fallback: genres from dataset → used when LLM не сработал или не вызывали
 */
function extractFeaturesRuleBased(movie) {
  return {
    sub_genre: Array.isArray(movie.genres) ? movie.genres : [],
    themes: []
  };
}

/**
 * One-movie LLM extract (Stage 2 for selected movie)
 */
async function extractFeaturesLLM(movie) {
  const description = movie.description || movie.title || "";
  const prompt =
    `You are a movie data processing assistant.\n` +
    `Extract the following features from the movie description below.\n` +
    `Return the answer as a JSON object ONLY. Do not include markdown, backticks, comments, or extra text.\n\n` +
    `Fields:\n` +
    `- "sub-genre": list of sub-genres (e.g., ["Space Opera"], ["Heist"], ["Romantic Comedy"])\n` +
    `- "themes": list of themes (e.g., ["Good vs Evil", "Coming of Age", "Redemption"])\n\n` +
    `Description: ${description}\n\n` +
    `Output format example (structure only): {"sub-genre": ["Space Opera"], "themes": ["Good vs Evil", "Hope"]}`;

  const raw = await callGeminiModel(prompt);

  try {
    const cleaned = cleanLLMJson(raw);
    const parsed = JSON.parse(cleaned);

    const rawSub = parsed["sub-genre"] || parsed["sub_genre"] || [];
    const rawThemes = parsed["themes"] || [];
    return {
      sub_genre: normalizeList(rawSub),
      themes: normalizeList(rawThemes)
    };
  } catch (e) {
    console.warn(
      "Failed to parse single-movie LLM output, falling back to dataset genres",
      e,
      raw
    );
    return extractFeaturesRuleBased(movie);
  }
}

/**
 * Stage 2 main entry for selected movie.
 * First try cache, then LLM, then fallback.
 */
async function extractFeatures(movie) {
  const cached = llmFeaturesById.get(movie.id);
  if (cached) return cached;

  if (USE_LLM) {
    const f = await extractFeaturesLLM(movie);
    llmFeaturesById.set(movie.id, f);

    // новые sub-genre/themes → пересоберём мастер-лист и вектора
    rebuildMasterAndVectors();
    return f;
  }
  return extractFeaturesRuleBased(movie);
}

/**
 * Features used for similarity for ANY movie:
 * если есть LLM-фичи — используем их,
 * иначе — rule-based genres.
 */
function getFeaturesForSimilarity(movie) {
  const f = llmFeaturesById.get(movie.id);
  if (f) return f;
  return extractFeaturesRuleBased(movie);
}

// ---------------------- Stage 3 & 4: master lists + one-hot ----------------------

function rebuildMasterAndVectors() {
  const subSet = new Set();
  const themeSet = new Set();

  // Stage 3: master keywords
  for (const m of movies) {
    const f = getFeaturesForSimilarity(m);
    (f.sub_genre || []).forEach(s => s && subSet.add(s));
    (f.themes || []).forEach(t => t && themeSet.add(t));
  }

  masterKeywords.subGenres = Array.from(subSet).sort();
  masterKeywords.themes = Array.from(themeSet).sort();

  subGenreIndex = new Map();
  masterKeywords.subGenres.forEach((g, idx) => {
    subGenreIndex.set(g, idx);
  });

  themeIndex = new Map();
  masterKeywords.themes.forEach((t, idx) => {
    themeIndex.set(t, idx);
  });

  // Stage 4: one-hot for all movies based on current master lists
  movieVectors = movies.map(m => {
    const f = getFeaturesForSimilarity(m);
    return encodeFeatures(f);
  });
}

/**
 * Encode sub_genre + themes into one-hot vector:
 * [ all subGenres..., all themes... ]
 */
function encodeFeatures(features) {
  const dim = masterKeywords.subGenres.length + masterKeywords.themes.length;
  const vec = new Float32Array(dim);

  (features.sub_genre || []).forEach(g => {
    const idx = subGenreIndex.get(g);
    if (idx !== undefined) vec[idx] = 1;
  });

  (features.themes || []).forEach(t => {
    const idx = themeIndex.get(t);
    if (idx !== undefined) {
      const offset = masterKeywords.subGenres.length;
      vec[offset + idx] = 1;
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

/**
 * baseVec — вектор выбранного фильма (из LLM фичей),
 * остальные — уже в movieVectors.
 */
function getSimilarMovies(baseMovieId, baseVec, topK = 10) {
  const scores = [];

  for (let i = 0; i < movies.length; i++) {
    const m = movies[i];
    if (m.id === baseMovieId) continue;
    const sim = cosineSimilarity(baseVec, movieVectors[i]);
    if (sim <= 0) continue;
    scores.push({ movie: m, score: sim });
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

  statusElement.textContent =
    "Running pipeline: Extract (Gemini) → Consolidate → Encode → Cosine similarity...";
  resultElement.textContent = "";

  // Stage 2: LLM для выбранного фильма
  const extracted = await extractFeatures(movie);

  // Stage 4: вектор выбранного фильма по LLM-фичам
  const encodedBase = encodeFeatures(extracted);

  // Recommendations: similarity по LLM-вектору против всех movieVectors
  const recs = getSimilarMovies(movieId, encodedBase, 10);

  // Render recs
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

  // Обновляем отображение 4 стадий
  updatePipelineView(movie, extracted, encodedBase);

  statusElement.textContent = "Done. You can try another movie.";
}

function populateMoviesDropdown() {
  const selectElement = document.getElementById("movie-select");

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

  s3In.textContent = "[…LLM features collected for processed movies…]";
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
  const preview = vectorArray.slice(0, 24);
  s4Vec.textContent =
    "[ " +
    preview.map(v => v.toFixed(0)).join(", ") +
    (vectorArray.length > 24 ? ", … ]" : " ]");
}

// ---------------------- Gemini Call ----------------------

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
      maxOutputTokens: 512
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
