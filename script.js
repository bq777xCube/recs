// --- Fast lookup state for cosine ---
let genreIndex = null;    // Map genre -> index
let movieVectors = null;  // Float32Array[] aligned with movies[]
let movieById = new Map();

// Initialize the application when the window loads
window.onload = async function () {
  try {
    // Display loading message
    const resultElement = document.getElementById('result');
    resultElement.textContent = "Loading movie data...";
    resultElement.className = 'loading';

    // Load data
    await loadData();

    // Prepare indexes & vectors for cosine
    prepareIndexes();

    // Populate dropdown and update status
    populateMoviesDropdown();
    resultElement.textContent = "Data loaded. Please select a movie.";
    resultElement.className = 'success';
  } catch (error) {
    console.error('Initialization error:', error);
    // Error message already set in data.js if needed
  }
};

// Build helpers: genre index, vectors and id map
function prepareIndexes() {
  movieById.clear();
  const genres = new Set();

  for (const m of movies) {
    movieById.set(m.id, m);
    (m.genres || []).forEach(g => genres.add(String(g).trim()));
  }

  genreIndex = new Map([...genres].map((g, i) => [g, i]));

  const G = genreIndex.size;
  movieVectors = movies.map(m => vectorizeGenres(m, G));
}

// Turn a movie's genres into a one-hot vector
function vectorizeGenres(movie, G) {
  const v = new Float32Array(G);
  if (!movie || !Array.isArray(movie.genres)) return v;
  for (const g of movie.genres) {
    const idx = genreIndex.get(String(g).trim());
    if (idx !== undefined) v[idx] = 1;
  }
  return v;
}

// Populate the movies dropdown with sorted movie titles
function populateMoviesDropdown() {
  const selectElement = document.getElementById('movie-select');

  // Clear existing options except the first placeholder
  while (selectElement.options.length > 1) {
    selectElement.remove(1);
  }

  // Sort movies alphabetically by title
  const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));

  // Add movies to dropdown
  sortedMovies.forEach(movie => {
    const option = document.createElement('option');
    option.value = movie.id;
    option.textContent = movie.title;
    selectElement.appendChild(option);
  });
}

// Main recommendation function
function getRecommendations() {
  const resultElement = document.getElementById('result');
  const cards = document.getElementById('cards');
  cards.innerHTML = '';

  try {
    // Step 1: Get user input
    const selectElement = document.getElementById('movie-select');
    const algoElement = document.getElementById('algo-select');
    const kInput = document.getElementById('k-input');

    const selectedMovieId = parseInt(selectElement.value, 10);
    const method = (algoElement.value || 'jaccard').toLowerCase();
    const k = Math.max(1, Math.min(10, parseInt(kInput.value, 10) || 2));

    if (isNaN(selectedMovieId)) {
      resultElement.textContent = "Please select a movie first.";
      resultElement.className = 'error';
      return;
    }

    // Step 2: Find the liked movie
    const likedMovie = movieById.get(selectedMovieId);
    if (!likedMovie) {
      resultElement.textContent = "Error: Selected movie not found in database.";
      resultElement.className = 'error';
      return;
    }

    // Show loading message while processing
    resultElement.textContent = "Calculating recommendations...";
    resultElement.className = 'loading';

    // Use setTimeout to allow the UI to update before heavy computation
    setTimeout(() => {
      try {
        // Score candidates by selected method
        const scoredMovies = scoreCandidates(likedMovie, method);

        // Step 5: Sort by score in descending order
        scoredMovies.sort((a, b) => b.score - a.score);

        // Step 6: Select top-K recommendations
        const topRecommendations = scoredMovies.slice(0, k);

        // Step 7: Display results
        if (topRecommendations.length > 0) {
          resultElement.textContent = `Because you liked "${likedMovie.title}", we recommend:`;
          resultElement.className = 'success';

          for (const rec of topRecommendations) {
            const li = document.createElement('li');
            li.className = 'card';
            li.innerHTML = `
              <h3>${escapeHtml(rec.title)}</h3>
              <div class="badges">
                ${(rec.genres || []).map(g => `<span class="badge">${escapeHtml(g)}</span>`).join('')}
              </div>
              <p class="muted" style="margin:.5rem 0 0;">Score (${method}): ${rec.score.toFixed(3)}</p>
            `;
            cards.appendChild(li);
          }
        } else {
          resultElement.textContent = `No recommendations found for "${likedMovie.title}".`;
          resultElement.className = 'error';
        }
      } catch (error) {
        console.error('Error in recommendation calculation:', error);
        resultElement.textContent = "An error occurred while calculating recommendations.";
        resultElement.className = 'error';
      }
    }, 60);
  } catch (error) {
    console.error('Error in getRecommendations:', error);
    resultElement.textContent = "An unexpected error occurred.";
    resultElement.className = 'error';
  }
}

// Compute similarity scores for all candidates by method
function scoreCandidates(likedMovie, method) {
  const candidates = movies.filter(m => m.id !== likedMovie.id);

  if (method === 'jaccard') {
    const likedGenres = new Set((likedMovie.genres || []).map(g => String(g).trim()));
    return candidates.map(candidate => {
      const cGenres = new Set((candidate.genres || []).map(g => String(g).trim()));
      const inter = intersectionSize(likedGenres, cGenres);
      const uni = unionSize(likedGenres, cGenres);
      const score = uni > 0 ? inter / uni : 0;
      return { ...candidate, score };
    });
  }

  if (method === 'cosine') {
    const G = genreIndex ? genreIndex.size : 0;
    const likedVec = vectorizeGenres(likedMovie, G);
    return candidates.map(c => {
      const idx = movies.findIndex(m => m.id === c.id);
      const cVec = movieVectors[idx] || vectorizeGenres(c, G);
      const score = cosineSim(likedVec, cVec);
      return { ...c, score };
    });
  }

  // Fallback
  return scoreCandidates(likedMovie, 'jaccard');
}

// --- Set operations for Jaccard ---
function intersectionSize(aSet, bSet) {
  let count = 0;
  for (const x of aSet) if (bSet.has(x)) count++;
  return count;
}
function unionSize(aSet, bSet) {
  const seen = new Set(aSet);
  for (const x of bSet) seen.add(x);
  return seen.size;
}

// --- Cosine similarity on Float32Array vectors ---
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  const L = Math.min(a.length, b.length);
  for (let i = 0; i < L; i++) {
    const ai = a[i], bi = b[i];
    dot += ai * bi;
    na += ai * ai;
    nb += bi * bi;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom ? (dot / denom) : 0;
}

// --- Utilities ---
function escapeHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;')
    .replace(/'/g,'&#039;');
}
