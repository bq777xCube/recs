// Global variables to store parsed data & metadata
let movies = [];       // [{ id, title, year }]
let ratings = [];      // [{ userId, movieId, rating }]
let numUsers = 0;
let numMovies = 0;

// Robust for embedding input dimensions
let maxUserId = 0;
let maxMovieId = 0;

// Domain for dropdown
let userIdsSorted = [];

/**
 * Load u.item and u.data from the same directory.
 */
async function loadData() {
  // Load movie data
  const movieResponse = await fetch('u.item');
  if (!movieResponse.ok) throw new Error(`Failed to load u.item: ${movieResponse.status}`);
  const movieText = await movieResponse.text();
  movies = parseItemData(movieText);
  numMovies = movies.length;
  maxMovieId = movies.reduce((m, x) => Math.max(m, x.id), 0);

  // Load rating data
  const ratingResponse = await fetch('u.data');
  if (!ratingResponse.ok) throw new Error(`Failed to load u.data: ${ratingResponse.status}`);
  const ratingText = await ratingResponse.text();
  ratings = parseRatingData(ratingText);

  // Users set, counts & max ID
  const userSet = new Set(ratings.map(r => r.userId));
  userIdsSorted = Array.from(userSet).sort((a, b) => a - b);
  numUsers = userSet.size;
  maxUserId = userIdsSorted.length ? Math.max(...userIdsSorted) : 0;

  console.log(
    `Loaded ${numMovies} movies (maxMovieId=${maxMovieId}) and ${ratings.length} ratings from ${numUsers} users (maxUserId=${maxUserId})`
  );

  return { movies, ratings, numUsers, numMovies, maxUserId, maxMovieId, userIdsSorted };
}

/** Parse MovieLens 100K u.item */
function parseItemData(text) {
  const lines = text.split('\n');
  const out = [];

  for (const line of lines) {
    if (!line.trim()) continue;
    const parts = line.split('|');
    if (parts.length >= 2) {
      const id = parseInt(parts[0], 10);
      const rawTitle = parts[1];
      const m = rawTitle.match(/(.+)\s+\((\d{4})\)$/);
      let title = rawTitle;
      let year = null;
      if (m) {
        title = m[1].trim();
        year = parseInt(m[2], 10);
      }
      out.push({ id, title, year });
    }
  }
  return out;
}

/** Parse MovieLens 100K u.data (tab-separated) */
function parseRatingData(text) {
  const lines = text.split('\n');
  const out = [];
  for (const line of lines) {
    if (!line.trim()) continue;
    const parts = line.split('\t');
    if (parts.length >= 3) {
      out.push({
        userId: parseInt(parts[0], 10),
        movieId: parseInt(parts[1], 10),
        rating: parseFloat(parts[2]),
      });
    }
  }
  return out;
}
