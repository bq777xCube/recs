// ===== Data parsing (MovieLens-like) =====
// We only need these globals in other files.
let movies = [];   // {id, title, genres[]}
let ratings = [];  // {userId, itemId, rating, timestamp}

// Genre names (your earlier setup; classic 100k has 19 w/ "unknown").
// Keep as-is for your u.item layout (our parser uses the LAST N columns).
const genreNames = [
  "Action", "Adventure", "Animation", "Children's", "Comedy",
  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
  "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
  "Thriller", "War", "Western"
];

async function loadData() {
  try {
    const moviesResponse = await fetch('u.item');
    if (!moviesResponse.ok) throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
    const moviesText = await moviesResponse.text();
    parseItemData(moviesText);

    const ratingsResponse = await fetch('u.data');
    if (!ratingsResponse.ok) throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
    const ratingsText = await ratingsResponse.text();
    parseRatingData(ratingsText);
  } catch (error) {
    console.error('Error loading data:', error);
    const statusLine = document.getElementById('status-line');
    if (statusLine) {
      statusLine.textContent = `Error: ${error.message}. Put u.item & u.data next to this page.`;
      statusLine.className = 'err';
    }
    throw error;
  }
}

// Robust u.item parser: assumes the last N fields are genre flags.
function parseItemData(text) {
  const lines = text.split('\n');
  for (const line of lines) {
    if (!line.trim()) continue;
    const fields = line.split('|');
    if (fields.length < 5) continue;

    const id = parseInt(fields[0], 10);
    const title = fields[1];

    const totalFields = fields.length;
    const N = genreNames.length;
    const startIdx = Math.max(0, totalFields - N);
    const genreBits = fields.slice(startIdx).map(v => Number.parseInt(v, 10) || 0);

    const genres = [];
    for (let i = 0; i < Math.min(N, genreBits.length); i++) {
      if (genreBits[i] === 1) genres.push(genreNames[i]);
    }
    movies.push({ id, title, genres });
  }
}

// u.data parser (userId, itemId, rating, timestamp)
function parseRatingData(text) {
  const lines = text.split('\n');
  for (const line of lines) {
    if (!line.trim()) continue;
    const [userId, itemId, rating, timestamp] = line.split('\t').map(x => Number(x));
    if (!Number.isFinite(userId) || !Number.isFinite(itemId) || !Number.isFinite(rating)) continue;
    ratings.push({ userId, itemId, rating, timestamp });
  }
}
