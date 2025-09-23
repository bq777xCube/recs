// Global variables for storing movie and rating data
let movies = [];
let ratings = [];

// Genre names as defined in your setup (make sure this matches your u.item layout).
// Note: classic MovieLens 100k has 19 genre flags (including "unknown").
// If your u.item omits "unknown", keep this list as-is.
const genreNames = [
  "Action", "Adventure", "Animation", "Children's", "Comedy",
  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
  "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
  "Thriller", "War", "Western"
];

// Primary function to load data from files
async function loadData() {
  try {
    // Load and parse movie data
    const moviesResponse = await fetch('u.item');
    if (!moviesResponse.ok) {
      throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
    }
    const moviesText = await moviesResponse.text();
    parseItemData(moviesText);

    // Load and parse rating data
    const ratingsResponse = await fetch('u.data');
    if (!ratingsResponse.ok) {
      throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
    }
    const ratingsText = await ratingsResponse.text();
    parseRatingData(ratingsText);
  } catch (error) {
    console.error('Error loading data:', error);
    const resultElement = document.getElementById('result');
    if (resultElement) {
      resultElement.textContent = `Error: ${error.message}. Please make sure u.item and u.data files are in the correct location.`;
      resultElement.className = 'error';
    }
    throw error; // Re-throw to allow script.js to handle the error
  }
}

// Parse movie data from u.item-like format
function parseItemData(text) {
  const lines = text.split('\n');

  for (const line of lines) {
    if (line.trim() === '') continue;

    const fields = line.split('|');
    if (fields.length < 5) continue; // Skip invalid lines

    const id = parseInt(fields[0], 10);
    const title = fields[1];

    // Robustly locate the genre bit-array as the LAST N columns
    const totalFields = fields.length;
    const N = genreNames.length; // expected number of genre flags
    const startIdx = Math.max(0, totalFields - N);
    const genreBitsRaw = fields.slice(startIdx, totalFields);
    const genreBits = genreBitsRaw.map(v => {
      const n = parseInt(v, 10);
      return Number.isFinite(n) ? n : 0;
    });

    // Build genres by matching 1-bits to genre names
    const genres = [];
    for (let i = 0; i < Math.min(N, genreBits.length); i++) {
      if (genreBits[i] === 1) genres.push(genreNames[i]);
    }

    movies.push({ id, title, genres });
  }
}

// Parse rating data from u.data format
function parseRatingData(text) {
  const lines = text.split('\n');

  for (const line of lines) {
    if (line.trim() === '') continue;

    const fields = line.split('\t');
    if (fields.length < 4) continue; // Skip invalid lines

    const userId = parseInt(fields[0], 10);
    const itemId = parseInt(fields[1], 10);
    const rating = parseFloat(fields[2]);
    const timestamp = parseInt(fields[3], 10);

    ratings.push({ userId, itemId, rating, timestamp });
  }
}
