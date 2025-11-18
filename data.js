// Global variables for storing movie and rating data
let movies = [];
let ratings = [];

// Genre names (MovieLens 100k without "unknown")
const genreNames = [
  "Action",
  "Adventure",
  "Animation",
  "Children's",
  "Comedy",
  "Crime",
  "Documentary",
  "Drama",
  "Fantasy",
  "Film-Noir",
  "Horror",
  "Musical",
  "Mystery",
  "Romance",
  "Sci-Fi",
  "Thriller",
  "War",
  "Western"
];

/**
 * Stage 1: Read Raw Data
 * Data engineering step: read CSV-like files and keep raw text.
 */
async function loadData() {
  const [itemsResp, ratingsResp] = await Promise.all([
    fetch("u.item"),
    fetch("u.data")
  ]);

  if (!itemsResp.ok || !ratingsResp.ok) {
    throw new Error("Failed to load u.item or u.data");
  }

  const [itemsText, ratingsText] = await Promise.all([
    itemsResp.text(),
    ratingsResp.text()
  ]);

  parseItemData(itemsText);
  parseRatingData(ratingsText);
}

// Parse movie data from u.item-like format
function parseItemData(text) {
  const lines = text.split("\n");

  for (const line of lines) {
    if (!line.trim()) continue;

    const fields = line.split("|");
    if (fields.length < 5) continue;

    const id = parseInt(fields[0], 10);
    const title = fields[1];

    // Treat the original line as raw CSV input for Stage 1 visualisation
    const rawLine = line;

    // Genre bits: last N columns
    const genreStart = fields.length - genreNames.length;
    const bitFields = fields.slice(genreStart);
    const bits = bitFields.map(v => parseInt(v, 10) || 0);

    // Convert bits to human-readable genres
    const genres = [];
    bits.forEach((b, idx) => {
      if (b === 1 && genreNames[idx]) {
        genres.push(genreNames[idx]);
      }
    });

    // For MovieLens 100k there is no overview; we use title as a placeholder
    const description = title;

    movies.push({
      id,
      title,
      description, // used as "raw text" for LLM stages
      rawLine,
      genreBits: bits,
      genres
    });
  }
}

// Parse ratings data (userId, itemId, rating, timestamp)
function parseRatingData(text) {
  const lines = text.split("\n");

  for (const line of lines) {
    if (!line.trim()) continue;

    const fields = line.split("\t");
    if (fields.length < 4) continue;

    const userId = parseInt(fields[0], 10);
    const itemId = parseInt(fields[1], 10);
    const rating = parseFloat(fields[2]);
    const timestamp = parseInt(fields[3], 10);

    ratings.push({ userId, itemId, rating, timestamp });
  }
}
