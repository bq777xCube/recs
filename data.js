// ======= Data & metadata =======
let movies = [];       // [{id,title,year}]
let ratings = [];      // [{userId,movieId,rating}]
let numUsers = 0;
let numMovies = 0;
let maxUserId = 0;
let maxMovieId = 0;
let userIdsSorted = [];

// Load u.item & u.data (MovieLens 100K)
async function loadData() {
  const movieResp = await fetch('u.item');
  if (!movieResp.ok) throw new Error(`Failed to load u.item: ${movieResp.status}`);
  movies = parseItemData(await movieResp.text());
  numMovies = movies.length;
  maxMovieId = movies.reduce((m, x) => Math.max(m, x.id), 0);

  const ratingResp = await fetch('u.data');
  if (!ratingResp.ok) throw new Error(`Failed to load u.data: ${ratingResp.status}`);
  ratings = parseRatingData(await ratingResp.text());

  const userSet = new Set(ratings.map(r => r.userId));
  userIdsSorted = Array.from(userSet).sort((a,b)=>a-b);
  numUsers = userSet.size;
  maxUserId = userIdsSorted.length ? Math.max(...userIdsSorted) : 0;

  return { movies, ratings, numUsers, numMovies, maxUserId, maxMovieId, userIdsSorted };
}

// Parse u.item
function parseItemData(text) {
  const out = [];
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    const parts = line.split('|');
    if (parts.length >= 2) {
      const id = parseInt(parts[0], 10);
      const rawTitle = parts[1];
      const m = rawTitle.match(/(.+)\s+\((\d{4})\)$/);
      out.push({
        id,
        title: m ? m[1].trim() : rawTitle,
        year: m ? parseInt(m[2], 10) : null
      });
    }
  }
  return out;
}

// Parse u.data (tab-separated)
function parseRatingData(text) {
  const out = [];
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    const p = line.split('\t');
    if (p.length >= 3) {
      out.push({ userId: +p[0], movieId: +p[1], rating: +p[2] });
    }
  }
  return out;
}
