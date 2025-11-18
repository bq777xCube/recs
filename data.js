// Global variables for storing movie data
let movies = [];
let ratings = []; // not used now, kept for compatibility

/**
 * Stage 1: Read Raw Data from movies_metadata.csv
 *
 * We use:
 * - id
 * - title / original_title
 * - overview
 * - genres (parsed from JSON-like string)
 * - popularity (for simple top-N filtering)
 */
async function loadData() {
  const resp = await fetch("movies_metadata.csv");
  if (!resp.ok) {
    throw new Error("Failed to load movies_metadata.csv");
  }
  const text = await resp.text();

  // PapaParse comes from CDN in index.html
  const parsed = Papa.parse(text, {
    header: true,
    skipEmptyLines: true
  });

  const tmp = [];

  parsed.data.forEach((row, idx) => {
    const idRaw = row.id;
    const id = Number.isFinite(parseInt(idRaw, 10)) ? parseInt(idRaw, 10) : idx;

    const title =
      row.title && row.title.trim()
        ? row.title.trim()
        : (row.original_title || "").trim();

    const overview = (row.overview || "").trim();

    // Combine title + overview for LLM description
    const description = title
      ? `${title}. ${overview}`
      : overview;

    // genres column is like: "[{'id': 16, 'name': 'Animation'}, ...]"
    let genres = [];
    const genresRaw = row.genres;
    if (genresRaw && genresRaw.trim() && genresRaw !== "[]") {
      try {
        // fix single quotes → double quotes
        const fixed = genresRaw.replace(/'/g, '"');
        const arr = JSON.parse(fixed);
        if (Array.isArray(arr)) {
          genres = arr
            .map(g => g && g.name)
            .filter(Boolean);
        }
      } catch (e) {
        console.warn("Failed to parse genres for row", idRaw, e);
      }
    }

    const popularity = parseFloat(row.popularity || "0") || 0;

    // compact rawLine for Stage 1 display
    const rawLine = `id=${idRaw}, title="${title}", genres=${genres.join(
      ", "
    )}`;

    if ((title || overview) && description) {
      tmp.push({
        id,
        title: title || `Movie ${id}`,
        description,
        overview,
        rawLine,
        genres,
        popularity
      });
    }
  });

  // Чтобы UI не умирал от десятков тысяч фильмов — возьмём топ-N по популярности
  tmp.sort((a, b) => b.popularity - a.popularity);
  const MAX_MOVIES = 300; // можно увеличить/уменьшить
  movies = tmp.slice(0, MAX_MOVIES);
  ratings = []; // не используем
}
