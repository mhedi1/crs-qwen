import { CATALOG, type CatalogMovie } from "./movies"
import type { Candidate, ChatResponse, Intent, Movie, Profile, Role, SelectedCandidate } from "./types"

const GENRE_KEYWORD_MAP: Record<string, string> = {
  horror: "Horror", scary: "Horror",
  comedy: "Comedy", comedic: "Comedy",
  action: "Action",
  thriller: "Thriller",
  romance: "Romance", romantic: "Romance",
  drama: "Drama", dramatic: "Drama",
  "sci-fi": "Sci-Fi", scifi: "Sci-Fi",
  fantasy: "Fantasy",
  mystery: "Mystery",
  animation: "Animation", animated: "Animation",
  documentary: "Documentary",
  western: "Western",
  war: "War",
  crime: "Crime",
  adventure: "Adventure",
}

const DECADE_PATTERNS: [RegExp, string][] = [
  [/\b50s\b|\b1950s\b|\bfifties\b/, "1950s"],
  [/\b60s\b|\b1960s\b|\bsixties\b/, "1960s"],
  [/\b70s\b|\b1970s\b|\bseventies\b/, "1970s"],
  [/\b80s\b|\b1980s\b|\beighties\b/, "1980s"],
  [/\b90s\b|\b1990s\b|\bnineties\b/, "1990s"],
  [/\b00s\b|\b2000s\b/, "2000s"],
  [/\b2010s\b/, "2010s"],
  [/\b2020s\b/, "2020s"],
]

const FOLLOW_UP_PATTERNS = [
  /\bwhy\b/, /\btell me more\b/, /\bwhat('| i)?s it about\b/, /\bwhat is it about\b/,
  /\bwho (directed|stars|made)\b/, /\bi (watched|saw|liked|loved|enjoyed) it\b/,
  /\bhow long\b/, /\bwhen (was|did)\b/, /\bmore about\b/, /\bwhat about (it|this)\b/,
  /\bsounds good\b/, /\bis it (scary|funny|sad|good|long)\b/,
]

export function extractGenres(text: string): string[] {
  const found: string[] = []
  const t = text.toLowerCase()
  for (const phrase of ["science fiction", "sci fi", "sci-fi"]) {
    if (t.includes(phrase) && !found.includes("Sci-Fi")) found.push("Sci-Fi")
  }
  for (const token of t.match(/[\w'-]+/g) ?? []) {
    const genre = GENRE_KEYWORD_MAP[token]
    if (genre && !found.includes(genre)) found.push(genre)
  }
  return found
}

export function extractDecades(text: string): string[] {
  const found: string[] = []
  const t = text.toLowerCase()
  for (const [pattern, decade] of DECADE_PATTERNS) {
    if (pattern.test(t) && !found.includes(decade)) found.push(decade)
  }
  return found
}

export function extractMentionedMovies(text: string): string[] {
  const found: string[] = []
  // Quoted titles
  for (const m of text.matchAll(/["“”‘’']([^"'“”]{3,60})["“”‘’']/g)) {
    const title = m[1].trim()
    if (title && !found.includes(title)) found.push(title)
  }
  // "like / loved / watched X"
  const re =
    /\b(?:like|loved?|watch(?:ed)?|seen|saw|enjoy(?:ed)?|similar to|such as)\s+([A-Z][A-Za-z0-9 :'!&-]{1,50}?)(?=[,.!?\n]|$|\s+(?:and|but|or|which|because)\b)/g
  for (const m of text.matchAll(re)) {
    const title = m[1].trim()
    if (title.length > 2 && !found.includes(title)) found.push(title)
  }
  return found
}

function classifyIntent(message: string, hasLastMovie: boolean): Intent {
  if (!hasLastMovie) return "NEW_PREFERENCE"
  const t = message.toLowerCase()
  // If the user names new preferences (genre/decade) or asks for something new, treat as new
  if (/\b(recommend|another|something (else|different)|suggest|give me|next|different)\b/.test(t)) {
    return "NEW_PREFERENCE"
  }
  for (const p of FOLLOW_UP_PATTERNS) {
    if (p.test(t)) return "FOLLOW_UP"
  }
  // Short reactions with no new signal → follow-up
  if (extractGenres(message).length === 0 && extractDecades(message).length === 0 && message.trim().split(/\s+/).length <= 5) {
    return "FOLLOW_UP"
  }
  return "NEW_PREFERENCE"
}

function scoreMovie(
  movie: CatalogMovie,
  genres: string[],
  decades: string[],
  mentioned: string[],
  fullText: string,
): number {
  let score = 0
  const t = fullText.toLowerCase()
  const g = genres.map((x) => x.toLowerCase())
  const movieGenres = movie.genre.toLowerCase()
  const kw = movie.keywords

  for (const genre of g) {
    if (movieGenres.includes(genre) || kw.includes(genre)) score += 15
  }
  for (const decade of decades) {
    if (movie.decade === decade) score += 12
  }
  for (const word of kw) {
    if (t.includes(word)) score += 2
  }
  // Soft popularity prior
  score += (movie.rating ?? 7) * 0.4
  return score
}

export interface EngineInput {
  message: string
  history: { role: Role; content: string }[]
  previouslyRecommended: string[]
  lastMovie: Movie | null
  turn: number
}

export function runPipeline(input: EngineInput): ChatResponse {
  const { message, history, previouslyRecommended, lastMovie } = input
  const turnNumber = input.turn + 1

  // Build cumulative profile from all user turns + current message
  const userTexts = [...history.filter((h) => h.role === "user").map((h) => h.content), message]
  const genres: string[] = []
  const decades: string[] = []
  const mentioned: string[] = []
  for (const txt of userTexts) {
    for (const x of extractGenres(txt)) if (!genres.includes(x)) genres.push(x)
    for (const x of extractDecades(txt)) if (!decades.includes(x)) decades.push(x)
    for (const x of extractMentionedMovies(txt)) if (!mentioned.includes(x)) mentioned.push(x)
  }

  const intent = classifyIntent(message, !!lastMovie)

  const profile: Profile = {
    genres,
    decades,
    mentioned_movies: mentioned,
    turn: turnNumber,
    seed_count: mentioned.length,
    fallback_used: false,
  }

  // ── FOLLOW_UP path: pipeline not re-run ─────────────────────────────
  if (intent === "FOLLOW_UP" && lastMovie) {
    return {
      response: followUpResponse(message, lastMovie),
      movie: lastMovie,
      candidates: [],
      selected_candidate: null,
      turn_number: turnNumber,
      profile,
      intent,
    }
  }

  // ── NEW_PREFERENCE path: full retrieval + rerank ────────────────────
  const fullText = userTexts.join(" ")
  const ranked = [...CATALOG]
    .map((m) => ({ m, score: scoreMovie(m, genres, decades, mentioned, fullText) }))
    .sort((a, b) => b.score - a.score)

  // Weak-seed fallback: no strong genre/decade/mention signal
  const weakSeed = genres.length === 0 && decades.length === 0 && mentioned.length === 0
  profile.fallback_used = weakSeed

  // Exclude already-recommended when possible
  const notRepeated = ranked.filter((r) => !previouslyRecommended.includes(r.m.title))
  const pool = notRepeated.length > 0 ? notRepeated : ranked

  const top5 = pool.slice(0, 5)
  const selected = top5[0].m

  const candidates: Candidate[] = pool.slice(0, 8).map((r) => ({
    title: r.m.title,
    genre: r.m.genre,
    decade: r.m.decade,
    score: Math.round(r.score * 10) / 10,
  }))

  const selectedCandidate: SelectedCandidate = {
    title: selected.title,
    genre: selected.genre,
    decade: selected.decade,
    kbrd_rank: 1,
    in_top5: true,
  }

  const movie: Movie = {
    title: selected.title,
    genre: selected.genre,
    decade: selected.decade,
    poster_url: selected.poster_url,
    rating: selected.rating,
    overview: selected.overview,
  }

  return {
    response: newRecommendationResponse(movie, { genres, decades, mentioned, weakSeed }),
    movie,
    candidates,
    selected_candidate: selectedCandidate,
    turn_number: turnNumber,
    profile,
    intent,
  }
}

function newRecommendationResponse(
  movie: Movie,
  ctx: { genres: string[]; decades: string[]; mentioned: string[]; weakSeed: boolean },
): string {
  const bits: string[] = []
  if (ctx.genres.length) bits.push(`your taste for ${humanList(ctx.genres.map((g) => g.toLowerCase()))}`)
  if (ctx.decades.length) bits.push(`the ${humanList(ctx.decades)} era`)
  if (ctx.mentioned.length) bits.push(`films in the vein of ${humanList(ctx.mentioned)}`)

  const rationale = bits.length
    ? `Based on ${humanList(bits)}, `
    : ctx.weakSeed
      ? "I didn't catch a strong preference yet, so here's a widely acclaimed pick — "
      : "Here's something I think you'll enjoy — "

  return (
    `${rationale}I'd recommend **${movie.title}** (${movie.decade}). ` +
    `${movie.overview} It sits squarely in the ${movie.genre.toLowerCase()} space. ` +
    `Want to know more about it, or should I suggest something different?`
  )
}

function followUpResponse(message: string, movie: Movie): string {
  const t = message.toLowerCase()
  if (/\bwhy\b/.test(t)) {
    return `I picked **${movie.title}** because it aligns closely with the preferences you've shared so far — a strong ${movie.genre.toLowerCase()} entry from the ${movie.decade}. ${movie.overview}`
  }
  if (/about|plot|story/.test(t)) {
    return `**${movie.title}** (${movie.decade}): ${movie.overview}`
  }
  if (/\b(watched|saw|liked|loved|enjoyed)\b/.test(t)) {
    return `Glad to hear it! Since you enjoyed **${movie.title}**, I can find something with a similar ${movie.genre.split(",")[0].toLowerCase()} feel — just say the word.`
  }
  return `Happy to go deeper on **${movie.title}** — it's a ${movie.genre.toLowerCase()} film from the ${movie.decade}. ${movie.overview} Ask me anything else, or I can recommend another title.`
}

function humanList(items: string[]): string {
  if (items.length === 0) return ""
  if (items.length === 1) return items[0]
  if (items.length === 2) return `${items[0]} and ${items[1]}`
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`
}
