export type Role = "user" | "system"

export interface Movie {
  title: string
  genre: string
  decade: string
  poster_url?: string | null
  rating?: number | null
  overview?: string
}

export interface Candidate {
  title: string
  genre?: string
  decade?: string
  score?: number
}

export interface SelectedCandidate {
  title: string
  genre?: string
  decade?: string
  kbrd_rank: number | null
  in_top5: boolean
}

export interface Profile {
  genres: string[]
  decades: string[]
  mentioned_movies: string[]
  turn: number
  seed_count: number
  fallback_used: boolean
}

export type Intent = "NEW_PREFERENCE" | "FOLLOW_UP"

export interface ChatResponse {
  response: string
  movie: Movie | null
  candidates: Candidate[]
  selected_candidate: SelectedCandidate | null
  turn_number: number
  profile: Profile
  intent: Intent
}

export interface ChatMessage {
  id: string
  role: Role
  content: string
  movie?: Movie | null
  candidates?: Candidate[]
  selectedCandidate?: SelectedCandidate | null
  intent?: Intent
  turnNumber?: number
}
