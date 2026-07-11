"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "motion/react"
import { ChevronDown, Star, Film, Info, GitBranch } from "lucide-react"
import type { Candidate, Movie, SelectedCandidate } from "@/lib/types"
import { cn } from "@/lib/utils"

interface MovieCardProps {
  movie: Movie
  candidates: Candidate[]
  selectedCandidate?: SelectedCandidate | null
  followUp: boolean
}

export function MovieCard({ movie, candidates, selectedCandidate, followUp }: MovieCardProps) {
  const [open, setOpen] = useState(false)
  const genres = movie.genre?.split(",").map((g) => g.trim()).filter(Boolean) ?? []

  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1], delay: 0.05 }}
      className="overflow-hidden rounded-2xl border border-border bg-card/80 shadow-2xl shadow-black/40 backdrop-blur-sm"
    >
      <div className="flex gap-4 p-4">
        {/* Poster */}
        <div className="group relative aspect-[2/3] w-24 shrink-0 overflow-hidden rounded-xl bg-muted shadow-lg shadow-black/40 ring-1 ring-border sm:w-28">
          {movie.poster_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={movie.poster_url || "/placeholder.svg"}
              alt={`Poster for ${movie.title}`}
              className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center">
              <Film className="size-6 text-muted-foreground" aria-hidden="true" />
            </div>
          )}
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/40 to-transparent" />
        </div>

        {/* Meta */}
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <h3 className="text-pretty text-lg font-semibold leading-tight text-card-foreground">{movie.title}</h3>
              {movie.decade && (
                <p className="mt-0.5 font-mono text-xs text-muted-foreground">{movie.decade}</p>
              )}
            </div>
            {typeof movie.rating === "number" && (
              <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-accent/15 px-2 py-1 text-xs font-bold text-accent ring-1 ring-accent/30">
                <Star className="size-3 fill-current" aria-hidden="true" />
                {movie.rating.toFixed(1)}
                <span className="font-medium text-accent/70">/10</span>
              </span>
            )}
          </div>

          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {genres.map((g) => (
              <span
                key={g}
                className="inline-flex items-center rounded-full bg-primary/15 px-2.5 py-1 text-xs font-medium text-primary ring-1 ring-primary/25"
              >
                {g}
              </span>
            ))}
          </div>

          {movie.overview && (
            <p className="mt-3 line-clamp-3 text-sm leading-relaxed text-muted-foreground">{movie.overview}</p>
          )}
        </div>
      </div>

      {/* Follow-up notice */}
      {followUp && (
        <div className="flex items-center gap-2 border-t border-border bg-secondary/30 px-4 py-2.5 font-mono text-[11px] text-muted-foreground">
          <Info className="size-3.5 text-primary" aria-hidden="true" />
          Follow-up: pipeline not re-run
        </div>
      )}

      {/* Pipeline details */}
      {!followUp && candidates.length > 0 && (
        <div className="border-t border-border">
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="flex w-full items-center justify-between px-4 py-3 text-left transition-colors hover:bg-secondary/40"
          >
            <span className="inline-flex items-center gap-2 font-mono text-xs font-medium text-muted-foreground">
              <GitBranch className="size-3.5 text-primary" aria-hidden="true" />
              Pipeline details
              <span className="rounded-full bg-secondary px-2 py-0.5 text-[10px] text-secondary-foreground">
                {candidates.length} candidates
              </span>
            </span>
            <ChevronDown
              className={cn("size-4 text-muted-foreground transition-transform duration-300", open && "rotate-180")}
              aria-hidden="true"
            />
          </button>

          <AnimatePresence initial={false}>
            {open && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
                className="overflow-hidden"
              >
                <div className="px-4 pb-4">
                  {selectedCandidate && (
                    <p className="mb-2.5 font-mono text-[11px] text-muted-foreground">
                      Selected <span className="text-primary">{selectedCandidate.title}</span>
                      {selectedCandidate.kbrd_rank != null && <> · KBRD rank #{selectedCandidate.kbrd_rank}</>}
                    </p>
                  )}
                  <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
                    {candidates.map((c, i) => {
                      const isSel = selectedCandidate?.title === c.title
                      return (
                        <div
                          key={`${c.title}-${i}`}
                          className={cn(
                            "shrink-0 rounded-xl px-3 py-2 transition-colors",
                            isSel
                              ? "bg-primary/15 ring-1 ring-primary/40"
                              : "bg-secondary/60 ring-1 ring-border",
                          )}
                        >
                          <div className="flex items-center gap-1.5">
                            <span className="font-mono text-[10px] text-muted-foreground">#{i + 1}</span>
                            <span
                              className={cn(
                                "whitespace-nowrap text-xs font-medium",
                                isSel ? "text-primary" : "text-secondary-foreground",
                              )}
                            >
                              {c.title}
                            </span>
                          </div>
                          {typeof c.score === "number" && (
                            <span className="mt-0.5 block font-mono text-[10px] text-muted-foreground">
                              score {c.score}
                            </span>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </motion.div>
  )
}
