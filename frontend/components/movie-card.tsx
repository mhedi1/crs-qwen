"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "motion/react"
import { ChevronDown, Star, Film, Calendar, Info, GitBranch } from "lucide-react"
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
    <div className="overflow-hidden rounded-xl border border-border bg-card shadow-lg shadow-black/20">
      <div className="flex gap-4 p-4">
        {/* Poster */}
        <div className="relative aspect-[2/3] w-24 shrink-0 overflow-hidden rounded-lg bg-muted ring-1 ring-border sm:w-28">
          {movie.poster_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={movie.poster_url || "/placeholder.svg"}
              alt={`Poster for ${movie.title}`}
              className="h-full w-full object-cover"
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center">
              <Film className="size-6 text-muted-foreground" aria-hidden="true" />
            </div>
          )}
        </div>

        {/* Meta */}
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-2">
            <h3 className="text-pretty text-lg font-semibold leading-tight text-card-foreground">{movie.title}</h3>
            {typeof movie.rating === "number" && (
              <span className="inline-flex shrink-0 items-center gap-1 rounded-md bg-accent/15 px-1.5 py-0.5 text-xs font-semibold text-accent ring-1 ring-accent/25">
                <Star className="size-3 fill-current" aria-hidden="true" />
                {movie.rating.toFixed(1)}
              </span>
            )}
          </div>

          <div className="mt-2 flex flex-wrap gap-1.5">
            {movie.decade && (
              <span className="inline-flex items-center gap-1 rounded-full border border-border bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground">
                <Calendar className="size-3" aria-hidden="true" />
                {movie.decade}
              </span>
            )}
            {genres.map((g) => (
              <span
                key={g}
                className="inline-flex items-center rounded-full border border-primary/25 bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary"
              >
                {g}
              </span>
            ))}
          </div>

          {movie.overview && (
            <p className="mt-2.5 line-clamp-3 text-sm leading-relaxed text-muted-foreground">{movie.overview}</p>
          )}
        </div>
      </div>

      {/* Follow-up notice */}
      {followUp && (
        <div className="flex items-center gap-2 border-t border-border bg-secondary/40 px-4 py-2 font-mono text-[11px] text-muted-foreground">
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
            className="flex w-full items-center justify-between px-4 py-2.5 text-left transition-colors hover:bg-secondary/40"
          >
            <span className="inline-flex items-center gap-2 font-mono text-xs font-medium text-muted-foreground">
              <GitBranch className="size-3.5 text-primary" aria-hidden="true" />
              Pipeline details
              <span className="rounded bg-secondary px-1.5 py-0.5 text-[10px] text-secondary-foreground">
                {candidates.length} candidates
              </span>
            </span>
            <ChevronDown
              className={cn("size-4 text-muted-foreground transition-transform", open && "rotate-180")}
              aria-hidden="true"
            />
          </button>

          <AnimatePresence initial={false}>
            {open && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.22, ease: "easeInOut" }}
                className="overflow-hidden"
              >
                <div className="px-4 pb-3.5">
                  {selectedCandidate && (
                    <p className="mb-2 font-mono text-[11px] text-muted-foreground">
                      Selected{" "}
                      <span className="text-primary">{selectedCandidate.title}</span>
                      {selectedCandidate.kbrd_rank != null && (
                        <> · KBRD rank #{selectedCandidate.kbrd_rank}</>
                      )}
                    </p>
                  )}
                  <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
                    {candidates.map((c, i) => {
                      const isSel = selectedCandidate?.title === c.title
                      return (
                        <div
                          key={`${c.title}-${i}`}
                          className={cn(
                            "shrink-0 rounded-lg border px-2.5 py-1.5",
                            isSel
                              ? "border-primary/50 bg-primary/10"
                              : "border-border bg-secondary/60",
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
    </div>
  )
}
