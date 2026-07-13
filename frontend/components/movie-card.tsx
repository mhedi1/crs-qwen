"use client"

import { useState, useRef } from "react"
import { motion, AnimatePresence } from "motion/react"
import type { Candidate, Movie, SelectedCandidate } from "@/lib/types"

interface MovieCardProps {
  movie: Movie
  candidates: Candidate[]
  selectedCandidate?: SelectedCandidate | null
  followUp: boolean
}

export function MovieCard({ movie, candidates, selectedCandidate, followUp }: MovieCardProps) {
  const [open, setOpen] = useState(false)
  const cardRef = useRef<HTMLDivElement>(null)
  const genres = movie.genre?.split(",").map((g) => g.trim()).filter(Boolean) ?? []

  return (
    <div ref={cardRef} className="mt-1.5 rounded-[16px] bg-card border border-border/80 shadow-sm overflow-hidden flex flex-col w-full max-w-full relative group transition-colors hover:border-primary/50">
      <div className="flex gap-3 p-3.5 items-stretch h-full">
        {/* Poster */}
        <div className="w-[80px] shrink-0 h-full flex items-center justify-center">
          {movie.poster_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={movie.poster_url || "/placeholder.svg"}
              alt={`Poster for ${movie.title}`}
              className="w-full h-auto rounded-[8px] object-cover block shadow-sm border border-border/40 transition-transform duration-500 group-hover:scale-105 group-hover:shadow-md"
            />
          ) : (
            <div className="w-full h-[120px] bg-[#f1f5f9] dark:bg-muted border border-[#cbd5e1] dark:border-border rounded-[8px] text-[#475569] dark:text-muted-foreground font-mono text-[0.6rem] text-center flex items-center justify-center leading-[1.3] p-1">
              POSTER<br />UNAVAILABLE
            </div>
          )}
        </div>

        {/* Meta */}
        <div className="flex-1 min-w-0 flex flex-col justify-center py-1">
          <div className="flex items-start justify-between gap-2">
            <h3 className="text-[1.1rem] font-bold text-foreground leading-tight tracking-tight mb-1.5">{movie.title}</h3>
            {typeof movie.rating === "number" && movie.rating > 0 && (
              <span className="shrink-0 inline-flex items-center gap-1.5 text-[0.75rem] font-bold bg-[#fef3c7] dark:bg-yellow-500/10 border border-[#fde68a] dark:border-yellow-500/20 text-[#d97706] dark:text-yellow-500 rounded-[6px] px-2 py-[2px]">
                <span className="text-[#f59e0b]">★</span> {movie.rating.toFixed(1)}
              </span>
            )}
          </div>
          
          <div className="flex gap-1.5 flex-wrap mb-1">
            {movie.decade && (
              <span className="inline-flex items-center gap-1 text-[0.7rem] font-medium bg-transparent border border-border/80 text-muted-foreground rounded-full px-2 py-0.5 leading-none">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="opacity-70"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
                {movie.decade}
              </span>
            )}
            {genres.map((g) => (
              <span
                key={g}
                className="inline-flex items-center text-[0.7rem] font-medium bg-[#eff6ff] dark:bg-blue-500/10 text-[#2563eb] dark:text-blue-400 rounded-full px-2 py-0.5 leading-none"
              >
                {g}
              </span>
            ))}
          </div>

          {movie.overview && (
            <p className="text-[0.75rem] text-muted-foreground/80 leading-[1.3] mt-1.5 italic">
              {movie.overview.split(/(?<=[.!?])\s+/)[0]}
            </p>
          )}

        </div>
      </div>

      {/* Follow-up notice */}
      {followUp && (
        <div className="px-4 pb-3">
          <span className="text-[0.75rem] text-muted-foreground opacity-80 italic">
            &#8629; Follow-up — pipeline not re-run
          </span>
        </div>
      )}

      {/* Pipeline details */}
      {!followUp && candidates.length > 0 && (
        <div className="px-4 pb-3">
          <button
            type="button"
            onClick={() => {
              setOpen((o) => {
                const next = !o
                if (next) {
                  setTimeout(() => {
                    cardRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" })
                  }, 250)
                }
                return next
              })
            }}
            aria-expanded={open}
            className="text-[0.75rem] font-medium text-muted-foreground transition-colors hover:text-foreground inline-flex items-center gap-1"
          >
            Pipeline Details 
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}>
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
          </button>

          <AnimatePresence initial={false}>
            {open && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.25, ease: "easeInOut" }}
                className="overflow-hidden"
              >
                <div className="mt-2 bg-secondary border border-border rounded-lg p-[10px_12px] text-[0.75rem] font-mono">
                  <div className="text-[0.65rem] font-bold tracking-[0.06em] uppercase text-muted-foreground mb-1.5">
                    Retrieved Candidates
                  </div>
                  <div className="flex flex-wrap gap-[6px] mb-1">
                    {candidates.map((c, i) => {
                      const isSel = selectedCandidate?.title === c.title
                      if (isSel) {
                        return (
                          <span key={i} className="text-[0.75rem] font-mono px-2.5 py-[3px] rounded-[20px] bg-[#d9770614] border-[1.5px] border-[#d97706] text-[#b45309] dark:text-[#fbbf24] whitespace-nowrap">
                            [+] {i + 1} {c.title}
                          </span>
                        )
                      }
                      if (c.source === "KBRD_NEURAL") {
                        return (
                          <span key={i} className="text-[0.75rem] font-mono px-2.5 py-[3px] rounded-[20px] bg-[#2563eb14] border border-[#2563eb4d] text-[#1d4ed8] dark:text-[#60a5fa] whitespace-nowrap">
                            {i + 1} {c.title}
                          </span>
                        )
                      }
                      return (
                        <span key={i} className="text-[0.75rem] font-mono px-2.5 py-[3px] rounded-[20px] bg-[#16a34a14] border border-[#16a34a4d] text-[#15803d] dark:text-[#4ade80] whitespace-nowrap">
                          {i + 1} {c.title}
                        </span>
                      )
                    })}
                  </div>
                  <div className="text-[0.7rem] text-muted-foreground font-mono italic mt-1 mb-3">
                    Showing top 5 of 50 retrieved candidates
                  </div>

                  {selectedCandidate && (
                    <div className="flex items-baseline flex-wrap gap-1.5 border-t border-[#d9770633] pt-[10px]">
                      <span className="text-[0.7rem] font-mono text-[var(--color-movie-gold)] whitespace-nowrap">Qwen selected:</span>
                      <span className="font-bold font-mono text-[var(--color-movie-gold)] text-[0.88rem]">{selectedCandidate.title}</span>
                      <span className="text-[0.78rem] font-mono text-muted-foreground">
                        {selectedCandidate.in_top5 && selectedCandidate.kbrd_rank != null ? (
                          <>
                            (ranked <strong className="text-[var(--color-movie-gold)]">#{selectedCandidate.kbrd_rank}</strong> in KBRD list){" "}
                            <span className="text-[0.72rem] bg-[#d9770614] border border-[#d9770633] rounded-[5px] px-1.5 py-[1px] text-[var(--color-movie-gold)]">(in top-5)</span>
                          </>
                        ) : selectedCandidate.in_top5 ? (
                          <span className="text-[0.72rem] bg-[#d9770614] border border-[#d9770633] rounded-[5px] px-1.5 py-[1px] text-[var(--color-movie-gold)]">(in top-5)</span>
                        ) : (
                          <span className="text-[0.72rem] bg-[#6b728014] border border-border rounded-[5px] px-1.5 py-[1px] text-muted-foreground">(ranked outside KBRD top-5)</span>
                        )}
                      </span>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}
