"use client"

import { Clapperboard, Trash2 } from "lucide-react"

export function Header({ onClear, disabled }: { onClear: () => void; disabled: boolean }) {
  return (
    <header className="flex items-center justify-between border-b border-border/70 px-4 py-3 md:px-6">
      <div className="flex items-center gap-3">
        <div className="grad-primary flex size-9 items-center justify-center rounded-xl text-primary-foreground shadow-lg shadow-primary/25">
          <Clapperboard className="size-5" aria-hidden="true" />
        </div>
        <div className="leading-tight">
          <h1 className="text-balance text-base font-semibold tracking-tight text-foreground">CineSeek</h1>
          <p className="font-mono text-[11px] text-muted-foreground">conversational recommender · KBRD + rerank</p>
        </div>
      </div>

      <button
        type="button"
        onClick={onClear}
        disabled={disabled}
        className="inline-flex items-center gap-1.5 rounded-full border border-border bg-secondary/60 px-3 py-1.5 text-sm font-medium text-secondary-foreground transition-all hover:border-border hover:bg-secondary disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Trash2 className="size-4" aria-hidden="true" />
        <span className="hidden sm:inline">Clear chat</span>
      </button>
    </header>
  )
}
