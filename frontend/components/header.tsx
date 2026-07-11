"use client"

import { Clapperboard, Trash2 } from "lucide-react"

export function Header({ onClear, disabled }: { onClear: () => void; disabled: boolean }) {
  return (
    <header className="flex items-center justify-between border-b border-border bg-panel/60 px-4 py-3 backdrop-blur-md md:px-6">
      <div className="flex items-center gap-3">
        <div className="flex size-9 items-center justify-center rounded-lg bg-primary/15 ring-1 ring-primary/30">
          <Clapperboard className="size-5 text-primary" aria-hidden="true" />
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
        className="inline-flex items-center gap-1.5 rounded-md border border-border bg-secondary px-3 py-1.5 text-sm font-medium text-secondary-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Trash2 className="size-4" aria-hidden="true" />
        <span className="hidden sm:inline">Clear chat</span>
      </button>
    </header>
  )
}
