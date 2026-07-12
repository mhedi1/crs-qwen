"use client"

import { Clapperboard, Trash2, Sun, Moon } from "lucide-react"
import { useTheme } from "next-themes"
import { useEffect, useState } from "react"
import Link from "next/link"

export function Header({ onClear, disabled }: { onClear: () => void; disabled: boolean }) {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  // Avoid hydration mismatch
  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <nav className="flex w-full items-center justify-between px-4 py-3 bg-card border-b border-border shrink-0">
      <Link href="/" className="flex items-center gap-3 transition-opacity hover:opacity-80 no-underline cursor-pointer">
        <div className="flex size-10 items-center justify-center rounded-[10px] bg-primary text-white">
          <Clapperboard className="size-5" aria-hidden="true" />
        </div>
        <div className="leading-tight">
          <h1 className="text-[17px] font-bold text-foreground m-0 tracking-normal">CineSeek</h1>
          <p className="text-[13px] text-muted-foreground m-0">Conversational recommender</p>
        </div>
      </Link>

      <div className="flex items-center gap-2">
        {mounted && (
          <button
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="inline-flex items-center gap-1.5 rounded-md border border-border bg-transparent px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            aria-label="Toggle theme"
          >
            {theme === "dark" ? (
              <>
                <Sun className="h-4 w-4" />
                <span className="hidden sm:inline">Light Mode</span>
              </>
            ) : (
              <>
                <Moon className="h-4 w-4" />
                <span className="hidden sm:inline">Dark Mode</span>
              </>
            )}
          </button>
        )}

        <button
          type="button"
          onClick={onClear}
          disabled={disabled}
          className="inline-flex items-center gap-1.5 rounded-md border border-border bg-transparent px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Trash2 className="size-4" aria-hidden="true" />
          <span className="hidden sm:inline">Clear chat</span>
        </button>
      </div>
    </nav>
  )
}
