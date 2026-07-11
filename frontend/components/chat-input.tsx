"use client"

import { useRef, useState } from "react"
import { ArrowUp } from "lucide-react"

const STARTERS = [
  "I love 80s sci-fi like Blade Runner",
  "Recommend a tense 90s crime thriller",
  "Something funny and quirky",
  "I want a scary movie for tonight",
]

export function ChatInput({
  onSend,
  loading,
  showStarters,
}: {
  onSend: (text: string) => void
  loading: boolean
  showStarters: boolean
}) {
  const [value, setValue] = useState("")
  const composingRef = useRef(false)

  function submit() {
    const text = value.trim()
    if (!text || loading) return
    onSend(text)
    setValue("")
  }

  return (
    <div className="border-t border-border bg-panel/60 p-3 backdrop-blur-md md:p-4">
      {showStarters && (
        <div className="mb-3 flex flex-wrap gap-2">
          {STARTERS.map((s) => (
            <button
              key={s}
              type="button"
              disabled={loading}
              onClick={() => onSend(s)}
              className="rounded-full border border-border bg-secondary/60 px-3 py-1.5 text-xs font-medium text-secondary-foreground transition-colors hover:border-primary/40 hover:bg-primary/10 hover:text-primary disabled:opacity-40"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      <div className="flex items-end gap-2 rounded-xl border border-border bg-card p-2 shadow-sm focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/30">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onCompositionStart={() => (composingRef.current = true)}
          onCompositionEnd={() => (composingRef.current = false)}
          onKeyDown={(e) => {
            if (
              e.key === "Enter" &&
              !e.shiftKey &&
              !composingRef.current &&
              e.nativeEvent.keyCode !== 229
            ) {
              e.preventDefault()
              submit()
            }
          }}
          rows={1}
          placeholder="Describe the kind of film you're in the mood for…"
          className="max-h-32 min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
        />
        <button
          type="button"
          onClick={submit}
          disabled={loading || !value.trim()}
          aria-label="Send message"
          className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <ArrowUp className="size-5" aria-hidden="true" />
        </button>
      </div>
    </div>
  )
}
