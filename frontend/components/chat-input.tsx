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
    <div className="px-3 pb-4 pt-2 md:px-4">
      {showStarters && (
        <div className="mb-3 flex flex-wrap gap-2">
          {STARTERS.map((s) => (
            <button
              key={s}
              type="button"
              disabled={loading}
              onClick={() => onSend(s)}
              className="glass rounded-full border border-border px-3.5 py-1.5 text-xs font-medium text-muted-foreground transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:text-foreground disabled:opacity-40"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      <div className="glass flex items-end gap-2 rounded-2xl border border-border p-2 transition-all focus-within:border-primary/50 focus-within:glow-primary">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onCompositionStart={() => (composingRef.current = true)}
          onCompositionEnd={() => (composingRef.current = false)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !composingRef.current && e.nativeEvent.keyCode !== 229) {
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
          className="grad-primary flex size-9 shrink-0 items-center justify-center rounded-xl text-primary-foreground shadow-lg shadow-primary/30 transition-all hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
        >
          <ArrowUp className="size-5" aria-hidden="true" />
        </button>
      </div>
    </div>
  )
}
