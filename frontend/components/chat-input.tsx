"use client"

import { useRef, useState } from "react"
import { ArrowUp } from "lucide-react"
import { motion } from "motion/react"

export function ChatInput({
  onSend,
  loading,
}: {
  onSend: (text: string) => void
  loading: boolean
}) {
  const [value, setValue] = useState("")
  const composingRef = useRef(false)

  const [focused, setFocused] = useState(false)

  function submit() {
    const text = value.trim()
    if (!text || loading) return
    onSend(text)
    setValue("")
  }

  return (
    <div className="relative w-full max-w-[760px] mx-auto group flex justify-center">
      <motion.div 
        layout
        transition={{ type: "spring", stiffness: 300, damping: 25 }}
        style={{ width: focused || value ? "100%" : "65%" }}
        className="relative flex items-center bg-card/80 backdrop-blur-xl border border-border/60 shadow-[0_8px_30px_rgb(0,0,0,0.06)] dark:shadow-[0_8px_30px_rgb(0,0,0,0.3)] rounded-[28px] px-3 py-2 transition-colors duration-300 focus-within:shadow-[0_16px_50px_rgba(37,99,235,0.2)] focus-within:border-primary/50 hover:border-border cursor-text"
      >
        <textarea
          value={value}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
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
          placeholder="Describe the film you want…"
          className="max-h-32 min-h-[44px] flex-1 resize-none bg-transparent px-4 py-3 text-[15px] text-foreground placeholder:text-muted-foreground focus:outline-none"
        />
        <button
          type="button"
          onClick={submit}
          disabled={loading || !value.trim()}
          className="flex size-9 shrink-0 items-center justify-center rounded-full bg-primary text-white transition-all hover:bg-primary/90 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-50 absolute right-3 bottom-3.5"
        >
          <ArrowUp className="size-5" strokeWidth={2.5} aria-hidden="true" />
        </button>
      </motion.div>
    </div>
  )
}
