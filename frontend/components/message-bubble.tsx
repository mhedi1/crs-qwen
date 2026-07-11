"use client"

import { motion } from "motion/react"
import { Sparkles, User } from "lucide-react"
import type { ChatMessage } from "@/lib/types"
import { MovieCard } from "./movie-card"
import { cn } from "@/lib/utils"

function renderText(text: string) {
  // Render **bold** segments
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={i} className="font-semibold text-foreground">
          {part.slice(2, -2)}
        </strong>
      )
    }
    return <span key={i}>{part}</span>
  })
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user"

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={cn("flex gap-3", isUser ? "justify-end" : "justify-start")}
    >
      {!isUser && (
        <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/15 ring-1 ring-primary/30">
          <Sparkles className="size-4 text-primary" aria-hidden="true" />
        </div>
      )}

      <div className={cn("flex max-w-[85%] flex-col gap-3", isUser && "items-end")}>
        <div
          className={cn(
            "rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
            isUser
              ? "rounded-br-sm bg-primary text-primary-foreground"
              : "rounded-bl-sm bg-secondary text-secondary-foreground",
          )}
        >
          {renderText(message.content)}
        </div>

        {!isUser && message.movie && (
          <MovieCard
            movie={message.movie}
            candidates={message.candidates ?? []}
            selectedCandidate={message.selectedCandidate}
            followUp={message.intent === "FOLLOW_UP"}
          />
        )}
      </div>

      {isUser && (
        <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-secondary ring-1 ring-border">
          <User className="size-4 text-muted-foreground" aria-hidden="true" />
        </div>
      )}
    </motion.div>
  )
}
