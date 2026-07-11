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
      initial={{ opacity: 0, y: 14, filter: "blur(4px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      className={cn("flex gap-3", isUser ? "justify-end" : "justify-start")}
    >
      {!isUser && (
        <div className="grad-primary mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-xl text-primary-foreground shadow-lg shadow-primary/25">
          <Sparkles className="size-4" aria-hidden="true" />
        </div>
      )}

      <div className={cn("flex max-w-[85%] flex-col gap-3", isUser && "items-end")}>
        <div
          className={cn(
            "px-4 py-2.5 text-sm leading-relaxed",
            isUser
              ? "grad-primary rounded-3xl rounded-br-md text-primary-foreground shadow-lg shadow-primary/20"
              : "glass rounded-3xl rounded-bl-md text-foreground",
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
        <div className="glass mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-xl text-muted-foreground">
          <User className="size-4" aria-hidden="true" />
        </div>
      )}
    </motion.div>
  )
}
