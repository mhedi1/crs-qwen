"use client"

import { motion } from "motion/react"
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

  if (isUser) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 15 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 400, damping: 25 }}
        className="flex mb-4 w-full justify-end"
      >
        <div
          className="max-w-[72%] p-[10px_14px] text-[0.88rem] leading-[1.5] shadow-[0_1px_3px_rgba(0,0,0,0.1)] bg-primary text-white rounded-[14px_14px_2px_14px] whitespace-pre-wrap inline-block"
          style={{ wordBreak: "break-word" }}
        >
          {message.content}
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98, y: 15 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 400, damping: 25, delay: 0.05 }}
      className="flex mb-4 w-full justify-start group"
    >
      <div className="flex flex-col w-full max-w-[600px] items-start">
        {message.turnNumber !== undefined && (
          <div className="text-[0.72rem] font-semibold text-muted-foreground tracking-[0.06em] uppercase mb-1.5 px-1">
            Turn {message.turnNumber}
          </div>
        )}
        
        <div
          className="p-[14px_16px] text-[0.88rem] leading-[1.5] shadow-sm w-full bg-card border border-border text-foreground rounded-[16px_16px_16px_4px]"
          style={{ wordBreak: "break-word" }}
        >
          <motion.div 
            initial={{ opacity: 0, filter: "blur(8px)" }}
            animate={{ opacity: 1, filter: "blur(0px)" }}
            transition={{ duration: 0.6, ease: "easeOut", delay: 0.15 }}
            className="md-content"
          >
            {renderText(message.content)}
          </motion.div>
        </div>

        {message.movie && message.intent !== "FOLLOW_UP" && (
          <div className="mt-2 w-full">
            <MovieCard
              movie={message.movie}
              candidates={message.candidates ?? []}
              selectedCandidate={message.selectedCandidate}
              followUp={message.intent === "FOLLOW_UP"}
            />
          </div>
        )}
      </div>
    </motion.div>
  )
}
