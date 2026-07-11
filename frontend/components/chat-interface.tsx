"use client"

import { useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { PanelRightClose, PanelRightOpen, Sparkles } from "lucide-react"
import type { ChatMessage, ChatResponse, Movie, Profile } from "@/lib/types"
import { Header } from "./header"
import { MessageBubble } from "./message-bubble"
import { ChatInput } from "./chat-input"
import { SystemIntel } from "./system-intel"

const EMPTY_PROFILE: Profile = {
  genres: [],
  decades: [],
  mentioned_movies: [],
  turn: 0,
  seed_count: 0,
  fallback_used: false,
}

function uid() {
  return Math.random().toString(36).slice(2)
}

function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="grad-primary mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-xl text-primary-foreground shadow-lg shadow-primary/25">
        <Sparkles className="size-4" aria-hidden="true" />
      </div>
      <div className="glass flex items-center gap-1.5 rounded-3xl rounded-bl-md px-4 py-3.5">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="size-2 rounded-full bg-primary"
            style={{ animation: "dot-pulse 1.2s ease-in-out infinite", animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
    </div>
  )
}

function EmptyState() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="flex flex-1 flex-col items-center justify-center px-6 py-16 text-center"
    >
      <div className="grad-primary mb-5 flex size-16 items-center justify-center rounded-2xl text-primary-foreground shadow-xl shadow-primary/40">
        <Sparkles className="size-8" aria-hidden="true" />
      </div>
      <h2 className="text-balance text-2xl font-semibold tracking-tight text-foreground md:text-3xl">
        What are you in the mood to watch?
      </h2>
      <p className="mt-3 max-w-md text-pretty text-sm leading-relaxed text-muted-foreground">
        Describe a genre, an era, or films you love. CineSeek runs a KBRD retrieval and rerank pipeline, then explains
        each pick — with full visibility into the candidates it considered.
      </p>
    </motion.div>
  )
}

export function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [profile, setProfile] = useState<Profile>(EMPTY_PROFILE)
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const lastMovieRef = useRef<Movie | null>(null)
  const prevRecommendedRef = useRef<string[]>([])
  const turnRef = useRef(0)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }, [messages, loading])

  async function handleSend(text: string) {
    if (loading) return
    const userMsg: ChatMessage = { id: uid(), role: "user", content: text }
    const history = messages.map((m) => ({ role: m.role, content: m.content }))
    setMessages((prev) => [...prev, userMsg])
    setLoading(true)

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history,
          previouslyRecommended: prevRecommendedRef.current,
          lastMovie: lastMovieRef.current,
          turn: turnRef.current,
        }),
      })
      if (!res.ok) throw new Error(`Request failed: ${res.status}`)
      const data = (await res.json()) as ChatResponse

      turnRef.current = data.turn_number
      setProfile(data.profile)
      if (data.movie) {
        lastMovieRef.current = data.movie
        if (data.intent === "NEW_PREFERENCE" && !prevRecommendedRef.current.includes(data.movie.title)) {
          prevRecommendedRef.current = [...prevRecommendedRef.current, data.movie.title]
        }
      }

      const systemMsg: ChatMessage = {
        id: uid(),
        role: "system",
        content: data.response,
        movie: data.movie,
        candidates: data.candidates,
        selectedCandidate: data.selected_candidate,
        intent: data.intent,
        turnNumber: data.turn_number,
      }
      setMessages((prev) => [...prev, systemMsg])
    } catch (err) {
      console.log("[v0] send error:", (err as Error).message)
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: "system",
          content: "I ran into an issue reaching the recommendation pipeline. Please try again.",
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function handleClear() {
    setMessages([])
    setProfile(EMPTY_PROFILE)
    lastMovieRef.current = null
    prevRecommendedRef.current = []
    turnRef.current = 0
  }

  return (
    <div className="app-aura flex h-dvh overflow-hidden bg-background">
      {/* Main column */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center">
          <div className="min-w-0 flex-1">
            <Header onClear={handleClear} disabled={messages.length === 0 && !loading} />
          </div>
          <button
            type="button"
            onClick={() => setSidebarOpen((o) => !o)}
            aria-label="Toggle System Intel panel"
            className="glass mr-3 flex size-9 items-center justify-center rounded-xl border border-border text-secondary-foreground transition-colors hover:bg-secondary lg:hidden"
          >
            {sidebarOpen ? <PanelRightClose className="size-4" /> : <PanelRightOpen className="size-4" />}
          </button>
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin">
          <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col px-4 py-6 md:px-6">
            {messages.length === 0 && !loading ? (
              <EmptyState />
            ) : (
              <div className="flex flex-col gap-6">
                {messages.map((m) => (
                  <MessageBubble key={m.id} message={m} />
                ))}
                {loading && <TypingIndicator />}
              </div>
            )}
          </div>
        </div>

        <div className="mx-auto w-full max-w-3xl">
          <ChatInput onSend={handleSend} loading={loading} showStarters={messages.length === 0} />
        </div>
      </div>

      {/* Desktop sidebar */}
      <div className="hidden w-80 shrink-0 lg:block">
        <SystemIntel profile={profile} />
      </div>

      {/* Mobile sidebar overlay */}
      <AnimatePresence>
        {sidebarOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSidebarOpen(false)}
              className="fixed inset-0 z-40 bg-black/50 lg:hidden"
            />
            <motion.div
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "spring", damping: 30, stiffness: 300 }}
              className="fixed inset-y-0 right-0 z-50 w-80 max-w-[85vw] lg:hidden"
            >
              <SystemIntel profile={profile} />
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}
