"use client"

import { useEffect, useRef, useState } from "react"
import { AnimatePresence, motion, useMotionValue, useTransform } from "motion/react"
import { PanelRightClose, PanelRightOpen, Sparkles, Clapperboard } from "lucide-react"
import type { ChatMessage, ChatResponse, Movie, Profile } from "@/lib/types"
import { Header } from "./header"
import { MessageBubble } from "./message-bubble"
import { ChatInput } from "./chat-input"
import { SystemIntel } from "./system-intel"

const STARTERS = [
  "I love 80s sci-fi like Blade Runner",
  "Recommend a tense 90s crime thriller",
  "Something funny and quirky",
  "I want a scary movie for tonight",
]

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
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex gap-4 items-end"
    >
      <div className="flex size-9 shrink-0 items-center justify-center rounded-[10px] bg-primary/10 text-primary shadow-sm border border-primary/20">
        <Sparkles className="size-4" aria-hidden="true" />
      </div>
      <div className="flex items-center gap-3 rounded-2xl rounded-bl-[10px] bg-card/60 backdrop-blur-md px-5 py-4 border border-border/50 shadow-sm">
        <div className="flex items-center gap-1.5">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="size-1.5 rounded-full bg-primary/70"
              style={{ animation: "dot-pulse 1.2s ease-in-out infinite", animationDelay: `${i * 0.15}s` }}
            />
          ))}
        </div>
        <span className="text-[0.85rem] font-medium text-muted-foreground animate-pulse">
          Analyzing request & consulting Qwen...
        </span>
      </div>
    </motion.div>
  )
}

function StarterCard({ text, onSend, index }: { text: string, onSend: (s:string)=>void, index: number }) {
  const x = useMotionValue(0)
  const y = useMotionValue(0)
  const rotateX = useTransform(y, [-50, 50], [8, -8])
  const rotateY = useTransform(x, [-150, 150], [-8, 8])

  function handleMouseMove(e: React.MouseEvent<HTMLButtonElement>) {
    const rect = e.currentTarget.getBoundingClientRect()
    x.set(e.clientX - rect.left - rect.width / 2)
    y.set(e.clientY - rect.top - rect.height / 2)
  }

  function handleMouseLeave() {
    x.set(0)
    y.set(0)
  }

  return (
    <motion.button
      style={{ rotateX, rotateY, transformPerspective: 1000 }}
      whileHover={{ scale: 1.02, zIndex: 10 }}
      whileTap={{ scale: 0.98 }}
      initial={{ opacity: 0, filter: "blur(10px)", y: 20 }}
      animate={{ opacity: 1, filter: "blur(0px)", y: 0 }}
      transition={{ delay: 0.2 + index * 0.1, duration: 0.5 }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={() => onSend(text)}
      className="flex items-center justify-between rounded-2xl border border-border/60 bg-card/60 backdrop-blur-xl p-4 text-left shadow-sm transition-colors hover:bg-muted/80 hover:shadow-xl hover:border-primary/30 cursor-pointer"
    >
      <span className="text-[0.9rem] font-medium text-foreground/80">{text}</span>
    </motion.button>
  )
}

function EmptyState({ onSend }: { onSend: (text: string) => void }) {
  const titleWords = "What are you in the mood to watch?".split(" ")

  return (
    <div className="flex flex-1 flex-col items-center justify-center px-4 py-10 md:py-16 text-center z-10 relative">
      <motion.div 
        initial={{ scale: 0.8, opacity: 0, filter: "blur(10px)" }}
        animate={{ scale: 1, opacity: 1, filter: "blur(0px)" }}
        transition={{ type: "spring", damping: 20, stiffness: 100 }}
        className="mb-8 flex size-16 items-center justify-center rounded-2xl bg-card/60 backdrop-blur-xl border border-border/60 shadow-lg"
      >
        <Clapperboard className="size-7 text-primary" aria-hidden="true" />
      </motion.div>
      <div className="text-balance text-[28px] md:text-[34px] font-bold tracking-tight text-foreground flex flex-wrap justify-center gap-x-[0.25em] mb-2">
        {titleWords.map((word, i) => (
          <motion.span
            key={i}
            initial={{ filter: "blur(12px)", opacity: 0, y: 15 }}
            animate={{ filter: "blur(0px)", opacity: 1, y: 0 }}
            transition={{ delay: 0.1 + i * 0.05, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          >
            {word}
          </motion.span>
        ))}
      </div>

      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4, duration: 0.8 }}
        className="mt-12 grid w-full max-w-[700px] grid-cols-1 gap-4 sm:grid-cols-2 perspective-[1000px]"
      >
        {STARTERS.map((s, i) => (
          <StarterCard key={s} text={s} onSend={onSend} index={i} />
        ))}
      </motion.div>
    </div>
  )
}

export function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [profile, setProfile] = useState<Profile>(EMPTY_PROFILE)
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [initialized, setInitialized] = useState(false)

  const lastMovieRef = useRef<Movie | null>(null)
  const prevRecommendedRef = useRef<string[]>([])
  const turnRef = useRef(0)
  const scrollRef = useRef<HTMLDivElement>(null)
  const dragConstraintsRef = useRef<HTMLDivElement>(null)

  // Load state on mount
  useEffect(() => {
    const saved = sessionStorage.getItem("chat_state")
    if (saved) {
      try {
        const data = JSON.parse(saved)
        if (data.messages) setMessages(data.messages)
        if (data.profile) setProfile(data.profile)
        if (data.lastMovie) lastMovieRef.current = data.lastMovie
        if (data.prevRecommended) prevRecommendedRef.current = data.prevRecommended
        if (data.turn) turnRef.current = data.turn
      } catch (e) {
        console.error("Failed to parse chat state", e)
      }
    }
    setInitialized(true)
  }, [])

  // Save state on change
  useEffect(() => {
    if (!initialized) return
    const state = {
      messages,
      profile,
      lastMovie: lastMovieRef.current,
      prevRecommended: prevRecommendedRef.current,
      turn: turnRef.current
    }
    sessionStorage.setItem("chat_state", JSON.stringify(state))
  }, [messages, profile, initialized])

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
        credentials: "include",
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

  async function handleClear() {
    try {
      await fetch('/api/clear', { 
        method: 'POST', 
        credentials: 'include' 
      })
    } catch (e) {
      console.error('Failed to clear backend state', e)
    }
    setMessages([])
    setProfile(EMPTY_PROFILE)
    lastMovieRef.current = null
    prevRecommendedRef.current = []
    turnRef.current = 0
    sessionStorage.removeItem("chat_state")
  }

  return (
    <div ref={dragConstraintsRef} className="flex h-dvh overflow-hidden bg-background relative">
      <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
        <div className="absolute -top-[20%] -left-[10%] w-[60vw] h-[60vw] rounded-full bg-primary/10 blur-[120px] animate-aurora-1 opacity-60 mix-blend-screen dark:opacity-30" />
        <div className="absolute top-[40%] -right-[10%] w-[50vw] h-[50vw] rounded-full bg-movie-gold/10 blur-[120px] animate-aurora-2 opacity-50 mix-blend-screen dark:opacity-20" />
        <div className="absolute -bottom-[20%] left-[20%] w-[70vw] h-[70vw] rounded-full bg-primary/10 blur-[120px] animate-aurora-3 opacity-40 mix-blend-screen dark:opacity-20" />
        <div className="absolute inset-0 bg-background/60 backdrop-blur-[40px]" />
      </div>

      {/* Main column */}
      <div className="flex min-w-0 flex-1 flex-col z-10 relative">
        <div className="flex items-center border-b border-border bg-card">
          <div className="min-w-0 flex-1">
            <Header onClear={handleClear} disabled={messages.length === 0 && !loading} />
          </div>
          <button
            type="button"
            onClick={() => setSidebarOpen((o) => !o)}
            aria-label="Toggle System Intel panel"
            className="mr-4 flex size-9 items-center justify-center rounded-md border border-border bg-card text-muted-foreground transition-colors hover:bg-muted hover:text-foreground cursor-pointer shadow-sm hover:shadow"
          >
            {sidebarOpen ? <PanelRightClose className="size-4" /> : <PanelRightOpen className="size-4" />}
          </button>
        </div>

        <div id="chat-scroll-container" ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin pb-32 scroll-pb-[140px]">
          <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col px-4 py-8 md:px-8">
            {messages.length === 0 && !loading ? (
              <EmptyState onSend={handleSend} />
            ) : (
              <div className="flex flex-col gap-8 pb-8">
                {messages.map((m) => (
                  <MessageBubble key={m.id} message={m} />
                ))}
                {loading && <TypingIndicator />}
              </div>
            )}
          </div>
        </div>

        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-background via-background to-transparent pb-6 pt-10 px-4 md:px-8">
          <ChatInput onSend={handleSend} loading={loading} />
        </div>
      </div>

      {/* Draggable Floating Widget */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            drag
            dragConstraints={dragConstraintsRef}
            dragElastic={0.1}
            dragMomentum={false}
            initial={{ opacity: 0, scale: 0.9, y: -20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: -20 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            className="absolute right-4 top-20 z-50 w-80 max-h-[70vh] flex flex-col bg-card/70 backdrop-blur-3xl border border-border/50 shadow-[0_20px_60px_rgba(0,0,0,0.15)] dark:shadow-[0_20px_60px_rgba(0,0,0,0.4)] rounded-[20px] overflow-hidden cursor-grab active:cursor-grabbing"
            style={{ touchAction: "none" }}
          >
            {/* Drag Handle Area */}
            <div className="w-full h-8 bg-black/5 dark:bg-white/5 flex items-center justify-center border-b border-border/30 shrink-0">
              <div className="w-10 h-1 bg-foreground/20 rounded-full" />
            </div>
            
            {/* Content Area - prevent dragging when interacting with scrollable content */}
            <div 
              className="flex-1 overflow-hidden" 
              onPointerDownCapture={(e) => e.stopPropagation()}
              style={{ cursor: "default" }}
            >
              <SystemIntel profile={profile} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
