"use client"

import { motion, AnimatePresence } from "motion/react"
import { Activity, Clapperboard, Clock, Film, Hash, Sparkles, Tag } from "lucide-react"
import type { Profile } from "@/lib/types"

function Section({
  icon,
  label,
  children,
}: {
  icon: React.ReactNode
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="border-b border-border px-4 py-3.5 last:border-b-0">
      <div className="mb-2 flex items-center gap-1.5">
        <span className="text-primary">{icon}</span>
        <h3 className="font-mono text-[11px] font-medium uppercase tracking-wider text-muted-foreground">{label}</h3>
      </div>
      {children}
    </div>
  )
}

function Chips({ items, empty }: { items: string[]; empty: string }) {
  if (items.length === 0) {
    return <p className="font-mono text-xs text-muted-foreground/50">{empty}</p>
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      <AnimatePresence initial={false}>
        {items.map((item) => (
          <motion.span
            key={item}
            initial={{ opacity: 0, scale: 0.85 }}
            animate={{ opacity: 1, scale: 1 }}
            className="rounded-full bg-secondary px-2.5 py-0.5 text-xs font-medium text-secondary-foreground ring-1 ring-border"
          >
            {item}
          </motion.span>
        ))}
      </AnimatePresence>
    </div>
  )
}

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: number }) {
  return (
    <div className="glass rounded-2xl border border-border px-4 py-3">
      <div className="flex items-center gap-1.5 text-muted-foreground">
        {icon}
        <span className="font-mono text-[10px] uppercase tracking-wider">{label}</span>
      </div>
      <p className="mt-1 font-mono text-2xl font-semibold text-foreground">{value}</p>
    </div>
  )
}

export function SystemIntel({ profile }: { profile: Profile }) {
  return (
    <aside className="flex h-full w-full flex-col overflow-hidden border-l border-border bg-panel/60 backdrop-blur-xl">
      <div className="flex items-center gap-2 border-b border-border px-4 py-3.5">
        <Activity className="size-4 text-primary" aria-hidden="true" />
        <h2 className="text-sm font-semibold tracking-tight text-foreground">System Intel</h2>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <div className="grid grid-cols-2 gap-3 px-4 py-4">
          <StatCard icon={<Clock className="size-3.5" aria-hidden="true" />} label="Turn" value={profile.turn} />
          <StatCard icon={<Hash className="size-3.5" aria-hidden="true" />} label="Seeds" value={profile.seed_count} />
        </div>

        <div className="border-t border-border">
          <Section icon={<Tag className="size-3.5" />} label="Genre preference">
            <Chips items={profile.genres} empty="none detected" />
          </Section>

          <Section icon={<Clock className="size-3.5" />} label="Decade preference">
            <Chips items={profile.decades} empty="none detected" />
          </Section>

          <Section icon={<Film className="size-3.5" />} label="Extracted seeds">
            <Chips items={profile.mentioned_movies} empty="none extracted yet" />
          </Section>

          <Section icon={<Clapperboard className="size-3.5" />} label="Mentioned films">
            <Chips items={profile.mentioned_movies} empty="no films referenced" />
          </Section>

          <Section icon={<Sparkles className="size-3.5" />} label="Pipeline status">
            <div className="flex items-center gap-2">
              <span className="relative flex size-2">
                <span
                  className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${profile.fallback_used ? "bg-accent" : "bg-primary"}`}
                  aria-hidden="true"
                />
                <span
                  className={`relative inline-flex size-2 rounded-full ${profile.fallback_used ? "bg-accent" : "bg-primary"}`}
                  aria-hidden="true"
                />
              </span>
              <span className="font-mono text-xs text-muted-foreground">
                {profile.fallback_used ? "weak-seed fallback active" : "seed-driven retrieval"}
              </span>
            </div>
          </Section>
        </div>
      </div>

      <div className="border-t border-border px-4 py-3">
        <p className="font-mono text-[10px] leading-relaxed text-muted-foreground/60">
          KBRD retrieval → Qwen rerank → response generation. Intent gating skips retrieval on follow-ups.
        </p>
      </div>
    </aside>
  )
}
