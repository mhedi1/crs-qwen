"use client"

import type { Profile } from "@/lib/types"

export function SystemIntel({ profile }: { profile: Profile }) {
  const isFallback = profile.fallback_used
  const genresStr = profile.genres.length > 0 ? profile.genres.join(", ") : "—"
  const decadesStr = profile.decades.length > 0 ? profile.decades.join(", ") : "—"
  
  const toShow = profile.mentioned_movies.slice(0, 6)
  const hidden = profile.mentioned_movies.slice(6)

  return (
    <div className="flex h-full w-full flex-col bg-transparent p-4 pb-10 overflow-y-auto scrollbar-thin">
      <div className="flex items-center mb-3">
        <div className="w-2 h-2 rounded-full bg-[#15803d] dark:bg-[#4ade80] mr-2" style={{ animation: "pulse-dot 2s infinite" }} />
        <span className="text-[0.85rem] font-bold tracking-[0.05em] text-primary">SYSTEM INTEL</span>
      </div>
      
      <div className="flex flex-col gap-2 mb-3">
        <div className="text-[0.7rem] font-semibold text-muted-foreground tracking-[0.06em]">CONVERSATION</div>
        <div className="flex justify-between items-baseline text-[0.8rem]">
          <span className="text-muted-foreground">Turn:</span>
          <span className="text-foreground font-mono px-1 py-0.5 rounded transition-colors">{profile.turn || "0"}</span>
        </div>
        <div className="flex justify-between items-baseline text-[0.8rem]">
          <span className="text-muted-foreground">Entity seeds:</span>
          <span className="text-foreground font-mono px-1 py-0.5 rounded transition-colors">{profile.seed_count || "0"}</span>
        </div>
        <div className="flex justify-between items-baseline text-[0.8rem]">
          <span className="text-muted-foreground">Seed source:</span>
          <span className={`font-mono px-1 py-0.5 rounded transition-colors ${isFallback ? "text-[#d97706] font-semibold" : "text-[#15803d] dark:text-[#4ade80]"}`}>
            {profile.turn === 0 ? "—" : isFallback ? "Qwen-assisted" : "Dialogue"}
          </span>
        </div>
      </div>
      
      <div className="h-[1px] bg-border my-4" />
      
      <div className="flex flex-col gap-2 mb-3">
        <div className="text-[0.7rem] font-semibold text-muted-foreground tracking-[0.06em]">USER PREFERENCES</div>
        <div className="flex justify-between items-baseline text-[0.8rem]">
          <span className="text-muted-foreground">Genre:</span>
          <span className="text-foreground font-mono px-1 py-0.5 rounded transition-colors">{genresStr}</span>
        </div>
        <div className="flex justify-between items-baseline text-[0.8rem]">
          <span className="text-muted-foreground">Decade:</span>
          <span className="text-foreground font-mono px-1 py-0.5 rounded transition-colors">{decadesStr}</span>
        </div>
      </div>
      
      <div className="h-[1px] bg-border my-4" />
      
      <div className="flex flex-col flex-1">
        <div className="text-[0.7rem] font-semibold text-muted-foreground tracking-[0.06em] mb-2">MENTIONED FILMS</div>
        <div className="flex flex-wrap gap-1.5 p-1 rounded transition-colors">
          {profile.mentioned_movies.length === 0 ? (
            <span className="text-muted-foreground text-sm">—</span>
          ) : (
            <>
              {toShow.map(m => (
                <span key={m} className="text-[0.7rem] bg-[#00000008] dark:bg-white/5 border border-border rounded px-1.5 py-0.5 text-foreground">
                  {m}
                </span>
              ))}
              {hidden.length > 0 && (
                <span className="text-[0.7rem] text-primary cursor-pointer hover:underline mt-1">
                  +{hidden.length} more
                </span>
              )}
            </>
          )}
        </div>
      </div>
      
      <div className="text-[0.7rem] text-muted-foreground text-center opacity-70 mt-auto pt-3">
        Mohamed Hedi Foughali
      </div>
    </div>
  )
}
