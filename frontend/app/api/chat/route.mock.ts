import { NextResponse } from "next/server"
import { runPipeline, type EngineInput } from "@/lib/engine"

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as Partial<EngineInput>
    const message = (body.message ?? "").trim()
    if (!message) {
      return NextResponse.json({ error: "Empty message" }, { status: 400 })
    }

    // Simulate pipeline latency for a realistic loading state
    await new Promise((r) => setTimeout(r, 650 + Math.random() * 500))

    const result = runPipeline({
      message,
      history: body.history ?? [],
      previouslyRecommended: body.previouslyRecommended ?? [],
      lastMovie: body.lastMovie ?? null,
      turn: body.turn ?? 0,
    })

    return NextResponse.json(result)
  } catch (err) {
    console.log("[v0] /api/chat error:", (err as Error).message)
    return NextResponse.json({ error: "Pipeline failure" }, { status: 500 })
  }
}
