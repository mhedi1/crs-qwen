"use client";

import Link from "next/link";
import { motion } from "motion/react";
import { Database, Brain, Zap, ArrowRight, Film } from "lucide-react";

export default function LandingPage() {
  return (
    <div className="flex h-screen w-full flex-col items-center justify-center bg-background px-4 overflow-hidden relative">
      {/* Background glow */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/10 via-transparent to-transparent z-0 opacity-50" />
      
      <main className="z-10 flex w-full max-w-4xl flex-col items-center text-center">
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-6 inline-flex items-center rounded-full border border-border bg-card/50 backdrop-blur-sm px-3 py-1 text-xs font-medium text-muted-foreground shadow-sm"
        >
          <span className="mr-2 flex h-2 w-2 rounded-full bg-primary" />
          CRS-Qwen Prototype
        </motion.div>

        <motion.h1 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mb-4 text-4xl font-extrabold tracking-tight text-foreground sm:text-5xl max-w-3xl"
        >
          Discover films through conversation.
        </motion.h1>
        
        <motion.p 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mb-10 max-w-2xl text-base text-muted-foreground sm:text-lg"
        >
          A state-of-the-art dialogue system bridging Knowledge-Grounded Neural Retrieval (KBRD) with Qwen LLM Semantic Reranking.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <Link href="/chat">
            <button className="group relative mb-14 inline-flex items-center justify-center overflow-hidden rounded-full bg-primary px-8 py-3.5 text-sm font-medium text-white shadow-md transition-all hover:scale-105 hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-background">
              <Film className="mr-2 h-4 w-4" />
              Start Demo
              <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
            </button>
          </Link>
        </motion.div>

        {/* Compact Features */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="grid w-full grid-cols-1 gap-4 sm:grid-cols-3"
        >
          <div className="flex flex-col items-center rounded-2xl border border-border bg-card/40 backdrop-blur-sm p-6 shadow-sm transition-all hover:bg-card hover:shadow-md">
            <Database className="mb-4 h-6 w-6 text-primary" />
            <h3 className="mb-2 text-sm font-bold text-foreground">Neural Retrieval</h3>
            <p className="text-xs text-muted-foreground">Extracts semantic entities and queries an R-GCN over 50k+ nodes.</p>
          </div>
          <div className="flex flex-col items-center rounded-2xl border border-border bg-card/40 backdrop-blur-sm p-6 shadow-sm transition-all hover:bg-card hover:shadow-md">
            <Brain className="mb-4 h-6 w-6 text-primary" />
            <h3 className="mb-2 text-sm font-bold text-foreground">Semantic Reranking</h3>
            <p className="text-xs text-muted-foreground">Applies dialogue context and entity constraints perfectly for the user.</p>
          </div>
          <div className="flex flex-col items-center rounded-2xl border border-border bg-card/40 backdrop-blur-sm p-6 shadow-sm transition-all hover:bg-card hover:shadow-md">
            <Zap className="mb-4 h-6 w-6 text-primary" />
            <h3 className="mb-2 text-sm font-bold text-foreground">Zero-Shot Gen</h3>
            <p className="text-xs text-muted-foreground">Powered by Qwen3.5 LLM to generate highly natural recommendations.</p>
          </div>
        </motion.div>
      </main>

      {/* Absolute footer to keep it out of the layout flow */}
      <footer className="absolute bottom-6 z-10 text-center text-[0.7rem] text-muted-foreground">
        Pristini School of AI & Universidad de Jaén <br />
        <span className="opacity-70">Developed by Mohamed Hedi Foughali</span>
      </footer>
    </div>
  );
}
