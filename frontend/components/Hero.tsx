"use client";

import Link from "next/link";

import { useAuth } from "@/lib/auth-context";

export function Hero() {
  const { user } = useAuth();

  return (
    <section className="relative overflow-hidden">
      <div
        aria-hidden
        className="absolute inset-0 -z-10 bg-mesh opacity-90"
      />
      <div
        aria-hidden
        className="absolute inset-0 -z-10 grid-bg opacity-30 mask-fade-y"
      />
      <div
        aria-hidden
        className="absolute -top-40 left-1/2 -translate-x-1/2 h-[420px] w-[820px] rounded-full bg-mint-500/15 blur-3xl"
      />

      <div className="container-page relative pt-20 pb-24 sm:pt-28 sm:pb-32">
        <div className="flex flex-col items-center text-center animate-fade-up">
          <span className="chip mb-6">
            <span className="h-2 w-2 rounded-full bg-mint-400 animate-pulse-glow" />
            Hybrid ResNet-50 + BiLSTM · 82 % test accuracy
          </span>

          <h1 className="heading-xl text-mint-50 max-w-4xl">
            Spot the{" "}
            <span className="text-mint-400 text-glow">manipulated</span> in
            seconds.
          </h1>
          <p className="mt-6 max-w-2xl text-lg text-ink-300 leading-relaxed">
            Veridex extracts faces, runs them through a fine-tuned hybrid
            CNN-LSTM, and returns a calibrated REAL / FAKE verdict for any
            short clip — entirely on your machine, no upload to third parties.
          </p>

          <div className="mt-10 flex flex-col sm:flex-row gap-3">
            <Link href="/detect" className="btn-primary">
              {user ? "Open the detector" : "Try it free"}
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-4 w-4"
                aria-hidden
              >
                <path d="M5 12h14" />
                <path d="m13 5 7 7-7 7" />
              </svg>
            </Link>
            <Link href="/about" className="btn-secondary">
              How it works
            </Link>
          </div>

          <p className="mt-5 text-xs text-ink-300/80">
            No file is sent to a third-party server. Inference runs locally
            against the Flask backend on your machine.
          </p>
        </div>

        <div className="mt-16 mx-auto max-w-4xl animate-fade-up">
          <HeroPreviewCard />
        </div>
      </div>
    </section>
  );
}

function HeroPreviewCard() {
  return (
    <div className="relative">
      <div
        aria-hidden
        className="absolute -inset-1 rounded-3xl bg-gradient-to-r from-mint-500/30 via-mint-400/0 to-mint-500/30 blur-2xl"
      />
      <div className="relative card overflow-hidden">
        <div className="px-5 py-3 border-b border-ink-500/60 flex items-center gap-2 text-xs text-ink-300 font-mono">
          <span className="inline-flex gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-rose-400/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-amber-400/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-mint-400/70" />
          </span>
          <span className="ml-3">veridex.local /detect</span>
        </div>
        <div className="grid md:grid-cols-2">
          <div className="p-6 sm:p-8 border-b md:border-b-0 md:border-r border-ink-500/60">
            <p className="label">Sample upload</p>
            <div className="rounded-xl border border-dashed border-mint-500/40 bg-ink-800/60 p-6 text-center">
              <div className="text-3xl">📼</div>
              <p className="mt-2 text-sm text-mint-50/85">
                clip_001.mp4 · 6.3 MB
              </p>
              <p className="text-xs text-ink-300 mt-1">
                32 frames · 30 fps · 224×224 face crops
              </p>
            </div>
            <ul className="mt-5 space-y-2 text-sm text-ink-300">
              <li className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-mint-400" />
                Frames extracted: 32 / 32
              </li>
              <li className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-mint-400" />
                Faces detected: 32
              </li>
              <li className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-mint-400" />
                Inference: ResNet-50 → BiLSTM(128, 1, bidir)
              </li>
            </ul>
          </div>
          <div className="p-6 sm:p-8 flex flex-col">
            <p className="label">Result</p>
            <div className="flex items-baseline gap-3">
              <span className="text-5xl font-semibold tracking-tight text-rose-400 text-glow">
                FAKE
              </span>
              <span className="text-xs uppercase tracking-[0.18em] text-ink-300">
                example
              </span>
            </div>
            <p className="text-sm text-ink-300 mt-2">
              99.27 % probability fake · threshold 0.575
            </p>
            <div className="mt-5">
              <div className="h-2 rounded-full bg-ink-600 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-mint-400 to-mint-500 rounded-full"
                  style={{ width: "99%" }}
                />
              </div>
              <p className="text-xs text-ink-300 mt-2">
                Confidence 99.27 %
              </p>
            </div>
            <dl className="mt-6 grid grid-cols-2 gap-3 text-sm">
              <div className="rounded-lg bg-ink-800/70 p-3">
                <dt className="text-xs text-ink-300">P(fake)</dt>
                <dd className="font-mono text-mint-100">99.27 %</dd>
              </div>
              <div className="rounded-lg bg-ink-800/70 p-3">
                <dt className="text-xs text-ink-300">P(real)</dt>
                <dd className="font-mono text-mint-100">0.73 %</dd>
              </div>
              <div className="rounded-lg bg-ink-800/70 p-3">
                <dt className="text-xs text-ink-300">Inference</dt>
                <dd className="font-mono text-mint-100">3.42 s</dd>
              </div>
              <div className="rounded-lg bg-ink-800/70 p-3">
                <dt className="text-xs text-ink-300">Device</dt>
                <dd className="font-mono text-mint-100">CPU</dd>
              </div>
            </dl>
          </div>
        </div>
      </div>
    </div>
  );
}
