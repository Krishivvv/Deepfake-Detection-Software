"use client";

import { useEffect, useRef, useState } from "react";

interface Stat {
  value: number;
  suffix?: string;
  label: string;
  decimals?: number;
}

const STATS: Stat[] = [
  { value: 82.0, suffix: " %", label: "Test accuracy", decimals: 1 },
  { value: 0.870, suffix: "", label: "ROC-AUC", decimals: 3 },
  { value: 1000, suffix: "", label: "Videos in study set" },
  { value: 32, suffix: "", label: "Face frames per video" },
];

export function StatBar() {
  return (
    <section className="border-y border-ink-500/60 bg-ink-800/40">
      <div className="container-page py-10">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-y-6 gap-x-4">
          {STATS.map((stat) => (
            <Counter key={stat.label} stat={stat} />
          ))}
        </div>
      </div>
    </section>
  );
}

function Counter({ stat }: { stat: Stat }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    let started = false;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting && !started) {
            started = true;
            const start = performance.now();
            const duration = 900;
            const tick = (now: number) => {
              const t = Math.min(1, (now - start) / duration);
              const eased = 1 - Math.pow(1 - t, 3);
              setDisplay(stat.value * eased);
              if (t < 1) requestAnimationFrame(tick);
              else setDisplay(stat.value);
            };
            requestAnimationFrame(tick);
            observer.disconnect();
          }
        }
      },
      { threshold: 0.4 },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [stat.value]);

  const decimals = stat.decimals ?? 0;
  const formatted = display.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });

  return (
    <div ref={ref} className="text-center">
      <p className="font-mono text-3xl sm:text-4xl font-semibold text-mint-300 text-glow">
        {formatted}
        {stat.suffix}
      </p>
      <p className="mt-1 text-xs sm:text-sm text-ink-300 uppercase tracking-[0.15em]">
        {stat.label}
      </p>
    </div>
  );
}
