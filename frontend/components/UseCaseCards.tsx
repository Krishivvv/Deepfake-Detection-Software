interface UseCase {
  emoji: string;
  title: string;
  blurb: string;
  score: number;
  verdict: "REAL" | "FAKE";
  scoreNote: string;
}

const USE_CASES: UseCase[] = [
  {
    emoji: "🎭",
    title: "Face-swap detection",
    blurb:
      "Spots identity transplants from FaceSwap-family models — even when blending boundaries are smoothed.",
    score: 97,
    verdict: "FAKE",
    scoreNote: "FaceSwap test clip · 97 % p(fake)",
  },
  {
    emoji: "🗣️",
    title: "Reenactment & lip-sync",
    blurb:
      "Catches Face2Face-style mouth-and-pose puppeteering by exploiting temporal artefacts across frames.",
    score: 99,
    verdict: "FAKE",
    scoreNote: "Face2Face test clip · 99 % p(fake)",
  },
  {
    emoji: "🧬",
    title: "Neural texture synthesis",
    blurb:
      "Detects neural-rendered textures even when single-frame artefacts are imperceptible to the human eye.",
    score: 95,
    verdict: "FAKE",
    scoreNote: "NeuralTextures test clip · 95 % p(fake)",
  },
  {
    emoji: "✅",
    title: "Authentic content",
    blurb:
      "Calibrated threshold prevents over-flagging — real videos get a confident REAL label without retraining.",
    score: 99,
    verdict: "REAL",
    scoreNote: "Untouched test clip · 0.4 % p(fake)",
  },
];

export function UseCaseCards() {
  return (
    <section id="use-cases" className="container-page py-20 sm:py-28 scroll-mt-page">
      <div className="max-w-2xl">
        <span className="chip">Use cases</span>
        <h2 className="heading-lg mt-4 text-mint-50">
          Built around the four manipulation families in FaceForensics++.
        </h2>
        <p className="mt-4 text-ink-300 leading-relaxed">
          The model was trained against deepfakes, face-swaps, reenactments,
          and neural-texture synthesis — and calibrated so the real class
          isn't drowned out by the 4-to-1 fake skew in the dataset.
        </p>
      </div>

      <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {USE_CASES.map((uc) => (
          <article key={uc.title} className="card-hover p-6">
            <div className="text-3xl">{uc.emoji}</div>
            <h3 className="mt-4 heading-md text-mint-50">{uc.title}</h3>
            <p className="mt-2 text-sm text-ink-300 leading-relaxed">
              {uc.blurb}
            </p>
            <div className="mt-5">
              <ScoreRing
                value={uc.score}
                tone={uc.verdict === "FAKE" ? "warn" : "ok"}
                label={uc.verdict}
              />
              <p className="mt-2 text-xs text-ink-300">{uc.scoreNote}</p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function ScoreRing({
  value,
  tone,
  label,
}: {
  value: number;
  tone: "ok" | "warn";
  label: string;
}) {
  const radius = 22;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;
  const stroke = tone === "warn" ? "#FB7185" : "#34D399";
  const labelTone =
    tone === "warn" ? "text-rose-400" : "text-mint-300";

  return (
    <div className="flex items-center gap-3">
      <svg width="56" height="56" viewBox="0 0 56 56" aria-hidden>
        <circle
          cx="28"
          cy="28"
          r={radius}
          stroke="rgba(255,255,255,0.08)"
          strokeWidth="6"
          fill="none"
        />
        <circle
          cx="28"
          cy="28"
          r={radius}
          stroke={stroke}
          strokeWidth="6"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 28 28)"
          fill="none"
        />
        <text
          x="50%"
          y="52%"
          textAnchor="middle"
          dominantBaseline="middle"
          className="fill-mint-50 font-mono text-[12px] font-semibold"
        >
          {value}%
        </text>
      </svg>
      <span
        className={[
          "text-xs font-semibold uppercase tracking-[0.16em]",
          labelTone,
        ].join(" ")}
      >
        {label}
      </span>
    </div>
  );
}
