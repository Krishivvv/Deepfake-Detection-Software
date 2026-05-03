const STEPS = [
  {
    n: "01",
    title: "Upload a clip",
    body:
      "Drag a short video into the detector, or click to pick. mp4, mov, mkv, webm and avi up to 50 MB.",
  },
  {
    n: "02",
    title: "Faces are extracted locally",
    body:
      "32 evenly-spaced frames are pulled from the video and passed through MTCNN to locate and crop a single face per frame.",
  },
  {
    n: "03",
    title: "Hybrid model runs on your CPU",
    body:
      "The fine-tuned ResNet-50 backbone produces a 2 048-d feature per face; a BiLSTM aggregates the temporal sequence into one logit.",
  },
  {
    n: "04",
    title: "Calibrated REAL / FAKE verdict",
    body:
      "The logit is compared to the val-tuned threshold (0.575). You get a label, confidence, frame counts, and timing — all in JSON.",
  },
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="border-y border-ink-500/60 bg-ink-800/40 scroll-mt-page">
      <div className="container-page py-20 sm:py-28">
        <div className="grid gap-12 lg:grid-cols-[1fr_1.1fr] lg:gap-20 items-start">
          <div>
            <span className="chip">How it works</span>
            <h2 className="heading-lg mt-4 text-mint-50">
              Four steps, ~15 seconds, fully local.
            </h2>
            <p className="mt-4 text-ink-300 leading-relaxed">
              The whole pipeline lives on your machine — feature extraction,
              face detection, model inference. No third-party API, no upload
              anywhere.
            </p>

            <ol className="mt-10 space-y-6">
              {STEPS.map((step) => (
                <li
                  key={step.n}
                  className="flex gap-5"
                >
                  <span className="font-mono text-sm font-semibold text-mint-300/90 pt-1 min-w-[2.5rem]">
                    {step.n}
                  </span>
                  <div>
                    <h3 className="text-mint-50 font-semibold">{step.title}</h3>
                    <p className="mt-1 text-sm text-ink-300 leading-relaxed">
                      {step.body}
                    </p>
                  </div>
                </li>
              ))}
            </ol>
          </div>

          <div className="relative">
            <div
              aria-hidden
              className="absolute -inset-4 rounded-3xl bg-gradient-to-br from-mint-500/20 via-transparent to-mint-500/10 blur-2xl"
            />
            <div className="relative card overflow-hidden">
              <div className="px-5 py-3 border-b border-ink-500/60 flex items-center justify-between">
                <span className="font-mono text-xs text-ink-300">
                  pipeline.trace
                </span>
                <span className="text-[10px] uppercase tracking-[0.16em] text-mint-300">
                  live
                </span>
              </div>
              <div className="p-6 space-y-3 font-mono text-xs sm:text-sm">
                <Line k="upload" v="clip_022.mp4 (4.8 MB)" />
                <Line k="frames" v="32 / 32 extracted" tone="ok" />
                <Line k="faces" v="32 / 32 detected (MTCNN)" tone="ok" />
                <Line k="backbone" v="ResNet-50 → 32 × 2048 features" />
                <Line k="head" v="BiLSTM(128, 1, bidir) → 1 logit" />
                <Line k="prob_fake" v="0.0044" tone="ok" />
                <Line k="threshold" v="0.575" />
                <Line k="verdict" v="REAL · 99.56 % confidence" tone="emph" />
                <Line k="elapsed" v="14.32s" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Line({
  k,
  v,
  tone,
}: {
  k: string;
  v: string;
  tone?: "ok" | "emph";
}) {
  const valueColor =
    tone === "ok"
      ? "text-mint-300"
      : tone === "emph"
        ? "text-mint-200 text-glow"
        : "text-mint-50/85";
  return (
    <div className="flex items-baseline gap-3">
      <span className="text-ink-300/90 w-24 sm:w-28 shrink-0">{k}</span>
      <span className="text-ink-300">»</span>
      <span className={valueColor}>{v}</span>
    </div>
  );
}
