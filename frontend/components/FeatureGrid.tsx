interface Feature {
  title: string;
  body: string;
  icon: React.ReactNode;
}

const iconProps = {
  fill: "none" as const,
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  className: "h-5 w-5",
  "aria-hidden": true,
};

const FEATURES: Feature[] = [
  {
    title: "Calibrated thresholds",
    body:
      "Decision threshold tuned on validation for macro-F1 — not a stock 0.5 — so the imbalanced class doesn't dominate.",
    icon: (
      <svg viewBox="0 0 24 24" {...iconProps}>
        <path d="M3 12h4l3-9 4 18 3-9h4" />
      </svg>
    ),
  },
  {
    title: "Honest about its limits",
    body:
      "Real-class recall is 73 % at the deployed threshold — we report it, alongside the rest, on the About page.",
    icon: (
      <svg viewBox="0 0 24 24" {...iconProps}>
        <path d="M12 9v4" />
        <path d="M12 17h.01" />
        <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" />
      </svg>
    ),
  },
  {
    title: "Local-only inference",
    body:
      "Uploads stay on your machine. The Flask backend runs the model on CPU and deletes the file after each prediction.",
    icon: (
      <svg viewBox="0 0 24 24" {...iconProps}>
        <rect x="3" y="11" width="18" height="11" rx="2" />
        <path d="M7 11V7a5 5 0 0 1 10 0v4" />
      </svg>
    ),
  },
  {
    title: "Reproducible pipeline",
    body:
      "Three scripts to retrain end-to-end on the same hardware: extract → train head → evaluate. Documented step-by-step.",
    icon: (
      <svg viewBox="0 0 24 24" {...iconProps}>
        <path d="M21 12a9 9 0 1 1-9-9 9 9 0 0 1 6.36 2.64" />
        <path d="M21 4v5h-5" />
      </svg>
    ),
  },
  {
    title: "Typed errors, not 500s",
    body:
      "Bad uploads — wrong type, no faces, too short — return JSON error codes the UI renders inline. No mystery crashes.",
    icon: (
      <svg viewBox="0 0 24 24" {...iconProps}>
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
        <path d="m9 11 3 3L22 4" />
      </svg>
    ),
  },
  {
    title: "Open source",
    body:
      "Every line of model, training, and frontend code is in the repo under MIT. Nothing is hidden behind an SDK.",
    icon: (
      <svg viewBox="0 0 24 24" {...iconProps}>
        <path d="m18 16 4-4-4-4" />
        <path d="m6 8-4 4 4 4" />
        <path d="m14.5 4-5 16" />
      </svg>
    ),
  },
];

export function FeatureGrid() {
  return (
    <section className="container-page py-20 sm:py-28">
      <div className="max-w-2xl">
        <span className="chip">Why Veridex</span>
        <h2 className="heading-lg mt-4 text-mint-50">
          A research demo that doesn't oversell itself.
        </h2>
        <p className="mt-4 text-ink-300 leading-relaxed">
          Most deepfake demos either hide behind an opaque API or quote
          accuracy figures without context. Veridex is built the other way:
          metrics are surfaced, threshold choices are explained, and you can
          retrain the whole thing on a laptop.
        </p>
      </div>

      <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {FEATURES.map((f) => (
          <div key={f.title} className="card-hover p-6">
            <div className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-mint-500/10 ring-1 ring-mint-500/30 text-mint-300">
              {f.icon}
            </div>
            <h3 className="mt-4 heading-md text-mint-50">{f.title}</h3>
            <p className="mt-2 text-sm text-ink-300 leading-relaxed">{f.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
