import Link from "next/link";

export default function AboutPage() {
  return (
    <div className="container-page py-16 sm:py-20">
      <header className="max-w-3xl">
        <span className="chip">About Veridex</span>
        <h1 className="heading-xl mt-4 text-mint-50">
          A research demo for detecting face-manipulated video.
        </h1>
        <p className="mt-5 text-ink-300 leading-relaxed text-lg">
          Veridex was built as a final-year project to study practical
          deepfake detection on commodity hardware. It pairs a fine-tuned
          ResNet-50 with a small bidirectional LSTM, calibrates the decision
          threshold on validation, and ships the result behind a Flask
          backend and this Next.js frontend.
        </p>
      </header>

      <section id="metrics" className="mt-16 scroll-mt-page">
        <h2 className="heading-lg text-mint-50">Performance</h2>
        <p className="mt-3 text-ink-300 max-w-3xl">
          Evaluated on 150 held-out test videos (4 800 face frames). Numbers
          below come from <code className="font-mono text-mint-200">outputs/hybrid_evaluation.txt</code>.
        </p>

        <div className="mt-8 overflow-hidden rounded-2xl border border-ink-500/60 bg-ink-700/50">
          <table className="w-full text-sm">
            <thead className="bg-ink-800/70">
              <tr>
                <th className="text-left px-5 py-3 font-medium text-mint-300">Metric</th>
                <th className="text-right px-5 py-3 font-medium text-mint-300">
                  Hybrid v3 (deployed)
                </th>
                <th className="text-right px-5 py-3 font-medium text-mint-300">
                  CNN baseline (alt)
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-ink-500/60">
              {METRIC_ROWS.map((row) => (
                <tr key={row.label}>
                  <td className="px-5 py-3 text-mint-50/85">{row.label}</td>
                  <td className="px-5 py-3 text-right font-mono text-mint-100">
                    {row.hybrid}
                  </td>
                  <td className="px-5 py-3 text-right font-mono text-ink-300">
                    {row.cnn}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="mt-4 text-xs text-ink-300">
          Decision threshold 0.575 was selected on the val split for
          macro-F1. ROC-AUC is threshold-independent.
        </p>
      </section>

      <section className="mt-20">
        <h2 className="heading-lg text-mint-50">Architecture</h2>
        <div className="mt-6 grid gap-5 lg:grid-cols-2">
          <div className="card p-6">
            <h3 className="heading-md text-mint-50">Backbone</h3>
            <p className="mt-3 text-sm text-ink-300 leading-relaxed">
              ResNet-50 with ImageNet pretraining; the last block was
              fine-tuned on the FaceForensics++ subset during the CNN
              baseline stage. At inference its global-average-pool layer
              produces a 2 048-d feature per face crop. The classification
              head is replaced by Identity so the backbone acts as a feature
              extractor.
            </p>
          </div>
          <div className="card p-6">
            <h3 className="heading-md text-mint-50">Temporal head</h3>
            <p className="mt-3 text-sm text-ink-300 leading-relaxed">
              A single-layer bidirectional LSTM (hidden 128, dropout 0.5)
              consumes the 32-frame feature sequence and produces a single
              sigmoid logit per video. Trained for 12 epochs on cached
              features with WeightedRandomSampler for class balance.
            </p>
          </div>
          <div className="card p-6">
            <h3 className="heading-md text-mint-50">Pipeline</h3>
            <ol className="mt-3 space-y-2 text-sm text-ink-300 list-decimal pl-5">
              <li>Upload validated for size, type and emptiness.</li>
              <li>OpenCV extracts 32 evenly-spaced frames.</li>
              <li>facenet-pytorch MTCNN detects + crops one face per frame.</li>
              <li>Backbone produces (32, 2 048); LSTM aggregates → logit.</li>
              <li>Sigmoid → P(fake) compared to threshold → REAL / FAKE.</li>
            </ol>
          </div>
          <div className="card p-6">
            <h3 className="heading-md text-mint-50">Calibration</h3>
            <p className="mt-3 text-sm text-ink-300 leading-relaxed">
              At threshold 0.5 the un-tuned CNN baseline labels almost
              everything as fake — real-recall 5.5 %. Sweeping thresholds on
              the validation split and picking the macro-F1 maximiser brings
              real-recall to 73 % at the deployed threshold of 0.575, with no
              retraining.
            </p>
          </div>
        </div>
      </section>

      <section id="limitations" className="mt-20 scroll-mt-page">
        <h2 className="heading-lg text-mint-50">Limitations</h2>
        <ul className="mt-6 space-y-4 max-w-3xl">
          {LIMITS.map((limit) => (
            <li key={limit.title} className="card p-5">
              <p className="font-semibold text-mint-50">{limit.title}</p>
              <p className="mt-1 text-sm text-ink-300 leading-relaxed">
                {limit.body}
              </p>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-20">
        <h2 className="heading-lg text-mint-50">Tech stack</h2>
        <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4 text-sm">
          {STACK.map((s) => (
            <div key={s.name} className="card p-4">
              <p className="font-semibold text-mint-50">{s.name}</p>
              <p className="mt-1 text-xs text-ink-300">{s.role}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-20 card p-8 sm:p-10 text-center">
        <h2 className="heading-md text-mint-50">Try it on a clip of your own.</h2>
        <p className="mt-2 text-ink-300 max-w-xl mx-auto text-sm">
          The detector runs entirely on this machine. No upload to a third
          party.
        </p>
        <Link href="/detect" className="btn-primary mt-6">
          Open the detector
        </Link>
      </section>
    </div>
  );
}

const METRIC_ROWS = [
  { label: "Test accuracy", hybrid: "82.00 %", cnn: "77.58 %" },
  { label: "ROC-AUC", hybrid: "0.8703", cnn: "0.7657" },
  { label: "Macro F1", hybrid: "0.7509", cnn: "0.6433" },
  { label: "F1 real", hybrid: "0.6197", cnn: "0.4258" },
  { label: "F1 fake", hybrid: "0.8821", cnn: "0.8607" },
  { label: "Recall real", hybrid: "0.7333", cnn: "0.4156" },
  { label: "Recall fake", hybrid: "0.8417", cnn: "0.8659" },
  { label: "Decision threshold", hybrid: "0.575 (val-tuned)", cnn: "0.75 (val-tuned)" },
];

const LIMITS = [
  {
    title: "Real-class recall is 73 %.",
    body:
      "Roughly 1 in 4 authentic videos still gets called fake at the deployed threshold. This is markedly better than the un-calibrated 5.5 %, but it isn't 100 %.",
  },
  {
    title: "Trained on 700 videos.",
    body:
      "Published FaceForensics results in the 90 %+ range fine-tune the backbone end-to-end on a GPU with the full dataset. CPU training caps practical accuracy here.",
  },
  {
    title: "Single-modality, single-face.",
    body:
      "The model only looks at 32 face crops per video at 224 × 224. No body-language cues, no audio, no compression-domain features.",
  },
  {
    title: "Not for forensic use.",
    body:
      "Designed as a research / class demo. Predictions are probabilistic; novel manipulation pipelines may transfer poorly.",
  },
];

const STACK = [
  { name: "PyTorch 2.x", role: "ResNet-50 + LSTM, training & inference" },
  { name: "facenet-pytorch", role: "MTCNN face detection" },
  { name: "OpenCV", role: "Frame extraction" },
  { name: "Flask + SQLAlchemy", role: "Backend, JWT auth, prediction API" },
  { name: "SQLite + bcrypt", role: "User storage, password hashing" },
  { name: "Next.js 14 (App Router)", role: "This frontend" },
  { name: "Tailwind CSS", role: "Styling system" },
  { name: "TypeScript", role: "Frontend types" },
];
