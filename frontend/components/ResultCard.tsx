import type { PredictionResult } from "@/lib/api";

export function ResultCard({ result }: { result: PredictionResult }) {
  const isFake = result.label === "FAKE";
  const tone = isFake ? "rose" : "mint";

  return (
    <div className="card overflow-hidden animate-fade-up">
      <div className="px-6 py-4 border-b border-ink-500/60 flex items-center justify-between">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-300">
            Verdict
          </p>
          <p
            className={[
              "text-3xl sm:text-4xl font-semibold tracking-tight mt-1",
              isFake ? "text-rose-400 text-glow" : "text-mint-300 text-glow",
            ].join(" ")}
          >
            {result.label}
          </p>
        </div>
        <div className="text-right">
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-300">
            Request
          </p>
          <p className="font-mono text-sm text-mint-50/80 mt-1">
            #{result.request_id}
          </p>
        </div>
      </div>

      <div className="p-6 sm:p-7">
        <ConfidenceGauge value={result.confidence_pct} tone={tone} />

        <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
          <Stat
            label="P(fake)"
            value={`${result.probability_fake_pct.toFixed(2)} %`}
          />
          <Stat
            label="P(real)"
            value={`${result.probability_real_pct.toFixed(2)} %`}
          />
          <Stat
            label="Threshold"
            value={result.threshold.toFixed(3)}
          />
          <Stat
            label="Device"
            value={result.device.toUpperCase()}
          />
          <Stat
            label="Inference"
            value={`${result.inference_seconds.toFixed(2)} s`}
          />
          <Stat
            label="Total"
            value={`${result.total_seconds.toFixed(2)} s`}
          />
          <Stat
            label="Frames sampled"
            value={`${result.video_info.frames_sampled} / ${result.video_info.total_video_frames}`}
          />
          <Stat
            label="Faces found"
            value={`${result.video_info.frames_with_faces} / ${result.video_info.frames_sampled}`}
          />
        </div>

        <div className="mt-6 rounded-xl border border-ink-500/70 bg-ink-800/60 p-4 text-xs text-ink-300 leading-relaxed">
          <p>
            <span className="font-semibold text-mint-50/90">How to read this:</span>{" "}
            P(fake) is compared to the calibrated decision threshold. The
            confidence shown above is `max(P(fake), P(real))`. The threshold
            (0.575) was selected on the validation set to maximise macro-F1 —
            it isn't tuned to your individual upload.
          </p>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-ink-800/70 border border-ink-500/50 p-3">
      <p className="text-xs text-ink-300">{label}</p>
      <p className="font-mono text-mint-50 mt-1">{value}</p>
    </div>
  );
}

function ConfidenceGauge({
  value,
  tone,
}: {
  value: number;
  tone: "rose" | "mint";
}) {
  const stroke = tone === "rose" ? "#FB7185" : "#34D399";
  const trackColor = "rgba(255,255,255,0.06)";
  const radius = 70;
  const circumference = Math.PI * radius;
  const offset = circumference - (Math.min(100, Math.max(0, value)) / 100) * circumference;

  return (
    <div className="flex flex-col sm:flex-row items-center sm:items-end gap-6">
      <svg width="180" height="100" viewBox="0 0 180 100" aria-hidden>
        <path
          d="M 20 90 A 70 70 0 0 1 160 90"
          fill="none"
          stroke={trackColor}
          strokeWidth="14"
          strokeLinecap="round"
        />
        <path
          d="M 20 90 A 70 70 0 0 1 160 90"
          fill="none"
          stroke={stroke}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
        <text
          x="90"
          y="78"
          textAnchor="middle"
          className="fill-mint-50 font-mono"
          fontSize="22"
          fontWeight="600"
        >
          {value.toFixed(1)}%
        </text>
      </svg>
      <div className="flex-1 w-full">
        <p className="font-mono text-xs uppercase tracking-[0.18em] text-ink-300">
          Confidence
        </p>
        <p className="text-mint-50 mt-1 text-sm leading-relaxed">
          The model is{" "}
          <span className={tone === "rose" ? "text-rose-300" : "text-mint-300"}>
            {value.toFixed(1)} %
          </span>{" "}
          confident in this decision based on a sigmoid output that exceeds
          the calibrated threshold.
        </p>
      </div>
    </div>
  );
}
