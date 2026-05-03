"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, fetchHealth, predictVideo, type PredictionResult } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { FileUploader } from "@/components/FileUploader";
import { ResultCard } from "@/components/ResultCard";

type Phase = "idle" | "uploading" | "analyzing" | "done" | "error";

export default function DetectPage() {
  const { user, ready } = useAuth();
  const [file, setFile] = useState<File | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<{ code: string; message: string } | null>(null);
  const [serverState, setServerState] = useState<
    "checking" | "online" | "model_loading" | "offline"
  >("checking");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const h = await fetchHealth();
        if (cancelled) return;
        setServerState(h.model_loaded ? "online" : "model_loading");
      } catch {
        if (!cancelled) setServerState("offline");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const onAnalyze = async () => {
    if (!file) return;
    setError(null);
    setResult(null);
    setProgress(0);
    setPhase("uploading");
    try {
      const res = await predictVideo(file, (pct) => {
        setProgress(pct);
        if (pct >= 99) setPhase("analyzing");
      });
      setResult(res);
      setPhase("done");
    } catch (err) {
      const apiErr = err as ApiError;
      setError({
        code: apiErr.code || "request_failed",
        message: apiErr.message || "Request failed.",
      });
      setPhase("error");
    }
  };

  const reset = () => {
    setFile(null);
    setProgress(0);
    setResult(null);
    setError(null);
    setPhase("idle");
  };

  return (
    <div className="container-page py-12 sm:py-16">
      <header className="mb-8">
        <span className="chip">Detector</span>
        <h1 className="heading-lg mt-3 text-mint-50">Run the model on your video.</h1>
        <p className="mt-2 text-ink-300 max-w-2xl">
          Pick a clip, hit analyze. The Flask backend extracts 32 face crops,
          runs the hybrid model, and returns a calibrated REAL / FAKE
          decision. Inference takes ~15 seconds on this machine.
        </p>
        {ready && !user && (
          <p className="mt-4 text-xs text-ink-300">
            <Link href="/signup" className="text-mint-300 hover:text-mint-200 underline-offset-4 hover:underline">
              Create a free account
            </Link>{" "}
            or{" "}
            <Link href="/login" className="text-mint-300 hover:text-mint-200 underline-offset-4 hover:underline">
              sign in
            </Link>{" "}
            — predictions still work without one, but signing in lets the
            server tag each request to your account in its logs.
          </p>
        )}
      </header>

      <ServerStateBanner state={serverState} />

      <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
        <div className="card p-6 sm:p-8 space-y-6">
          <FileUploader
            file={file}
            onFileSelected={(f) => {
              setFile(f);
              setResult(null);
              setError(null);
              setPhase(f ? "idle" : "idle");
            }}
            disabled={phase === "uploading" || phase === "analyzing"}
          />

          {(phase === "uploading" || phase === "analyzing") && (
            <ProgressArea progress={progress} phase={phase} />
          )}

          <div className="flex flex-col sm:flex-row gap-3">
            <button
              type="button"
              className="btn-primary flex-1"
              disabled={!file || phase === "uploading" || phase === "analyzing"}
              onClick={onAnalyze}
            >
              {phase === "uploading"
                ? `Uploading ${progress}%`
                : phase === "analyzing"
                  ? "Analysing on CPU…"
                  : "Analyze video"}
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={reset}
              disabled={phase === "uploading" || phase === "analyzing"}
            >
              Reset
            </button>
          </div>

          {error && (
            <div className="rounded-xl border border-rose-400/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
              <p className="font-medium">{error.message}</p>
              <p className="mt-1 text-xs text-rose-300/80 font-mono">
                code: {error.code}
              </p>
            </div>
          )}
        </div>

        <aside className="space-y-4">
          <div className="card p-6">
            <h2 className="font-semibold text-mint-50">Tips for best results</h2>
            <ul className="mt-4 space-y-3 text-sm text-ink-300">
              <li className="flex gap-3">
                <span className="mt-1 h-1.5 w-1.5 rounded-full bg-mint-400 shrink-0" />
                Use clips with at least 1 second of clear face footage.
              </li>
              <li className="flex gap-3">
                <span className="mt-1 h-1.5 w-1.5 rounded-full bg-mint-400 shrink-0" />
                Keep file size under 50 MB; longer videos work — only 32
                evenly-spaced frames are sampled.
              </li>
              <li className="flex gap-3">
                <span className="mt-1 h-1.5 w-1.5 rounded-full bg-mint-400 shrink-0" />
                Front-facing single-subject framing maximises MTCNN's success
                rate.
              </li>
              <li className="flex gap-3">
                <span className="mt-1 h-1.5 w-1.5 rounded-full bg-mint-400 shrink-0" />
                Real-class recall is 73 %. The model can mis-flag a
                meaningful share of authentic videos as fake.
              </li>
            </ul>
          </div>

          <div className="card p-6 text-sm text-ink-300">
            <p className="font-semibold text-mint-50 mb-2">Response time</p>
            <p>
              First prediction can be slower while the model warms up. The
              backend keeps the model loaded in memory between requests.
            </p>
          </div>
        </aside>
      </div>

      {result && (
        <div className="mt-8">
          <ResultCard result={result} />
        </div>
      )}
    </div>
  );
}

function ProgressArea({
  progress,
  phase,
}: {
  progress: number;
  phase: Phase;
}) {
  return (
    <div className="rounded-xl border border-mint-500/30 bg-mint-500/5 px-4 py-4">
      <div className="flex items-center justify-between text-xs text-mint-200">
        <span>
          {phase === "uploading" ? "Uploading…" : "Running inference on CPU…"}
        </span>
        <span className="font-mono">{progress}%</span>
      </div>
      <div className="mt-2 h-2 rounded-full bg-ink-800 overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-mint-400 to-mint-500 transition-[width] duration-200"
          style={{
            width:
              phase === "analyzing" && progress >= 99
                ? "100%"
                : `${progress}%`,
          }}
        />
      </div>
    </div>
  );
}

function ServerStateBanner({
  state,
}: {
  state: "checking" | "online" | "model_loading" | "offline";
}) {
  if (state === "online") return null;

  const tone =
    state === "offline"
      ? "border-rose-400/40 bg-rose-500/10 text-rose-200"
      : "border-amber-400/40 bg-amber-500/10 text-amber-100";

  const message =
    state === "checking"
      ? "Checking backend status…"
      : state === "model_loading"
        ? "Backend reachable but model is still initialising. Try again in a few seconds."
        : "Could not reach the backend. Make sure the Flask server is running on the configured NEXT_PUBLIC_API_URL.";

  return (
    <div className={["mb-6 rounded-xl border px-4 py-3 text-sm", tone].join(" ")}>
      {message}
    </div>
  );
}
