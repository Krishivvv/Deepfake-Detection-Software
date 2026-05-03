"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface Props {
  onFileSelected: (file: File | null) => void;
  file: File | null;
  disabled?: boolean;
  maxBytes?: number;
  accept?: string[];
}

const DEFAULT_MAX = 50 * 1024 * 1024;
const DEFAULT_ACCEPT = ["mp4", "avi", "mov", "mkv", "webm"];

export function FileUploader({
  onFileSelected,
  file,
  disabled = false,
  maxBytes = DEFAULT_MAX,
  accept = DEFAULT_ACCEPT,
}: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (file) setError(null);
  }, [file]);

  const validate = useCallback(
    (f: File): string | null => {
      if (f.size === 0) return "The selected file is empty.";
      if (f.size > maxBytes) {
        const mb = Math.round(maxBytes / (1024 * 1024));
        return `Video must be under ${mb} MB.`;
      }
      const ext = f.name.split(".").pop()?.toLowerCase() || "";
      if (!accept.includes(ext)) {
        return `Please upload a video (${accept.map((e) => "." + e).join(", ")}).`;
      }
      return null;
    },
    [accept, maxBytes],
  );

  const handle = useCallback(
    (f: File | null) => {
      if (!f) {
        setError(null);
        onFileSelected(null);
        return;
      }
      const err = validate(f);
      if (err) {
        setError(err);
        onFileSelected(null);
        return;
      }
      setError(null);
      onFileSelected(f);
    },
    [onFileSelected, validate],
  );

  const onPick = () => inputRef.current?.click();

  return (
    <div>
      <input
        ref={inputRef}
        type="file"
        accept={accept.map((e) => "." + e).join(",")}
        className="hidden"
        disabled={disabled}
        onChange={(e) => handle(e.target.files?.[0] || null)}
      />

      <div
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
        onClick={() => !disabled && onPick()}
        onKeyDown={(e) => {
          if (disabled) return;
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onPick();
          }
        }}
        onDragOver={(e) => {
          if (disabled) return;
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setDragOver(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (disabled) return;
          handle(e.dataTransfer.files?.[0] || null);
        }}
        className={[
          "rounded-2xl border-2 border-dashed p-10 text-center cursor-pointer transition-all",
          "focus:outline-none focus-visible:ring-2 focus-visible:ring-mint-400/60",
          dragOver
            ? "border-mint-400 bg-mint-500/5 scale-[1.01]"
            : "border-ink-500 bg-ink-800/40 hover:border-mint-500/50 hover:bg-ink-800/70",
          disabled && "opacity-60 cursor-not-allowed",
        ]
          .filter(Boolean)
          .join(" ")}
      >
        <div className="mx-auto inline-flex h-12 w-12 items-center justify-center rounded-xl bg-mint-500/10 ring-1 ring-mint-500/30 text-mint-300">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-5 w-5"
            aria-hidden
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>
        <p className="mt-4 text-mint-50 font-medium">
          {file ? file.name : "Drop a video, or click to choose"}
        </p>
        <p className="mt-1 text-xs text-ink-300">
          {file
            ? `${(file.size / (1024 * 1024)).toFixed(1)} MB · ${file.type || "video/*"}`
            : `${accept.map((e) => "." + e).join(", ")} · up to ${Math.round(maxBytes / (1024 * 1024))} MB`}
        </p>
        {file && (
          <button
            type="button"
            className="mt-4 text-xs text-mint-300 hover:text-mint-200 underline-offset-4 hover:underline"
            onClick={(e) => {
              e.stopPropagation();
              handle(null);
            }}
          >
            Choose a different file
          </button>
        )}
      </div>

      {error && (
        <div className="mt-3 rounded-xl border border-rose-400/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
          {error}
        </div>
      )}
    </div>
  );
}
