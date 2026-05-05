import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t border-ink-500/60 bg-ink-900">
      <div className="container-page py-14">
        <div className="grid gap-10 md:grid-cols-4">
          <div className="md:col-span-2">
            <Link href="/" className="inline-flex items-center gap-2.5 mb-4">
              <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-mint-500/15 ring-1 ring-mint-500/40">
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-4 w-4 text-mint-300"
                  aria-hidden
                >
                  <path d="M12 2 4 6v6c0 5 3.5 9.4 8 10 4.5-.6 8-5 8-10V6l-8-4z" />
                  <path d="m9 12 2 2 4-4" />
                </svg>
              </span>
              <span className="font-semibold tracking-tight text-mint-50">Veridex</span>
            </Link>
            <p className="text-sm text-ink-300 max-w-md leading-relaxed">
              A research demo for video deepfake detection. Upload a clip, get
              a calibrated REAL / FAKE prediction in seconds. Built on a
              hybrid ResNet-50 + BiLSTM trained on FaceForensics++.
            </p>
            <p className="text-xs text-ink-300/80 mt-4">
              © {new Date().getFullYear()} Veridex. Educational use only.
            </p>
          </div>

          <div>
            <h3 className="text-xs font-semibold uppercase tracking-[0.16em] text-mint-300 mb-4">
              Tool
            </h3>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="/detect" className="text-mint-50/75 hover:text-mint-300 transition">
                  Detect a video
                </Link>
              </li>
              <li>
                <Link href="/about" className="text-mint-50/75 hover:text-mint-300 transition">
                  How it works
                </Link>
              </li>
              <li>
                <Link href="/about#metrics" className="text-mint-50/75 hover:text-mint-300 transition">
                  Performance metrics
                </Link>
              </li>
              <li>
                <Link href="/about#limitations" className="text-mint-50/75 hover:text-mint-300 transition">
                  Limitations
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-xs font-semibold uppercase tracking-[0.16em] text-mint-300 mb-4">
              Account
            </h3>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="/login" className="text-mint-50/75 hover:text-mint-300 transition">
                  Sign in
                </Link>
              </li>
              <li>
                <Link href="/signup" className="text-mint-50/75 hover:text-mint-300 transition">
                  Create account
                </Link>
              </li>
              <li>
                <Link href="/#faq" className="text-mint-50/75 hover:text-mint-300 transition">
                  FAQ
                </Link>
              </li>
              <li>
                <a
                  href="https://github.com/ondyari/FaceForensics"
                  target="_blank"
                  rel="noreferrer"
                  className="text-mint-50/75 hover:text-mint-300 transition"
                >
                  Dataset (FaceForensics++)
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-10 pt-6 border-t border-ink-500/60 flex flex-col sm:flex-row gap-3 sm:items-center sm:justify-between">
          <p className="text-xs text-ink-300">
            Built with PyTorch, Flask, Next.js. Open-source MIT.
          </p>
          <div className="flex items-center gap-4">
            <a
              href="https://github.com/Krishivvv/Deepfake-Detection-Software"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 text-xs text-ink-300 hover:text-mint-300 transition"
              aria-label="View source on GitHub"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" className="h-4 w-4" aria-hidden>
                <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.1.79-.25.79-.56v-2.04c-3.2.7-3.87-1.37-3.87-1.37-.52-1.32-1.27-1.67-1.27-1.67-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.02 1.76 2.69 1.25 3.34.95.1-.74.4-1.25.72-1.54-2.55-.29-5.24-1.27-5.24-5.66 0-1.25.45-2.27 1.18-3.07-.12-.29-.51-1.45.11-3.03 0 0 .96-.31 3.15 1.17a10.9 10.9 0 0 1 5.74 0c2.19-1.48 3.15-1.17 3.15-1.17.62 1.58.23 2.74.11 3.03.74.8 1.18 1.82 1.18 3.07 0 4.4-2.69 5.37-5.25 5.65.41.36.78 1.07.78 2.16v3.2c0 .31.21.67.8.56C20.71 21.39 24 17.08 24 12 24 5.65 18.85.5 12 .5Z"/>
              </svg>
              GitHub
            </a>
            <p className="text-xs text-ink-300">
              Not for forensic use. Predictions are probabilistic.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}
