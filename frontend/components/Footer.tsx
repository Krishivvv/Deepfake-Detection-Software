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
          <p className="text-xs text-ink-300">
            Not for forensic use. Predictions are probabilistic.
          </p>
        </div>
      </div>
    </footer>
  );
}
