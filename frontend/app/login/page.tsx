"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

import { ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";

export default function LoginPage() {
  return (
    <Suspense fallback={<LoginFallback />}>
      <LoginInner />
    </Suspense>
  );
}

function LoginFallback() {
  return (
    <div className="container-page py-16 sm:py-24">
      <div className="mx-auto max-w-md text-center text-ink-300 text-sm">
        Loading…
      </div>
    </div>
  );
}

function LoginInner() {
  const router = useRouter();
  const search = useSearchParams();
  const { user, ready, signIn } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const next = search.get("next") || "/detect";

  useEffect(() => {
    if (ready && user) router.replace(next);
  }, [ready, user, next, router]);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await signIn(email.trim(), password);
      router.replace(next);
    } catch (err) {
      const apiErr = err as ApiError;
      setError(apiErr.message || "Sign-in failed.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="container-page py-16 sm:py-24">
      <div className="mx-auto max-w-md">
        <header className="mb-8 text-center">
          <span className="chip">Sign in</span>
          <h1 className="heading-lg mt-3 text-mint-50">Welcome back.</h1>
          <p className="mt-2 text-ink-300 text-sm">
            Sign in to tag your detection requests in the server logs.
          </p>
        </header>

        <form onSubmit={onSubmit} className="card p-6 sm:p-7 space-y-5" noValidate>
          <div>
            <label className="label" htmlFor="email">
              Email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              required
              className="input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={submitting}
            />
          </div>
          <div>
            <label className="label" htmlFor="password">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              required
              className="input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={submitting}
            />
          </div>

          {error && (
            <div className="rounded-xl border border-rose-400/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
              {error}
            </div>
          )}

          <button type="submit" className="btn-primary w-full" disabled={submitting}>
            {submitting ? "Signing in…" : "Sign in"}
          </button>

          <p className="text-center text-sm text-ink-300">
            Don't have an account?{" "}
            <Link
              href={`/signup${next ? `?next=${encodeURIComponent(next)}` : ""}`}
              className="text-mint-300 hover:text-mint-200"
            >
              Create one
            </Link>
          </p>
        </form>

        <p className="mt-6 text-center text-xs text-ink-300/80">
          Predictions also work without an account — visit{" "}
          <Link href="/detect" className="text-mint-300 hover:text-mint-200">
            /detect
          </Link>{" "}
          directly.
        </p>
      </div>
    </div>
  );
}
