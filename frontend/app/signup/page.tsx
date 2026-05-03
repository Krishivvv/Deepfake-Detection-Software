"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

import { ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";

export default function SignupPage() {
  return (
    <Suspense fallback={<SignupFallback />}>
      <SignupInner />
    </Suspense>
  );
}

function SignupFallback() {
  return (
    <div className="container-page py-16 sm:py-24">
      <div className="mx-auto max-w-md text-center text-ink-300 text-sm">
        Loading…
      </div>
    </div>
  );
}

function SignupInner() {
  const router = useRouter();
  const search = useSearchParams();
  const { user, ready, signUp } = useAuth();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const next = search.get("next") || "/detect";

  useEffect(() => {
    if (ready && user) router.replace(next);
  }, [ready, user, next, router]);

  const passwordHelp =
    password.length === 0
      ? "Minimum 8 characters."
      : password.length < 8
        ? `${8 - password.length} more character${8 - password.length === 1 ? "" : "s"} to go.`
        : "Looks good.";

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }
    setSubmitting(true);
    try {
      await signUp(email.trim(), password, name.trim() || undefined);
      router.replace(next);
    } catch (err) {
      const apiErr = err as ApiError;
      setError(apiErr.message || "Sign-up failed.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="container-page py-16 sm:py-24">
      <div className="mx-auto max-w-md">
        <header className="mb-8 text-center">
          <span className="chip">Get started</span>
          <h1 className="heading-lg mt-3 text-mint-50">Create your account.</h1>
          <p className="mt-2 text-ink-300 text-sm">
            Stored locally in the project's SQLite database. Email and a
            password — that's it.
          </p>
        </header>

        <form onSubmit={onSubmit} className="card p-6 sm:p-7 space-y-5" noValidate>
          <div>
            <label className="label" htmlFor="name">
              Display name <span className="text-ink-300/70 normal-case">(optional)</span>
            </label>
            <input
              id="name"
              type="text"
              autoComplete="name"
              maxLength={120}
              className="input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={submitting}
            />
          </div>
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
              autoComplete="new-password"
              required
              minLength={8}
              className="input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={submitting}
            />
            <p
              className={[
                "mt-2 text-xs",
                password.length === 0
                  ? "text-ink-300"
                  : password.length < 8
                    ? "text-amber-300"
                    : "text-mint-300",
              ].join(" ")}
            >
              {passwordHelp}
            </p>
          </div>

          {error && (
            <div className="rounded-xl border border-rose-400/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
              {error}
            </div>
          )}

          <button type="submit" className="btn-primary w-full" disabled={submitting}>
            {submitting ? "Creating account…" : "Create account"}
          </button>

          <p className="text-center text-sm text-ink-300">
            Already have one?{" "}
            <Link
              href={`/login${next ? `?next=${encodeURIComponent(next)}` : ""}`}
              className="text-mint-300 hover:text-mint-200"
            >
              Sign in
            </Link>
          </p>
        </form>

        <p className="mt-6 text-center text-xs text-ink-300/80">
          By signing up you agree this is a research demo and predictions
          are not for forensic use.
        </p>
      </div>
    </div>
  );
}
