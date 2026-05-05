"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { useAuth } from "@/lib/auth-context";

const NAV_LINKS = [
  { href: "/", label: "Home" },
  { href: "/detect", label: "Detect" },
  { href: "/about", label: "About" },
];

export function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, ready, signOut } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 4);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  const linkClass = (href: string) => {
    const active = pathname === href || (href !== "/" && pathname.startsWith(href));
    return [
      "px-3 py-2 rounded-lg text-sm font-medium transition-colors",
      active
        ? "text-mint-300"
        : "text-mint-50/70 hover:text-mint-200",
    ].join(" ");
  };

  const handleSignOut = () => {
    signOut();
    router.push("/");
  };

  return (
    <header
      className={[
        "sticky top-0 z-40 w-full backdrop-blur transition-all duration-200",
        scrolled
          ? "bg-ink-900/80 border-b border-ink-500/60"
          : "bg-ink-900/30 border-b border-transparent",
      ].join(" ")}
    >
      <div className="container-page flex h-16 items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5 group">
          <Image
            src="/communication-skills.png"
            alt=""
            width={40}
            height={40}
            priority
            className="h-9 w-9 object-contain transition-transform group-hover:scale-105"
          />
          <span className="font-semibold tracking-tight text-mint-50">
            Veridex
          </span>
        </Link>

        <nav className="hidden md:flex items-center gap-1">
          {NAV_LINKS.map((link) => (
            <Link key={link.href} href={link.href} className={linkClass(link.href)}>
              {link.label}
            </Link>
          ))}
        </nav>

        <div className="hidden md:flex items-center gap-2">
          {!ready ? (
            <div className="h-9 w-32 rounded-lg bg-ink-700/50 animate-pulse" />
          ) : user ? (
            <>
              <span className="text-xs text-ink-300 hidden lg:inline">
                {user.email}
              </span>
              <button onClick={handleSignOut} className="btn-ghost">
                Sign out
              </button>
              <Link href="/detect" className="btn-primary">
                Detect now
              </Link>
            </>
          ) : (
            <>
              <Link href="/login" className="btn-ghost">
                Sign in
              </Link>
              <Link href="/signup" className="btn-primary">
                Get started
              </Link>
            </>
          )}
        </div>

        <button
          type="button"
          className="md:hidden inline-flex h-9 w-9 items-center justify-center rounded-lg border border-ink-500 text-mint-100"
          aria-label="Toggle navigation"
          aria-expanded={mobileOpen}
          onClick={() => setMobileOpen((v) => !v)}
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-5 w-5"
            aria-hidden
          >
            {mobileOpen ? (
              <path d="M6 6l12 12M18 6L6 18" />
            ) : (
              <>
                <path d="M3 6h18" />
                <path d="M3 12h18" />
                <path d="M3 18h18" />
              </>
            )}
          </svg>
        </button>
      </div>

      {mobileOpen && (
        <div className="md:hidden border-t border-ink-500/60 bg-ink-900/95">
          <div className="container-page py-3 flex flex-col gap-1">
            {NAV_LINKS.map((link) => (
              <Link key={link.href} href={link.href} className={linkClass(link.href)}>
                {link.label}
              </Link>
            ))}
            {ready && user ? (
              <>
                <Link href="/detect" className="btn-primary mt-2">
                  Detect now
                </Link>
                <button
                  type="button"
                  onClick={handleSignOut}
                  className="btn-secondary"
                >
                  Sign out ({user.email})
                </button>
              </>
            ) : (
              <>
                <Link href="/login" className="btn-secondary mt-2">
                  Sign in
                </Link>
                <Link href="/signup" className="btn-primary">
                  Get started
                </Link>
              </>
            )}
          </div>
        </div>
      )}
    </header>
  );
}
