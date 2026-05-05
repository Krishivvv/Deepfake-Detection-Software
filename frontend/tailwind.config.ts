import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Surfaces, borders and muted text. Token names kept for stability;
        // values inverted for a bright, Figma-modern light theme.
        ink: {
          900: "#F7F8FC", // page background (cool off-white)
          800: "#FFFFFF", // input background
          700: "#FFFFFF", // card background
          600: "#F1F2FA", // hover / elevated surface
          500: "#E5E7F0", // borders
          400: "#CBD0DC", // strong dividers
          300: "#64748B", // muted text on light surfaces
        },
        // Primary accent + body text. Names kept; repurposed:
        // 50–200 are dark text on light surfaces, 300–900 are the vibrant
        // indigo/violet accent that drives buttons, gradients and glows.
        mint: {
          50:  "#0B1020", // primary text
          100: "#1F2235", // secondary text / numeric data
          200: "#3A3F5C", // tertiary text / hover from accent
          300: "#6D28D9", // accent (chips, links, stats) – deep violet
          400: "#8B5CF6", // bright violet (hero highlight, hover bg)
          500: "#6366F1", // primary indigo (button bg, gradient stop)
          600: "#4F46E5", // primary hover
          700: "#4338CA",
          800: "#3730A3",
          900: "#312E81",
        },
        // Warning. 100/300 redefined to dark amber so they stay readable
        // when used as text on a pale `bg-amber-500/10` tint.
        amber: {
          100: "#78350F",
          300: "#B45309",
          400: "#FBBF24",
          500: "#F59E0B",
        },
        // Error. 200/300 redefined to dark rose for the same reason.
        rose: {
          200: "#9F1239",
          300: "#BE123C",
          400: "#F43F5E",
          500: "#E11D48",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(99, 102, 241, 0.45), 0 12px 32px -10px rgba(99, 102, 241, 0.35)",
        "glow-sm": "0 0 0 1px rgba(99, 102, 241, 0.30), 0 6px 18px -6px rgba(99, 102, 241, 0.25)",
        "glow-lg": "0 0 0 1px rgba(99, 102, 241, 0.55), 0 30px 80px -20px rgba(139, 92, 246, 0.55)",
        sink: "0 1px 2px rgba(15, 23, 42, 0.04), 0 8px 24px -12px rgba(15, 23, 42, 0.10)",
      },
      backgroundImage: {
        "grid-fade":
          "radial-gradient(circle at 50% 0%, rgba(99,102,241,0.18), transparent 60%), linear-gradient(rgba(15,23,42,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(15,23,42,0.06) 1px, transparent 1px)",
        "mesh":
          "radial-gradient(at 20% 10%, rgba(99,102,241,0.18) 0, transparent 55%), radial-gradient(at 80% 0%, rgba(139,92,246,0.14) 0, transparent 55%), radial-gradient(at 50% 100%, rgba(244,114,182,0.12) 0, transparent 60%)",
      },
      animation: {
        "pulse-glow": "pulseGlow 2.4s ease-in-out infinite",
        "float": "float 6s ease-in-out infinite",
        "fade-up": "fadeUp 0.7s ease-out both",
        "fade-in": "fadeIn 0.6s ease-out both",
      },
      keyframes: {
        pulseGlow: {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(99,102,241,0.55)" },
          "50%": { boxShadow: "0 0 0 14px rgba(99,102,241,0)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-8px)" },
        },
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
