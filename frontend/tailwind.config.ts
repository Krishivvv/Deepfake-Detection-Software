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
        ink: {
          900: "#07090A",
          800: "#0C1012",
          700: "#10161A",
          600: "#161E24",
          500: "#1F2A31",
          400: "#2C3A43",
          300: "#3F525E",
        },
        mint: {
          50: "#E8FFF6",
          100: "#C8FBE6",
          200: "#9CF2D0",
          300: "#65E3B3",
          400: "#34D399",
          500: "#10B981",
          600: "#059669",
          700: "#047857",
          800: "#065F46",
          900: "#064E3B",
        },
        amber: {
          400: "#FBBF24",
          500: "#F59E0B",
        },
        rose: {
          400: "#FB7185",
          500: "#F43F5E",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(16, 185, 129, 0.45), 0 0 32px -4px rgba(16, 185, 129, 0.25)",
        "glow-sm": "0 0 0 1px rgba(16, 185, 129, 0.35), 0 0 18px -4px rgba(16, 185, 129, 0.2)",
        "glow-lg": "0 0 0 1px rgba(16, 185, 129, 0.55), 0 0 80px -10px rgba(16, 185, 129, 0.45)",
        sink: "inset 0 1px 0 rgba(255,255,255,0.04), inset 0 0 0 1px rgba(255,255,255,0.02)",
      },
      backgroundImage: {
        "grid-fade":
          "radial-gradient(circle at 50% 0%, rgba(16,185,129,0.18), transparent 60%), linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px)",
        "mesh":
          "radial-gradient(at 20% 10%, rgba(16,185,129,0.16) 0, transparent 50%), radial-gradient(at 80% 0%, rgba(52,211,153,0.10) 0, transparent 50%), radial-gradient(at 50% 100%, rgba(5,150,105,0.10) 0, transparent 60%)",
      },
      animation: {
        "pulse-glow": "pulseGlow 2.4s ease-in-out infinite",
        "float": "float 6s ease-in-out infinite",
        "fade-up": "fadeUp 0.7s ease-out both",
        "fade-in": "fadeIn 0.6s ease-out both",
      },
      keyframes: {
        pulseGlow: {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(16,185,129,0.55)" },
          "50%": { boxShadow: "0 0 0 14px rgba(16,185,129,0)" },
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
