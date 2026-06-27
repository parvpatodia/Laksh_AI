import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Primary action color -- sky blue, consistent with brand identity.
        brand: {
          50:  "#f0f9ff",
          400: "#38bdf8",
          500: "#0ea5e9",
          600: "#0284c7",
          700: "#0369a1",
          900: "#0c4a6e",
        },
        // Performance accent -- amber/gold for elite-tier highlights.
        perf: {
          400: "#fbbf24",
          500: "#f59e0b",
          600: "#d97706",
        },
        // Surface hierarchy -- deep navy for premium dark-mode display and TV.
        surface: {
          950: "#03080f",
          900: "#080f1a",
          850: "#0c1524",
          800: "#111c30",
          750: "#162035",
          700: "#1c2a40",
          600: "#273a55",
          500: "#364e6e",
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
      letterSpacing: {
        widest2: "0.25em",
      },
      animation: {
        "pulse-slow":  "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-up":     "fadeUp 0.6s ease-out both",
        "fade-in":     "fadeIn 0.4s ease-out both",
      },
      keyframes: {
        fadeUp: {
          "0%":   { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          "0%":   { opacity: "0" },
          "100%": { opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
