/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        space: {
          950: "#030712",
          900: "#060919",
          850: "#0a0f29",
          800: "#0f1738",
          700: "#172554",
        },
        cyan: {
          400: "#22d3ee",
          500: "#06b6d4",
          glow: "#00e5ff",
        },
        amber: {
          glow: "#ffb300",
        },
        crimson: {
          glow: "#ff334b",
        }
      },
      animation: {
        "meteor-effect": "meteor 5s linear infinite",
        "pulse-glow": "pulseGlow 2.5s ease-in-out infinite",
        "shimmer": "shimmer 2s linear infinite",
      },
      keyframes: {
        meteor: {
          "0%": { transform: "rotate(215deg) translateX(0)", opacity: "1" },
          "70%": { opacity: "1" },
          "100%": {
            transform: "rotate(215deg) translateX(-500px)",
            opacity: "0",
          },
        },
        pulseGlow: {
          "0%, 100%": { opacity: "1", filter: "drop-shadow(0 0 15px rgba(0,229,255,0.6))" },
          "50%": { opacity: "0.6", filter: "drop-shadow(0 0 5px rgba(0,229,255,0.2))" }
        },
        shimmer: {
          from: { backgroundPosition: "0 0" },
          to: { backgroundPosition: "-200% 0" },
        }
      }
    },
  },
  plugins: [],
}
