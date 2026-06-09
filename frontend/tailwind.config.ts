import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0f172a",
        mist: "#e2e8f0",
        aurora: "#7c3aed"
      },
      boxShadow: {
        glow: "0 20px 80px rgba(124, 58, 237, 0.22)"
      }
    }
  },
  plugins: []
};

export default config;
