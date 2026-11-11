/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: "#0d0f13",
          panel: "#14181f",
          border: "#262c36",
          ink: "#f0efec",
          muted: "#8b949e",
          faint: "#5a6370",
        },
        series: {
          1: "#3987e5",
          2: "#d95926",
          3: "#199e70",
          4: "#c98500",
          5: "#d55181",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
