/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: '#12121a',
          dark: '#0a0a0f',
          light: '#1a1a2e',
        },
        neon: {
          green: '#00ff88',
          cyan: '#00ccff',
          magenta: '#ff00ff',
          amber: '#ffaa00',
          red: '#ff3355',
        },
        border: {
          DEFAULT: '#1e1e2e',
          hover: '#2e2e4e',
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'neon-green': '0 0 20px rgba(0, 255, 136, 0.3)',
        'neon-cyan': '0 0 20px rgba(0, 204, 255, 0.3)',
        'neon-magenta': '0 0 20px rgba(255, 0, 255, 0.3)',
        'neon-amber': '0 0 20px rgba(255, 170, 0, 0.3)',
        'glow': '0 0 40px rgba(0, 255, 136, 0.15)',
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        }
      }
    },
  },
  plugins: [],
}
