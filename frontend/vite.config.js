import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Backend to proxy to. `node start.cjs --backend http://localhost:5002 --port 5174` runs a
// second copy of the app next to the usual one (see .claude/launch.json)
const backend = process.env.BACKEND_URL || 'http://localhost:5001'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/video_feed': backend,
      '/toggle_detection': backend,
      '/toggle_batched_mode': backend,
      '/toggle_tracking': backend,
      '/toggle_auto_fps': backend,
      '/set_target_fps': backend,
      '/get_config': backend,
      '/api': backend,
    }
  }
})
