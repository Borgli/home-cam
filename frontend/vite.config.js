import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/video_feed': 'http://localhost:5001',
      '/toggle_detection': 'http://localhost:5001',
      '/toggle_batched_mode': 'http://localhost:5001',
      '/toggle_tracking': 'http://localhost:5001',
      '/toggle_auto_fps': 'http://localhost:5001',
      '/set_target_fps': 'http://localhost:5001',
      '/get_config': 'http://localhost:5001',
      '/api': 'http://localhost:5001',
    }
  }
})
