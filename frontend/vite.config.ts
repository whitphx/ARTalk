import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiTarget = process.env.ARTALK_API_TARGET ?? 'http://127.0.0.1:8961'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
    proxy: {
      '/api': apiTarget,
    },
  },
})
