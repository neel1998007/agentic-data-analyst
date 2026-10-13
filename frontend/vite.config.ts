import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Proxy API routes to the local FastAPI backend during development so
    // relative URLs like fetch('/ask') work without a running Nginx.
    proxy: {
      '/ask':    'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})

