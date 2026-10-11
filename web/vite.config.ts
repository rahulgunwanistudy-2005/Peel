import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Relative base so the same build works on Vercel, on a custom domain and under the Space root.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { target: 'es2022', assetsInlineLimit: 0 },
  server: { port: 5173 },
})
