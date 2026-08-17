import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// En developpement, Vite (:5173) proxifie /api vers Flask (:8000).
// Le code frontend appelle donc toujours des chemins relatifs (/api/...),
// ce qui le rend identique en dev et en demo (ou Flask sert le build).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
