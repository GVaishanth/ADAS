import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Builds the merged ADAS frontend as a static, client-only SPA that the
// FastAPI backend (port 7000) serves. Base is './' so asset URLs are relative
// and the same bundle works whether it's served at "/" or "/dashboard".
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { outDir: 'dist', assetsDir: 'assets' },
  server: { host: true, port: 3000 },
});
