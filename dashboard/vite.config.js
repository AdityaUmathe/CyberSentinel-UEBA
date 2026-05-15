import { defineConfig } from "vite";

/**
 * CyberSentinel UEBA dashboard — Vite config.
 *
 *  - `root` is the dashboard/ directory; entry is dashboard/index.html.
 *  - Build output goes to dashboard/dist/ which Flask serves at `/`.
 *  - During `npm run dev`, the Vite dev server proxies /api/* to the Flask
 *    server on port 3026 so the dashboard fetches still work end-to-end.
 */
export default defineConfig({
  root: ".",
  // Absolute base so /assets/* resolves from root no matter which tab path
  // (e.g. /feed, /users) the browser is currently on.
  base: "/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    assetsDir: "assets",
    sourcemap: false,
    target: "es2020",
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:3026",
        changeOrigin: false,
        secure: false,
      },
    },
  },
  preview: {
    port: 4173,
  },
});
