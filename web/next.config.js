/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow @mediapipe/tasks-vision WASM assets to be served correctly.
  // The vision bundle ships its own wasm files that must not be re-processed.
  webpack: (config) => {
    config.resolve.fallback = { fs: false, path: false };
    return config;
  },
  // Do NOT use next.config rewrites to proxy /api/* → Fly. Vercel **Deployment Protection**
  // (SSO / password on previews) runs at the edge *before* rewrites; `/api/laksh/*` then
  // returns 401 HTML instead of reaching Next.js. The browser calls Fly directly instead
  // (see web/lib/api.ts + CORS on the FastAPI app).
  //
  // Do NOT set Cross-Origin-Embedder-Policy: require-corp globally (breaks cross-origin fetch).
};

module.exports = nextConfig;
