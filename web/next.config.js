/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow @mediapipe/tasks-vision WASM assets to be served correctly.
  // The vision bundle ships its own wasm files that must not be re-processed.
  webpack: (config) => {
    config.resolve.fallback = { fs: false, path: false };
    return config;
  },
  // Headers: allow SharedArrayBuffer for WASM multi-threading if needed.
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
