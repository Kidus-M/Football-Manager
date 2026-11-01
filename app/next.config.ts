import type { NextConfig } from "next";

// The Python API (uvicorn recruit_intel.api.main:app) serves /api/*; the app
// proxies to it so the browser only ever talks to one origin.
const API_ORIGIN = process.env.RECRUIT_API_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_ORIGIN}/api/:path*` }];
  },
};

export default nextConfig;
