/** @type {import('next').NextConfig} */

function computeApiBaseUrl() {
  const apiUrl = process.env.API_URL;
  if (!apiUrl || apiUrl.length === 0) {
    return "http://localhost:8000";
  }
  if (apiUrl.startsWith("http://") || apiUrl.startsWith("https://")) {
    return apiUrl;
  }
  // When API_URL is just a hostname (e.g., from Render's fromService.host)
  return `https://${apiUrl}`;
}

const nextConfig = {
  experimental: {
    reactCompiler: true,
  },
  compiler: {
    removeConsole: {
      exclude: ["error"],
    },
  },
  reactStrictMode: false,
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "**",
      },
      {
        protocol: "http",
        hostname: "**",
      },
    ],
  },
  rewrites: async () => {
    const API_BASE = computeApiBaseUrl();
    return [
      {
        source: "/api/:path*",
        destination: `${API_BASE}/api/:path*`,
      },
      {
        source: "/docs",
        destination: `${API_BASE}/docs`,
      },
      {
        source: "/openapi.json",
        destination: `${API_BASE}/openapi.json`,
      },
    ];
  },
};

module.exports = nextConfig;
