const fs = require("fs");

// Production browser traffic always stays on the Vercel origin. The Vercel
// Function at /api/backend injects the private HF token server-side.
const apiBaseUrl = process.env.VERCEL
  ? "/api/backend"
  : process.env.OMNIRAG_API_BASE_URL || process.env.VITE_API_BASE_URL || "";

fs.writeFileSync(
  "config.js",
  `window.OMNIRAG_API_BASE_URL = ${JSON.stringify(apiBaseUrl.replace(/\/$/, ""))};\n`,
);
