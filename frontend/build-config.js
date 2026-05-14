const fs = require("fs");

const apiBaseUrl = process.env.OMNIRAG_API_BASE_URL || process.env.VITE_API_BASE_URL || "";

fs.writeFileSync(
  "config.js",
  `window.OMNIRAG_API_BASE_URL = ${JSON.stringify(apiBaseUrl.replace(/\/$/, ""))};\n`,
);
