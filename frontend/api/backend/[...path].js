/**
 * Same-origin gateway for a private Hugging Face Space.
 *
 * The browser calls /api/backend/<backend-route>. This function adds the HF
 * bearer token server-side, so the token is never present in frontend assets.
 */

const ALLOWED_PATHS = new Set([
  "",
  "healthz",
  "api-status",
  "upload",
  "remove-file",
  "clear-files",
  "new-chat",
  "query",
]);

function readPath(req) {
  const value = req.query?.path;
  const parts = Array.isArray(value) ? value : value ? [value] : [];
  if (parts.length) return parts.join("/").replace(/^\/+|\/+$/g, "");

  // Vercel's plain Node function runtime does not always populate req.query
  // for a catch-all route. Fall back to the original URL so `/api/backend/upload`
  // is always forwarded as `/upload`, never as the HF Space root `/`.
  const pathname = new URL(req.url || "/", "http://localhost").pathname;
  const prefix = "/api/backend";
  return pathname.startsWith(prefix)
    ? pathname.slice(prefix.length).replace(/^\/+|\/+$/g, "")
    : "";
}

function copyResponseHeaders(upstream, res) {
  for (const name of ["content-type", "cache-control"]) {
    const value = upstream.headers.get(name);
    if (value) res.setHeader(name, value);
  }
}

module.exports = async function handler(req, res) {
  const spaceUrl = (process.env.HF_SPACE_URL || "").replace(/\/$/, "");
  // HF_TOKEN is accepted temporarily for existing deployments. Prefer the
  // explicit HF_SPACE_READ_TOKEN name in new Vercel environment settings.
  const token = process.env.HF_SPACE_READ_TOKEN || process.env.HF_TOKEN;
  const path = readPath(req);

  if (!spaceUrl || !token) {
    return res.status(500).json({
      error: "Vercel proxy is not configured. Set HF_SPACE_URL and HF_SPACE_READ_TOKEN.",
    });
  }
  if (!ALLOWED_PATHS.has(path)) {
    return res.status(404).json({ error: "Unknown backend route." });
  }

  const headers = {
    Authorization: `Bearer ${token}`,
  };
  for (const name of ["content-type", "x-omnirag-session-id"]) {
    const value = req.headers[name];
    if (value) headers[name] = value;
  }

  try {
    const method = req.method || "GET";
    const requestOptions = { method, headers };
    if (!["GET", "HEAD"].includes(method)) {
      // Forward multipart uploads and JSON without parsing/re-encoding them.
      requestOptions.body = req;
      requestOptions.duplex = "half";
    }

    const upstream = await fetch(`${spaceUrl}/${path}`, requestOptions);
    copyResponseHeaders(upstream, res);
    res.status(upstream.status);
    res.send(Buffer.from(await upstream.arrayBuffer()));
  } catch (error) {
    console.error("HF Space proxy request failed:", error);
    res.status(502).json({ error: "Unable to reach the private backend." });
  }
};

// Preserve upload streams instead of letting Vercel parse multipart bodies.
module.exports.config = {
  api: { bodyParser: false },
  maxDuration: 60,
};
