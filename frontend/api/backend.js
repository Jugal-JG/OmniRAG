/**
 * Same-origin gateway for a private Hugging Face Space.
 *
 * Requests are rewritten from /api/backend/<backend-route> to this single
 * function. Keeping one concrete function path avoids platform-specific
 * catch-all routing differences for nested paths such as /auth/me.
 */

const ALLOWED_PATHS = new Set([
  "",
  "healthz",
  "auth/me",
  "api-status",
  "files",
  "admin/users",
  "upload",
  "upload-chunk",
  "remove-file",
  "clear-files",
  "new-chat",
  "query",
]);

function readPath(req) {
  const value = req.query?.path;
  const parts = Array.isArray(value) ? value : value ? [value] : [];
  if (parts.length) return parts.join("/").replace(/^\/+|\/+$/g, "");
  return "";
}

function copyResponseHeaders(upstream, res) {
  for (const name of ["content-type", "cache-control"]) {
    const value = upstream.headers.get(name);
    if (value) res.setHeader(name, value);
  }
}

module.exports = async function handler(req, res) {
  const spaceUrl = (process.env.HF_SPACE_URL || "").replace(/\/$/, "");
  const token = process.env.HF_SPACE_READ_TOKEN || process.env.HF_TOKEN;
  const path = readPath(req);

  if (!spaceUrl || !token) {
    return res.status(500).json({
      error: "Vercel proxy is not configured. Set HF_SPACE_URL and HF_SPACE_READ_TOKEN.",
    });
  }
  const isAdminDeletion = /^admin\/users\/[a-f0-9]{32}(?:\/workspace|\/files\/[^/]+)?$/.test(path);
  if (!ALLOWED_PATHS.has(path) && !isAdminDeletion) {
    return res.status(404).json({ error: "Unknown backend route." });
  }

  const headers = { Authorization: `Bearer ${token}` };
  for (const name of ["content-type", "x-omnirag-auth", "x-omnirag-session-id"]) {
    const value = req.headers[name];
    if (value) headers[name] = value;
  }

  try {
    const method = req.method || "GET";
    const requestOptions = { method, headers };
    if (!["GET", "HEAD"].includes(method)) {
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

module.exports.config = {
  api: { bodyParser: false },
  maxDuration: 60,
};
