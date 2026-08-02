/* ── OmniRAG login page ──────────────────────────────────────────────────────
 * Static presentation only: renders the document-mosaic backdrop and applies
 * the app's existing theme. The "Continue with Google" button is inert by
 * design — no auth flow is wired up here.
 */

const THEME_STORAGE_KEY = "omnirag_theme";
const AUTH_STORAGE_KEY = "omnirag_google_id_token";
const API_BASE_URL = (window.OMNIRAG_API_BASE_URL || "").replace(/\/$/, "");
document.documentElement.dataset.theme = localStorage.getItem(THEME_STORAGE_KEY) || "light";

const TILE_PRESETS = [
  { ext: "PDF",  ac: "#d0563f", ic: "bi bi-file-earmark-pdf",         chip: "rgba(208,86,63,.14)" },
  { ext: "XLSX", ac: "#2f9e6b", ic: "bi bi-file-earmark-spreadsheet", chip: "rgba(47,158,107,.14)" },
  { ext: "DOCX", ac: "#4a83c9", ic: "bi bi-file-earmark-word",        chip: "rgba(74,131,201,.14)" },
  { ext: "PNG",  ac: "#b98bd9", ic: "bi bi-file-earmark-image",       chip: "rgba(185,139,217,.14)" },
  { ext: "PPTX", ac: "#d08a3f", ic: "bi bi-file-earmark-slides",      chip: "rgba(208,138,63,.14)" },
  { ext: "CSV",  ac: "#3fb0b0", ic: "bi bi-filetype-csv",             chip: "rgba(63,176,176,.14)" },
  { ext: "MD",   ac: "#8b95a8", ic: "bi bi-markdown",                 chip: "rgba(139,149,168,.14)" },
  { ext: "JPG",  ac: "#c98fae", ic: "bi bi-file-earmark-image",       chip: "rgba(201,143,174,.14)" },
  { ext: "TXT",  ac: "#9aa3b2", ic: "bi bi-file-earmark-text",        chip: "rgba(154,163,178,.14)" },
  { ext: "JSON", ac: "#c8a86a", ic: "bi bi-filetype-json",            chip: "rgba(200,168,106,.14)" },
];
const TILE_HEIGHTS = [150, 120, 182, 138, 108, 168, 132, 158, 126, 176];

function mosaicColumns() {
  if (window.innerWidth <= 560) return 2;
  if (window.innerWidth <= 900) return 4;
  return 8;
}

function mosaicTileCount() {
  // The mosaic starts above and extends below the viewport. Calculate enough
  // tiles to cover that full rotated canvas instead of relying on a fixed 56.
  const averageTileHeight = 160; // tile height plus its vertical gap
  const rows = Math.ceil((window.innerHeight * 1.78) / averageTileHeight);
  return Math.max(56, mosaicColumns() * (rows + 2));
}

function buildTiles(count = mosaicTileCount()) {
  const tiles = [];
  for (let i = 0; i < count; i++) {
    const preset = TILE_PRESETS[(i * 3 + 1) % TILE_PRESETS.length];
    const h = TILE_HEIGHTS[(i * 5 + 2) % TILE_HEIGHTS.length];
    tiles.push({ ...preset, h });
  }
  return tiles;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMosaic() {
  const mosaic = document.getElementById("loginMosaic");
  if (!mosaic) return;
  mosaic.innerHTML = buildTiles().map(t => `
    <div class="login-tile" style="height:${t.h}px">
      <div class="login-tile-bar" style="background:${t.ac}"></div>
      <div class="login-tile-body">
        <div class="login-tile-chip" style="color:${t.ac};background:${t.chip}">
          <i class="${t.ic}"></i>${escapeHtml(t.ext)}
        </div>
        <div class="login-tile-line" style="width:92%"></div>
        <div class="login-tile-line" style="width:78%"></div>
        <div class="login-tile-line" style="width:60%"></div>
        <i class="${t.ic} login-tile-ghost" style="color:${t.ac}"></i>
      </div>
    </div>`).join("");
}

renderMosaic();

let mosaicResizeTimer;
window.addEventListener("resize", () => {
  window.clearTimeout(mosaicResizeTimer);
  mosaicResizeTimer = window.setTimeout(renderMosaic, 150);
});

function showLoginError(message) {
  const error = document.getElementById("loginError");
  if (!error) return;
  error.textContent = message;
  error.hidden = false;
}

function destination() {
  const next = new URLSearchParams(window.location.search).get("next");
  return next && next.startsWith("/") && !next.startsWith("//") ? next : "/";
}

async function onGoogleCredential(response) {
  if (!response?.credential) {
    showLoginError("Google did not return a sign-in token. Please try again.");
    return;
  }
  try {
    const res = await fetch(`${API_BASE_URL}/auth/me`, {
      headers: { "X-Omnirag-Auth": response.credential },
    });
    if (!res.ok) throw new Error("The server could not verify this Google account.");
    localStorage.setItem(AUTH_STORAGE_KEY, response.credential);
    window.location.replace(destination());
  } catch (error) {
    localStorage.removeItem(AUTH_STORAGE_KEY);
    showLoginError(error.message || "Sign-in failed. Please try again.");
  }
}

function initializeGoogleSignIn() {
  const clientId = window.OMNIRAG_GOOGLE_CLIENT_ID || "";
  const target = document.getElementById("googleSignIn");
  if (!clientId) {
    showLoginError("Google sign-in is not configured. Set GOOGLE_OAUTH_CLIENT_ID and redeploy.");
    return;
  }
  if (!window.google?.accounts?.id) {
    window.setTimeout(initializeGoogleSignIn, 50);
    return;
  }
  google.accounts.id.initialize({
    client_id: clientId,
    callback: onGoogleCredential,
    auto_select: false,
    cancel_on_tap_outside: true,
  });
  google.accounts.id.renderButton(target, {
    theme: "outline",
    size: "large",
    shape: "rectangular",
    text: "continue_with",
    width: Math.min(400, Math.max(260, target.clientWidth || 400)),
  });
}

initializeGoogleSignIn();
