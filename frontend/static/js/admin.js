const API_BASE_URL = (window.OMNIRAG_API_BASE_URL || "").replace(/\/$/, "");
const AUTH_STORAGE_KEY = "omnirag_google_id_token";
const SESSION_STORAGE_KEY = "omnirag_session_id";

function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const credential = localStorage.getItem(AUTH_STORAGE_KEY);
  if (credential) headers.set("X-Omnirag-Auth", credential);
  return fetch(`${API_BASE_URL}${path}`, { ...options, headers }).then(response => {
    if (response.status === 401) {
      localStorage.removeItem(AUTH_STORAGE_KEY);
      window.location.replace("/login?next=%2Fadmin");
    }
    return response;
  });
}

function escapeHtml(value) {
  return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function formatBytes(value) {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / (1024 ** index)).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function formatDate(seconds) {
  return seconds ? new Date(seconds * 1000).toLocaleString() : "—";
}

function showError(message) {
  const error = document.getElementById("adminError");
  error.textContent = message;
  error.hidden = false;
}

async function json(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function renderSummary(totals = {}, users = []) {
  const safeUsers = Array.isArray(users) ? users : [];
  const cachedSpreadsheets = safeUsers.reduce(
    (sum, user) => sum + (Array.isArray(user.cached_spreadsheets) ? user.cached_spreadsheets.length : 0), 0
  );
  document.getElementById("summaryCards").innerHTML = [
    ["Registered users", totals.users || 0],
    ["Uploaded files", safeUsers.reduce((sum, user) => sum + (Array.isArray(user.files) ? user.files.length : 0), 0)],
    ["Cached spreadsheets", cachedSpreadsheets],
    ["Upload storage", formatBytes(totals.upload_bytes || 0)],
    ["Index/cache storage", formatBytes(totals.cache_bytes || 0)],
  ].map(([label, value]) => `<article class="summary-card"><span>${label}</span><strong>${value}</strong></article>`).join("");
}

function userCard(user) {
  const userFiles = Array.isArray(user.files) ? user.files : [];
  const cachedSpreadsheets = Array.isArray(user.cached_spreadsheets) ? user.cached_spreadsheets : [];
  const files = userFiles.length
    ? userFiles.map(file => `<div class="file-row"><i class="bi bi-file-earmark"></i><span class="file-name">${escapeHtml(file.name)}</span><span class="file-size">${formatBytes(file.bytes)}</span><button class="file-delete" data-action="file" data-key="${user.account_key}" data-email="${escapeHtml(user.email)}" data-file="${encodeURIComponent(file.name)}">Delete</button></div>`).join("")
    : '<p class="file-empty">No uploaded files.</p>';
  const cachedFiles = cachedSpreadsheets.length
    ? cachedSpreadsheets.map(file => `<div class="cache-row"><i class="bi bi-database"></i><span class="file-name">${escapeHtml(file.name)}</span><span class="cache-detail">${Number(file.rows || 0).toLocaleString()} rows · ${escapeHtml(file.status || "unknown")}</span><span class="cache-state ${file.is_active_upload ? "active" : "retained"}">${file.is_active_upload ? "Active upload" : "Retained cache"}</span></div>`).join("")
    : '<p class="file-empty">No cached spreadsheets.</p>';
  return `<article class="user-card">
    <div class="user-top"><div class="user-identity"><h3 class="user-name">${escapeHtml(user.name)}</h3><div class="user-email">${escapeHtml(user.email)}</div><div class="user-meta">First seen: ${formatDate(user.first_seen)} · Last seen: ${formatDate(user.last_seen)}</div></div>
      <div class="user-storage"><div class="storage-stat"><span>Uploads</span><strong>${formatBytes(user.upload_bytes)}</strong></div><div class="storage-stat"><span>Cache</span><strong>${formatBytes(user.cache_bytes)}</strong></div><div class="storage-stat"><span>Total</span><strong>${formatBytes(user.total_bytes)}</strong></div></div>
    </div><div class="user-files"><h4>Active uploads</h4>${files}</div><div class="user-files cache-files"><h4>Cached spreadsheet records</h4><p class="cache-note">Retained SQLite metadata, including records whose source upload was removed.</p>${cachedFiles}</div><div class="user-footer"><button class="danger-button" data-action="workspace" data-key="${user.account_key}" data-email="${escapeHtml(user.email)}">Delete workspace data</button><button class="danger-button forget-button" data-action="forget" data-key="${user.account_key}" data-email="${escapeHtml(user.email)}">Forget user completely</button></div>
  </article>`;
}

function renderUsers(users) {
  window.adminUsers = users;
  document.getElementById("userCount").textContent = `${users.length} registered user${users.length === 1 ? "" : "s"}`;
  document.getElementById("userList").innerHTML = users.length ? users.map(userCard).join("") : '<div class="admin-error">No authenticated users have been recorded yet.</div>';
}

async function loadUsers() {
  const response = await apiFetch("/admin/users");
  const data = await json(response);
  const users = Array.isArray(data.users) ? data.users : [];
  renderSummary(data.totals, users);
  renderUsers(users);
}

async function deleteData(action, key, email, encodedFile) {
  const target = action === "file" ? decodeURIComponent(encodedFile) : email;
  const label = action === "file" ? `file “${target}”` : action === "workspace" ? `all OmniRAG data for ${email}` : `all OmniRAG data and the user record for ${email}`;
  if (action !== "file" && prompt(`Type ${email} to permanently delete ${label}:`) !== email) return;
  if (action === "file" && !confirm(`Permanently delete ${label}? Its derived cache will also be removed.`)) return;
  const path = action === "file"
    ? `/admin/users/${key}/files/${encodedFile}`
    : action === "workspace" ? `/admin/users/${key}/workspace` : `/admin/users/${key}`;
  await json(await apiFetch(path, { method: "DELETE" }));
  await loadUsers();
}

document.getElementById("refreshUsers").addEventListener("click", () => loadUsers().catch(error => showError(error.message)));
document.getElementById("adminSignOut").addEventListener("click", () => {
  localStorage.removeItem(AUTH_STORAGE_KEY);
  localStorage.removeItem(SESSION_STORAGE_KEY);
  window.location.replace("/login");
});
document.getElementById("userList").addEventListener("click", event => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  deleteData(button.dataset.action, button.dataset.key, button.dataset.email, button.dataset.file).catch(error => showError(error.message));
});

(async () => {
  document.documentElement.dataset.theme = localStorage.getItem("omnirag_theme") || "light";
  try {
    const profile = await json(await apiFetch("/auth/me"));
    if (!profile.is_admin) throw new Error("This account is not allowed to access the owner console.");
    await loadUsers();
    document.getElementById("adminContent").hidden = false;
  } catch (error) {
    showError(error.message);
  }
})();
