/* ── Config ───────────────────────────────────────────────────────────────── */
const ENGINE_BADGE_CLASS = {
  "basic_rag":      "basic-rag",
  "router_engine":  "router-engine",
  "subquestion":    "subquestion",
  "multi_document": "multi-document",
  "multimodal":     "multimodal",
  "react":          "react",
  "merged":         "merged",
};

const ENGINE_ICON = {
  "basic_rag":      "bi-search",
  "router_engine":  "bi-signpost-split",
  "subquestion":    "bi-question-diamond",
  "multi_document": "bi-files",
  "multimodal":     "bi-image",
  "react":          "bi-cpu",
  "merged":         "bi-intersect",
};

const API_BASE_URL = (window.OMNIRAG_API_BASE_URL || "").replace(/\/$/, "");
const SESSION_STORAGE_KEY = "omnirag_session_id";
const AUTH_STORAGE_KEY = "omnirag_google_id_token";

function getSessionId() {
  let sid = localStorage.getItem(SESSION_STORAGE_KEY);
  if (!sid) {
    sid = crypto.randomUUID();
    localStorage.setItem(SESSION_STORAGE_KEY, sid);
  }
  return sid;
}

function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-Omnirag-Session-Id", getSessionId());
  const credential = localStorage.getItem(AUTH_STORAGE_KEY);
  if (credential) headers.set("X-Omnirag-Auth", credential);

  // No `credentials: "include"`: the session is tracked entirely via the
  // X-Omnirag-Session-Id header above, so cross-site cookies aren't needed.
  // Avoiding credentialed mode sidesteps fragile third-party-cookie handling
  // and the stricter preflight rules that go with it.
  return fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  }).then(res => {
    if (res.status === 401) {
      localStorage.removeItem(AUTH_STORAGE_KEY);
      const next = `${window.location.pathname}${window.location.search}${window.location.hash}`;
      window.location.replace(`/login?next=${encodeURIComponent(next)}`);
    }
    return res;
  });
}

// The Vercel proxy in front of the private HF Space rejects request bodies
// over ~4.5 MB before our function runs, so files larger than one chunk are
// uploaded in ≤4 MB pieces and reassembled server-side. MAX_UPLOAD_BYTES is
// the backend's limit on the final reassembled file.
const UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024;    // per-request cap (< Vercel's ~4.5 MB)
const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;     // reassembled-file cap (Flask MAX_CONTENT_LENGTH)

function formatBytes(n) {
  return n >= 1024 * 1024 ? `${(n / (1024 * 1024)).toFixed(1)} MB` : `${Math.ceil(n / 1024)} KB`;
}

/**
 * Parse a backend response as JSON, translating non-JSON error pages
 * (Vercel 413s, gateway 502/503s, Flask HTML error pages) into readable
 * errors instead of "Unexpected token '<'".
 */
async function readJsonResponse(res) {
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    if (res.status === 413) {
      throw new Error(`file too large — the server accepts at most ${formatBytes(MAX_UPLOAD_BYTES)} per upload`);
    }
    if ([502, 503, 504].includes(res.status)) {
      throw new Error("the backend is unreachable or still waking up — try again in a moment");
    }
    throw new Error(`the server returned an unexpected ${res.status} response`);
  }
}

/* ── State ────────────────────────────────────────────────────────────────── */
let uploadedFiles = [];
let selectedFiles = new Set();
let isLoading = false;
let abortController = null;   // active AbortController while a query is in flight

/* ── DOM refs ─────────────────────────────────────────────────────────────── */
const dropZone      = document.getElementById("dropZone");
const fileInput     = document.getElementById("fileInput");
const fileList      = document.getElementById("fileList");
const chatContainer = document.getElementById("chatContainer");
const welcomeMsg    = document.getElementById("welcomeMsg");
const queryInput    = document.getElementById("queryInput");
const sendBtn       = document.getElementById("sendBtn");
const sendIcon      = document.getElementById("sendIcon");
const multiDocToggle = document.getElementById("multiDocToggle");
const thinkingToggle = document.getElementById("thinkingToggle");
const multiDocCard  = document.getElementById("multiDocCard");
const thinkingCard  = document.getElementById("thinkingCard");
const activeModes   = document.getElementById("activeModes");
const clearFilesBtn = document.getElementById("clearFilesBtn");
const newChatBtn    = document.getElementById("newChatBtn");
const adminBtn      = document.getElementById("adminBtn");
const signOutBtn    = document.getElementById("signOutBtn");
const themeToggle   = document.getElementById("themeToggle");
const THEME_STORAGE_KEY = "omnirag_theme";

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const dark = theme === "dark";
  themeToggle.setAttribute("aria-label", `Switch to ${dark ? "light" : "dark"} theme`);
  themeToggle.title = `Switch to ${dark ? "light" : "dark"} theme`;
}

function animateThemeChange(nextTheme) {
  const rect = themeToggle.getBoundingClientRect();
  const wipe = document.createElement("div");
  wipe.className = `theme-genie theme-genie-${nextTheme}`;
  wipe.style.left = `${rect.left + rect.width / 2}px`;
  wipe.style.top = `${rect.top + rect.height / 2}px`;
  document.body.appendChild(wipe);

  requestAnimationFrame(() => wipe.classList.add("animate"));
  window.setTimeout(() => applyTheme(nextTheme), 280);
  window.setTimeout(() => wipe.remove(), 1180);
}

themeToggle.addEventListener("click", () => {
  const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
  animateThemeChange(nextTheme);
});

applyTheme(localStorage.getItem(THEME_STORAGE_KEY) || "light");

/* ── Bootstrap Toast ──────────────────────────────────────────────────────── */
const toastEl = document.getElementById("toast");
const toastBody = document.getElementById("toastBody");
const bsToast = new bootstrap.Toast(toastEl, { delay: 3500 });

function showToast(msg, type = "success") {
  toastEl.classList.remove("bg-success", "bg-danger", "bg-warning", "bg-info");
  toastEl.classList.add(`bg-${type}`);
  toastBody.textContent = msg;
  bsToast.show();
}

/* ── API status ───────────────────────────────────────────────────────────── */
async function loadApiStatus() {
  try {
    const res = await apiFetch("/api-status");
    const data = await readJsonResponse(res);
    const container = document.getElementById("api-status");
    const labels = { mistral: "Mistral", groq: "Groq", google: "Gemini" };
    container.innerHTML = Object.entries(labels).map(([k, label]) =>
      `<span class="provider-state ${data[k] ? 'available' : 'unavailable'}">${label}</span>`
    ).join("");
  } catch {}
}

/* ── File upload ──────────────────────────────────────────────────────────── */
function fileIcon(name) {
  const ext = name.split(".").pop().toLowerCase();
  const map = { pdf: "bi-file-pdf text-danger", txt: "bi-file-text text-secondary",
    png: "bi-file-image text-warning", jpg: "bi-file-image text-warning",
    jpeg: "bi-file-image text-warning", gif: "bi-file-image text-warning",
    html: "bi-filetype-html text-orange", md: "bi-markdown text-info",
    csv: "bi-filetype-csv text-success", xlsx: "bi-file-earmark-spreadsheet text-success" };
  return map[ext] || "bi-file-earmark text-secondary";
}

function renderFileList() {
  fileList.innerHTML = uploadedFiles.map(f =>
    `<div class="file-chip">
       <input class="file-select form-check-input m-0" type="checkbox" aria-label="Use ${escapeHtml(f)} in answers"
              ${selectedFiles.has(f) ? "checked" : ""} ${isLoading ? "disabled" : ""}
              onchange="toggleFileSelection('${escapeHtml(f)}', this.checked)">
       <i class="bi ${fileIcon(f)} file-icon"></i>
       <span class="file-name" title="${f}">${f}</span>
       <button class="file-remove-btn" title="Remove file" ${isLoading ? "disabled" : ""} onclick="removeFile('${escapeHtml(f)}')">
         <i class="bi bi-x"></i>
       </button>
     </div>`
  ).join("");
  clearFilesBtn.hidden = uploadedFiles.length === 0;
  setQueryAvailability();
}

function setQueryAvailability() {
  const hasSelectedFiles = selectedFiles.size > 0;
  queryInput.disabled = isLoading || !hasSelectedFiles;
  sendBtn.disabled = !isLoading && !hasSelectedFiles;
  queryInput.placeholder = hasSelectedFiles
    ? "Ask anything about your documents..."
    : "Select a document to start chatting";
  document.querySelectorAll(".example-btn").forEach(btn => {
    btn.disabled = isLoading || !hasSelectedFiles;
  });
}

function syncUploadedFiles(files, selectNewFiles = false) {
  const previousFiles = new Set(uploadedFiles);
  uploadedFiles = files;
  selectedFiles = new Set(files.filter(f => selectedFiles.has(f) || (selectNewFiles && !previousFiles.has(f))));
}

function toggleFileSelection(fname, selected) {
  if (isLoading) return;
  if (selected) selectedFiles.add(fname);
  else selectedFiles.delete(fname);
  renderFileList();
}

async function removeFile(fname) {
  try {
    const res = await apiFetch("/remove-file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: fname }),
    });
    const data = await readJsonResponse(res);
    if (data.success) {
      syncUploadedFiles(data.files);
      renderFileList();
      showToast(`Removed: ${fname}`, "info");
    }
  } catch (e) {
    showToast("Failed to remove file: " + e.message, "danger");
  }
}

// Upload one large file as a sequence of ≤4 MB chunks. The backend stages the
// chunks and reassembles them once the final one arrives; that final response
// carries the updated file list, which we return to the caller.
async function uploadFileInChunks(file) {
  const uploadId = crypto.randomUUID();
  const totalChunks = Math.max(1, Math.ceil(file.size / UPLOAD_CHUNK_BYTES));
  let data = null;

  for (let index = 0; index < totalChunks; index++) {
    const start = index * UPLOAD_CHUNK_BYTES;
    const blob = file.slice(start, start + UPLOAD_CHUNK_BYTES);

    const formData = new FormData();
    formData.append("upload_id", uploadId);
    formData.append("filename", file.name);
    formData.append("chunk_index", String(index));
    formData.append("total_chunks", String(totalChunks));
    formData.append("chunk", blob, file.name);

    const res = await apiFetch("/upload-chunk", { method: "POST", body: formData });
    data = await readJsonResponse(res);
    if (!res.ok || !data.success) {
      throw new Error(data.error || `upload failed at chunk ${index + 1}/${totalChunks} (${res.status})`);
    }
  }
  return data;
}

async function uploadFiles(files) {
  // Reject files over the backend's reassembled-file limit up front, rather
  // than uploading every chunk only to have the server reject the last one.
  const oversized = files.filter(f => f.size > MAX_UPLOAD_BYTES);
  if (oversized.length) {
    showToast(
      `Skipped (over the ${formatBytes(MAX_UPLOAD_BYTES)} upload limit): ${oversized.map(f => f.name).join(", ")}`,
      "danger"
    );
    files = files.filter(f => f.size <= MAX_UPLOAD_BYTES);
    if (!files.length) return;
  }

  try {
    let data = null;
    for (const file of files) {
      if (file.size > UPLOAD_CHUNK_BYTES) {
        // Too big for a single Vercel request — stream it in chunks.
        data = await uploadFileInChunks(file);
      } else {
        // Small enough to send in one request via the existing /upload route.
        const formData = new FormData();
        formData.append("files", file);
        const res = await apiFetch("/upload", { method: "POST", body: formData });
        data = await readJsonResponse(res);
        if (!res.ok || !data.success) {
          throw new Error(data.error || `upload was rejected (${res.status})`);
        }
      }
    }
    if (data?.success) {
      syncUploadedFiles(data.files, true);
      renderFileList();
      if (data.errors?.length) showToast(`${data.errors[0]}`, "warning");
      else if (data.warnings?.length) showToast(`${data.warnings[0]}`, "warning");
      else showToast(`${files.length} file(s) uploaded`, "success");
    }
  } catch (e) {
    showToast("Upload failed: " + e.message, "danger");
  }
}

dropZone.addEventListener("click", (e) => {
  // Don't trigger if the click came from the Browse label/input — it already opens the dialog
  if (e.target.closest("label") || e.target === fileInput) return;
  fileInput.click();
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    uploadFiles([...fileInput.files]);
    fileInput.value = "";   // reset so re-selecting the same file still triggers "change"
  }
});

dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) uploadFiles([...e.dataTransfer.files]);
});

clearFilesBtn.addEventListener("click", async () => {
  try {
    await apiFetch("/clear-files", { method: "POST" });
    uploadedFiles = [];
    selectedFiles.clear();
    renderFileList();
    showToast("Files cleared", "info");
  } catch {}
});

newChatBtn.addEventListener("click", async () => {
  try {
    await apiFetch("/new-chat", { method: "POST" });
    // Remove all message elements but keep welcome splash hidden
    [...chatContainer.children].forEach(el => {
      if (el.id !== "welcomeMsg") el.remove();
    });
    if (welcomeMsg) welcomeMsg.style.display = "";
    showToast("New chat started — uploaded files kept", "info");
  } catch {}
});

signOutBtn.addEventListener("click", () => {
  // Auth uses a verified Google ID token, not a server-side cookie session.
  // Removing it locally prevents any further authenticated API requests.
  localStorage.removeItem(AUTH_STORAGE_KEY);
  localStorage.removeItem(SESSION_STORAGE_KEY);
  window.location.replace("/login");
});

/* ── Toggle state ─────────────────────────────────────────────────────────── */
function updateToggles() {
  const multi = multiDocToggle.checked;
  const think = thinkingToggle.checked;

  multiDocCard.classList.toggle("active", multi && !think);
  thinkingCard.classList.toggle("active-react", think);
  thinkingCard.classList.remove("active");

  const setBlocked = (card, toggle, blocked, reason) => {
    toggle.disabled = blocked;
    card.classList.toggle("mode-blocked", blocked);
    card.title = blocked ? reason : "";
    card.setAttribute("aria-disabled", String(blocked));
  };

  // Only one explicit engine mode can be active. The unavailable card remains
  // visible and explains why it cannot be selected when hovered.
  if (isLoading) {
    setBlocked(multiDocCard, multiDocToggle, true, "Engine mode is locked while an answer is being generated.");
    setBlocked(thinkingCard, thinkingToggle, true, "Engine mode is locked while an answer is being generated.");
    multiDocCard.style.opacity = "";
    thinkingCard.style.opacity = "";
  } else if (think) {
    setBlocked(multiDocCard, multiDocToggle, true, "Turn off Thinking Mode before selecting Multi-Document Agent.");
    setBlocked(thinkingCard, thinkingToggle, false, "");
    multiDocCard.style.opacity = "";
    thinkingCard.style.opacity = "";
  } else if (multi) {
    setBlocked(multiDocCard, multiDocToggle, false, "");
    setBlocked(thinkingCard, thinkingToggle, true, "Turn off Multi-Document Agent before selecting Thinking Mode.");
    multiDocCard.style.opacity = "";
    thinkingCard.style.opacity = "";
  } else {
    setBlocked(multiDocCard, multiDocToggle, false, "");
    setBlocked(thinkingCard, thinkingToggle, false, "");
    multiDocCard.style.opacity = "";
    thinkingCard.style.opacity = "";
  }

  // Active mode chips in input bar
  const chips = [];
  if (multi && !think)
    chips.push(`<span class="active-mode-chip bg-purple text-dark"><i class="bi bi-files"></i> Multi-Doc</span>`);
  if (think)
    chips.push(`<span class="active-mode-chip bg-danger text-white"><i class="bi bi-cpu"></i> Thinking (ReAct)</span>`);
  activeModes.innerHTML = chips.join("");
}

multiDocToggle.addEventListener("change", () => {
  if (multiDocToggle.checked) thinkingToggle.checked = false;
  updateToggles();
});
thinkingToggle.addEventListener("change", () => {
  if (thinkingToggle.checked) multiDocToggle.checked = false;
  updateToggles();
});

/* ── Auto-resize textarea ─────────────────────────────────────────────────── */
queryInput.addEventListener("input", () => {
  queryInput.style.height = "auto";
  queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + "px";
});

queryInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!isLoading) sendQuery();   // Enter only sends; stop requires clicking the button
  }
});

/* ── Example queries ──────────────────────────────────────────────────────── */
document.querySelectorAll(".example-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    queryInput.value = btn.dataset.query;
    queryInput.dispatchEvent(new Event("input"));
    queryInput.focus();
  });
});

/* ── Chat rendering ───────────────────────────────────────────────────────── */
function hideWelcome() {
  if (welcomeMsg) welcomeMsg.style.display = "none";
}

function appendUserMessage(text) {
  hideWelcome();
  const div = document.createElement("div");
  div.className = "msg-user";
  div.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  chatContainer.appendChild(div);
  scrollToBottom();
}

function appendThinkingBubble() {
  hideWelcome();
  const div = document.createElement("div");
  div.className = "msg-assistant";
  div.id = "thinkingBubble";
  div.innerHTML = `
    <div class="response-card">
      <div class="thinking-dots">
        <div class="thinking-dot"></div>
        <div class="thinking-dot"></div>
        <div class="thinking-dot"></div>
      </div>
    </div>`;
  chatContainer.appendChild(div);
  scrollToBottom();
}

function removeThinkingBubble() {
  const el = document.getElementById("thinkingBubble");
  if (el) el.remove();
}

function appendResponse(data) {
  const label = data.approach_label || "basic_rag";
  const badgeClass = ENGINE_BADGE_CLASS[label] || "basic-rag";
  const icon = ENGINE_ICON[label] || "bi-cpu";

  const hasThinking = data.thinking_steps?.length > 0;
  const hasSources  = data.sources?.length > 0;

  const answerHtml = renderMarkdown(data.answer || "");

  const thinkingHtml = hasThinking ? `
    <div class="section-toggle" onclick="toggleSection(this)">
      <i class="bi bi-lightbulb text-warning"></i>
      Thinking / Intermediate Steps
      <span class="ms-1 badge bg-secondary opacity-75">${data.thinking_steps.length}</span>
      <i class="bi bi-chevron-down chevron"></i>
    </div>
    <div class="section-content">
      <div class="thinking-steps">
        ${data.thinking_steps.map(s => `<div class="thinking-step">${escapeHtml(s)}</div>`).join("")}
      </div>
    </div>` : "";

  const sourcesHtml = hasSources ? `
    <div class="section-toggle" onclick="toggleSection(this)">
      <i class="bi bi-journals text-info"></i>
      Sources
      <span class="ms-1 badge bg-secondary opacity-75">${data.sources.length}</span>
      <i class="bi bi-chevron-down chevron"></i>
    </div>
    <div class="section-content">
      <div class="source-list">
        ${data.sources.map(s => `
          <div class="source-item">
            ${s.file ? `<div class="source-file"><i class="bi bi-file-earmark me-1"></i>${escapeHtml(s.file)} ${s.score !== null ? `<span class="source-score">score: ${s.score}</span>` : ''}</div>` : ""}
            <div>${escapeHtml((s.text || "").substring(0, 250))}${(s.text||"").length > 250 ? "…" : ""}</div>
          </div>`).join("")}
      </div>
    </div>` : "";

  const div = document.createElement("div");
  div.className = "msg-assistant";
  div.innerHTML = `
    <div class="response-card">
      <div class="response-header">
        <span class="engine-badge ${badgeClass}">
          <i class="bi ${icon} me-1"></i>${data.approach}
        </span>
      </div>

      <!-- Router reasoning (always visible, collapsible) -->
      <div class="section-toggle open" onclick="toggleSection(this)">
        <i class="bi bi-signpost text-secondary"></i>
        Router Reasoning
        <i class="bi bi-chevron-down chevron" style="transform:rotate(180deg)"></i>
      </div>
      <div class="section-content show">
        <p class="router-reason mb-0"><i class="bi bi-arrow-right-circle me-1 text-secondary"></i>${escapeHtml(data.router_reason || "")}</p>
      </div>

      <!-- Answer -->
      <div class="response-body">
        <div class="answer-content">${answerHtml}</div>
      </div>

      ${thinkingHtml}
      ${sourcesHtml}
    </div>`;

  chatContainer.appendChild(div);
  // Render any LaTeX math blocks in the answer after the element is in the DOM
  const answerEl = div.querySelector(".answer-content");
  if (answerEl) {
    renderMath(answerEl);
    // Models do not consistently use h1 for the response title. Mark the
    // first heading explicitly so the visual title is stable across answers.
    answerEl.querySelector("h1, h2, h3")?.classList.add("answer-title");
    answerEl.querySelectorAll("li strong:first-child, p > strong:first-child").forEach(strong => {
      const label = strong.textContent.trim();
      const nextText = strong.nextSibling?.textContent?.trimStart() || "";
      const hasTrailingColon = nextText.startsWith(":");
      const parentText = strong.parentElement?.textContent?.trim() || "";
      const listText = strong.closest("li")?.textContent?.trim() || "";
      const isStandalone = parentText === label || listText === label;
      const isShortLabel = /^[\p{L}\p{N}\s-]{2,80}:?$/u.test(label);
      if (isStandalone || (isShortLabel && (label.endsWith(":") || hasTrailingColon))) {
        strong.classList.add("inline-subheading");
      }
    });
    decorateNumericValues(answerEl);
  }
  scrollToBottom();
}

function appendError(msg, approach, reason) {
  const div = document.createElement("div");
  div.className = "msg-assistant";
  div.innerHTML = `
    <div class="response-card border-danger">
      <div class="response-header">
        <span class="engine-badge react"><i class="bi bi-exclamation-triangle me-1"></i>${approach || "Error"}</span>
      </div>
      ${reason ? `<div class="section-content show"><p class="router-reason mb-0">${escapeHtml(reason)}</p></div>` : ""}
      <div class="response-body">
        <div class="alert alert-danger mb-0 py-2"><i class="bi bi-x-circle me-2"></i>${escapeHtml(msg)}</div>
      </div>
    </div>`;
  chatContainer.appendChild(div);
  scrollToBottom();
}

/* ── Send / Stop ──────────────────────────────────────────────────────────── */
function enterLoadingMode() {
  isLoading = true;
  sendBtn.classList.add("stop-mode");
  sendBtn.title = "Stop generation";
  sendIcon.className = "bi bi-stop-fill";
  queryInput.disabled = true;
  updateToggles();
  renderFileList();
}

function exitLoadingMode() {
  isLoading = false;
  abortController = null;
  sendBtn.classList.remove("stop-mode");
  sendBtn.title = "Send (Enter)";
  sendIcon.className = "bi bi-send-fill";
  setQueryAvailability();
  updateToggles();
  renderFileList();
  if (uploadedFiles.length) queryInput.focus();
}

async function sendQuery() {
  // If already loading, act as stop button
  if (isLoading) {
    if (abortController) abortController.abort();
    removeThinkingBubble();
    exitLoadingMode();
    showToast("Generation stopped", "warning");
    return;
  }

  if (!selectedFiles.size) {
    showToast("Select at least one document before asking a question", "info");
    return;
  }

  const query = queryInput.value.trim();
  if (!query) return;

  abortController = new AbortController();
  enterLoadingMode();

  appendUserMessage(query);
  queryInput.value = "";
  queryInput.style.height = "auto";
  appendThinkingBubble();

  try {
    const res = await apiFetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        selected_files: [...selectedFiles],
        multi_doc: multiDocToggle.checked,
        thinking: thinkingToggle.checked,
      }),
      signal: abortController.signal,
    });

    const data = await readJsonResponse(res);
    removeThinkingBubble();

    if (!res.ok || data.error) {
      appendError(data.error || "Server error", data.approach, data.router_reason);
    } else {
      appendResponse(data);
    }
  } catch (e) {
    removeThinkingBubble();
    if (e.name !== "AbortError") {
      appendError("Network error: " + e.message);
    }
    // AbortError means user clicked stop — already handled above, no error card needed
  } finally {
    exitLoadingMode();
  }
}

sendBtn.addEventListener("click", sendQuery);

/* ── Collapsible sections ─────────────────────────────────────────────────── */
function toggleSection(toggleEl) {
  const content = toggleEl.nextElementSibling;
  const isOpen = toggleEl.classList.toggle("open");
  content.classList.toggle("show", isOpen);
}

/* ── Helpers ──────────────────────────────────────────────────────────────── */
function scrollToBottom() {
  requestAnimationFrame(() => {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// GFM enables tables/strikethrough; breaks makes single newlines from LLM
// answers render as line breaks instead of gluing sentences together.
if (typeof marked !== "undefined") {
  marked.use({ gfm: true, breaks: true });
}

/**
 * Heuristic: does the text between two $…$ delimiters actually look like
 * LaTeX?  Guards against a stray literal "$" pairing with a later one and
 * swallowing plain prose into a fake math span.
 */
function looksLikeMath(body) {
  const trimmed = body.trim();
  if (!trimmed) return false;
  if (/[\\^_{}=<>]/.test(trimmed)) return true;        // \frac, x^2, a_{i}, =, <
  return /^[A-Za-z][A-Za-z0-9]{0,2}$/.test(trimmed);   // single symbol: $n$, $x1$
}

function renderMarkdown(text) {
  try {
    // Protect monetary amounts before interpreting $...$ as LaTeX. Without this,
    // text such as "$389 million ... $2,714 million" becomes one giant math span.
    const currencyBlocks = [];
    let currencySafe = text.replace(
      /\$\s?\d[\d,]*(?:\.\d+)?(?:\s*(?:[kmbt]|thousand|million|billion|trillion)\b)?(?=(?:\s|[,.!?;:)\]}]|[–-]\s*\$?\d|$))/gi,
      (match) => {
        const idx = currencyBlocks.length;
        currencyBlocks.push(match);
        return `OMNIRAGCURRENCYBLOCK${idx}END`;
      }
    );

    // ── Step 1: Lift all math blocks out before marked sees them ───────────
    // marked.parse() aggressively processes $ symbols and backslashes,
    // mangling LaTeX content between $...$ delimiters. We stash each math
    // expression as a unique placeholder, run marked on the safe remainder,
    // then splice the original LaTeX back in so KaTeX finds it untouched.
    const mathBlocks = [];
    const literalBlocks = [];

    function stash(match) {
      const idx = mathBlocks.length;
      mathBlocks.push(match);
      // Use plain alphanumeric tokens so Marked leaves them untouched.
      return `OMNIRAGMATHBLOCK${idx}END`;
    }

    // Non-math dollar spans are stashed too, but restored inside a .no-math
    // span so KaTeX's auto-render can't re-pair the $ signs at the DOM stage.
    function stashLiteral(match) {
      const idx = literalBlocks.length;
      literalBlocks.push(match);
      return `OMNIRAGLITERALBLOCK${idx}END`;
    }

    // Order matters: extract $$...$$ display math before $...$ inline math.
    // Inline math must stay on one line — otherwise a stray "$" pairs with a
    // later one and swallows entire sentences into a fake math span.
    let safe = currencySafe
      .replace(/\$\$([\s\S]*?)\$\$/g, stash)                                  // $$...$$ block
      .replace(/\$([^\$\n][^\$\n]*?)\$/g,
        (match, inner) => (looksLikeMath(inner) ? stash(match) : stashLiteral(match)))
      .replace(/\\\[([\s\S]*?)\\\]/g, stash)                                   // \[...\] block
      .replace(/\\\(([\s\S]*?)\\\)/g, stash);                                  // \(...\) inline

    // ── Step 2: Run marked on the math-free text ───────────────────────────
    let html = marked.parse(safe);

    // ── Step 3: Restore original math expressions ──────────────────────────
    // HTML-escape the raw LaTeX: "<" or "&" inside math (e.g. $x < y$) would
    // otherwise be parsed as markup on innerHTML insertion and silently eat
    // the rest of the answer. The browser decodes the entities back to plain
    // text, so KaTeX still sees the original characters.
    html = html.replace(/OMNIRAGMATHBLOCK(\d+)END/g, (_, i) => escapeHtml(mathBlocks[+i]));
    html = html.replace(
      /OMNIRAGLITERALBLOCK(\d+)END/g,
      (_, i) => `<span class="no-math">${escapeHtml(literalBlocks[+i])}</span>`
    );
    html = html.replace(
      /OMNIRAGCURRENCYBLOCK(\d+)END/g,
      (_, i) => `<span class="no-math currency-amount">${escapeHtml(currencyBlocks[+i])}</span>`
    );

    return html;
  } catch {
    return `<p>${escapeHtml(text)}</p>`;
  }
}

/**
 * Run KaTeX over an already-inserted DOM element to render any LaTeX math.
 * Supports both $...$ inline and $$...$$ display delimiters, as well as
 * \(...\) and \[...\] which LLMs sometimes emit for academic papers.
 */
function renderMath(el) {
  if (typeof renderMathInElement !== "function") return;
  try {
    renderMathInElement(el, {
      delimiters: [
        { left: "$$",  right: "$$",  display: true  },
        { left: "$",   right: "$",   display: false },
        { left: "\\[", right: "\\]", display: true  },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,   // never crash the page on bad LaTeX
      strict: false,
      ignoredClasses: ["no-math"],
    });
    // Style only the rendered equation, not the paragraph that may introduce it.
    el.querySelectorAll(".katex-display").forEach(display => {
      display.classList.add("formula-callout");
    });
  } catch (e) {
    console.warn("[KaTeX] render failed:", e);
  }
}

function decorateNumericValues(container) {
  container.querySelectorAll(".currency-amount").forEach(el => el.classList.add("numeric-value"));
  const numericPattern = /(?:[$€£]\s?\d[\d,]*(?:\.\d+)?(?:\s*(?:thousand|million|billion|trillion))?|\b\d[\d,]*(?:\.\d+)?%)/gi;
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const textNodes = [];

  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (node.parentElement?.closest(".katex, .no-math, .numeric-value, pre, code")) continue;
    if (numericPattern.test(node.textContent)) textNodes.push(node);
    numericPattern.lastIndex = 0;
  }

  textNodes.forEach(node => {
    const fragment = document.createDocumentFragment();
    let lastIndex = 0;
    node.textContent.replace(numericPattern, (match, offset) => {
      fragment.append(document.createTextNode(node.textContent.slice(lastIndex, offset)));
      const value = document.createElement("span");
      value.className = "numeric-value";
      value.textContent = match;
      fragment.append(value);
      lastIndex = offset + match.length;
      return match;
    });
    fragment.append(document.createTextNode(node.textContent.slice(lastIndex)));
    node.replaceWith(fragment);
  });
}

/* ── Init ─────────────────────────────────────────────────────────────────── */
async function initializeApp() {
  const next = window.location.pathname + window.location.search + window.location.hash;
  try {
    const res = await apiFetch("/auth/me");
    if (!res.ok) {
      // A 401 is already redirected by apiFetch. Any other failure must not
      // leave the unauthenticated application UI visible behind an error.
      window.location.replace(`/login?next=${encodeURIComponent(next)}`);
      return;
    }
    const profile = await readJsonResponse(res);
    adminBtn.hidden = !profile.is_admin;
  } catch {
    window.location.replace(`/login?next=${encodeURIComponent(next)}`);
    return;
  }
  loadApiStatus();
  updateToggles();
  renderFileList();
}

initializeApp();
