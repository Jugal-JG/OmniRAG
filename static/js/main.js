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

/* ── State ────────────────────────────────────────────────────────────────── */
let uploadedFiles = [];
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
    const res = await fetch("/api-status");
    const data = await res.json();
    const container = document.getElementById("api-status");
    const labels = { mistral: "Mistral", google: "Google", groq: "Groq" };
    container.innerHTML = Object.entries(labels).map(([k, label]) =>
      `<span class="api-pill ${data[k] ? 'ok' : 'err'}">${label}</span>`
    ).join("");
  } catch {}
}

/* ── File upload ──────────────────────────────────────────────────────────── */
function fileIcon(name) {
  const ext = name.split(".").pop().toLowerCase();
  const map = { pdf: "bi-file-pdf text-danger", txt: "bi-file-text text-secondary",
    png: "bi-file-image text-warning", jpg: "bi-file-image text-warning",
    jpeg: "bi-file-image text-warning", gif: "bi-file-image text-warning",
    html: "bi-filetype-html text-orange", md: "bi-markdown text-info" };
  return map[ext] || "bi-file-earmark text-secondary";
}

function renderFileList() {
  fileList.innerHTML = uploadedFiles.map(f =>
    `<div class="file-chip">
       <i class="bi ${fileIcon(f)} file-icon"></i>
       <span class="file-name" title="${f}">${f}</span>
       <button class="file-remove-btn" title="Remove file" onclick="removeFile('${escapeHtml(f)}')">
         <i class="bi bi-x"></i>
       </button>
     </div>`
  ).join("");
}

async function removeFile(fname) {
  try {
    const res = await fetch("/remove-file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: fname }),
    });
    const data = await res.json();
    if (data.success) {
      uploadedFiles = data.files;
      renderFileList();
      showToast(`Removed: ${fname}`, "info");
    }
  } catch (e) {
    showToast("Failed to remove file: " + e.message, "danger");
  }
}

async function uploadFiles(files) {
  const formData = new FormData();
  files.forEach(f => formData.append("files", f));

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (data.success) {
      uploadedFiles = data.files;
      renderFileList();
      if (data.errors?.length) showToast(`${data.errors[0]}`, "warning");
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
    await fetch("/clear-files", { method: "POST" });
    uploadedFiles = [];
    renderFileList();
    showToast("Files cleared", "info");
  } catch {}
});

newChatBtn.addEventListener("click", async () => {
  try {
    await fetch("/new-chat", { method: "POST" });
    // Remove all message elements but keep welcome splash hidden
    [...chatContainer.children].forEach(el => {
      if (el.id !== "welcomeMsg") el.remove();
    });
    if (welcomeMsg) welcomeMsg.style.display = "";
    showToast("New chat started — uploaded files kept", "info");
  } catch {}
});

/* ── Toggle state ─────────────────────────────────────────────────────────── */
function updateToggles() {
  const multi = multiDocToggle.checked;
  const think = thinkingToggle.checked;

  multiDocCard.classList.toggle("active", multi && !think);
  thinkingCard.classList.toggle("active-react", think);
  thinkingCard.classList.remove("active");

  // mutual exclusivity visual cue
  if (think) {
    multiDocToggle.disabled = true;
    multiDocCard.style.opacity = ".45";
  } else {
    multiDocToggle.disabled = false;
    multiDocCard.style.opacity = "";
  }

  // Active mode chips in input bar
  const chips = [];
  if (multi && !think)
    chips.push(`<span class="active-mode-chip bg-purple text-dark"><i class="bi bi-files"></i> Multi-Doc</span>`);
  if (think)
    chips.push(`<span class="active-mode-chip bg-danger text-white"><i class="bi bi-cpu"></i> Thinking (ReAct)</span>`);
  activeModes.innerHTML = chips.join("");
}

multiDocToggle.addEventListener("change", updateToggles);
thinkingToggle.addEventListener("change", updateToggles);

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
}

function exitLoadingMode() {
  isLoading = false;
  abortController = null;
  sendBtn.classList.remove("stop-mode");
  sendBtn.title = "Send (Enter)";
  sendIcon.className = "bi bi-send-fill";
  queryInput.disabled = false;
  queryInput.focus();
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

  const query = queryInput.value.trim();
  if (!query) return;

  abortController = new AbortController();
  enterLoadingMode();

  appendUserMessage(query);
  queryInput.value = "";
  queryInput.style.height = "auto";
  appendThinkingBubble();

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        multi_doc: multiDocToggle.checked,
        thinking: thinkingToggle.checked,
      }),
      signal: abortController.signal,
    });

    const data = await res.json();
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

function renderMarkdown(text) {
  try {
    return marked.parse(text);
  } catch {
    return `<p>${escapeHtml(text)}</p>`;
  }
}

/* ── Init ─────────────────────────────────────────────────────────────────── */
loadApiStatus();
updateToggles();
queryInput.focus();
