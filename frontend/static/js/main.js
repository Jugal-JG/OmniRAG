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

function getSessionId() {
  let sid = localStorage.getItem(SESSION_STORAGE_KEY);
  if (!sid) {
    sid = crypto.randomUUID();
    localStorage.setItem(SESSION_STORAGE_KEY, sid);
  }
  return sid;
}

function apiUrl(path) {
  return `${API_BASE_URL}${path}`;
}

function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-Omnirag-Session-Id", getSessionId());

  // No `credentials: "include"`: the session is tracked entirely via the
  // X-Omnirag-Session-Id header above, so cross-site cookies aren't needed.
  // Avoiding credentialed mode sidesteps fragile third-party-cookie handling
  // and the stricter preflight rules that go with it.
  return fetch(apiUrl(path), {
    ...options,
    headers,
  });
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
    const data = await res.json();
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
    const data = await res.json();
    if (data.success) {
      syncUploadedFiles(data.files);
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
    const res = await apiFetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (data.success) {
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

    function stash(match) {
      const idx = mathBlocks.length;
      mathBlocks.push(match);
      // Use plain alphanumeric tokens so Marked leaves them untouched.
      return `OMNIRAGMATHBLOCK${idx}END`;
    }

    // Order matters: extract $$...$$ display math before $...$ inline math.
    let safe = currencySafe
      .replace(/\$\$([\s\S]*?)\$\$/g, stash)           // $$...$$ block
      .replace(/\$([^\$\n][^\$]*?)\$/g, stash)          // $...$ inline (no newlines)
      .replace(/\\\[([\s\S]*?)\\\]/g, stash)             // \[...\] block
      .replace(/\\\(([\s\S]*?)\\\)/g, stash);            // \(...\) inline

    // ── Step 2: Run marked on the math-free text ───────────────────────────
    let html = marked.parse(safe);

    // ── Step 3: Restore original math expressions ──────────────────────────
    html = html.replace(/OMNIRAGMATHBLOCK(\d+)END/g, (_, i) => mathBlocks[+i]);
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
loadApiStatus();
updateToggles();
renderFileList();
