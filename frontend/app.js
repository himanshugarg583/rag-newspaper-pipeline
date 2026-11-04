/* Newspaper AI frontend — talks to the FastAPI backend at /api/v1 */
"use strict";

const API = "/api/v1";

// ---------- DOM refs ----------
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");
const docList = document.getElementById("docList");
const refreshDocsBtn = document.getElementById("refreshDocs");
const healthDot = document.getElementById("healthDot");
const healthText = document.getElementById("healthText");
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");

// ---------- State ----------
let sessionId = null;
let pollTimer = null;

// ---------- Helpers ----------
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/* Render clean answer without any [Source ...] markers. */
function renderAnswer(text) {
  const cleaned = text.replace(/\[(?:Source|Sources)[^\]]*\]/gi, "").trim();
  return escapeHtml(cleaned);
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

function toast(message, ok) {
  const node = el("div", "toast " + (ok ? "ok" : "err"), message);
  uploadStatus.prepend(node);
  setTimeout(() => node.remove(), 6000);
}

async function apiError(response) {
  try {
    const body = await response.json();
    return body?.error?.message || body?.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}

// ---------- Health ----------
async function checkHealth() {
  try {
    const res = await fetch(`${API}/health`);
    const body = await res.json();
    const ok = body.status === "healthy";
    healthDot.className = "status-dot " + (ok ? "ok" : "bad");
    healthText.textContent = ok
      ? "Backend connected"
      : `Degraded (db: ${body.database}, redis: ${body.redis})`;
  } catch {
    healthDot.className = "status-dot bad";
    healthText.textContent = "Backend unreachable";
  }
}

// ---------- Documents ----------
async function loadDocuments() {
  refreshDocsBtn.classList.add("spinning");
  try {
    const res = await fetch(`${API}/documents?page=1&page_size=50`);
    if (!res.ok) throw new Error(await apiError(res));
    const body = await res.json();
    renderDocuments(body.items);

    // Keep polling while anything is still processing.
    const processing = body.items.some((d) => d.status === "PROCESSING");
    clearTimeout(pollTimer);
    if (processing) pollTimer = setTimeout(loadDocuments, 4000);
  } catch (err) {
    console.error("Failed to load documents:", err);
  } finally {
    setTimeout(() => refreshDocsBtn.classList.remove("spinning"), 400);
  }
}

function renderDocuments(items) {
  docList.innerHTML = "";
  if (!items.length) {
    docList.appendChild(el("li", "doc-empty", "No newspapers uploaded yet"));
    return;
  }
  for (const doc of items) {
    const li = el("li", "doc-item");
    li.appendChild(el("span", "doc-icon", "📄"));

    const info = el("div", "doc-info");
    info.appendChild(el("div", "doc-name", doc.filename));
    const metaParts = [formatBytes(doc.file_size)];
    if (doc.total_pages) metaParts.push(doc.total_pages + " pages");
    if (doc.total_chunks) metaParts.push(doc.total_chunks + " chunks");
    info.appendChild(el("div", "doc-meta", metaParts.join(" · ")));
    if (doc.status === "FAILED" && doc.error_message) {
      const errLine = el("div", "doc-meta", "⚠ " + doc.error_message);
      errLine.style.color = "var(--red)";
      info.appendChild(errLine);
    }
    li.appendChild(info);

    li.appendChild(
      el("span", "badge " + doc.status.toLowerCase(), doc.status)
    );

    if (doc.language && doc.language !== "unknown") {
      li.appendChild(
        el("span", "badge lang-" + doc.language.toLowerCase(), doc.language.toUpperCase())
      );
    }

    const delBtn = el("button", "doc-del", "✕");
    delBtn.title = "Delete newspaper";
    delBtn.addEventListener("click", () => deleteDocument(doc.id, doc.filename));
    li.appendChild(delBtn);

    docList.appendChild(li);
  }
}

async function deleteDocument(id, filename) {
  if (!confirm(`Delete "${filename}" and all its data?`)) return;
  try {
    const res = await fetch(`${API}/documents/${id}`, { method: "DELETE" });
    if (!res.ok) throw new Error(await apiError(res));
    toast(`Deleted ${filename}`, true);
    loadDocuments();
  } catch (err) {
    toast(`Delete failed: ${err.message}`, false);
  }
}

// ---------- Upload ----------
async function uploadFiles(files) {
  for (const file of files) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      toast(`${file.name}: only PDF files are accepted`, false);
      continue;
    }
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API}/documents/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(await apiError(res));
      toast(`${file.name} uploaded — processing…`, true);
    } catch (err) {
      toast(`${file.name}: ${err.message}`, false);
    }
  }
  loadDocuments();
}

uploadZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) uploadFiles([...fileInput.files]);
  fileInput.value = "";
});

["dragenter", "dragover"].forEach((evt) =>
  uploadZone.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  uploadZone.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
  })
);
uploadZone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) uploadFiles([...e.dataTransfer.files]);
});

// ---------- Chat ----------
function clearWelcome() {
  const welcome = chatMessages.querySelector(".welcome");
  if (welcome) welcome.remove();
}

function addMessage(role, html) {
  clearWelcome();
  const msg = el("div", "msg " + role);
  const avatar = el("div", "avatar", role === "user" ? "🧑" : "🤖");
  const bubble = el("div", "bubble");
  bubble.innerHTML = html;
  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return bubble;
}

function addTyping() {
  const bubble = addMessage(
    "assistant",
    '<div class="typing"><span></span><span></span><span></span></div>'
  );
  return bubble.closest(".msg");
}

async function askQuestion(question) {
  addMessage("user", escapeHtml(question));
  const typingMsg = addTyping();
  sendBtn.disabled = true;
  questionInput.disabled = true;

  try {
    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question }),
    });
    if (!res.ok) throw new Error(await apiError(res));
    const body = await res.json();
    sessionId = body.session_id;

    typingMsg.remove();
    addMessage("assistant", renderAnswer(body.answer));
  } catch (err) {
    typingMsg.remove();
    const bubble = addMessage("assistant", escapeHtml("⚠ " + err.message));
    bubble.closest(".msg").classList.add("error");
  } finally {
    sendBtn.disabled = false;
    questionInput.disabled = false;
    questionInput.focus();
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  questionInput.style.height = "auto";
  askQuestion(question);
});

// Enter sends, Shift+Enter adds a newline.
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

// Auto-grow textarea.
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + "px";
});

// Suggestion chips.
chatMessages.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (chip) askQuestion(chip.dataset.q);
});

// New chat resets the session (documents stay).
newChatBtn.addEventListener("click", () => {
  sessionId = null;
  chatMessages.innerHTML = `
    <div class="welcome">
      <div class="welcome-icon">🗞️</div>
      <h3>New conversation started</h3>
      <p>Ask anything about your uploaded newspapers. Previous chat context has been cleared.</p>
    </div>`;
});

refreshDocsBtn.addEventListener("click", loadDocuments);

// ---------- Init ----------
checkHealth();
setInterval(checkHealth, 30000);
loadDocuments();
questionInput.focus();
