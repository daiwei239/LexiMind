const SESSION_KEY = "leximind_session_id";

const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");
const chatLog = document.getElementById("chat-log");
const welcome = document.getElementById("welcome");
const sourcesList = document.getElementById("sources-list");
const sourcesCount = document.getElementById("sources-count");
const newSessionBtn = document.getElementById("new-session-btn");
const messageTemplate = document.getElementById("message-template");
const sourceTemplate = document.getElementById("source-template");

let isSending = false;

function getSessionId() {
  let sessionId = localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, sessionId);
  }
  return sessionId;
}

function resetSession() {
  localStorage.removeItem(SESSION_KEY);
  chatLog.innerHTML = "";
  welcome.hidden = false;
  renderSources([]);
}

function formatTime(date = new Date()) {
  return date.toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function appendMessage(role, text, options = {}) {
  welcome.hidden = true;

  const node = messageTemplate.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  if (options.loading) node.classList.add("loading");
  if (options.error) node.classList.add("error");
  if (options.streaming) node.classList.add("streaming");

  node.querySelector(".role").textContent = role === "user" ? "你" : "LexiMind";
  node.querySelector("time").textContent = formatTime();
  node.querySelector(".message-body").textContent = text;

  chatLog.appendChild(node);
  chatLog.scrollTop = chatLog.scrollHeight;
  return node;
}

function renderSources(sources) {
  sourcesList.innerHTML = "";
  sourcesCount.textContent = String(sources.length);

  if (!sources.length) {
    const hint = document.createElement("p");
    hint.className = "empty-hint";
    hint.textContent = "发送问题后，相关法条会显示在这里。";
    sourcesList.appendChild(hint);
    return;
  }

  for (const source of sources) {
    const card = sourceTemplate.content.firstElementChild.cloneNode(true);
    card.querySelector(".source-title").textContent =
      source.title || source.parent_article_id || "相关法条";
    card.querySelector(".source-content").textContent =
      source.content || source.full_text || "暂无摘要";
    sourcesList.appendChild(card);
  }
}

function setSendingState(active) {
  isSending = active;
  sendBtn.disabled = active;
  questionInput.disabled = active;
  sendBtn.querySelector(".send-label").textContent = active ? "生成中…" : "发送";
}

function parseSseBlock(block) {
  const lines = block.split("\n");
  let event = "message";
  let data = "";

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      data = line.slice(5).trim();
    }
  }

  if (!data) {
    return null;
  }

  return { event, data: JSON.parse(data) };
}

async function consumeChatStream(question, pendingNode) {
  const response = await fetch("/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: getSessionId(),
      question,
    }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = payload.detail || payload.message || `请求失败（${response.status}）`;
    throw new Error(detail);
  }

  if (!response.body) {
    throw new Error("浏览器不支持流式响应。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answerText = "";
  let started = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";

    for (const part of parts) {
      const parsed = parseSseBlock(part.trim());
      if (!parsed) {
        continue;
      }

      const { event, data } = parsed;

      if (event === "meta") {
        pendingNode.classList.remove("loading");
        pendingNode.classList.add("streaming");
        renderSources(data.sources || []);
        if (!started) {
          pendingNode.querySelector(".message-body").textContent = "";
          started = true;
        }
      } else if (event === "token") {
        if (!started) {
          pendingNode.classList.remove("loading");
          pendingNode.classList.add("streaming");
          pendingNode.querySelector(".message-body").textContent = "";
          started = true;
        }
        answerText += data.text || "";
        pendingNode.querySelector(".message-body").textContent = answerText;
        chatLog.scrollTop = chatLog.scrollHeight;
      } else if (event === "done") {
        answerText = data.answer || answerText;
        pendingNode.classList.remove("streaming");
        pendingNode.querySelector(".message-body").textContent = answerText;
      } else if (event === "error") {
        throw new Error(data.message || "流式生成失败");
      }
    }
  }

  pendingNode.classList.remove("streaming");
  if (!answerText) {
    pendingNode.querySelector(".message-body").textContent = "未生成回答，请稍后重试。";
  }
}

async function sendQuestion(question) {
  const trimmed = question.trim();
  if (!trimmed || isSending) return;

  appendMessage("user", trimmed);
  questionInput.value = "";
  questionInput.style.height = "auto";

  const pending = appendMessage("assistant", "正在检索法条并生成回答…", { loading: true });
  setSendingState(true);

  try {
    await consumeChatStream(trimmed, pending);
  } catch (error) {
    pending.classList.remove("loading", "streaming");
    pending.classList.add("error");
    pending.querySelector(".message-body").textContent =
      error.message || "无法连接后端服务，请确认 uvicorn 已启动。";
  } finally {
    setSendingState(false);
    questionInput.focus();
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion(questionInput.value);
});

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 160)}px`;
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

newSessionBtn.addEventListener("click", resetSession);

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    const prompt = button.getAttribute("data-prompt");
    if (prompt) sendQuestion(prompt);
  });
});

getSessionId();
