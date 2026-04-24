import { createButton } from "../../components/button.js";

function getApiBase() {
  // Prefer runtime-configured API base for deployment
  const fromWindow = globalThis?.BOOKRECOMMEND_API_BASE;
  if (typeof fromWindow === "string" && fromWindow.trim()) return fromWindow.trim().replace(/\/+$/, "");

  const meta = document.querySelector('meta[name="bookrecommend-api-base"]');
  const fromMeta = meta?.getAttribute("content");
  if (typeof fromMeta === "string" && fromMeta.trim()) return fromMeta.trim().replace(/\/+$/, "");

  // Local dev default
  return "http://127.0.0.1:8001";
}

/**
 * @param {string} prompt
 * @param {any} intentV2
 * @param {{ excludeIds?: string[], excludeSignatures?: string[] } | undefined} opts
 */
async function fetchRecommend(prompt, intentV2, opts) {
  const API_BASE = getApiBase();
  const query = String(prompt || "");
  const custom_text = intentV2?.custom_text ?? null;
  const exclude_book_ids = Array.isArray(opts?.excludeIds) ? opts.excludeIds : [];
  const exclude_signatures = Array.isArray(opts?.excludeSignatures) ? opts.excludeSignatures : [];
  const res = await fetch(`${API_BASE}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      top_k: 3,
      intent_v2: intentV2 ?? null,
      custom_text,
      // 预留给后端：排除当前已展示书籍后再推荐
      exclude_book_ids,
      exclude_signatures,
    }),
  });

  if (!res.ok) {
    let detail = null;
    try {
      const errJson = await res.json();
      if (errJson && typeof errJson.detail === "string") detail = errJson.detail;
    } catch {
      // ignore parse errors
    }
    const err = new Error(`HTTP ${res.status}`);
    // @ts-ignore
    err.status = res.status;
    // @ts-ignore
    err.detail = detail;
    throw err;
  }

  const json = await res.json();
  if (!json || json.success !== true || !Array.isArray(json.books)) {
    throw new Error("bad payload");
  }
  return json;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * @param {object} params
 * @param {any} params.intentV2
 * @param {string | null} params.finalPrompt
 * @param {() => void} params.onBackToScene
 */
export function renderRecommendView({ intentV2, finalPrompt, onBackToScene } = {}) {
  const CACHE_KEY = "bookrecommend:last_recommend_v1";
  const CACHE_TTL_MS = 24 * 60 * 60 * 1000;
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;

  const root = document.createElement("section");
  root.className = "flow-step flow-step--recommend";

  const card = document.createElement("section");
  card.className = "prompt-preview fade-enter";

  const actions = document.createElement("div");
  actions.className = "prompt-preview__actions";

  const back = createButton({
    label: "返回修改",
    variant: "ghost",
    onClick: () => onBackToScene?.(),
  });
  back.className = "flow-back flow-back--bottom";

  const statusBox = el("section", "recommend-status");
  const statusText = el("p", "recommend-status__text");
  const progress = el("div", "recommend-progress");
  progress.setAttribute("role", "progressbar");
  progress.setAttribute("aria-label", "正在加载");
  progress.hidden = true;
  statusBox.append(statusText, progress);

  // 成功态专用状态栏（分割线与卡片之间）
  const successStatusBox = el("section", "recommend-status");
  successStatusBox.classList.add("fade-enter");
  const successStatusText = el("p", "recommend-status__text");
  const successProgress = el("div", "recommend-progress");
  successProgress.setAttribute("role", "progressbar");
  successProgress.setAttribute("aria-label", "正在加载");
  successProgress.hidden = true;
  successStatusBox.append(successStatusText, successProgress);
  successStatusBox.hidden = true;

  const retryBtn = createButton({
    label: "重试",
    variant: "ghost",
    onClick: () => run({ mode: "initial" }),
  });
  retryBtn.className = "recommend-status__retry";

  const results = el("section", "recommend-results");

  const refreshBtn = createButton({
    label: "⟳",
    variant: "ghost",
    onClick: () => run({ mode: "refresh" }),
  });
  refreshBtn.className = "flow-back flow-refresh";
  refreshBtn.setAttribute("aria-label", "刷新");
  refreshBtn.setAttribute("title", "刷新");
  refreshBtn.hidden = true;

  actions.append(back);
  card.append(statusBox, successStatusBox, results, refreshBtn, actions);

  let running = false;
  let restoredFromCache = false;
  /** @type {any[] | null} */
  let lastBooks = null;
  /** @type {Set<string>} */
  const excludeIdSet = new Set();

  function normalizeBook(raw) {
    const enriched = raw?.book_enriched && typeof raw.book_enriched === "object" ? raw.book_enriched : {};
    const subject_id = String(raw?.subject_id || raw?.book_id || enriched?.subject_id || raw?.id || "").trim() || null;
    const title = String(raw?.title || enriched?.title || "未命名书籍");
    const author = String(raw?.author || enriched?.author || "未知作者");
    const style_tags = Array.isArray(raw?.style_tags)
      ? raw.style_tags
      : Array.isArray(enriched?.style_tags)
        ? enriched.style_tags
        : [];
    const details_url = String(raw?.details_url || enriched?.details_url || "");
    const content_intro = String(raw?.content_intro || enriched?.content_intro || "");
    return { raw, enriched, subject_id, title, author, style_tags, details_url, content_intro };
  }

  function getBookIdForExclude(book) {
    return normalizeBook(book).subject_id;
  }

  function getBookSignatureForExclude(book) {
    const b = normalizeBook(book);
    const sig = `${String(b.title || "").trim()}|${String(b.author || "").trim()}`.trim();
    return sig === "|" ? null : sig;
  }

  function setSuccessStatusIdle() {
    successStatusBox.hidden = false;
    successStatusText.textContent = "可以试试这些书";
    successProgress.hidden = true;
    successStatusBox.classList.add("fade-enter--active");
  }

  function setSuccessStatusRefreshing() {
    successStatusBox.hidden = false;
    successStatusText.textContent = "正在获取推荐结果";
    successProgress.hidden = false;
    successStatusBox.classList.add("fade-enter--active");
  }

  function setSuccessStatusExhausted(detail) {
    successStatusBox.hidden = false;
    successStatusText.textContent = detail;
    successProgress.hidden = true;
    successStatusBox.classList.add("fade-enter--active");
  }

  function setLoading({ preserveResults = false } = {}) {
    refreshBtn.disabled = true;
    retryBtn.remove();
    refreshBtn.hidden = false;

    if (preserveResults) {
      // 刷新 loading：保持旧卡片，状态栏只在成功态区域展示
      statusBox.hidden = true;
      setSuccessStatusRefreshing();
      return;
    }

    // 初次 loading：使用原 statusBox（失败态/初次加载统一）
    successStatusBox.hidden = true;
    statusBox.hidden = false;
    statusBox.className = "recommend-status recommend-status--loading";
    statusText.textContent = "正在获取推荐结果";
    statusText.hidden = false;
    progress.hidden = false;
    results.replaceChildren();
  }

  function setError() {
    successStatusBox.hidden = true;
    statusBox.hidden = false;
    statusBox.className = "recommend-status recommend-status--error";
    statusText.textContent = "无法获取推荐，请点击返回修改";
    statusText.hidden = false;
    progress.hidden = true;
    refreshBtn.disabled = false;
    retryBtn.remove();
    results.replaceChildren();
    refreshBtn.hidden = true;
  }

  function renderTags(tags) {
    const wrap = el("div", "recommend-book__tags");
    for (const t of tags || []) {
      wrap.append(el("span", "recommend-tag", t));
    }
    return wrap;
  }

  function showBookModal(book) {
    const b = normalizeBook(book);
    const enriched = b.enriched || {};
    const title = b.title;
    const author = b.author;
    const tags = b.style_tags || [];
    const detailsUrl = b.details_url || "";

    const authorEra = String(enriched?.author_era_nationality || "").trim();
    const translator = String(enriched?.translator || "").trim();
    const publisher = String(enriched?.publisher || "").trim();
    const pages = enriched?.pages != null ? String(enriched.pages).trim() : "";
    const intro = String(b.content_intro || "").trim();

    const overlay = document.createElement("div");
    overlay.className = "ui-modal__overlay";
    overlay.style.setProperty("--accent", "#4aa3ff");
    overlay.classList.add("fade-enter");

    const dialog = document.createElement("div");
    dialog.className = "ui-modal ui-modal--book";
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");

    const h = document.createElement("div");
    h.className = "ui-modal__title";
    h.textContent = title;

    const content = document.createElement("div");
    content.className = "recommend-modal__content";

    const right = document.createElement("div");
    right.className = "recommend-modal__right";

    const authorLine = document.createElement("div");
    authorLine.className = "recommend-modal__author";
    authorLine.textContent = authorEra ? `${author}（${authorEra}）` : author;

    const tagLine = renderTags(tags);
    tagLine.classList.add("recommend-book__tags--modal");

    const meta = document.createElement("div");
    meta.className = "recommend-modal__meta";
    const rows = [];
    if (translator) rows.push(["译者", translator]);
    if (publisher) rows.push(["出版社", publisher]);
    if (pages) rows.push(["页数", pages]);
    for (const [k, v] of rows) {
      const row = document.createElement("div");
      row.className = "recommend-modal__meta-row";
      const kk = document.createElement("div");
      kk.className = "recommend-modal__meta-k";
      kk.textContent = k;
      const vv = document.createElement("div");
      vv.className = "recommend-modal__meta-v";
      vv.textContent = v;
      row.append(kk, vv);
      meta.append(row);
    }

    const introWrap = document.createElement("div");
    introWrap.className = "recommend-modal__intro";
    if (intro) {
      const t = document.createElement("div");
      t.className = "recommend-modal__section-title";
      t.textContent = "内容简介";
      const p = document.createElement("div");
      p.className = "recommend-modal__intro-text";
      p.textContent = intro;
      introWrap.append(t, p);
    }

    right.append(authorLine, tagLine);
    if (meta.childElementCount) right.append(meta);
    if (introWrap.childElementCount) right.append(introWrap);

    content.append(right);

    const actions = document.createElement("div");
    actions.className = "ui-modal__actions";

    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "flow-back";
    closeBtn.textContent = "关闭";

    const doubanBtn = document.createElement("button");
    doubanBtn.type = "button";
    doubanBtn.className = "flow-back ui-modal__ok";
    doubanBtn.textContent = "前往豆瓣";
    doubanBtn.disabled = !detailsUrl;
    doubanBtn.addEventListener("click", () => {
      if (!detailsUrl) return;
      const w = window.open(detailsUrl, "_blank", "noopener,noreferrer");
      if (w) w.opener = null;
    });

    const disclaimer = document.createElement("div");
    disclaimer.className = "recommend-modal__disclaimer";
    disclaimer.textContent = "该链接仅用于笔试";
    disclaimer.hidden = !detailsUrl;

    const close = () => {
      if (reduceMotion) {
        overlay.remove();
        document.removeEventListener("keydown", onKeyDown, true);
        return;
      }
      overlay.classList.remove("fade-enter", "fade-enter--active");
      overlay.classList.add("fade-exit");
      requestAnimationFrame(() => overlay.classList.add("fade-exit--active"));
      overlay.addEventListener(
        "transitionend",
        () => {
          overlay.remove();
          document.removeEventListener("keydown", onKeyDown, true);
        },
        { once: true },
      );
    };
    const onKeyDown = (e) => {
      if (e.key === "Escape") close();
    };

    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) close();
    });
    closeBtn.addEventListener("click", close);
    document.addEventListener("keydown", onKeyDown, true);

    actions.append(closeBtn, doubanBtn);
    dialog.append(h, content, actions, disclaimer);
    overlay.append(dialog);
    document.body.append(overlay);
    requestAnimationFrame(() => overlay.classList.add("fade-enter--active"));
    requestAnimationFrame(() => (detailsUrl ? doubanBtn : closeBtn).focus());
  }

  function renderBook(b) {
    const bb = normalizeBook(b);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "recommend-book";
    btn.setAttribute("aria-label", `查看《${bb.title || "未命名书籍"}》详情`);
    btn.addEventListener("click", () => showBookModal(b));

    const body = el("div", "recommend-book__body");
    const h = el("div", "recommend-book__title", bb.title || "未命名书籍");
    const by = el("div", "recommend-book__author", bb.author || "未知作者");
    const tags = bb.style_tags || [];
    body.append(h, by, renderTags(tags));

    btn.append(body);
    return btn;
  }

  function setSuccess(payload, { skipCacheWrite = false } = {}) {
    successStatusBox.hidden = false;
    setSuccessStatusIdle();
    statusBox.hidden = true;
    refreshBtn.disabled = false;
    retryBtn.remove();
    refreshBtn.hidden = false;
    lastBooks = Array.isArray(payload?.books) ? payload.books : null;
    // 连续刷新累计排除：把本次推荐的 subject_id 加入集合
    for (const b of lastBooks || []) {
      const id = getBookIdForExclude(b);
      if (id) excludeIdSet.add(id);
    }
    results.replaceChildren(...payload.books.map(renderBook));

    if (!skipCacheWrite) {
      try {
        localStorage.setItem(
          CACHE_KEY,
          JSON.stringify({
            savedAt: Date.now(),
            payload,
            intentV2Snapshot: intentV2 ?? null,
            finalPromptSnapshot: finalPrompt ?? null,
          }),
        );
      } catch {
        // ignore storage failures
      }
    }
  }

  function tryRestoreCache() {
    try {
      const raw = localStorage.getItem(CACHE_KEY);
      if (!raw) return false;
      const parsed = JSON.parse(raw);
      const savedAt = Number(parsed?.savedAt);
      const payload = parsed?.payload;
      const cachedPrompt = typeof parsed?.finalPromptSnapshot === "string" ? parsed.finalPromptSnapshot : null;
      const currentPrompt = String(finalPrompt || "").trim();

      if (!Number.isFinite(savedAt) || Date.now() - savedAt > CACHE_TTL_MS) {
        localStorage.removeItem(CACHE_KEY);
        return false;
      }
      if (!payload || !Array.isArray(payload.books)) {
        localStorage.removeItem(CACHE_KEY);
        return false;
      }

      // 仅用于“刷新恢复”或“同一提示词回看”
      if (currentPrompt) {
        if (!cachedPrompt) {
          return false;
        }
        if (currentPrompt !== cachedPrompt) {
          return false;
        }
      }

      setSuccess(payload, { skipCacheWrite: true });
      restoredFromCache = true;
      return true;
    } catch {
      try {
        localStorage.removeItem(CACHE_KEY);
      } catch {
        // ignore
      }
      return false;
    }
  }

  async function run({ mode = "initial" } = {}) {
    if (running) return;
    const prompt = (finalPrompt || "").trim();
    if (!prompt) {
      setError();
      return;
    }

    running = true;
    setLoading({ preserveResults: mode === "refresh" });
    try {
      /** @type {string[]} */
      const excludeIds = [];
      /** @type {string[]} */
      const excludeSignatures = [];
      if (mode === "refresh") {
        // 持续排除：把历史所有已推荐过的 subject_id 全部带上
        excludeIds.push(...Array.from(excludeIdSet));
        // 兜底：仍把当前三本的 signature 带上（后端可忽略）
        for (const b of lastBooks || []) {
          const sig = getBookSignatureForExclude(b);
          if (sig) excludeSignatures.push(sig);
        }
      }

      const payload = await fetchRecommend(prompt, intentV2, { excludeIds, excludeSignatures });
      if (mode === "refresh" && !reduceMotion && results.childElementCount) {
        results.classList.remove("fade-enter", "fade-enter--active");
        results.classList.add("fade-exit");
        requestAnimationFrame(() => results.classList.add("fade-exit--active"));
        results.addEventListener(
          "transitionend",
          () => {
            results.classList.remove("fade-exit", "fade-exit--active");
            setSuccess(payload);
            results.classList.add("fade-enter");
            requestAnimationFrame(() => results.classList.add("fade-enter--active"));
          },
          { once: true },
        );
      } else {
        setSuccess(payload);
      }
    } catch (e) {
      const status = /** @type {any} */ (e)?.status;
      const detail = /** @type {any} */ (e)?.detail;
      if (mode === "refresh" && status === 409) {
        refreshBtn.disabled = true;
        statusBox.hidden = true;
        setSuccessStatusExhausted(
          typeof detail === "string" && detail.trim() ? detail : "抱歉，由于书籍库限制，无法再推荐新书",
        );
        return;
      }
      setError();
    } finally {
      running = false;
    }
  }

  requestAnimationFrame(() => {
    void card.offsetWidth;
    requestAnimationFrame(() => {
      card.classList.add("fade-enter--active");
      if (!tryRestoreCache()) run({ mode: "initial" });
    });
  });

  root.append(card);
  return root;
}

