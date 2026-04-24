import { createButton } from "../../components/button.js";
import { generatePromptV2 } from "../rules/prompt-engine-v2.js";

function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function pickBatch({ pool, seen, count }) {
  const available = pool.filter((t) => !seen.has(t));
  if (available.length === 0) return { batch: [], exhausted: true };
  const batch = shuffle(available).slice(0, count);
  for (const t of batch) seen.add(t);
  const exhausted = pool.every((t) => seen.has(t));
  return { batch, exhausted };
}

/**
 * @param {object} params
 * @param {{ mood_tags: string[], scene_tags: string[], style_tags: string[], difficulty: number, pace: number, length_type: string }} params.intentV2
 * @param {(partialIntent: any) => void} params.onBack
 * @param {(payload: { prompt: string, dirty: boolean, intentV2: any }) => void} params.onSendPrompt
 */
export function renderSceneView({ intentV2, onBack, onSendPrompt } = {}) {
  const root = document.createElement("section");
  root.className = "flow-step flow-step--scene";

  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;

  const animateBatchRefresh = ({ gridEl, toolsEl, doRender }) => {
    if (reduceMotion) {
      doRender();
      return;
    }

    const staticBtn = toolsEl.querySelector(".flow-refresh");
    if (staticBtn && "disabled" in staticBtn) staticBtn.disabled = true;

    gridEl.classList.add("scene-button-grid--transitioning");
    const toolTargets = Array.from(toolsEl.children).filter((n) => !n.classList?.contains("flow-refresh"));
    for (const el of [gridEl, ...toolTargets]) {
      el.classList.remove("fade-enter", "fade-enter--active");
      el.classList.add("fade-exit");
    }

    requestAnimationFrame(() => {
      for (const el of [gridEl, ...toolTargets]) el.classList.add("fade-exit--active");
    });

    const onEnd = (e) => {
      if (e.target !== gridEl) return;
      gridEl.removeEventListener("transitionend", onEnd);

      gridEl.classList.remove("fade-exit", "fade-exit--active", "scene-button-grid--transitioning");
      for (const el of toolTargets) el.classList.remove("fade-exit", "fade-exit--active");

      doRender();

      const newToolTargets = Array.from(toolsEl.children).filter((n) => !n.classList?.contains("flow-refresh"));
      for (const el of [gridEl, ...newToolTargets]) {
        el.classList.add("fade-enter");
        el.classList.remove("fade-enter--active");
      }

      requestAnimationFrame(() => {
        void gridEl.offsetWidth;
        requestAnimationFrame(() => {
          for (const el of [gridEl, ...newToolTargets]) el.classList.add("fade-enter--active");
          const btn = toolsEl.querySelector(".flow-refresh");
          if (btn && "disabled" in btn) btn.disabled = false;
        });
      });
    };

    gridEl.addEventListener("transitionend", onEnd);
  };

  const styleHeading = document.createElement("h2");
  styleHeading.className = "flow-step__title";
  styleHeading.textContent = "你想要什么样的内容与风格";
  styleHeading.classList.add("fade-enter");

  const styleGrid = document.createElement("section");
  styleGrid.className = "scene-button-grid scene-button-grid--two";
  styleGrid.classList.add("fade-enter");

  const STYLE = ["文学", "小说", "社会观察", "悬疑", "科幻", "散文", "历史", "爱情", "人物传记", "哲学思考"];
  const selectedStyle = new Set(Array.isArray(intentV2?.style_tags) ? intentV2.style_tags : []);
  const seenStyle = new Set();
  let exhaustedStyle = false;
  /** @type {string[]} */
  let lastStyleHintTags = [];
  let showCustom = false;
  /** @type {string | null} */
  let customStyleText = (intentV2?.custom_style_text || "").trim() || null;

  const state = {
    mood_tags: Array.isArray(intentV2?.mood_tags) ? intentV2.mood_tags.slice() : [],
    scene_tags: Array.isArray(intentV2?.scene_tags) ? intentV2.scene_tags.slice() : [],
    style_tags: Array.isArray(intentV2?.style_tags) ? intentV2.style_tags.slice() : [],
    difficulty: Number(intentV2?.difficulty ?? 3),
    pace: Number(intentV2?.pace ?? 3),
    length_type: intentV2?.length_type || "中篇",
    custom_text: (intentV2?.custom_text || "").trim() || null,
    custom_scene_text: (intentV2?.custom_scene_text || "").trim() || null,
    custom_mood_text: (intentV2?.custom_mood_text || "").trim() || null,
    custom_style_text: (intentV2?.custom_style_text || "").trim() || null,
  };

  let difficultyTouched = false;
  let paceTouched = false;

  function paceLabel(pace) {
    const v = Math.max(1, Math.min(5, Number(pace || 3)));
    if (v <= 2) return "舒缓";
    if (v === 3) return "适中";
    return "紧凑";
  }

  function lengthDifficultyRange(lengthType) {
    const lt = String(lengthType || "");
    if (lt === "短篇") return { min: 2, max: 3 };
    if (lt === "中篇") return { min: 3, max: 3 };
    if (lt === "长篇") return { min: 3, max: 4 };
    if (lt === "超长篇") return { min: 4, max: 5 };
    return { min: 3, max: 3 };
  }

  function stylePaceRange(styleTags) {
    const set = new Set((styleTags || []).filter(Boolean));
    const fast = ["悬疑", "科幻", "小说"];
    const slow = ["散文", "哲学思考"];
    if (fast.some((t) => set.has(t))) return { min: 4, max: 5, reason: "与你选择的风格更匹配：紧凑推进" };
    if (slow.some((t) => set.has(t))) return { min: 1, max: 2, reason: "与你选择的风格更匹配：舒缓慢读" };
    return { min: 3, max: 3, reason: "默认节奏：适中" };
  }

  function midInt(min, max) {
    return Math.round((Number(min) + Number(max)) / 2);
  }

  function setSliderValue({ inputEl, valueEl, value, onChange, markTouched }) {
    inputEl.value = String(value);
    if (valueEl) valueEl.textContent = String(value);
    onChange?.(Number(value));
    if (markTouched) markTouched();
  }

  function renderSelectedHints({ tags, accent, onRemove }) {
    if (!tags?.length) return null;
    const wrap = document.createElement("div");
    wrap.className = "selected-hints";
    wrap.style.setProperty("--accent", accent);
    for (const t of tags) {
      const chip = document.createElement("span");
      chip.className = "selected-chip";
      chip.style.setProperty("--accent", accent);

      const label = document.createElement("span");
      label.className = "selected-chip__label";
      label.textContent = t;

      const close = document.createElement("button");
      close.type = "button";
      close.className = "selected-chip__close";
      close.textContent = "×";
      close.setAttribute("aria-label", `删除 ${t}`);
      close.setAttribute("title", "删除");
      close.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        onRemove?.(t, chip);
      });

      chip.append(label, close);
      wrap.append(chip);
    }
    return wrap;
  }

  const showConfirmModal = ({ title, detail, accent }) =>
    new Promise((resolve) => {
      const overlay = document.createElement("div");
      overlay.className = "ui-modal__overlay";
      overlay.style.setProperty("--accent", accent);

      const dialog = document.createElement("div");
      dialog.className = "ui-modal";
      dialog.setAttribute("role", "dialog");
      dialog.setAttribute("aria-modal", "true");

      const h = document.createElement("div");
      h.className = "ui-modal__title";
      h.textContent = title;

      const p = document.createElement("div");
      p.className = "ui-modal__desc";
      p.textContent = detail;

      const actions = document.createElement("div");
      actions.className = "ui-modal__actions";

      const cancelBtn = document.createElement("button");
      cancelBtn.type = "button";
      cancelBtn.className = "flow-back";
      cancelBtn.textContent = "取消";

      const okBtn = document.createElement("button");
      okBtn.type = "button";
      okBtn.className = "flow-back ui-modal__ok";
      okBtn.textContent = "确认";

      const close = (v) => {
        overlay.remove();
        document.removeEventListener("keydown", onKeyDown, true);
        resolve(v);
      };

      const onKeyDown = (e) => {
        if (e.key === "Escape") close(false);
      };

      overlay.addEventListener("click", (e) => {
        if (e.target === overlay) close(false);
      });
      cancelBtn.addEventListener("click", () => close(false));
      okBtn.addEventListener("click", () => close(true));
      document.addEventListener("keydown", onKeyDown, true);

      actions.append(cancelBtn, okBtn);
      dialog.append(h, p, actions);
      overlay.append(dialog);
      document.body.append(overlay);
      requestAnimationFrame(() => okBtn.focus());
    });

  function renderInlineCustom({ accent, placeholder, value, onConfirm }) {
    const wrap = document.createElement("div");
    wrap.className = "inline-custom";
    wrap.style.setProperty("--accent", accent);

    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = placeholder;
    input.className = "inline-custom__input";
    input.value = value || "";

    const confirm = document.createElement("button");
    confirm.type = "button";
    confirm.className = "inline-custom__confirm";
    confirm.textContent = "✓";
    confirm.setAttribute("aria-label", "确认自定义内容");
    confirm.setAttribute("title", "确认");

    let open = false;
    let confirmed = Boolean((value || "").trim());
    let confirming = false;
    const sync = () => {
      wrap.classList.toggle("inline-custom--open", open);
      wrap.classList.toggle("inline-custom--confirmed", confirmed);
      input.disabled = confirmed;
    };
    sync();

    const openNow = () => {
      if (confirmed) return;
      open = true;
      sync();
      requestAnimationFrame(() => input.focus());
    };

    wrap.addEventListener("click", (e) => {
      if (e.target === confirm) return;
      openNow();
    });

    confirm.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!open && !confirmed) openNow();
      const v = (input.value || "").trim();
      if (!v) return;
      if (confirmed || confirming) return;
      confirming = true;
      showConfirmModal({ title: "确认使用该自定义内容？", detail: v, accent }).then((ok) => {
        confirming = false;
        if (!ok) {
          requestAnimationFrame(() => input.focus());
          return;
        }
        confirmed = true;
        sync();
        onConfirm?.(v);
      });
    });

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        confirm.click();
      }
    });

    wrap.append(input, confirm);
    return wrap;
  }

  function mkTagButton(label, accent = "#9ad0ff") {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ui-scene-button";
    btn.dataset.tag = label;
    btn.style.setProperty("--accent", accent);

    const icon = document.createElement("span");
    icon.className = "ui-scene-button__icon";
    icon.textContent = "";

    const textWrap = document.createElement("span");
    textWrap.className = "ui-scene-button__label";
    textWrap.textContent = label;

    const check = document.createElement("span");
    check.className = "ui-scene-button__check";
    check.textContent = "✓";

    btn.append(icon, textWrap, check);

    const sync = () => {
      const on = selectedStyle.has(label);
      btn.classList.toggle("ui-scene-button--selected", on);
      btn.setAttribute("aria-pressed", on ? "true" : "false");
    };
    sync();

    btn.addEventListener("click", () => {
      if (selectedStyle.has(label)) selectedStyle.delete(label);
      else selectedStyle.add(label);
      sync();
      maybeNudgePace();
      updateHints();
      updatePrompt();
    });

    return btn;
  }

  const styleTools = document.createElement("div");
  styleTools.className = "prompt-preview__actions fade-enter";

  /** @type {HTMLButtonElement | null} */
  let styleRefreshBtn = null;
  const clearToolsKeepRefresh = () => {
    for (const child of Array.from(styleTools.children)) {
      if (child.classList?.contains("flow-refresh")) continue;
      child.remove();
    }
  };

  function renderStyleBatch({ viaRefresh } = {}) {
    styleGrid.replaceChildren();
    clearToolsKeepRefresh();

    // 立刻回流后可能打破 exhausted 状态，这里每次渲染都重新计算一次
    exhaustedStyle = STYLE.every((t) => seenStyle.has(t));
    styleGrid.classList.remove("scene-button-grid--one", "scene-button-grid--two", "scene-button-grid--three-strict");
    styleGrid.classList.add("scene-button-grid--two");

    if (viaRefresh) lastStyleHintTags = Array.from(selectedStyle);

    const hintWrap = renderSelectedHints({
      tags: lastStyleHintTags,
      accent: "#9ad0ff",
      onRemove: (tag, chipEl) => {
        selectedStyle.delete(tag);
        // 立刻回流：从 seen 中移除，确保下次刷新可再次出现
        seenStyle.delete(tag);
        exhaustedStyle = false;
        lastStyleHintTags = lastStyleHintTags.filter((x) => x !== tag);
        chipEl?.remove();
        updatePrompt();

        const btn = styleGrid.querySelector(`.ui-scene-button[data-tag="${tag}"]`);
        if (btn) {
          btn.classList.remove("ui-scene-button--selected");
          btn.setAttribute("aria-pressed", "false");
        }
      },
    });

    if (styleRefreshBtn) {
      styleRefreshBtn.hidden = false;
      styleRefreshBtn.style.display = "";
    }

    // exhausted 且用户再次刷新后进入自定义输入
    if (showCustom) {
      if (styleRefreshBtn) {
        styleRefreshBtn.hidden = true;
        styleRefreshBtn.style.display = "none";
      }
      styleTools.append(
        renderInlineCustom({
          accent: "#9ad0ff",
          placeholder: "输入你的补充需求",
          value: customStyleText,
          onConfirm: (v) => {
            customStyleText = v;
            state.custom_style_text = customStyleText;
            state.custom_text =
              [
                state.custom_scene_text ? `补充场景：${state.custom_scene_text}` : null,
                state.custom_mood_text ? `补充氛围：${state.custom_mood_text}` : null,
                state.custom_style_text ? `补充风格：${state.custom_style_text}` : null,
              ]
                .filter(Boolean)
                .join("\n") || null;
            updatePrompt();
          },
        }),
      );
      if (hintWrap) styleTools.append(hintWrap);
      return;
    }

    if (exhaustedStyle) {
      if (hintWrap) styleTools.append(hintWrap);
      // 主+回收都耗尽：仍显示刷新，下一次刷新进入 showCustom
      if (styleRefreshBtn && !styleRefreshBtn.isConnected) styleTools.append(styleRefreshBtn);
      return;
    }

    if (hintWrap) styleTools.append(hintWrap);
    const { batch, exhausted } = pickBatch({ pool: STYLE, seen: seenStyle, count: 4 });
    exhaustedStyle = exhausted;
    if (batch.length === 1) {
      styleGrid.classList.remove("scene-button-grid--two");
      styleGrid.classList.add("scene-button-grid--one");
    } else if (batch.length === 3) {
      styleGrid.classList.remove("scene-button-grid--two");
      styleGrid.classList.add("scene-button-grid--three-strict");
    }
    for (const s of batch) styleGrid.append(mkTagButton(s, "#9ad0ff"));
    if (styleRefreshBtn && !styleRefreshBtn.isConnected) styleTools.append(styleRefreshBtn);
  }

  styleRefreshBtn = createButton({
    label: "⟳",
    variant: "ghost",
    onClick: () => {
      animateBatchRefresh({
        gridEl: styleGrid,
        toolsEl: styleTools,
        doRender: () => {
          if (exhaustedStyle) {
            if (!showCustom) {
              showCustom = true;
              renderStyleBatch({ viaRefresh: true });
              return;
            }
          }
          renderStyleBatch({ viaRefresh: true });
        },
      });
    },
  });
  styleRefreshBtn.className = "flow-back flow-refresh";
  styleRefreshBtn.setAttribute("aria-label", "刷新");
  styleRefreshBtn.setAttribute("title", "刷新");

  renderStyleBatch();

  const promptPreview = document.createElement("section");
  promptPreview.className = "prompt-preview fade-enter";

  const promptTitle = document.createElement("h3");
  promptTitle.className = "prompt-preview__title";
  promptTitle.classList.add("prompt-preview__title--step");
  promptTitle.textContent = "可修改或补充提示词";

  const promptEmpty = document.createElement("p");
  promptEmpty.className = "prompt-preview__empty";
  promptEmpty.textContent = "请选择至少一个场景以生成提示词。";

  const promptTextarea = document.createElement("textarea");
  promptTextarea.className = "prompt-preview__textarea";
  promptTextarea.rows = 1;
  promptTextarea.spellcheck = false;

  /** @type {string} */
  let lastGeneratedPrompt = "";

  const promptHint = document.createElement("p");
  promptHint.className = "prompt-preview__hint";
  promptHint.remove();

  const promptActions = document.createElement("div");
  promptActions.className = "prompt-preview__actions";

  const controls = document.createElement("section");
  controls.className = "prompt-preview fade-enter";
  controls.classList.add("prompt-preview--v2-controls");

  const controlsTitle = document.createElement("h3");
  controlsTitle.className = "prompt-preview__title";
  controlsTitle.classList.add("prompt-preview__title--step");
  controlsTitle.textContent = "你想有什么样的阅读强度与节奏";

  const infoBtn = document.createElement("button");
  infoBtn.type = "button";
  infoBtn.className = "v2-info";
  infoBtn.textContent = "i";
  infoBtn.setAttribute("aria-label", "为什么会影响推荐？");
  infoBtn.setAttribute("title", "为什么会影响推荐？");

  const showInfoModal = () => {
    const overlay = document.createElement("div");
    overlay.className = "ui-modal__overlay";
    overlay.style.setProperty("--accent", "#4aa3ff");

    const dialog = document.createElement("div");
    dialog.className = "ui-modal";
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");

    const h = document.createElement("div");
    h.className = "ui-modal__title";
    h.textContent = "我们如何推荐阅读难度与节奏？";

    const p = document.createElement("div");
    p.className = "ui-modal__desc";
    p.textContent =
      "我们的书籍库中：\n" +
      "1) 书籍的篇幅和难度通常强相关：短篇更常见为低～中等难度（2–3）；中篇多为中等难度（3）；长篇更常见为中～偏高难度（3–4）；超长篇更常见为偏高难度（4–5）。\n" +
      "2) 阅读节奏更常随风格变化：悬疑 / 科幻 / 强情节小说更常偏快（4–5）；散文 / 哲学思考更常偏慢（1–2）；其它风格通常适中（3）。";

    const actions = document.createElement("div");
    actions.className = "ui-modal__actions";
    const okBtn = document.createElement("button");
    okBtn.type = "button";
    okBtn.className = "flow-back ui-modal__ok";
    okBtn.textContent = "知道了";

    const close = () => {
      overlay.remove();
      document.removeEventListener("keydown", onKeyDown, true);
    };
    const onKeyDown = (e) => {
      if (e.key === "Escape") close();
    };
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) close();
    });
    okBtn.addEventListener("click", close);
    document.addEventListener("keydown", onKeyDown, true);

    actions.append(okBtn);
    dialog.append(h, p, actions);
    overlay.append(dialog);
    document.body.append(overlay);
    requestAnimationFrame(() => okBtn.focus());
  };
  infoBtn.addEventListener("click", showInfoModal);

  const sliders = document.createElement("div");
  sliders.className = "v2-sliders";

  function mkSlider(label, value, onChange, { onTouch } = {}) {
    const wrap = document.createElement("label");
    wrap.className = "v2-slider";
    const t = document.createElement("div");
    t.className = "v2-slider__label";
    t.textContent = label;
    const input = document.createElement("input");
    input.type = "range";
    input.min = "1";
    input.max = "5";
    input.step = "1";
    input.value = String(value ?? 3);
    input.addEventListener("input", () => {
      onTouch?.();
      onChange?.(Number(input.value));
    });
    wrap.append(t, input);
    return { wrap, input, val: null };
  }

  const warning = document.createElement("div");
  warning.className = "v2-warning";
  warning.hidden = true;

  const difficultySlider = mkSlider(
    "阅读难度",
    state.difficulty,
    (v) => {
      state.difficulty = v;
      updatePrompt();
      updateHints();
    },
    {
      onTouch: () => {
        difficultyTouched = true;
      },
    },
  );

  const paceSlider = mkSlider(
    "阅读节奏",
    state.pace,
    (v) => {
      state.pace = v;
      updatePrompt();
      updateHints();
    },
    {
      onTouch: () => {
        paceTouched = true;
      },
    },
  );

  function mkTicks(labels) {
    const ticks = document.createElement("div");
    ticks.className = "v2-ticks";
    for (const text of labels) {
      const s = document.createElement("span");
      s.className = "v2-tick";
      s.textContent = text;
      ticks.append(s);
    }
    return ticks;
  }

  function maybeNudgeDifficulty() {
    if (difficultyTouched) return;
    const r = lengthDifficultyRange(state.length_type);
    const want = state.length_type === "短篇" ? 2 : midInt(r.min, r.max);
    setSliderValue({
      inputEl: difficultySlider.input,
      value: want,
      onChange: (v) => {
        state.difficulty = v;
        updatePrompt();
      },
    });
  }

  function maybeNudgePace() {
    if (paceTouched) return;
    const r = stylePaceRange(Array.from(selectedStyle));
    const want = midInt(r.min, r.max);
    setSliderValue({
      inputEl: paceSlider.input,
      value: want,
      onChange: (v) => {
        state.pace = v;
        updatePrompt();
      },
    });
  }

  function updateHints() {
    // 按需求仅保留黄条提示，不展示“推荐难度 / 当前节奏 / 建议节奏”等文本模块
    const warn =
      (state.length_type === "短篇" && Number(state.difficulty) >= 4) ||
      (state.length_type === "超长篇" && Number(state.difficulty) <= 2);
    warning.hidden = !warn;
    if (warn) {
      warning.textContent = "该组合在书库中较少见，推荐结果可能不稳定。可考虑调整难度或篇幅。";
    }
  }

  const difficultyWrap = document.createElement("div");
  difficultyWrap.className = "v2-control v2-control--primary";
  difficultyWrap.append(difficultySlider.wrap, mkTicks(["简单", "适中", "挑战"]));

  const paceWrap = document.createElement("div");
  paceWrap.className = "v2-control v2-control--secondary";
  paceWrap.append(paceSlider.wrap, mkTicks(["舒缓", "适中", "紧凑"]));

  sliders.append(difficultyWrap, paceWrap);

  const lengthWrap = document.createElement("div");
  lengthWrap.className = "v2-length";
  const lengthLabel = document.createElement("div");
  lengthLabel.className = "v2-length__label";
  lengthLabel.textContent = "篇幅";
  const lengthBtns = document.createElement("div");
  lengthBtns.className = "v2-length__btns";
  const LENGTH = ["短篇", "中篇", "长篇", "超长篇"];
  const lengthBtnEls = new Map();
  function syncLength() {
    for (const [k, el] of lengthBtnEls.entries()) {
      el.classList.toggle("v2-length__btn--selected", k === state.length_type);
      el.setAttribute("aria-pressed", k === state.length_type ? "true" : "false");
    }
  }
  for (const l of LENGTH) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "v2-length__btn";
    b.textContent = l;
    b.addEventListener("click", () => {
      state.length_type = l;
      syncLength();
      maybeNudgeDifficulty();
      updateHints();
      updatePrompt();
    });
    lengthBtnEls.set(l, b);
    lengthBtns.append(b);
  }
  syncLength();
  lengthWrap.append(lengthLabel, lengthBtns);

  controls.append(infoBtn, controlsTitle, warning, lengthWrap, sliders);

  function autosizeTextarea() {
    promptTextarea.style.height = "auto";
    promptTextarea.style.height = `${promptTextarea.scrollHeight}px`;
  }

  // 当窗口/容器尺寸变化导致换行变化时，重新 autosize，避免遮挡
  const __ro =
    typeof ResizeObserver !== "undefined"
      ? new ResizeObserver(() => {
          if (!promptTextarea.isConnected) {
            __ro.disconnect();
            return;
          }
          requestAnimationFrame(autosizeTextarea);
        })
      : null;
  __ro?.observe(promptPreview);

  function __onResize() {
    if (!promptTextarea.isConnected) {
      window.removeEventListener("resize", __onResize);
      return;
    }
    requestAnimationFrame(autosizeTextarea);
  }
  window.addEventListener("resize", __onResize);

  promptTextarea.addEventListener("input", () => {
    autosizeTextarea();
    updateSendButtonLabel();
  });

  const sendBtn = document.createElement("button");
  sendBtn.type = "button";
  sendBtn.className = "prompt-send";
  sendBtn.textContent = "发送";
  sendBtn.disabled = true;

  function isDirty() {
    const current = promptTextarea.value || "";
    const base = lastGeneratedPrompt || "";
    // 不做 trim：用户的“补充/换行/空格”也算编辑，应触发“确认并发送”
    return current.length > 0 && current !== base;
  }

  function updateSendButtonLabel() {
    if (sendBtn.disabled) return;
    // 需求调整：按钮文案始终为“发送”，不随编辑状态变化
    sendBtn.textContent = "发送";
  }

  sendBtn.addEventListener("click", () => {
    if (sendBtn.disabled) return;
    const prompt = (promptTextarea.value || "").trim();
    if (!prompt) return;

    sendBtn.disabled = true;
    sendBtn.classList.add("is-submitting");

    const dirty = isDirty();
    sendBtn.textContent = "提交成功";
    sendBtn.classList.remove("is-submitting");
    sendBtn.classList.add("is-success");

    window.setTimeout(() => {
      state.style_tags = Array.from(selectedStyle);
      onSendPrompt?.({ prompt, dirty, intentV2: state });
    }, 1000);
  });

  promptActions.append(sendBtn);
  promptPreview.append(promptTitle, promptEmpty, promptTextarea, promptHint, promptActions);

  function updatePrompt() {
    state.style_tags = Array.from(selectedStyle);
    const { prompt } = generatePromptV2(state);
    const extra = (state.custom_text || "").trim();
    const finalPrompt = extra ? `${prompt}\n${extra}` : prompt;
    promptTextarea.value = finalPrompt;
    lastGeneratedPrompt = finalPrompt;
    promptEmpty.hidden = true;
    promptTextarea.hidden = false;
    promptHint.hidden = false;
    sendBtn.disabled = false;
    sendBtn.classList.remove("is-success", "is-submitting");
    updateSendButtonLabel();
    autosizeTextarea();
  }

  updatePrompt();
  maybeNudgeDifficulty();
  maybeNudgePace();
  updateHints();

  const back = createButton({
    label: "返回上一步",
    variant: "ghost",
    onClick: () => {
      if (reduceMotion) {
        state.style_tags = Array.from(selectedStyle);
        onBack?.(state);
        return;
      }

      back.disabled = true;
      back.style.pointerEvents = "none";

      // 先让主要模块淡出，再返回上一步
      for (const el of [styleHeading, styleGrid, controls, promptPreview]) {
        el.classList.remove("fade-enter", "fade-enter--active");
        el.classList.add("fade-exit");
      }

      // 选择区按既有节奏淡出
      styleGrid.classList.add("scene-button-grid--transitioning");
      requestAnimationFrame(() => {
        for (const el of [styleHeading, styleGrid, controls, promptPreview]) el.classList.add("fade-exit--active");
      });

      // 等按钮淡出完成后再切回 goal，避免突兀
      const lastBtn = styleGrid.querySelector(".ui-scene-button:last-child");
      (lastBtn ?? styleGrid).addEventListener(
        "transitionend",
        () => {
          state.style_tags = Array.from(selectedStyle);
          onBack?.(state);
        },
        { once: true },
      );
    },
  });
  back.className = "flow-back flow-back--bottom";
  back.classList.add("fade-enter");

  requestAnimationFrame(() => {
    // 强制读取一次样式，确保初始态已提交到渲染管线
    void styleHeading.offsetWidth;
    void styleGrid.offsetWidth;
    void controls.offsetWidth;
    void back.offsetWidth;

    requestAnimationFrame(() => {
      styleHeading.classList.add("fade-enter--active");
      styleGrid.classList.add("fade-enter--active");
      controls.classList.add("fade-enter--active");
      back.classList.add("fade-enter--active");
      promptPreview.classList.add("fade-enter--active");
      styleTools.classList.add("fade-enter--active");
    });
  });

  root.append(styleHeading, styleGrid, styleTools, controls, promptPreview, back);
  return root;
}

