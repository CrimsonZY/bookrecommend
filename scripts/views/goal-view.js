import { createButton } from "../../components/button.js";

function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function buildGroupIndex(exclusionGroups) {
  const map = new Map();
  (exclusionGroups || []).forEach((group, gi) => {
    for (const t of group || []) map.set(t, gi);
  });
  return map;
}

function pickBatch({ pool, seen, count, groupIndex }) {
  const available = pool.filter((t) => !seen.has(t));
  if (available.length === 0) return { batch: [], exhausted: true };

  const byGroup = new Map();
  const ungrouped = [];
  for (const t of available) {
    const gi = groupIndex?.get(t);
    if (gi == null) ungrouped.push(t);
    else {
      if (!byGroup.has(gi)) byGroup.set(gi, []);
      byGroup.get(gi).push(t);
    }
  }

  const batch = [];
  // 先从不同互斥组各取一个
  for (const gi of shuffle(Array.from(byGroup.keys()))) {
    const items = byGroup.get(gi);
    if (!items?.length) continue;
    batch.push(shuffle(items)[0]);
    if (batch.length >= count) break;
  }
  // 不足时从未分组补齐
  if (batch.length < count) {
    for (const t of shuffle(ungrouped)) {
      if (batch.length >= count) break;
      batch.push(t);
    }
  }
  // 仍不足时从剩余可用补齐
  if (batch.length < count) {
    for (const t of shuffle(available)) {
      if (batch.length >= count) break;
      if (!batch.includes(t)) batch.push(t);
    }
  }

  for (const t of batch) seen.add(t);
  const exhausted = pool.every((t) => seen.has(t));
  return { batch, exhausted };
}

/**
 * @param {object} params
 * @param {boolean} params.returningFromStyleMore
 * @param {{ mood_tags: string[], scene_tags: string[] }} params.intentV2
 * @param {(partialIntent: { mood_tags: string[], scene_tags: string[] }) => void} params.onNext
 */
export function renderGoalView({ onNext, returningFromStyleMore, intentV2 } = {}) {
  const root = document.createElement("section");
  root.className = "flow-step flow-step--goal";

  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;

  /** @type {HTMLButtonElement | null} */
  let moodRefreshBtn = null;
  /** @type {HTMLButtonElement | null} */
  let sceneRefreshBtn = null;

  const clearToolsKeepRefresh = (toolsEl) => {
    for (const child of Array.from(toolsEl.children)) {
      if (child.classList?.contains("flow-refresh")) continue;
      child.remove();
    }
  };

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

  const sceneTitle = document.createElement("h2");
  sceneTitle.className = "flow-step__title";
  sceneTitle.textContent = "你想在什么时候阅读";
  sceneTitle.classList.add("fade-enter");

  const sceneGrid = document.createElement("section");
  sceneGrid.className = "scene-button-grid fade-enter";

  const sceneTools = document.createElement("div");
  sceneTools.className = "prompt-preview__actions prompt-preview__tools prompt-preview__tools--between fade-enter";

  const moodTitle = document.createElement("h2");
  moodTitle.className = "flow-step__title";
  moodTitle.textContent = "你想有什么样的阅读氛围";
  moodTitle.classList.add("fade-enter");

  const moodGrid = document.createElement("section");
  moodGrid.className = "scene-button-grid fade-enter";

  const moodTools = document.createElement("div");
  moodTools.className = "prompt-preview__actions prompt-preview__tools fade-enter";

  const selectedMood = new Set(intentV2?.mood_tags || []);
  const selectedScene = new Set(intentV2?.scene_tags || []);

  /** @type {string[]} */
  let lastSceneHintTags = [];
  /** @type {string[]} */
  let lastMoodHintTags = [];

  const MOOD = ["治愈", "温暖", "平静", "热血", "沉重", "悲伤", "哲思", "孤独", "希望", "幽默", "浪漫"];
  const SCENE = ["睡前", "通勤", "周末下午", "长期阅读", "旅行途中", "碎片阅读"];
  const moodGroupIndex = buildGroupIndex([
    ["治愈", "温暖"],
    ["沉重", "悲伤"],
    ["平静", "哲思"],
    ["热血", "希望"],
    ["幽默", "浪漫"],
    ["孤独", "沉重"],
  ]);

  const sceneGroupIndex = buildGroupIndex([]);

  const seenMood = new Set();
  const seenScene = new Set();

  let exhaustedMood = false;
  let exhaustedScene = false;

  /** @type {string | null} */
  let customMoodText = (intentV2?.custom_mood_text || "").trim() || null;
  /** @type {string | null} */
  let customSceneText = (intentV2?.custom_scene_text || "").trim() || null;

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
      showConfirmModal({
        title: "确认使用该自定义内容？",
        detail: v,
        accent,
      }).then((ok) => {
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

  function mkTagButton(label, selectedSet, colorVar) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ui-scene-button";
    btn.dataset.tag = label;
    btn.style.setProperty("--accent", colorVar);
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
      const on = selectedSet.has(label);
      btn.classList.toggle("ui-scene-button--selected", on);
      btn.setAttribute("aria-pressed", on ? "true" : "false");
    };
    sync();

    btn.addEventListener("click", () => {
      if (selectedSet.has(label)) selectedSet.delete(label);
      else selectedSet.add(label);
      sync();
      nextBtn.disabled = selectedMood.size === 0 && selectedScene.size === 0;
    });
    return btn;
  }

  function renderMoodBatch({ viaRefresh } = {}) {
    moodGrid.replaceChildren();
    clearToolsKeepRefresh(moodTools);
    moodGrid.classList.remove("scene-button-grid--one", "scene-button-grid--two", "scene-button-grid--three", "scene-button-grid--three-strict");

    // 立刻回流后可能打破 exhausted 状态，这里每次渲染都重新计算一次
    exhaustedMood = MOOD.every((t) => seenMood.has(t));

    if (viaRefresh) lastMoodHintTags = Array.from(selectedMood);

    const hintWrap = renderSelectedHints({
      tags: lastMoodHintTags,
      accent: "#9ad0ff",
      onRemove: (tag, chipEl) => {
        selectedMood.delete(tag);
        // 立刻回流：从 seen 中移除，确保下次刷新可再次出现
        seenMood.delete(tag);
        exhaustedMood = false;
        lastMoodHintTags = lastMoodHintTags.filter((x) => x !== tag);
        nextBtn.disabled = selectedMood.size === 0 && selectedScene.size === 0 && !customMoodText && !customSceneText;
        chipEl?.remove();

        const btn = moodGrid.querySelector(`.ui-scene-button[data-tag="${tag}"]`);
        if (btn) {
          btn.classList.remove("ui-scene-button--selected");
          btn.setAttribute("aria-pressed", "false");
        }
      },
    });

    if (exhaustedMood) {
      if (moodRefreshBtn) {
        moodRefreshBtn.hidden = true;
        moodRefreshBtn.style.display = "none";
      }
      const customBtn = createButton({
        label: "自定义",
        variant: "ghost",
        onClick: () => {},
      });
      moodTools.append(
        renderInlineCustom({
          accent: "#9ad0ff",
          placeholder: "输入你的情绪需求",
          value: customMoodText,
          onConfirm: (v) => {
            customMoodText = v;
            nextBtn.disabled =
              selectedMood.size === 0 && selectedScene.size === 0 && !customMoodText && !customSceneText;
          },
        }),
      );
      if (hintWrap) moodTools.append(hintWrap);
      return;
    }

    if (moodRefreshBtn) {
      moodRefreshBtn.hidden = false;
      moodRefreshBtn.style.display = "";
    }
    if (hintWrap) moodTools.append(hintWrap);
    const { batch, exhausted } = pickBatch({ pool: MOOD, seen: seenMood, count: 4, groupIndex: moodGroupIndex });
    exhaustedMood = exhausted;
    if (batch.length === 1) moodGrid.classList.add("scene-button-grid--one");
    else if (batch.length === 2) moodGrid.classList.add("scene-button-grid--two");
    else if (batch.length === 3) moodGrid.classList.add("scene-button-grid--three-strict");
    for (const m of batch) moodGrid.append(mkTagButton(m, selectedMood, "#9ad0ff"));

    if (moodRefreshBtn && !moodRefreshBtn.isConnected) moodTools.append(moodRefreshBtn);
    if (exhaustedMood) {
    }
  }

  function renderSceneBatch({ viaRefresh } = {}) {
    sceneGrid.classList.remove("scene-button-grid--one", "scene-button-grid--two", "scene-button-grid--three-strict");
    sceneGrid.replaceChildren();
    clearToolsKeepRefresh(sceneTools);

    // 立刻回流后可能打破 exhausted 状态，这里每次渲染都重新计算一次
    exhaustedScene = SCENE.every((t) => seenScene.has(t));

    if (viaRefresh) lastSceneHintTags = Array.from(selectedScene);

    const hintWrap = renderSelectedHints({
      tags: lastSceneHintTags,
      accent: "#c6b3ff",
      onRemove: (tag, chipEl) => {
        selectedScene.delete(tag);
        // 立刻回流：从 seen 中移除，确保下次刷新可再次出现
        seenScene.delete(tag);
        exhaustedScene = false;
        lastSceneHintTags = lastSceneHintTags.filter((x) => x !== tag);
        nextBtn.disabled = selectedMood.size === 0 && selectedScene.size === 0 && !customMoodText && !customSceneText;
        chipEl?.remove();

        const btn = sceneGrid.querySelector(`.ui-scene-button[data-tag="${tag}"]`);
        if (btn) {
          btn.classList.remove("ui-scene-button--selected");
          btn.setAttribute("aria-pressed", "false");
        }
      },
    });

    if (exhaustedScene) {
      if (sceneRefreshBtn) {
        sceneRefreshBtn.hidden = true;
        sceneRefreshBtn.style.display = "none";
      }
      const customBtn = createButton({
        label: "自定义",
        variant: "ghost",
        onClick: () => {},
      });
      sceneTools.append(
        renderInlineCustom({
          accent: "#c6b3ff",
          placeholder: "输入你的场景描述",
          value: customSceneText,
          onConfirm: (v) => {
            customSceneText = v;
            nextBtn.disabled =
              selectedMood.size === 0 && selectedScene.size === 0 && !customMoodText && !customSceneText;
          },
        }),
      );
      if (hintWrap) sceneTools.append(hintWrap);
      return;
    }

    if (sceneRefreshBtn) {
      sceneRefreshBtn.hidden = false;
      sceneRefreshBtn.style.display = "";
    }
    if (hintWrap) sceneTools.append(hintWrap);
    const { batch, exhausted } = pickBatch({ pool: SCENE, seen: seenScene, count: 3, groupIndex: sceneGroupIndex });
    exhaustedScene = exhausted;
    if (batch.length === 1) sceneGrid.classList.add("scene-button-grid--one");
    else if (batch.length === 2) sceneGrid.classList.add("scene-button-grid--two");
    else if (batch.length === 3) sceneGrid.classList.add("scene-button-grid--three-strict");
    for (const s of batch) sceneGrid.append(mkTagButton(s, selectedScene, "#c6b3ff"));

    if (sceneRefreshBtn && !sceneRefreshBtn.isConnected) sceneTools.append(sceneRefreshBtn);
    if (exhaustedScene) {
    }
  }

  moodRefreshBtn = createButton({
    label: "⟳",
    variant: "ghost",
    onClick: () =>
      animateBatchRefresh({
        gridEl: moodGrid,
        toolsEl: moodTools,
        doRender: () => renderMoodBatch({ viaRefresh: true }),
      }),
  });
  moodRefreshBtn.className = "flow-back flow-refresh";
  moodRefreshBtn.setAttribute("aria-label", "刷新");
  moodRefreshBtn.setAttribute("title", "刷新");

  sceneRefreshBtn = createButton({
    label: "⟳",
    variant: "ghost",
    onClick: () =>
      animateBatchRefresh({
        gridEl: sceneGrid,
        toolsEl: sceneTools,
        doRender: () => renderSceneBatch({ viaRefresh: true }),
      }),
  });
  sceneRefreshBtn.className = "flow-back flow-refresh";
  sceneRefreshBtn.setAttribute("aria-label", "刷新");
  sceneRefreshBtn.setAttribute("title", "刷新");

  renderSceneBatch();
  renderMoodBatch();

  const actions = document.createElement("div");
  actions.className = "prompt-preview__actions fade-enter";

  const nextBtn = createButton({
    label: "下一步",
    variant: "ghost",
    onClick: () => {
      const mood_tags = Array.from(selectedMood);
      const scene_tags = Array.from(selectedScene);
      // 自定义仅作为额外文本使用，不写入 intent_v2 tags
      const payload = {
        mood_tags,
        scene_tags,
        custom_scene_text: customSceneText,
        custom_mood_text: customMoodText,
        // 兼容旧字段：用于后续拼接/后端 custom_text
        custom_text:
          [
            customSceneText ? `补充场景：${customSceneText}` : null,
            customMoodText ? `补充氛围：${customMoodText}` : null,
            (intentV2?.custom_style_text || "").trim() ? `补充风格：${String(intentV2.custom_style_text).trim()}` : null,
          ]
            .filter(Boolean)
            .join("\n") || null,
      };
      onNext?.(payload);
    },
  });
  nextBtn.className = "prompt-send";
  nextBtn.disabled = selectedMood.size === 0 && selectedScene.size === 0 && !customMoodText && !customSceneText;
  actions.append(nextBtn);

  if (returningFromStyleMore) {
    requestAnimationFrame(() => {
      moodTitle.classList.add("fade-enter--active");
      moodGrid.classList.add("fade-enter--active");
      moodTools.classList.add("fade-enter--active");
      sceneTitle.classList.add("fade-enter--active");
      sceneGrid.classList.add("fade-enter--active");
      sceneTools.classList.add("fade-enter--active");
      actions.classList.add("fade-enter--active");
    });
  } else {
    requestAnimationFrame(() => {
      moodTitle.classList.add("fade-enter--active");
      moodGrid.classList.add("fade-enter--active");
      moodTools.classList.add("fade-enter--active");
      sceneTitle.classList.add("fade-enter--active");
      sceneGrid.classList.add("fade-enter--active");
      sceneTools.classList.add("fade-enter--active");
      actions.classList.add("fade-enter--active");
    });
  }

  root.append(sceneTitle, sceneGrid, sceneTools, moodTitle, moodGrid, moodTools, actions);
  return root;
}

