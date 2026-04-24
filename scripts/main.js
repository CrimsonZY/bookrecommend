/**
 * 应用入口：路由、状态与页面逻辑从此处组织或挂载。
 */

import { renderGoalView } from "./views/goal-view.js";
import { renderSceneView } from "./views/scene-view.js";
import { renderRecommendView } from "./views/recommend-view.js";
import { createNavBar } from "../components/NavBar.js";

const main = document.getElementById("main");
const topbar = document.getElementById("topbar");

if (topbar) {
  topbar.replaceChildren(
    createNavBar({
      title: "书籍推荐",
      subtitle: "选择你的目标，开始发现适合的书籍",
    }),
  );
}

const appState = {
  intentV2: {
    /** @type {string[]} */
    mood_tags: [],
    /** @type {string[]} */
    scene_tags: [],
    /** @type {string[]} */
    style_tags: [],
    /** @type {number} */
    difficulty: 3,
    /** @type {number} */
    pace: 3,
    /** @type {"短篇"|"中篇"|"长篇"|"超长篇"} */
    length_type: "中篇",
    /** @type {string | null} */
    custom_text: null,
    /** @type {string | null} */
    custom_scene_text: null,
    /** @type {string | null} */
    custom_mood_text: null,
    /** @type {string | null} */
    custom_style_text: null,
  },
  returningFromStyleMore: false,
  /** @type {string | null} */
  finalPrompt: null,
};

function normalizeHash() {
  const hash = (location.hash || "").replace("#", "");
  if (hash === "mood-scene" || hash === "style-more" || hash === "recommend") return hash;
  return "mood-scene";
}

function setHash(step) {
  if (step === "mood-scene") location.hash = "#mood-scene";
  else if (step === "style-more") location.hash = "#style-more";
  else location.hash = "#recommend";
}

function render() {
  if (!main) return;

  const step = normalizeHash();
  document.body.classList.toggle("page-style-more", step === "style-more");

  if (step === "mood-scene") {
    main.replaceChildren(
      renderGoalView({
        returningFromStyleMore: appState.returningFromStyleMore,
        intentV2: appState.intentV2,
        onNext: (partialIntent) => {
          appState.intentV2 = { ...appState.intentV2, ...partialIntent };
          appState.returningFromStyleMore = false;
          appState.finalPrompt = null;
          setHash("style-more");
        },
      }),
    );
    return;
  }

  if (step === "style-more") {
    main.replaceChildren(
      renderSceneView({
        intentV2: appState.intentV2,
        onBack: (partialIntent) => {
          appState.intentV2 = { ...appState.intentV2, ...partialIntent };
          appState.returningFromStyleMore = true;
          setHash("mood-scene");
        },
        onSendPrompt: ({ prompt, intentV2 }) => {
          appState.finalPrompt = prompt;
          appState.intentV2 = intentV2;
          setHash("recommend");
        },
      }),
    );
    return;
  }

  main.replaceChildren(
    renderRecommendView({
      intentV2: appState.intentV2,
      finalPrompt: appState.finalPrompt,
      onBackToScene: () => setHash("style-more"),
    }),
  );
}

if (main) {
  if (!location.hash) setHash("mood-scene");
  window.addEventListener("hashchange", render);
  render();
}
