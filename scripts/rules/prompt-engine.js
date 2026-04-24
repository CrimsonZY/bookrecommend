/**
 * 目标×场景 → 提示词 规则引擎
 *
 * - 先命中专用模板：goalId + sceneKey
 * - 否则使用目标通用模板 + 合并后的场景特点
 * - 场景多选时合并为一条提示词
 */

/** @typedef {{ id: string, title: string, subtitle?: string }} Goal */

/** @type {Record<string, { label: string, traits: string[] }>} */
const SCENES = {
  commute: {
    label: "通勤",
    traits: ["轻松", "节奏明快", "篇幅适中", "可随时中断并继续"],
  },
  bedtime: {
    label: "睡前",
    traits: ["轻松", "舒缓", "氛围感强", "容易让人放松"],
  },
  weekend: {
    label: "周末沉浸阅读",
    traits: ["更有深度", "更连贯", "章节结构清晰", "代入感强"],
  },
  quick: {
    label: "快速阅读",
    traits: ["简短", "信息密度高", "开篇抓人", "节奏紧凑"],
  },
};

/** @type {Record<string, string>} */
const GOALS = {
  resume_reading: "恢复阅读",
  gift_book: "送一本书",
  explore_new_books: "读新书",
};

/** @type {Record<string, Record<string, string>>} */
const SPECIFIC_TEMPLATES = {
  resume_reading: {
    commute: "推荐适合{goal}时在{scene}阅读的书籍，内容{sceneTraits}。",
    bedtime: "推荐适合{goal}时在{scene}阅读的书籍，内容{sceneTraits}。",
  },
  gift_book: {
    commute: "为{goal}挑选一本适合{scene}时阅读的书籍，内容{sceneTraits}。",
  },
  explore_new_books: {
    quick: "推荐适合{goal}的书籍，内容{sceneTraits}，适合{scene}。",
  },
};

/** @type {Record<string, string>} */
const GOAL_DEFAULT_TEMPLATES = {
  resume_reading: "推荐适合{goal}的书籍，内容{sceneTraits}。",
  gift_book: "为{goal}挑选一本书，阅读体验{sceneTraits}。",
  explore_new_books: "推荐适合{goal}的书籍，内容{sceneTraits}。",
};

function uniq(arr) {
  return Array.from(new Set(arr.filter(Boolean)));
}

function joinChinese(items) {
  const xs = items.filter(Boolean);
  if (xs.length <= 1) return xs[0] || "";
  if (xs.length === 2) return `${xs[0]}以及${xs[1]}`;
  return `${xs.slice(0, -1).join("、")}以及${xs[xs.length - 1]}`;
}

function render(template, vars) {
  return template.replace(/\{(\w+)\}/g, (_, k) => String(vars[k] ?? ""));
}

function normalizeSceneKey(sceneLabelOrKey) {
  if (!sceneLabelOrKey) return null;
  if (SCENES[sceneLabelOrKey]) return sceneLabelOrKey;
  const entry = Object.entries(SCENES).find(([, v]) => v.label === sceneLabelOrKey);
  return entry?.[0] ?? null;
}

function deriveSceneTraits(sceneKeys) {
  const traits = [];
  for (const k of sceneKeys) {
    const t = SCENES[k]?.traits ?? [];
    traits.push(...t);
  }
  return uniq(traits);
}

function sceneTraitsTextForKey(sceneKey) {
  const traits = uniq(SCENES[sceneKey]?.traits ?? []);
  return traits.length ? joinChinese(traits) : "贴合你的阅读节奏";
}

function renderMultiScenePrompt({ goalLabel, sceneKeys }) {
  const prefix = `为了${goalLabel}挑选一本书。`;

  const parts = sceneKeys.map((k, idx) => {
    const sceneLabel = SCENES[k]?.label ?? "该场景";
    const traitsText = sceneTraitsTextForKey(k);
    const lead = idx === 0 ? "我可以在" : "也可以在";
    const readWord = sceneLabel.endsWith("阅读") ? "" : "阅读";
    return `${lead}${sceneLabel}${readWord}，书籍内容${traitsText}。`;
  });

  return `${prefix}${parts.join("")}`;
}

/**
 * @param {object} params
 * @param {Goal | null} params.goal
 * @param {string[]} params.scenes 场景 key 或中文 label 均可
 * @returns {{ prompt: string, meta: { goalId: string | null, sceneKeys: string[], templateKey: string, fallback: boolean } }}
 */
export function generatePrompt({ goal, scenes } = {}) {
  const goalId = goal?.id ?? null;
  const goalLabel = (goalId && GOALS[goalId]) || goal?.title || "目标";

  const sceneKeys = uniq((scenes || []).map(normalizeSceneKey).filter(Boolean));
  const sceneLabels = sceneKeys.map((k) => SCENES[k].label);
  const sceneLabelMerged = sceneLabels.length ? joinChinese(sceneLabels) : "当前场景";

  // 多场景：按场景分段生成（避免 traits 混写冲突）
  if (sceneKeys.length >= 2) {
    return {
      prompt: renderMultiScenePrompt({ goalLabel, sceneKeys }),
      meta: {
        goalId,
        sceneKeys,
        templateKey: "multiScene.segmented",
        fallback: false,
      },
    };
  }

  const sceneTraitsMerged = deriveSceneTraits(sceneKeys);
  const sceneTraitsText = sceneTraitsMerged.length ? joinChinese(sceneTraitsMerged) : "贴合你的阅读节奏";

  const specific = goalId && sceneKeys.length === 1 ? SPECIFIC_TEMPLATES?.[goalId]?.[sceneKeys[0]] : null;
  const goalDefault = goalId ? GOAL_DEFAULT_TEMPLATES?.[goalId] : null;

  const template = specific || goalDefault || "推荐适合{goal}的书籍，内容{sceneTraits}。";
  const prompt = render(template, {
    goal: goalLabel,
    scene: sceneLabelMerged,
    sceneTraits: sceneTraitsText,
  });

  return {
    prompt,
    meta: {
      goalId,
      sceneKeys,
      templateKey: specific
        ? `specific.${goalId}.${sceneKeys[0]}`
        : goalDefault
          ? `goalDefault.${goalId}`
          : "fallback.default",
      fallback: !specific,
    },
  };
}

