function uniq(arr) {
  return Array.from(new Set((arr || []).filter(Boolean)));
}

function joinChinese(items) {
  const xs = (items || []).filter(Boolean);
  if (xs.length === 0) return "";
  if (xs.length === 1) return xs[0];
  if (xs.length === 2) return `${xs[0]} / ${xs[1]}`;
  return xs.join(" / ");
}

function paceDesc(pace) {
  const v = Math.max(1, Math.min(5, Number(pace || 3)));
  if (v <= 2) return "慢";
  if (v === 3) return "适中";
  return "快";
}

function difficultyDesc(difficulty) {
  const v = Math.max(1, Math.min(5, Number(difficulty || 3)));
  if (v <= 2) return "较低";
  if (v === 3) return "适中";
  return "较高";
}

/**
 * @param {object} intentV2
 * @param {string[]} intentV2.mood_tags
 * @param {string[]} intentV2.scene_tags
 * @param {string[]} intentV2.style_tags
 * @param {number} intentV2.difficulty
 * @param {number} intentV2.pace
 * @param {string} intentV2.length_type
 * @returns {{ prompt: string, meta: any }}
 */
export function generatePromptV2(intentV2 = {}) {
  const mood = uniq(intentV2.mood_tags);
  const scene = uniq(intentV2.scene_tags);
  const style = uniq(intentV2.style_tags);

  const moodText = mood.length ? `【${joinChinese(mood)}】` : "【—】";
  const styleText = style.length ? `【${joinChinese(style)}】` : "【—】";
  const sceneText = scene.length ? `【${joinChinese(scene)}】` : "【—】";

  const pDesc = paceDesc(intentV2.pace);
  const dDesc = difficultyDesc(intentV2.difficulty);
  const len = intentV2.length_type || "中篇";

  const prompt =
    `我现在希望阅读一本${moodText}风格的${styleText}类书籍，` +
    `阅读场景是${sceneText}，节奏偏【${pDesc}】，难度${dDesc}，篇幅为【${len}】。`;

  const extraLines = [
    (intentV2.custom_scene_text || "").trim() ? `补充场景：${String(intentV2.custom_scene_text).trim()}` : null,
    (intentV2.custom_mood_text || "").trim() ? `补充氛围：${String(intentV2.custom_mood_text).trim()}` : null,
    (intentV2.custom_style_text || "").trim() ? `补充风格：${String(intentV2.custom_style_text).trim()}` : null,
  ].filter(Boolean);

  return {
    prompt,
    meta: {
      intentV2: {
        mood_tags: mood,
        scene_tags: scene,
        style_tags: style,
        difficulty: Number(intentV2.difficulty || 3),
        pace: Number(intentV2.pace || 3),
        length_type: len,
        custom_scene_text: (intentV2.custom_scene_text || "").trim() || null,
        custom_mood_text: (intentV2.custom_mood_text || "").trim() || null,
        custom_style_text: (intentV2.custom_style_text || "").trim() || null,
        custom_text: extraLines.join("\n") || null,
      },
    },
  };
}

