/**
 * 通用图标+文本按钮组件（目标选择按钮等可复用）
 *
 * @param {object} options
 * @param {string} options.title - 主标题（短）
 * @param {string} [options.subtitle] - 副标题（更短，可选）
 * @param {string} options.icon - SVG 字符串（建议 24x24 viewBox）
 * @param {string} [options.accent] - 主题色（例如 "#9ad0ff"）
 * @param {() => void} [options.onClick]
 * @param {string} [options.ariaLabel]
 */
export function createButtonComponent({
  title,
  subtitle,
  icon,
  accent = "#9ad0ff",
  onClick,
  ariaLabel,
} = {}) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ui-goal-button";
  button.style.setProperty("--accent", accent);

  if (ariaLabel) button.setAttribute("aria-label", ariaLabel);
  else if (title && subtitle) button.setAttribute("aria-label", `${title}：${subtitle}`);
  else if (title) button.setAttribute("aria-label", title);

  const iconWrap = document.createElement("span");
  iconWrap.className = "ui-goal-button__icon";
  iconWrap.setAttribute("aria-hidden", "true");
  iconWrap.innerHTML = icon ?? "";

  const textWrap = document.createElement("span");
  textWrap.className = "ui-goal-button__text";

  const titleEl = document.createElement("span");
  titleEl.className = "ui-goal-button__title";
  titleEl.textContent = title ?? "";
  textWrap.appendChild(titleEl);

  if (subtitle) {
    const subEl = document.createElement("span");
    subEl.className = "ui-goal-button__subtitle";
    subEl.textContent = subtitle;
    textWrap.appendChild(subEl);
  }

  button.appendChild(iconWrap);
  button.appendChild(textWrap);

  if (typeof onClick === "function") {
    button.addEventListener("click", onClick);
  }

  return button;
}

