/**
 * 场景按钮组件（图标 + 文本）
 *
 * @param {object} options
 * @param {string} options.label
 * @param {string} options.icon - SVG 字符串（建议 24x24 viewBox）
 * @param {string} [options.accent]
 * @param {boolean} [options.selected]
 * @param {(nextSelected: boolean) => void} [options.onToggle]
 * @param {string} [options.ariaLabel]
 */
export function createSceneButtonComponent({
  label,
  icon,
  accent = "#9ad0ff",
  selected = false,
  onToggle,
  ariaLabel,
} = {}) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ui-scene-button";
  button.style.setProperty("--accent", accent);
  button.classList.toggle("ui-scene-button--selected", Boolean(selected));
  button.setAttribute("aria-pressed", String(Boolean(selected)));

  if (ariaLabel) button.setAttribute("aria-label", ariaLabel);
  else if (label) button.setAttribute("aria-label", label);

  const iconWrap = document.createElement("span");
  iconWrap.className = "ui-scene-button__icon";
  iconWrap.setAttribute("aria-hidden", "true");
  iconWrap.innerHTML = icon ?? "";

  const text = document.createElement("span");
  text.className = "ui-scene-button__label";
  text.textContent = label ?? "";

  const check = document.createElement("span");
  check.className = "ui-scene-button__check";
  check.setAttribute("aria-hidden", "true");
  check.innerHTML = `
<svg viewBox="0 0 24 24" width="18" height="18" role="img" focusable="false" aria-hidden="true">
  <path fill="currentColor" d="M9.2 16.2 4.9 11.9a.75.75 0 1 1 1.06-1.06l3.24 3.24 8.84-8.84a.75.75 0 1 1 1.06 1.06L9.2 16.2Z"/>
</svg>
`;

  button.append(iconWrap, text, check);

  button.addEventListener("click", () => {
    const next = !button.classList.contains("ui-scene-button--selected");
    button.classList.toggle("ui-scene-button--selected", next);
    button.setAttribute("aria-pressed", String(next));
    if (typeof onToggle === "function") onToggle(next);
  });

  return button;
}

