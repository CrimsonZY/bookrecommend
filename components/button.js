/**
 * 可复用按钮组件（示例 API，可按设计系统扩展）。
 * @param {object} options
 * @param {string} options.label
 * @param {string} [options.variant]
 * @param {() => void} [options.onClick]
 */
export function createButton({ label, variant = "primary", onClick } = {}) {
  const el = document.createElement("button");
  el.type = "button";
  el.textContent = label ?? "";
  el.dataset.variant = variant;
  if (typeof onClick === "function") {
    el.addEventListener("click", onClick);
  }
  return el;
}
