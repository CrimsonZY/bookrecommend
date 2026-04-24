import { createSceneButtonComponent } from "../SceneButtonComponent.js";

const stopwatchIcon = `
<svg viewBox="0 0 24 24" width="24" height="24" role="img" focusable="false" aria-hidden="true">
  <path fill="currentColor" d="M10 2.75c0-.41.34-.75.75-.75h2.5c.41 0 .75.34.75.75s-.34.75-.75.75h-.5V5a7.5 7.5 0 1 1-3.24.75V3.5h-.5A.75.75 0 0 1 10 2.75ZM12 7a6 6 0 1 0 0 12 6 6 0 0 0 0-12Zm.75 2.5a.75.75 0 0 1 .75.75v2.7l1.6 1.6a.75.75 0 1 1-1.06 1.06l-1.82-1.82a.75.75 0 0 1-.22-.53v-3.01a.75.75 0 0 1 .75-.75Z"/>
</svg>
`;

export function createQuickReadingSceneButton({ selected, onToggle } = {}) {
  return createSceneButtonComponent({
    label: "快速阅读",
    icon: stopwatchIcon,
    accent: "#ffcf9a",
    selected,
    onToggle,
  });
}

