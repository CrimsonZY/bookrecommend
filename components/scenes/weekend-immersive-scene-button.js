import { createSceneButtonComponent } from "../SceneButtonComponent.js";

const sofaIcon = `
<svg viewBox="0 0 24 24" width="24" height="24" role="img" focusable="false" aria-hidden="true">
  <path fill="currentColor" d="M7.25 7A3.25 3.25 0 0 0 4 10.25V13a2 2 0 0 0 1.25 1.86V18a.75.75 0 0 0 1.5 0v-3h10.5v3a.75.75 0 0 0 1.5 0v-3.14A2 2 0 0 0 20 13v-2.75A3.25 3.25 0 0 0 16.75 7H7.25ZM5.5 10.25c0-.97.78-1.75 1.75-1.75h9.5c.97 0 1.75.78 1.75 1.75V13a.5.5 0 0 1-.5.5H6a.5.5 0 0 1-.5-.5v-2.75Z"/>
</svg>
`;

export function createWeekendImmersiveSceneButton({ selected, onToggle } = {}) {
  return createSceneButtonComponent({
    label: "周末沉浸阅读",
    icon: sofaIcon,
    accent: "#b7e6c0",
    selected,
    onToggle,
  });
}

