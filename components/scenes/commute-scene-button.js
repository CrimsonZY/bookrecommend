import { createSceneButtonComponent } from "../SceneButtonComponent.js";

const commuteIcon = `
<svg viewBox="0 0 24 24" width="24" height="24" role="img" focusable="false" aria-hidden="true">
  <path fill="currentColor" d="M7 3.5C5.62 3.5 4.5 4.62 4.5 6v9.5c0 1.1.9 2 2 2h.25l-.6 1.2a.75.75 0 0 0 1.35.67l.94-1.87h7.12l.94 1.87a.75.75 0 0 0 1.35-.67l-.6-1.2H17.5c1.1 0 2-.9 2-2V6c0-1.38-1.12-2.5-2.5-2.5H7Zm0 2H17c.28 0 .5.22.5.5V11H6.5V6c0-.28.22-.5.5-.5ZM6.5 13h11v2.5a.5.5 0 0 1-.5.5H7a.5.5 0 0 1-.5-.5V13Zm2.25 1.25a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm6.5 0a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Z"/>
</svg>
`;

export function createCommuteSceneButton({ selected, onToggle } = {}) {
  return createSceneButtonComponent({
    label: "通勤",
    icon: commuteIcon,
    accent: "#9ad0ff",
    selected,
    onToggle,
  });
}

