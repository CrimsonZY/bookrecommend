import { createSceneButtonComponent } from "../SceneButtonComponent.js";

const moonIcon = `
<svg viewBox="0 0 24 24" width="24" height="24" role="img" focusable="false" aria-hidden="true">
  <path fill="currentColor" d="M12.2 3.25a.75.75 0 0 1 .66.96 7.25 7.25 0 0 0 8.93 9.15.75.75 0 0 1 .9.93A9.25 9.25 0 1 1 12.2 3.25Zm-2.06 2.3a7.75 7.75 0 1 0 10.2 10.5A8.75 8.75 0 0 1 10.14 5.55Z"/>
</svg>
`;

export function createBedtimeSceneButton({ selected, onToggle } = {}) {
  return createSceneButtonComponent({
    label: "睡前",
    icon: moonIcon,
    accent: "#d7c4ff",
    selected,
    onToggle,
  });
}

