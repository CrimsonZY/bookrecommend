import { createButtonComponent } from "../ButtonComponent.js";

const newBookIcon = `
<svg viewBox="0 0 24 24" width="24" height="24" role="img" focusable="false" aria-hidden="true">
  <path fill="currentColor" d="M6 4c-1.38 0-2.5 1.12-2.5 2.5v12.25c0 .97.78 1.75 1.75 1.75H18a2.5 2.5 0 0 0 2.5-2.5V6.5C20.5 5.12 19.38 4 18 4H6Zm0 2H18c.28 0 .5.22.5.5V18a.5.5 0 0 1-.5.5H6a2.5 2.5 0 0 0-.5.05V6.5c0-.28.22-.5.5-.5Zm5.25 2.25a.75.75 0 0 1 .75-.75h1v-1a.75.75 0 0 1 1.5 0v1h1a.75.75 0 0 1 0 1.5h-1v1a.75.75 0 0 1-1.5 0v-1h-1a.75.75 0 0 1-.75-.75ZM7.5 12.75a.75.75 0 0 1 .75-.75h8.5a.75.75 0 0 1 0 1.5h-8.5a.75.75 0 0 1-.75-.75Zm0 3a.75.75 0 0 1 .75-.75h8.5a.75.75 0 0 1 0 1.5h-8.5a.75.75 0 0 1-.75-.75Z"/>
</svg>
`;

export function createExploreNewBooksButton({ onClick } = {}) {
  return createButtonComponent({
    title: "探索新书",
    subtitle: "探索新书",
    icon: newBookIcon,
    accent: "#a8e6b1",
    onClick,
  });
}

