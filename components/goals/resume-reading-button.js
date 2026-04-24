import { createButtonComponent } from "../ButtonComponent.js";

const bookIcon = `
<svg viewBox="0 0 24 24" width="24" height="24" role="img" focusable="false" aria-hidden="true">
  <path fill="currentColor" d="M6 3.5c-1.38 0-2.5 1.12-2.5 2.5v13c0 .83.67 1.5 1.5 1.5H18a2.5 2.5 0 0 0 2.5-2.5V6c0-1.38-1.12-2.5-2.5-2.5H6Zm0 2H18c.28 0 .5.22.5.5v12a.5.5 0 0 1-.5.5H6a2.5 2.5 0 0 0-.5.05V6c0-.28.22-.5.5-.5Zm1.5 2.25a.75.75 0 0 1 .75-.75h8.5a.75.75 0 0 1 0 1.5h-8.5a.75.75 0 0 1-.75-.75Zm0 3a.75.75 0 0 1 .75-.75h8.5a.75.75 0 0 1 0 1.5h-8.5a.75.75 0 0 1-.75-.75Z"/>
</svg>
`;

export function createResumeReadingButton({ onClick } = {}) {
  return createButtonComponent({
    title: "恢复阅读",
    subtitle: "继续阅读",
    icon: bookIcon,
    accent: "#9ad0ff",
    onClick,
  });
}

