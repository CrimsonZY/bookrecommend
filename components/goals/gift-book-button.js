import { createButtonComponent } from "../ButtonComponent.js";

const giftIcon = `
<svg viewBox="0 0 24 24" width="24" height="24" role="img" focusable="false" aria-hidden="true">
  <path fill="currentColor" d="M20 7h-1.1c.38-.44.6-1 .6-1.6A2.4 2.4 0 0 0 17.1 3c-1.06 0-1.95.54-2.53 1.25-.42.52-.8 1.14-1.07 1.64-.27-.5-.65-1.12-1.07-1.64C11.86 3.54 10.96 3 9.9 3A2.4 2.4 0 0 0 7.5 5.4c0 .6.22 1.16.6 1.6H7a2 2 0 0 0-2 2v2.25c0 .41.34.75.75.75H11v7h2v-7h5.25c.41 0 .75-.34.75-.75V9a2 2 0 0 0-2-2Zm-9.16-1.6c-.28-.35-.7-.9-1.1-1.22-.26-.2-.55-.28-.84-.28-.5 0-.9.4-.9.9s.4.9.9.9h1.94ZM15.16 7h-1.94c.28-.35.7-.9 1.1-1.22.26-.2.55-.28.84-.28.5 0 .9.4.9.9s-.4.9-.9.9ZM6.5 9c0-.28.22-.5.5-.5h4v2.25H6.5V9Zm11 1.75H13V8.5h4c.28 0 .5.22.5.5v1.75Z"/>
</svg>
`;

export function createGiftBookButton({ onClick } = {}) {
  return createButtonComponent({
    title: "送一本书",
    subtitle: "送给特别的TA",
    icon: giftIcon,
    accent: "#ffcf9a",
    onClick,
  });
}

