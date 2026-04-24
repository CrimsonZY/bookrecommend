/** @deprecated 已整合到首页单页流程（scripts/main.js）。 */

const main = document.querySelector(".page-scene-selection .app-main");

if (main) {
  const p = document.createElement("p");
  p.textContent = "该页面已整合到首页流程，正在跳转…";
  main.replaceChildren(p);
}

location.replace("../index.html#style-more");
