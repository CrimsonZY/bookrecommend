/**
 * 顶部导航栏（标题 + 副标题）
 */
export function createNavBar({
  title = "书籍推荐",
  subtitle = "选择你的目标，开始发现适合的书籍",
} = {}) {
  const wrap = document.createElement("div");
  wrap.className = "nav-bar";

  const h1 = document.createElement("h1");
  h1.className = "nav-bar__title";
  h1.textContent = title;

  const p = document.createElement("p");
  p.className = "nav-bar__subtitle";
  p.textContent = subtitle;

  wrap.append(h1, p);
  return wrap;
}

