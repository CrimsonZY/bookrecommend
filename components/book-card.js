/**
 * 书籍卡片组件
 * @param {object} book
 * @param {string} book.title
 * @param {string} [book.author]
 * @param {string} [book.coverUrl] — 对应 assets 下图片路径
 */
export function createBookCard(book = {}) {
  const article = document.createElement("article");
  article.className = "book-card";

  const title = document.createElement("h2");
  title.className = "book-card__title";
  title.textContent = book.title ?? "";

  article.appendChild(title);

  if (book.author) {
    const author = document.createElement("p");
    author.className = "book-card__author";
    author.textContent = book.author;
    article.appendChild(author);
  }

  if (book.coverUrl) {
    const img = document.createElement("img");
    img.className = "book-card__cover";
    img.src = book.coverUrl;
    img.alt = book.title ? `${book.title} 封面` : "";
    article.insertBefore(img, title);
  }

  return article;
}
