from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from ml.model_loader import get_model

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


MODEL_NAME = "BAAI/bge-small-zh-v1.5"
INPUT_JSON = Path("data/books_enriched_v2.json")
OUT_EMB = Path("data/book_embeddings_v2.npy")
OUT_INDEX = Path("data/books_vector_index_v2.json")


def _clean_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def build_retrieval_book_text(book: Dict[str, Any]) -> str:
    title = _clean_text(book.get("title"))
    author = _clean_text(book.get("author"))
    era_nat = _clean_text(book.get("author_era_nationality"))
    intro = _clean_text(book.get("content_intro"))
    pages = _clean_text(book.get("pages"))

    mood_tags = ", ".join(book.get("mood_tags") or [])
    scene_tags = ", ".join(book.get("scene_tags") or [])
    style_tags = ", ".join(book.get("style_tags") or [])
    difficulty = _clean_text(book.get("difficulty"))
    pace = _clean_text(book.get("pace"))
    length_type = _clean_text(book.get("length_type"))

    # Keep deterministic, information-dense text for embedding.
    return "\n".join(
        [
            f"书名：{title}",
            f"作者：{author}",
            f"作者国籍/年代：{era_nat}",
            f"内容简介：{intro}",
            f"页数：{pages}",
            f"情绪标签：{mood_tags}",
            f"阅读场景：{scene_tags}",
            f"风格标签：{style_tags}",
            f"难度：{difficulty}",
            f"节奏：{pace}",
            f"篇幅：{length_type}",
        ]
    ).strip()


def main() -> int:
    if not INPUT_JSON.exists():
        raise SystemExit(f"input not found: {INPUT_JSON}")

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    books: List[Dict[str, Any]] = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    if not isinstance(books, list):
        raise SystemExit("input json must be a list")

    texts: List[str] = []
    index_items: List[Dict[str, str]] = []

    for i, b in enumerate(books):
        texts.append(build_retrieval_book_text(b))
        index_items.append(
            {
                "index": i,
                "id": _clean_text(b.get("subject_id")),
                "title": _clean_text(b.get("title")),
                "author": _clean_text(b.get("author")),
            }
        )

    print(f"[emb_v2] model = {MODEL_NAME}")
    print(f"[emb_v2] input_books = {len(books)}")
    print(f"[emb_v2] output = {OUT_EMB} , {OUT_INDEX}")

    model = get_model()
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    emb = normalize(emb, norm="l2")
    emb = emb.astype(np.float32, copy=False)

    np.save(OUT_EMB, emb)
    OUT_INDEX.write_text(json.dumps(index_items, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[emb_v2] saved embeddings shape = {emb.shape}, dtype = {emb.dtype}")
    return 0


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    raise SystemExit(main())

