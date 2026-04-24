from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.preprocessing import normalize

from ml.model_loader import get_model

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
INPUT_JSON = Path("data/douban_top100_clean_refilled.json")
OUT_EMB = Path("data/book_embeddings.npy")
OUT_INDEX = Path("data/books_vector_index.json")


DYNASTIES = {
    "夏",
    "商",
    "周",
    "秦",
    "汉",
    "西汉",
    "东汉",
    "三国",
    "魏",
    "蜀",
    "吴",
    "晋",
    "西晋",
    "东晋",
    "南北朝",
    "隋",
    "唐",
    "五代",
    "宋",
    "北宋",
    "南宋",
    "元",
    "明",
    "清",
    "民国",
    "当代",
    "现代",
}


def split_author_era_nationality(value: str) -> Tuple[str, str]:
    """
    Input field `author_era_nationality` in this project is a single string.
    Heuristic split:
      - if it matches known dynasties => era
      - otherwise => nationality
    """
    v = (value or "").strip()
    if not v:
        return "", ""
    if v in DYNASTIES:
        return "", v
    return v, ""


def build_recommend_text(book: Dict) -> str:
    title = (book.get("title") or "").strip()
    author = (book.get("author") or "").strip()
    era_nat = (book.get("author_era_nationality") or "").strip()
    nationality, era = split_author_era_nationality(era_nat)
    intro = (book.get("content_intro") or "").strip()
    pages = book.get("pages")

    # Keep it robust even if pages is not int.
    pages_str = str(pages).strip() if pages is not None else ""

    return "\n".join(
        [
            f"书名：{title}",
            f"作者：{author}",
            f"作者国籍：{nationality}",
            f"作者年代：{era}",
            f"内容简介：{intro}",
            f"页数：{pages_str}",
        ]
    ).strip()


def main() -> int:
    if not INPUT_JSON.exists():
        raise SystemExit(f"input not found: {INPUT_JSON}")

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    books: List[Dict] = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    if not isinstance(books, list):
        raise SystemExit("input json must be a list")

    texts: List[str] = []
    index_items: List[Dict] = []

    for i, b in enumerate(books):
        texts.append(build_recommend_text(b))
        index_items.append(
            {
                "index": i,
                "id": (b.get("subject_id") or "").strip(),
                "title": (b.get("title") or "").strip(),
                "author": (b.get("author") or "").strip(),
            }
        )

    print(f"[embedding] model = {MODEL_NAME}")
    print(f"[embedding] input_books = {len(books)}")
    print(f"[embedding] output = {OUT_EMB} , {OUT_INDEX}")

    # Enforce singleton loader to avoid accidental duplicate instances.
    model = get_model()
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Normalize for cosine similarity search by default.
    emb = normalize(emb, norm="l2")
    emb = emb.astype(np.float32, copy=False)

    np.save(OUT_EMB, emb)
    OUT_INDEX.write_text(json.dumps(index_items, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[embedding] saved embeddings shape = {emb.shape}, dtype = {emb.dtype}")
    return 0


if __name__ == "__main__":
    # Make relative paths stable when invoked from anywhere.
    os.chdir(Path(__file__).resolve().parents[1])
    raise SystemExit(main())

