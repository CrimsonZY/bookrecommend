from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from ml.model_loader import get_model

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMB_PATH = Path("data/book_embeddings_v2.npy")
INDEX_PATH = Path("data/books_vector_index_v2.json")
BOOKS_PATH = Path("data/books_enriched_v2.json")


@dataclass(frozen=True)
class BookRecommendation:
    rank: int
    score: float
    subject_id: str
    title: str
    author: str
    publisher: str
    content_intro: str
    cover_image: str


_MODEL: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        # sentence-transformers uses HF cache; if already cached, it won't re-download.
        _MODEL = get_model()
    return _MODEL


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vec, axis=-1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return vec / denom


def _load_embeddings() -> np.ndarray:
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"missing embeddings: {EMB_PATH}")
    emb = np.load(EMB_PATH)
    if emb.ndim != 2:
        raise ValueError(f"invalid embeddings shape: {emb.shape}")
    # Expect already normalized float32 from training, but keep it robust.
    emb = emb.astype(np.float32, copy=False)
    emb = _l2_normalize(emb)
    return emb


def _load_index_items() -> List[Dict[str, Any]]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"missing index: {INDEX_PATH}")
    items = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("books_vector_index.json must be a list")
    return items


def _load_books_by_id() -> Dict[str, Dict[str, Any]]:
    if not BOOKS_PATH.exists():
        raise FileNotFoundError(f"missing books data: {BOOKS_PATH}")
    books = json.loads(BOOKS_PATH.read_text(encoding="utf-8"))
    if not isinstance(books, list):
        raise ValueError("douban_top100_clean_refilled.json must be a list")
    by_id: Dict[str, Dict[str, Any]] = {}
    for b in books:
        if not isinstance(b, dict):
            continue
        sid = str(b.get("subject_id") or "").strip()
        if sid:
            by_id[sid] = b
    return by_id


def recommend_books(user_prompt: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Return a list of book dicts (for FastAPI usage).
    Each item includes: title, author, content_intro, cover_image, publisher, plus score/id.
    """
    prompt = (user_prompt or "").strip()
    if not prompt:
        raise ValueError("user_prompt 不能为空")
    if top_k <= 0:
        return []

    model = _get_model()
    emb_books = _load_embeddings()
    index_items = _load_index_items()
    books_by_id = _load_books_by_id()

    q = model.encode([prompt], convert_to_numpy=True, show_progress_bar=False)
    q = q.astype(np.float32, copy=False)
    q = _l2_normalize(q)[0]  # (dim,)

    # Cosine similarity for normalized vectors is dot product.
    scores = emb_books @ q  # (n,)
    k = min(int(top_k), scores.shape[0])
    top_idx = np.argsort(-scores)[:k]

    results: List[Dict[str, Any]] = []
    for rank, i in enumerate(top_idx, start=1):
        i_int = int(i)
        score = float(scores[i_int])

        item = index_items[i_int] if i_int < len(index_items) else {}
        sid = str(item.get("id") or "").strip()

        book = books_by_id.get(sid, {})
        results.append(
            {
                "rank": rank,
                "score": score,
                "subject_id": sid,
                "title": (book.get("title") or item.get("title") or "").strip(),
                "author": (book.get("author") or item.get("author") or "").strip(),
                "publisher": (book.get("publisher") or "").strip(),
                "content_intro": (book.get("content_intro") or "").strip(),
                "cover_image": (book.get("cover_image") or "").strip(),
            }
        )

    return results


def _pretty_print(recs: List[Dict[str, Any]]) -> None:
    if not recs:
        print("没有推荐结果。")
        return

    print("\n=== Top 推荐结果 ===\n")
    for r in recs:
        rank = r.get("rank")
        score = r.get("score")
        title = r.get("title") or ""
        author = r.get("author") or ""
        publisher = r.get("publisher") or ""
        intro = r.get("content_intro") or ""
        cover = r.get("cover_image") or ""
        sid = r.get("subject_id") or ""

        intro_short = intro.replace("\n", " ").strip()
        if len(intro_short) > 160:
            intro_short = intro_short[:160] + "…"

        print(f"{rank}. {title}")
        print(f"   作者：{author}")
        if publisher:
            print(f"   出版社：{publisher}")
        print(f"   相似度：{float(score):.4f}   id={sid}")
        print(f"   简介：{intro_short}")
        if cover:
            print(f"   封面：{cover}")
        print()


def main() -> int:
    # Best-effort UTF-8 output on Windows terminals.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("请输入你的中文需求（直接回车将使用示例）：")
    try:
        prompt = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。")
        return 0

    if not prompt:
        prompt = "最近焦虑，睡前想读点温柔的书"

    recs = recommend_books(prompt, top_k=3)
    _pretty_print(recs)
    return 0


if __name__ == "__main__":
    # Make relative paths stable when invoked from anywhere.
    os.chdir(Path(__file__).resolve().parents[1])
    raise SystemExit(main())

