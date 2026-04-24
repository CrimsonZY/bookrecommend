from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from ml.diversity_selector_v1 import DiversitySelectorV1
from ml.intent_tags_v2 import SentenceTransformerEmbedder, TagEnricherV2
from ml.reranker_v1 import RerankerV1
from ml.retrieval_v2 import RetrievalEngineV2

from ml.system_config import load_system_config

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_BOOKS = BASE_DIR / "data/books_enriched_v2.json"


@dataclass
class RecommendResult:
    book_id: str
    title: str
    author: str
    cover_url: str
    summary: str
    book_enriched: Dict[str, Any]
    score: float


class RecommendService:
    """
    Lazy-load, request-safe (single-process) recommendation pipeline.
    """

    _instance: Optional["RecommendService"] = None

    def __init__(self):
        # Must stay lightweight for low-memory environments (e.g. Render 512MB).
        # Heavy objects (SentenceTransformer, embeddings, full book json) must be lazy-loaded.
        self._cfg = load_system_config()
        self._embedder: Optional[SentenceTransformerEmbedder] = None
        self._intent: Optional[TagEnricherV2] = None
        self._retrieval: Optional[RetrievalEngineV2] = None
        self._reranker: Optional[RerankerV1] = None
        self._diversity: Optional[DiversitySelectorV1] = None
        self._book_map: Optional[Dict[str, Dict[str, Any]]] = None

    def _get_embedder(self) -> SentenceTransformerEmbedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder()
        return self._embedder

    def _get_intent(self) -> TagEnricherV2:
        if self._intent is None:
            self._intent = TagEnricherV2(embedder=self._get_embedder(), embedding_threshold=0.5)
        return self._intent

    def _get_retrieval(self) -> RetrievalEngineV2:
        if self._retrieval is None:
            self._retrieval = RetrievalEngineV2()
        return self._retrieval

    def _get_reranker(self) -> RerankerV1:
        if self._reranker is None:
            self._reranker = RerankerV1()
        return self._reranker

    def _get_diversity(self) -> DiversitySelectorV1:
        if self._diversity is None:
            self._diversity = DiversitySelectorV1()
        return self._diversity

    def _get_book_map(self) -> Dict[str, Dict[str, Any]]:
        if self._book_map is None:
            self._book_map = self._load_book_map()
        return self._book_map

    def _dbg_emit(self, message: str, data: Dict[str, Any]) -> None:
        if not self._cfg.debug_mode:
            return
        if not self._cfg.debug_emit_stdout_ndjson():
            return
        try:
            payload = {
                "ts_ms": int(time.time() * 1000),
                "system_version": self._cfg.system_version,
                "message": str(message),
                "data": data,
            }
            print(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass

    def _slim_candidate(self, x: Dict[str, Any], *, meta_fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = x.get("metadata") if isinstance(x.get("metadata"), dict) else (meta_fallback or {})
        return {
            "book_id": str(x.get("book_id") or ""),
            "score": float(x.get("score") or 0.0),
            "embedding_score": float(x.get("embedding_score") or 0.0),
            "final_score": float(x.get("final_score") or 0.0),
            "metadata": {
                "mood_tags": list(meta.get("mood_tags") or []),
                "scene_tags": list(meta.get("scene_tags") or []),
                "style_tags": list(meta.get("style_tags") or []),
                "difficulty": int(meta.get("difficulty") or 0),
                "pace": int(meta.get("pace") or 0),
                "length_type": str(meta.get("length_type") or ""),
            },
            "score_breakdown": x.get("score_breakdown") if isinstance(x.get("score_breakdown"), dict) else None,
            "rerank_breakdown": x.get("rerank_breakdown") if isinstance(x.get("rerank_breakdown"), dict) else None,
            "selection_reason": str(x.get("selection_reason") or ""),
        }

    @classmethod
    def get(cls) -> "RecommendService":
        if cls._instance is None:
            cls._instance = RecommendService()
        return cls._instance

    def _load_book_map(self) -> Dict[str, Dict[str, Any]]:
        if not DATA_BOOKS.exists():
            raise FileNotFoundError(f"missing data file: {DATA_BOOKS}")
        data = json.loads(DATA_BOOKS.read_text(encoding="utf-8"))
        m: Dict[str, Dict[str, Any]] = {}
        for b in data:
            if not isinstance(b, dict):
                continue
            sid = str(b.get("subject_id") or "").strip()
            if sid:
                m[sid] = b
        return m

    def _exclude_set(self, exclude_book_ids: Optional[Iterable[str]]) -> Set[str]:
        out: Set[str] = set()
        for x in exclude_book_ids or []:
            s = str(x or "").strip()
            if s:
                out.add(s)
        return out

    def recommend(
        self,
        query: str,
        top_k: int = 3,
        *,
        exclude_book_ids: Optional[Iterable[str]] = None,
    ) -> List[RecommendResult]:
        q = (query or "").strip()
        if not q:
            return []

        excludes = self._exclude_set(exclude_book_ids)
        self._dbg_emit(
            "debug_mode_on",
            {"query": q[:200], "top_k": int(top_k), "exclude_book_ids": sorted(list(excludes))[:50]},
        )
        intent = self._get_intent().parse(q)
        self._dbg_emit("intent", {"intent": intent})

        # Retrieve with escalating top_k to survive hard excludes on small dataset.
        retrieval_ks = [20, 50, 100]
        top20: List[Dict[str, Any]] = []
        for k in retrieval_ks:
            cands = self._get_retrieval().retrieve(q, intent, top_k=int(k))
            if excludes:
                cands = [c for c in cands if str(c.get("book_id") or "").strip() not in excludes]
            top20 = cands
            if len(top20) >= 20 or k == retrieval_ks[-1]:
                break

        self._dbg_emit("retrieval_top20", {"items": [self._slim_candidate(x) for x in top20[:20]]})

        top10 = self._get_reranker().rerank(q, intent, top20, top_k=10)
        if excludes:
            top10 = [c for c in top10 if str(c.get("book_id") or "").strip() not in excludes]
        self._dbg_emit("rerank_top10", {"items": [self._slim_candidate(x) for x in top10[:10]]})

        topN = self._get_diversity().select(q, intent, top10, top_k=top_k)
        if excludes:
            topN = [c for c in topN if str(c.get("book_id") or "").strip() not in excludes]

        # Exhaustion policy: hard not-enough OR soft low-final-score.
        need_k = max(1, int(top_k))
        if len(topN) < need_k:
            raise ExhaustedRecommendationError("not_enough_candidates")
        thr = float(self._cfg.refresh_min_final_score() or 0.0)
        if thr > 0.0:
            top1 = float(topN[0].get("final_score") or 0.0)
            topk_score = float(topN[need_k - 1].get("final_score") or 0.0) if len(topN) >= need_k else 0.0
            if top1 < thr or topk_score < thr:
                raise ExhaustedRecommendationError("score_too_low")

        items = []
        for x in topN[: max(1, int(top_k))]:
            bid = str(x.get("book_id") or "").strip()
            meta_fb = {}
            if bid:
                meta_fb = self._get_book_map().get(bid, {}) or {}
            items.append(self._slim_candidate(x, meta_fallback=meta_fb))
        self._dbg_emit("diversity_selected", {"items": items})

        out: List[RecommendResult] = []
        for x in topN:
            bid = str(x.get("book_id") or "").strip()
            enriched = self._get_book_map().get(bid, {})
            title = str(enriched.get("title") or "")
            author = str(enriched.get("author") or "")
            cover = str(enriched.get("cover_image") or "")
            intro = str(enriched.get("content_intro") or "")
            summary = intro[:300] if intro else ""
            out.append(
                RecommendResult(
                    book_id=bid,
                    title=title,
                    author=author,
                    cover_url=cover,
                    summary=summary,
                    book_enriched=enriched,
                    score=float(x.get("final_score") or 0.0),
                )
            )
        return out


class ExhaustedRecommendationError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = str(reason or "")


def timed_recommend(query: str, top_k: int, *, exclude_book_ids: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    svc = RecommendService.get()
    books = svc.recommend(query, top_k=top_k, exclude_book_ids=exclude_book_ids)
    ms = (time.perf_counter() - t0) * 1000.0
    return {
        "elapsed_ms": ms,
        "books": books,
    }

