from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ml.system_config import load_system_config

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

EMB_PATH = Path("data/book_embeddings_v2.npy")
INDEX_PATH = Path("data/books_vector_index_v2.json")
BOOKS_PATH = Path("data/books_enriched_v2.json")


MOOD_CANDIDATES = ["治愈", "温暖", "平静", "热血", "沉重", "悲伤", "哲思", "孤独", "希望", "幽默", "浪漫"]
SCENE_CANDIDATES = ["睡前", "通勤", "周末下午", "长期阅读", "旅行途中", "碎片阅读"]
STYLE_CANDIDATES = ["文学", "小说", "社会观察", "悬疑", "科幻", "散文", "历史", "爱情", "人物传记", "哲学思考"]
LENGTH_TYPES = ["短篇", "中篇", "长篇", "超长篇"]


TAG_DESCRIPTIONS: Dict[str, str] = {
    # mood
    "治愈": "治愈、疗愈、抚慰、放松、助眠",
    "温暖": "温暖、温柔、善意、陪伴、亲情友情",
    "平静": "平静、安静、宁静、舒缓、从容",
    "热血": "热血、冒险、战斗、燃、强情节推进",
    "沉重": "沉重、压抑、残酷现实、创伤",
    "悲伤": "悲伤、哀伤、离别、失去",
    "哲思": "哲思、思辨、反思、意义、存在",
    "孤独": "孤独、寂寞、疏离、自我对话",
    "希望": "希望、救赎、重生、勇气",
    "幽默": "幽默、搞笑、讽刺、荒诞",
    "浪漫": "浪漫、爱情、心动",
    # scene
    "睡前": "睡前阅读，助眠、放松、温柔治愈",
    "通勤": "通勤阅读，地铁公交，短小易中断继续",
    "周末下午": "周末下午，沉浸式阅读，中长篇故事性强",
    "长期阅读": "长期阅读，长篇大部头、经典、信息量大",
    "旅行途中": "旅行途中，轻松或冒险，故事推进强",
    "碎片阅读": "碎片阅读，短篇、随笔散文，随时读一点",
    # style
    "文学": "文学性强、严肃叙事、经典作品等",
    "小说": "以故事叙事为主的虚构作品",
    "社会观察": "社会观察，现实题材、阶层与制度",
    "悬疑": "悬疑推理，命案、谜团、反转",
    "科幻": "科幻，宇宙、未来、外星、科技文明",
    "散文": "散文随笔，片段式、抒情",
    "历史": "历史题材，王朝、时代变迁",
    "爱情": "爱情题材，恋爱、婚姻、浪漫",
    "人物传记": "人物传记，自传回忆录、真实人物",
    "哲学思考": "哲学思考，存在与意义、伦理思想",
}


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return x / denom


class QueryEmbedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = str(model_name or MODEL_NAME)
        self.model = None

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        self._ensure_model()
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        emb = emb.astype(np.float32, copy=False)
        return _l2_normalize(emb)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class TagQueryExpander:
    def __init__(self, tag_descriptions: Dict[str, str]):
        self.tag_descriptions = tag_descriptions

    def _expand(self, prefix: str, tags: List[Dict[str, Any]], max_tags: int = 3) -> str:
        # tags: [{"tag": "...", "confidence": 0.x}]
        items = []
        for x in tags or []:
            t = str(x.get("tag") or "").strip()
            conf = float(x.get("confidence") or 0.0)
            if not t:
                continue
            desc = self.tag_descriptions.get(t, t)
            items.append((t, conf, desc))
        items.sort(key=lambda y: y[1], reverse=True)
        items = items[:max_tags]
        if not items:
            return ""
        joined = "；".join([f"{t}（{desc}）" for t, _, desc in items])
        return f"{prefix}：{joined}"

    def build_queries(self, query: str, intent: Dict[str, Any]) -> Dict[str, str]:
        base = query.strip()
        mood = self._expand("情绪", intent.get("mood_tags") or [])
        scene = self._expand("场景", intent.get("scene_tags") or [])
        style = self._expand("风格", intent.get("style_tags") or [])
        return {
            "base": base,
            "mood": mood or base,
            "scene": scene or base,
            "style": style or base,
        }


class MultiVectorScorer:
    def __init__(self, book_embeddings: np.ndarray):
        self.E = _l2_normalize(book_embeddings.astype(np.float32, copy=False))

    def score(self, q_base: np.ndarray, q_mood: np.ndarray, q_scene: np.ndarray, q_style: np.ndarray) -> Dict[str, np.ndarray]:
        # cosine similarity for normalized vectors = dot product
        return {
            "query_score": self.E @ q_base,
            "mood_score": self.E @ q_mood,
            "scene_score": self.E @ q_scene,
            "style_score": self.E @ q_style,
        }


class ScoreAggregator:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "query_score": 0.4,
            "mood_score": 0.2,
            "scene_score": 0.2,
            "style_score": 0.2,
        }

    def _difficulty_penalty(self, want: int, have: int) -> float:
        diff = abs(int(want) - int(have))
        if diff == 0:
            return 0.0
        if diff == 1:
            return -0.1
        if diff == 2:
            return -0.2
        return -0.3

    def _pace_penalty(self, want: int, have: int) -> float:
        diff = abs(int(want) - int(have))
        if diff == 0:
            return 0.0
        if diff == 1:
            return -0.1
        return -0.2

    def _length_penalty(self, want: str, have: str) -> float:
        if not want or want not in LENGTH_TYPES:
            return 0.0
        if want == have:
            return 0.0
        return -0.1

    def aggregate(
        self,
        sims: Dict[str, np.ndarray],
        books: List[Dict[str, Any]],
        intent: Dict[str, Any],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        # base weighted score
        wq = self.weights["query_score"]
        wm = self.weights["mood_score"]
        ws = self.weights["scene_score"]
        wst = self.weights["style_score"]

        base = wq * sims["query_score"] + wm * sims["mood_score"] + ws * sims["scene_score"] + wst * sims["style_score"]

        want_diff = int(intent.get("difficulty", {}).get("value", 3) or 3)
        want_pace = int(intent.get("pace", {}).get("value", 3) or 3)
        want_len = str(intent.get("length_type", {}).get("value", "") or "")

        breakdowns: List[Dict[str, Any]] = []
        final = np.array(base, dtype=np.float32, copy=True)

        for i, b in enumerate(books):
            have_diff = int(b.get("difficulty", 3) or 3)
            have_pace = int(b.get("pace", 3) or 3)
            have_len = str(b.get("length_type", "") or "")

            p_diff = self._difficulty_penalty(want_diff, have_diff)
            p_pace = self._pace_penalty(want_pace, have_pace)
            p_len = self._length_penalty(want_len, have_len)

            penalty = p_diff + p_pace + p_len
            final[i] = float(final[i] + penalty)

            breakdowns.append(
                {
                    "query_score": float(sims["query_score"][i]),
                    "mood_score": float(sims["mood_score"][i]),
                    "scene_score": float(sims["scene_score"][i]),
                    "style_score": float(sims["style_score"][i]),
                    "weighted": {
                        "query": float(wq * sims["query_score"][i]),
                        "mood": float(wm * sims["mood_score"][i]),
                        "scene": float(ws * sims["scene_score"][i]),
                        "style": float(wst * sims["style_score"][i]),
                    },
                    "penalty": {
                        "difficulty": float(p_diff),
                        "pace": float(p_pace),
                        "length": float(p_len),
                        "total": float(penalty),
                    },
                }
            )

        return final, breakdowns


class RetrievalEngineV2:
    def __init__(
        self,
        *,
        embedder: Optional[QueryEmbedder] = None,
        book_embeddings: Optional[np.ndarray] = None,
        index_items: Optional[List[Dict[str, Any]]] = None,
        books: Optional[List[Dict[str, Any]]] = None,
        embeddings_path: Path = EMB_PATH,
        index_path: Path = INDEX_PATH,
        books_path: Path = BOOKS_PATH,
        weights: Optional[Dict[str, float]] = None,
    ):
        cfg = load_system_config()
        if weights is None:
            weights = cfg.retrieval_weights()

        self.embedder = embedder or QueryEmbedder()
        self.expander = TagQueryExpander(TAG_DESCRIPTIONS)
        self.aggregator = ScoreAggregator(weights=weights)

        self.book_embeddings = (
            book_embeddings.astype(np.float32, copy=False)
            if book_embeddings is not None
            else np.load(embeddings_path).astype(np.float32, copy=False)
        )
        self.index_items = (
            index_items if index_items is not None else json.loads(index_path.read_text(encoding="utf-8"))
        )
        self.books = books if books is not None else json.loads(books_path.read_text(encoding="utf-8"))

        # Build map by subject_id for metadata lookup
        self.books_by_id: Dict[str, Dict[str, Any]] = {}
        for b in self.books:
            if isinstance(b, dict):
                sid = str(b.get("subject_id") or "").strip()
                if sid:
                    self.books_by_id[sid] = b

        self.scorer = MultiVectorScorer(self.book_embeddings)

    def retrieve(self, query: str, intent: Dict[str, Any], top_k: int = 20) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        qs = self.expander.build_queries(q, intent or {})
        q_vecs = self.embedder.embed_texts([qs["base"], qs["mood"], qs["scene"], qs["style"]])
        q_base, q_mood, q_scene, q_style = q_vecs

        sims = self.scorer.score(q_base, q_mood, q_scene, q_style)
        scores, breakdowns = self.aggregator.aggregate(sims, self._aligned_books(), intent or {})

        k = min(int(top_k), scores.shape[0])
        order = np.argsort(-scores)[:k]

        out: List[Dict[str, Any]] = []
        aligned = self._aligned_books()
        for i in order:
            idx = int(i)
            b = aligned[idx]
            ev = self.book_embeddings[idx]
            weighted = breakdowns[idx]["weighted"]
            embedding_score = float(
                float(weighted.get("query", 0.0))
                + float(weighted.get("mood", 0.0))
                + float(weighted.get("scene", 0.0))
                + float(weighted.get("style", 0.0))
            )
            out.append(
                {
                    "book_id": str(b.get("subject_id") or ""),
                    "title": str(b.get("title") or ""),
                    "embedding_score": embedding_score,
                    "score": float(scores[idx]),
                    "embedding_vector": ev.astype(float, copy=False).tolist(),
                    "score_breakdown": {
                        "query_score": float(breakdowns[idx]["query_score"]),
                        "mood_score": float(breakdowns[idx]["mood_score"]),
                        "scene_score": float(breakdowns[idx]["scene_score"]),
                        "style_score": float(breakdowns[idx]["style_score"]),
                        "weighted": weighted,
                        "penalty": breakdowns[idx]["penalty"],
                    },
                    "metadata": {
                        "mood_tags": list(b.get("mood_tags") or []),
                        "scene_tags": list(b.get("scene_tags") or []),
                        "style_tags": list(b.get("style_tags") or []),
                        "difficulty": int(b.get("difficulty", 0) or 0),
                        "pace": int(b.get("pace", 0) or 0),
                        "length_type": str(b.get("length_type") or ""),
                    },
                }
            )
        return out

    def _aligned_books(self) -> List[Dict[str, Any]]:
        """
        Ensure order matches embedding rows using books_vector_index_v2.json.
        """
        aligned: List[Dict[str, Any]] = []
        for it in self.index_items:
            sid = str(it.get("id") or "").strip()
            aligned.append(self.books_by_id.get(sid, {"subject_id": sid, "title": it.get("title", "")}))
        return aligned


def demo() -> None:
    from ml.intent_tags_v2 import TagEnricherV2, SentenceTransformerEmbedder  # local import to avoid circular in tests

    enricher = TagEnricherV2(embedder=SentenceTransformerEmbedder(), embedding_threshold=0.5)
    engine = RetrievalEngineV2()
    q = "最近焦虑，睡前想读点温柔的书"
    intent = enricher.parse(q)
    res = engine.retrieve(q, intent, top_k=5)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    demo()

