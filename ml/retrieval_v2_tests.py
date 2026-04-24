from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Allow running as a plain script: python ml/retrieval_v2_tests.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.retrieval_v2 import RetrievalEngineV2, _l2_normalize  # noqa: E402


class KeywordEmbedder:
    """
    一个完全确定性的 mock embedder：
    - 文本包含某些关键词 -> 触发某个维度为 1
    - 其余为 0
    - 最终做 L2 normalize，便于直接点积当 cosine
    """

    def __init__(self):
        # dim0: base(默认) ; dim1: mood平静 ; dim2: scene通勤 ; dim3: style科幻
        self.dim = 4

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            s = (t or "").lower()
            out[i, 0] = 1.0
            if ("平静" in s) or ("助眠" in s) or ("安静" in s):
                out[i, 1] = 1.0
            if ("通勤" in s) or ("地铁" in s) or ("公交" in s):
                out[i, 2] = 1.0
            if ("科幻" in s) or ("宇宙" in s) or ("外星" in s) or ("未来" in s):
                out[i, 3] = 1.0
        return _l2_normalize(out)


def _make_engine() -> RetrievalEngineV2:
    # 4 本书，embedding 维度=4；每本书偏向不同维度
    E = np.array(
        [
            [1, 1, 0, 0],  # B0: 平静
            [1, 0, 1, 0],  # B1: 通勤
            [1, 0, 0, 1],  # B2: 科幻
            [1, 0, 0, 0],  # B3: 泛化
        ],
        dtype=np.float32,
    )
    E = _l2_normalize(E)

    books = [
        {"subject_id": "b0", "title": "平静之书", "difficulty": 3, "pace": 3, "length_type": "中篇"},
        {"subject_id": "b1", "title": "通勤之书", "difficulty": 3, "pace": 3, "length_type": "中篇"},
        {"subject_id": "b2", "title": "科幻之书", "difficulty": 3, "pace": 3, "length_type": "中篇"},
        {"subject_id": "b3", "title": "泛化之书", "difficulty": 3, "pace": 3, "length_type": "中篇"},
    ]
    index_items = [
        {"index": 0, "id": "b0", "title": "平静之书", "author": ""},
        {"index": 1, "id": "b1", "title": "通勤之书", "author": ""},
        {"index": 2, "id": "b2", "title": "科幻之书", "author": ""},
        {"index": 3, "id": "b3", "title": "泛化之书", "author": ""},
    ]

    return RetrievalEngineV2(embedder=KeywordEmbedder(), book_embeddings=E, index_items=index_items, books=books)


def _intent(
    *,
    mood: List[Dict[str, Any]] | None = None,
    scene: List[Dict[str, Any]] | None = None,
    style: List[Dict[str, Any]] | None = None,
    difficulty: int = 3,
    pace: int = 3,
    length_type: str = "中篇",
) -> Dict[str, Any]:
    return {
        "mood_tags": mood or [],
        "scene_tags": scene or [],
        "style_tags": style or [],
        "difficulty": {"value": difficulty, "confidence": 0.6},
        "pace": {"value": pace, "confidence": 0.6},
        "length_type": {"value": length_type, "confidence": 0.6},
    }


def test_1_single_query_low_intent() -> None:
    engine = _make_engine()
    intent = _intent()
    res = engine.retrieve("随便推荐一本", intent, top_k=20)
    assert isinstance(res, list)
    assert len(res) == 4  # 只有 4 本书
    for x in res:
        assert "book_id" in x and "title" in x and "score" in x and "score_breakdown" in x
        bd = x["score_breakdown"]
        assert set(["query_score", "mood_score", "scene_score", "style_score", "penalty"]).issubset(bd.keys())


def test_2_strong_mood_affects_ranking() -> None:
    engine = _make_engine()
    intent = _intent(mood=[{"tag": "平静", "confidence": 0.9}])
    res = engine.retrieve("想要睡前更平静", intent, top_k=3)
    assert res[0]["book_id"] == "b0"
    assert res[0]["score_breakdown"]["mood_score"] >= res[1]["score_breakdown"]["mood_score"]


def test_3_strong_scene_affects_ranking() -> None:
    engine = _make_engine()
    intent = _intent(scene=[{"tag": "通勤", "confidence": 0.9}])
    res = engine.retrieve("通勤路上读什么", intent, top_k=3)
    assert res[0]["book_id"] == "b1"
    assert res[0]["score_breakdown"]["scene_score"] >= res[1]["score_breakdown"]["scene_score"]


def test_4_multi_tags_fusion_breakdown_present() -> None:
    engine = _make_engine()
    intent = _intent(
        mood=[{"tag": "平静", "confidence": 0.8}],
        scene=[{"tag": "通勤", "confidence": 0.7}],
        style=[{"tag": "科幻", "confidence": 0.9}],
    )
    res = engine.retrieve("想要平静但又想看点科幻，通勤读", intent, top_k=4)
    assert len(res) == 4
    for x in res:
        bd = x["score_breakdown"]
        assert "weighted" in bd and "penalty" in bd
        w = bd["weighted"]
        assert set(["query", "mood", "scene", "style"]).issubset(w.keys())
        p = bd["penalty"]
        assert set(["difficulty", "pace", "length", "total"]).issubset(p.keys())
        assert math.isfinite(float(x["score"]))


def test_5_penalty_applies_on_mismatch() -> None:
    engine = _make_engine()
    # 目标难度=5，书的难度=3 -> diff=2 => -0.2
    intent = _intent(difficulty=5, pace=3, length_type="中篇")
    res = engine.retrieve("随便推荐一本", intent, top_k=1)
    bd = res[0]["score_breakdown"]
    assert abs(float(bd["penalty"]["difficulty"]) - (-0.2)) < 1e-6
    assert abs(float(bd["penalty"]["total"]) - (-0.2)) < 1e-6


def main() -> None:
    test_1_single_query_low_intent()
    test_2_strong_mood_affects_ranking()
    test_3_strong_scene_affects_ranking()
    test_4_multi_tags_fusion_breakdown_present()
    test_5_penalty_applies_on_mismatch()
    print("[retrieval_v2_tests] OK")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()

