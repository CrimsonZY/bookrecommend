from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from ml.reranker_v1 import RerankerV1  # noqa: E402


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


def _cand(
    book_id: str,
    embedding_score: float,
    *,
    mood_tags: List[str] | None = None,
    scene_tags: List[str] | None = None,
    style_tags: List[str] | None = None,
    difficulty: int = 3,
    pace: int = 3,
    length_type: str = "中篇",
) -> Dict[str, Any]:
    return {
        "book_id": book_id,
        "title": book_id,
        "embedding_score": float(embedding_score),
        "score_breakdown": {},
        "metadata": {
            "mood_tags": mood_tags or [],
            "scene_tags": scene_tags or [],
            "style_tags": style_tags or [],
            "difficulty": int(difficulty),
            "pace": int(pace),
            "length_type": str(length_type),
        },
    }


def test_1_embedding_high_but_tag_no_match() -> None:
    rr = RerankerV1()
    intent = _intent(mood=[{"tag": "平静", "confidence": 0.9}], style=[{"tag": "科幻", "confidence": 0.8}])
    c1 = _cand("A_high_embed_no_tag", 0.94, mood_tags=["热血"], style_tags=["悬疑"])
    c2 = _cand("B_mid_embed_good_tag", 0.65, mood_tags=["平静"], style_tags=["科幻"])
    res = rr.rerank("query", intent, [c1, c2], top_k=2)
    assert res[0]["book_id"] == "B_mid_embed_good_tag"


def test_2_tag_match_high_beats_mid_embedding() -> None:
    rr = RerankerV1()
    intent = _intent(mood=[{"tag": "治愈", "confidence": 0.8}], style=[{"tag": "散文", "confidence": 0.9}])
    c1 = _cand("A_mid_embed_good_tag", 0.60, mood_tags=["治愈"], style_tags=["散文"])
    c2 = _cand("B_higher_embed_bad_tag", 0.75, mood_tags=["沉重"], style_tags=["历史"])
    res = rr.rerank("query", intent, [c1, c2], top_k=2)
    assert res[0]["book_id"] == "A_mid_embed_good_tag"
    bd = res[0]["rerank_breakdown"]
    assert bd["tag_match_score"] > 0


def test_3_scene_strong_related() -> None:
    rr = RerankerV1()
    intent = _intent(scene=[{"tag": "通勤", "confidence": 0.9}])
    c1 = _cand("A_same_embed_scene_match", 0.60, scene_tags=["通勤"])
    c2 = _cand("B_same_embed_no_scene", 0.60, scene_tags=["睡前"])
    res = rr.rerank("query", intent, [c2, c1], top_k=2)
    assert res[0]["book_id"] == "A_same_embed_scene_match"
    assert res[0]["rerank_breakdown"]["scene_match_score"] >= 0.29


def test_4_difficulty_mismatch_penalty() -> None:
    rr = RerankerV1()
    intent = _intent(difficulty=5)
    c1 = _cand("A_higher_embed_diff_mismatch", 0.90, difficulty=3)  # abs>=1 -> penalty 0.15
    c2 = _cand("B_lower_embed_diff_match", 0.72, difficulty=5)  # penalty 0
    res = rr.rerank("query", intent, [c1, c2], top_k=2)
    assert res[0]["book_id"] == "B_lower_embed_diff_match"
    assert abs(float(res[1]["rerank_breakdown"]["penalty_score"]) - 0.15) < 1e-6


def test_5_mixed_conflicts_should_correct() -> None:
    rr = RerankerV1()
    intent = _intent(
        mood=[{"tag": "平静", "confidence": 0.9}],
        scene=[{"tag": "睡前", "confidence": 0.9}],
        style=[{"tag": "经典文学", "confidence": 0.9}],
        difficulty=3,
        pace=3,
        length_type="中篇",
    )
    # A: embedding 高但 scene 不匹配 + length 不匹配
    cA = _cand(
        "A_high_embed_bad_scene_length",
        0.92,
        mood_tags=["平静"],
        scene_tags=["通勤"],
        style_tags=["经典文学"],
        length_type="长篇",
    )
    # B: embedding 中等但 mood/style/scene 都匹配，无 penalty
    cB = _cand(
        "B_mid_embed_all_match",
        0.70,
        mood_tags=["平静"],
        scene_tags=["睡前"],
        style_tags=["经典文学"],
        length_type="中篇",
    )
    res = rr.rerank("query", intent, [cA, cB], top_k=2)
    assert res[0]["book_id"] == "B_mid_embed_all_match"
    assert res[0]["rerank_breakdown"]["penalty_score"] == 0.0


def test_6_strong_intent_hard_filter_prefers_style_scene_length() -> None:
    rr = RerankerV1()
    # High-confidence constraints: style=科幻, scene=通勤, length=短篇
    intent = {
        "mood_tags": [],
        "scene_tags": [{"tag": "通勤", "confidence": 0.85}],
        "style_tags": [{"tag": "科幻", "confidence": 0.85}],
        "difficulty": {"value": 3, "confidence": 0.5},
        "pace": {"value": 3, "confidence": 0.5},
        "length_type": {"value": "短篇", "confidence": 0.8},
    }
    c1 = _cand("A_high_embed_wrong", 0.95, scene_tags=["睡前"], style_tags=["经典文学"], length_type="短篇")
    c2 = _cand("B_match_all", 0.60, scene_tags=["通勤"], style_tags=["科幻"], length_type="短篇")
    c3 = _cand("C_match_style_only", 0.80, scene_tags=["睡前"], style_tags=["科幻"], length_type="短篇")
    res = rr.rerank("q", intent, [c1, c2, c3], top_k=2)
    # Strict filter should allow B only; relax may include C, but A should be filtered out.
    ids = [x["book_id"] for x in res]
    assert "A_high_embed_wrong" not in ids
    assert "B_match_all" in ids


def main() -> None:
    test_1_embedding_high_but_tag_no_match()
    test_2_tag_match_high_beats_mid_embedding()
    test_3_scene_strong_related()
    test_4_difficulty_mismatch_penalty()
    test_5_mixed_conflicts_should_correct()
    test_6_strong_intent_hard_filter_prefers_style_scene_length()
    print("[reranker_v1_tests] OK")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()

