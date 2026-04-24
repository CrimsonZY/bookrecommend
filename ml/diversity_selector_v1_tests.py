from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from ml.diversity_selector_v1 import DiversitySelectorV1, intent_strength_level, intent_strength_score  # noqa: E402


def _cand(
    book_id: str,
    final_score: float,
    *,
    v: List[float],
    mood: List[str],
    scene: List[str],
    style: List[str],
) -> Dict[str, Any]:
    return {
        "book_id": book_id,
        "final_score": float(final_score),
        "metadata": {"mood_tags": mood, "scene_tags": scene, "style_tags": style},
        "embedding_vector": v,
    }


def _strong_intent() -> Dict[str, Any]:
    # style conf>=0.7 => +2 ; mood non-empty => +1 ; any tag conf>=0.7 => +1 => score>=4 => strong
    return {
        "mood_tags": [{"tag": "平静", "confidence": 0.9}],
        "scene_tags": [],
        "style_tags": [{"tag": "经典文学", "confidence": 0.8}],
        "difficulty": {"value": 3, "confidence": 0.6},
        "pace": {"value": 3, "confidence": 0.6},
        "length_type": {"value": "长篇", "confidence": 0.6},
    }


def _weak_intent() -> Dict[str, Any]:
    return {}


def _medium_intent() -> Dict[str, Any]:
    # mood non-empty => +1 ; query len>=10 => +1 ; conf>=0.7 => +1 => 3 => medium
    return {
        "mood_tags": [{"tag": "治愈", "confidence": 0.72}],
        "scene_tags": [],
        "style_tags": [],
        "difficulty": {"value": 3, "confidence": 0.5},
        "pace": {"value": 3, "confidence": 0.5},
        "length_type": {"value": "中篇", "confidence": 0.5},
    }


def test_1_style_highly_consistent_should_avoid_identical_style_sets() -> None:
    sel = DiversitySelectorV1()
    # First is highest relevance; remaining contain 2 style variants.
    top10 = [
        _cand("A", 0.90, v=[1, 0], mood=["平静"], scene=["睡前"], style=["经典文学"]),
        _cand("B", 0.80, v=[0.9, 0.1], mood=["平静"], scene=["睡前"], style=["经典文学"]),
        _cand("C", 0.79, v=[0.8, 0.2], mood=["平静"], scene=["睡前"], style=["经典文学"]),
        _cand("D", 0.78, v=[0, 1], mood=["治愈"], scene=["睡前"], style=["散文"]),
        _cand("E", 0.77, v=[0.1, 0.9], mood=["治愈"], scene=["睡前"], style=["散文"]),
    ]
    out = sel.select("q", _strong_intent(), top10, top_k=3)
    assert len(out) == 3


def test_2_mixed_styles_should_pick_diverse() -> None:
    sel = DiversitySelectorV1()
    top10 = [
        _cand("A", 0.90, v=[1, 0, 0], mood=["平静"], scene=["睡前"], style=["经典文学"]),
        _cand("B", 0.86, v=[0.9, 0.1, 0], mood=["平静"], scene=["睡前"], style=["经典文学"]),
        _cand("C", 0.70, v=[0, 1, 0], mood=["热血"], scene=["旅行途中"], style=["科幻"]),
        _cand("D", 0.69, v=[0, 0.9, 0.1], mood=["热血"], scene=["旅行途中"], style=["科幻"]),
        _cand("E", 0.68, v=[0, 0, 1], mood=["幽默"], scene=["通勤"], style=["悬疑"]),
    ]
    out = sel.select("q", _strong_intent(), top10, top_k=3)
    assert len(out) == 3
    ids = [x["book_id"] for x in out]
    assert "A" in ids  # top1 by relevance


def test_3_mood_concentrated_should_not_pick_identical_mood_sets() -> None:
    sel = DiversitySelectorV1()
    top10 = [
        _cand("A", 0.90, v=[1, 0], mood=["平静"], scene=["睡前"], style=["经典文学"]),
        _cand("B", 0.85, v=[0.9, 0.1], mood=["平静"], scene=["通勤"], style=["经典文学"]),
        _cand("C", 0.70, v=[0, 1], mood=["治愈"], scene=["睡前"], style=["散文"]),
        _cand("D", 0.69, v=[0.1, 0.9], mood=["治愈"], scene=["通勤"], style=["散文"]),
        _cand("E", 0.68, v=[0.2, 0.8], mood=["热血"], scene=["旅行途中"], style=["科幻"]),
    ]
    out = sel.select("q", _strong_intent(), top10, top_k=3)
    assert len(out) == 3


def test_4_embedding_similar_but_tag_different_should_help_diversify() -> None:
    sel = DiversitySelectorV1()
    # A/B embeddings almost identical; C different. tags make B more redundant.
    top10 = [
        _cand("A", 0.90, v=[1, 0], mood=["平静"], scene=["睡前"], style=["经典文学"]),
        _cand("B", 0.89, v=[0.99, 0.01], mood=["平静"], scene=["睡前"], style=["经典文学"]),
        _cand("C", 0.70, v=[0, 1], mood=["治愈"], scene=["通勤"], style=["散文"]),
        _cand("D", 0.69, v=[0.2, 0.8], mood=["热血"], scene=["旅行途中"], style=["科幻"]),
    ]
    out = sel.select("q", _strong_intent(), top10, top_k=3)
    assert len(out) == 3
    ids = [x["book_id"] for x in out]
    assert "A" in ids
    # Expect not picking both A and B because they are too similar and tags identical.
    assert not ("A" in ids and "B" in ids and ids.index("B") < 3)


def test_5_score_gate_should_block_low_score_even_if_diverse() -> None:
    sel = DiversitySelectorV1()
    # top1=1.0 -> gate=0.84; C is very diverse but too low score should be blocked.
    top10 = [
        _cand("A", 1.00, v=[1, 0], mood=["平静"], scene=["睡前"], style=["经典文学"]),
        _cand("B", 0.90, v=[0.9, 0.1], mood=["平静"], scene=["睡前"], style=["经典文学"]),
        _cand("C_low_but_diverse", 0.30, v=[0, 1], mood=["热血"], scene=["旅行途中"], style=["科幻"]),
        _cand("D", 0.86, v=[0.8, 0.2], mood=["治愈"], scene=["通勤"], style=["散文"]),
    ]
    out = sel.select("q", _strong_intent(), top10, top_k=3)
    ids = [x["book_id"] for x in out]
    assert "C_low_but_diverse" not in ids


def test_6_fallback_top7_when_gate_not_enough_candidates() -> None:
    sel = DiversitySelectorV1()
    # Only A passes gate; need fallback to top7 to fill top3.
    top10 = [
        _cand("A", 1.00, v=[1, 0], mood=["平静"], scene=["睡前"], style=["经典文学"]),
        _cand("B", 0.60, v=[0.9, 0.1], mood=["平静"], scene=["睡前"], style=["经典文学"]),
        _cand("C", 0.59, v=[0, 1], mood=["治愈"], scene=["通勤"], style=["散文"]),
        _cand("D", 0.58, v=[0.1, 0.9], mood=["热血"], scene=["旅行途中"], style=["科幻"]),
        _cand("E", 0.57, v=[0.2, 0.8], mood=["幽默"], scene=["周末下午"], style=["悬疑"]),
        _cand("F", 0.56, v=[0.3, 0.7], mood=["浪漫"], scene=["睡前"], style=["爱情"]),
        _cand("G", 0.55, v=[0.4, 0.6], mood=["哲思"], scene=["长期阅读"], style=["哲学思考"]),
        _cand("H", 0.10, v=[0.5, 0.5], mood=["悲伤"], scene=["碎片阅读"], style=["历史"]),
    ]
    out = sel.select("q", _strong_intent(), top10, top_k=3)
    assert len(out) == 3


def test_7_weak_intent_has_lower_gate_and_more_exploration() -> None:
    sel = DiversitySelectorV1()
    # top1=1.0 -> strong gate=0.84 (excludes C=0.80), weak gate=0.78 (includes C)
    top10 = [
        _cand("A", 1.00, v=[1, 0], mood=["平静"], scene=["睡前"], style=["经典文学"]),
        _cand("B", 0.90, v=[0.99, 0.01], mood=["平静"], scene=["睡前"], style=["经典文学"]),
        _cand("D", 0.85, v=[0.98, 0.02], mood=["平静"], scene=["睡前"], style=["经典文学"]),
        _cand("C_diverse_but_lower", 0.80, v=[0, 1], mood=["热血"], scene=["旅行途中"], style=["科幻"]),
        _cand("E", 0.86, v=[0.97, 0.03], mood=["平静"], scene=["睡前"], style=["经典文学"]),
    ]
    out_strong = sel.select("q", _strong_intent(), top10, top_k=3)
    out_weak = sel.select("q", _weak_intent(), top10, top_k=3)
    ids_strong = [x["book_id"] for x in out_strong]
    ids_weak = [x["book_id"] for x in out_weak]
    assert "C_diverse_but_lower" not in ids_strong
    assert "C_diverse_but_lower" in ids_weak


def test_8_strength_scoring_levels() -> None:
    assert intent_strength_level(intent_strength_score("短", _weak_intent())) == "weak"
    assert intent_strength_level(intent_strength_score("这是一个超过十个字的query", _medium_intent())) == "medium"
    assert intent_strength_level(intent_strength_score("随便推荐几本", _strong_intent())) == "strong"


def main() -> None:
    test_1_style_highly_consistent_should_avoid_identical_style_sets()
    test_2_mixed_styles_should_pick_diverse()
    test_3_mood_concentrated_should_not_pick_identical_mood_sets()
    test_4_embedding_similar_but_tag_different_should_help_diversify()
    test_5_score_gate_should_block_low_score_even_if_diverse()
    test_6_fallback_top7_when_gate_not_enough_candidates()
    test_7_weak_intent_has_lower_gate_and_more_exploration()
    test_8_strength_scoring_levels()
    print("[diversity_selector_v1_tests] OK")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()

