from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence

import numpy as np

# allow running as plain script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from intent_tags_v2 import (  # type: ignore  # noqa: E402
    LENGTH_TYPES,
    MOOD_CANDIDATES,
    SCENE_CANDIDATES,
    STYLE_CANDIDATES,
    TagEnricherV2,
)


class KeywordEmbedder:
    """
    Deterministic fake embedder.
    If query contains certain keywords, it will be closer to corresponding prototype text.
    """

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            s = (t or "")
            v = np.zeros(self.dim, dtype=np.float32)
            hit_any = False
            # Map keywords to dimensions
            def hit(words: List[str], idx: int):
                nonlocal hit_any
                if any(w in s for w in words):
                    v[idx] += 5.0
                    hit_any = True

            hit(["睡前", "助眠"], 0)
            hit(["通勤", "地铁", "公交"], 1)
            hit(["三体", "科幻", "宇宙", "外星"], 2)
            hit(["悬疑", "推理", "命案", "凶手"], 3)
            hit(["治愈", "疗愈", "温柔"], 4)
            hit(["悲伤", "难过", "哀伤"], 5)
            hit(["平静", "安静", "宁静"], 6)
            hit(["浪漫", "爱情", "恋爱"], 7)
            hit(["文学", "文学性", "名著", "经典"], 8)
            hit(["碎片阅读", "短篇", "随笔", "散文"], 9)
            hit(["哲学思考", "哲学", "意义", "存在"], 10)
            # For texts without any known keywords, push to a dedicated dimension
            # so similarity with prototypes remains low (to test fallback behavior).
            if not hit_any:
                v[-1] = 1.0
            n = float(np.linalg.norm(v))
            vecs.append(v / (n if n else 1.0))
        return np.stack(vecs, axis=0)


def assert_schema(out: Dict):
    assert set(out.keys()) == {"mood_tags", "scene_tags", "difficulty", "pace", "length_type", "style_tags"}
    assert isinstance(out["mood_tags"], list) and len(out["mood_tags"]) <= 3
    assert isinstance(out["scene_tags"], list) and len(out["scene_tags"]) <= 3
    assert isinstance(out["style_tags"], list) and len(out["style_tags"]) <= 3

    for x in out["mood_tags"]:
        assert x["tag"] in MOOD_CANDIDATES
        assert 0.0 <= float(x["confidence"]) <= 1.0
    for x in out["scene_tags"]:
        assert x["tag"] in SCENE_CANDIDATES
        assert 0.0 <= float(x["confidence"]) <= 1.0
    for x in out["style_tags"]:
        assert x["tag"] in STYLE_CANDIDATES
        assert 0.0 <= float(x["confidence"]) <= 1.0

    assert 1 <= int(out["difficulty"]["value"]) <= 5
    assert 0.0 <= float(out["difficulty"]["confidence"]) <= 1.0
    assert 1 <= int(out["pace"]["value"]) <= 5
    assert 0.0 <= float(out["pace"]["confidence"]) <= 1.0
    assert out["length_type"]["value"] in LENGTH_TYPES
    assert 0.0 <= float(out["length_type"]["confidence"]) <= 1.0


def run_cases():
    enricher = TagEnricherV2(embedder=KeywordEmbedder(), embedding_threshold=0.5)
    enricher_no_embed = TagEnricherV2(embedder=None, embedding_threshold=0.5)

    cases = [
        # 1. 明确情绪输入
        ("明确情绪", "想读治愈温柔一点的书", {"mood_contains": "治愈"}),
        # 2. 明确场景输入
        ("明确场景", "通勤路上适合看的书", {"scene_contains": "通勤"}),
        # 3. 风格强约束输入
        ("风格强约束", "类似三体那种风格", {"style_contains": "科幻"}),
        # 4. 模糊输入
        ("模糊输入", "随便看看", {"fallback": True}),
        # 5. 多意图输入
        ("多意图", "睡前想看平静一点，通勤也能看", {"scene_any": ["睡前", "通勤"], "mood_contains": "平静"}),
        # 6. 负向情绪输入
        ("负向情绪", "最近很悲伤，想读点书", {"mood_contains": "悲伤"}),
    ]

    for name, q, exp in cases:
        # For the fuzzy input case, force no-embedding to validate fallback defaults deterministically.
        if exp.get("fallback"):
            out = enricher_no_embed.parse(q)
        else:
            out = enricher.parse(q)
        assert_schema(out)

        if exp.get("mood_contains"):
            assert exp["mood_contains"] in [x["tag"] for x in out["mood_tags"]], (name, out)
        if exp.get("scene_contains"):
            assert exp["scene_contains"] in [x["tag"] for x in out["scene_tags"]], (name, out)
        if exp.get("style_contains"):
            assert exp["style_contains"] in [x["tag"] for x in out["style_tags"]], (name, out)
        if exp.get("scene_any"):
            tags = [x["tag"] for x in out["scene_tags"]]
            assert any(t in tags for t in exp["scene_any"]), (name, out)
        if exp.get("fallback"):
            # For vague inputs, should fall back due to <0.5
            assert out["mood_tags"][0]["tag"] == "平静"
            assert out["scene_tags"][0]["tag"] == "碎片阅读"
            assert out["style_tags"][0]["tag"] == "文学"
            assert out["difficulty"]["value"] == 3
            assert out["pace"]["value"] == 3
            assert out["length_type"]["value"] == "中篇"

    print("All TagEnricherV2 test cases passed.")


if __name__ == "__main__":
    run_cases()

