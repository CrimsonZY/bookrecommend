from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence

import numpy as np

# Allow running as a plain script without requiring ml/ to be a Python package.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from intent_tags_v1 import (  # type: ignore  # noqa: E402
    LENGTH_TYPES,
    MOOD_CANDIDATES,
    SCENE_CANDIDATES,
    STYLE_CANDIDATES,
    TagEnricherV1,
)


class FakeEmbedder:
    """
    Deterministic tiny embedder for tests.
    It does NOT reflect semantic meaning; it only ensures:
    - encode() returns stable normalized vectors
    - code paths using embedding can run in tests without downloading models
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for ch in (t or ""):
                idx = (ord(ch) * 1315423911) % self.dim
                v[idx] += 1.0
            n = float(np.linalg.norm(v))
            if n == 0:
                v[0] = 1.0
                n = 1.0
            vecs.append(v / n)
        return np.stack(vecs, axis=0)


def assert_output_schema(out: Dict):
    assert isinstance(out, dict)
    assert set(out.keys()) == {"mood_tags", "scene_tags", "difficulty", "pace", "length_type", "style_tags"}

    assert isinstance(out["mood_tags"], list)
    assert isinstance(out["scene_tags"], list)
    assert isinstance(out["style_tags"], list)

    assert len(out["mood_tags"]) <= 2
    assert len(out["scene_tags"]) <= 2
    assert len(out["style_tags"]) <= 2

    for t in out["mood_tags"]:
        assert t in MOOD_CANDIDATES
    for t in out["scene_tags"]:
        assert t in SCENE_CANDIDATES
    for t in out["style_tags"]:
        assert t in STYLE_CANDIDATES

    assert isinstance(out["difficulty"], int) and 1 <= out["difficulty"] <= 5
    assert isinstance(out["pace"], int) and 1 <= out["pace"] <= 5
    assert out["length_type"] in LENGTH_TYPES


def run_cases():
    # Use rule-first behavior for stable tests; embedding is covered separately.
    enricher = TagEnricherV1(embedder=None, embedding_threshold=0.35)

    cases: List[Dict] = [
        {
            "name": "明确场景-睡前平静",
            "q": "想看睡前能让人平静下来的书",
            "expect_scene_contains": "睡前",
            "expect_mood_contains": "平静",
        },
        {
            "name": "明确场景-通勤轻松",
            "q": "通勤路上适合看的轻松小说",
            "expect_scene_contains": "通勤",
        },
        {
            "name": "风格型-类似三体",
            "q": "类似三体那种风格，最好停不下来",
            "expect_style_contains": "科幻",
            "expect_pace_min": 4,
        },
        {
            "name": "情绪型-孤独",
            "q": "最近很孤独，想读点能陪伴自己的书",
            "expect_mood_contains": "孤独",
        },
        {
            "name": "模糊输入",
            "q": "随便推荐点书",
            "expect_defaults": True,
        },
        {
            "name": "混合输入",
            "q": "睡前想看治愈温暖一点，别太长，节奏别太快",
            "expect_scene_contains": "睡前",
            "expect_any_mood": ["治愈", "温暖"],
        },
    ]

    for c in cases:
        out = enricher.parse(c["q"])
        assert_output_schema(out)

        if c.get("expect_scene_contains"):
            assert c["expect_scene_contains"] in out["scene_tags"], (c["name"], out)
        if c.get("expect_mood_contains"):
            assert c["expect_mood_contains"] in out["mood_tags"], (c["name"], out)
        if c.get("expect_style_contains"):
            assert c["expect_style_contains"] in out["style_tags"], (c["name"], out)
        if c.get("expect_pace_min"):
            assert out["pace"] >= c["expect_pace_min"], (c["name"], out)
        if c.get("expect_any_mood"):
            assert any(m in out["mood_tags"] for m in c["expect_any_mood"]), (c["name"], out)
        if c.get("expect_defaults"):
            # Default policy from spec when cannot判断.
            assert out["mood_tags"] == ["平静"], out
            assert out["scene_tags"] == ["碎片阅读"], out
            assert out["difficulty"] == 3, out
            assert out["pace"] == 3, out
            assert out["length_type"] == "中篇", out
            assert out["style_tags"] == ["经典文学"], out

    print("All intent tag test cases passed.")


if __name__ == "__main__":
    run_cases()

