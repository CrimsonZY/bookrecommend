from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np


MODEL_NAME = "/root/models/minilm"

MOOD_CANDIDATES = ["治愈", "温暖", "平静", "热血", "沉重", "悲伤", "哲思", "孤独", "希望", "幽默", "浪漫"]
SCENE_CANDIDATES = ["睡前", "通勤", "周末下午", "长期阅读", "旅行途中", "碎片阅读"]
STYLE_CANDIDATES = ["文学", "小说", "社会观察", "悬疑", "科幻", "散文", "历史", "爱情", "人物传记", "哲学思考"]
LENGTH_TYPES = ["短篇", "中篇", "长篇", "超长篇"]


DEFAULT_FALLBACK = {
    "mood_tag": ("平静", 0.4),
    "scene_tag": ("碎片阅读", 0.4),
    "difficulty": (3, 0.5),
    "pace": (3, 0.5),
    "length_type": ("中篇", 0.5),
    "style_tag": ("文学", 0.4),
}


def _clean_text(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _clip_int(x: int, lo: int = 1, hi: int = 5) -> int:
    return max(lo, min(hi, int(x)))


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return x / denom


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = str(model_name or MODEL_NAME)
        self.model = None

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        # Lazy import + lazy model load (avoid loading torch weights at service startup).
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_model()
        emb = self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        emb = emb.astype(np.float32, copy=False)
        return _l2_normalize(emb)


@dataclass(frozen=True)
class TagProto:
    tag: str
    text: str


class EmbeddingTagMatcher:
    """
    - query embedding
    - tag embedding (predefined tag descriptions)
    - cosine similarity
    - output top-k tags + mapped confidence
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    @staticmethod
    def similarity_to_conf(sim: float) -> float:
        # cosine similarity typically in [-1,1], clamp to [0,1]
        sim01 = _clip(max(0.0, float(sim)), 0.0, 1.0)
        # map to [0.3, 0.8]
        return _clip(0.3 + 0.5 * sim01, 0.3, 0.8)

    def match(self, query: str, protos: List[TagProto], top_k: int = 3) -> List[Dict[str, float]]:
        q = _clean_text(query)
        if not q:
            return []
        texts = [q] + [p.text for p in protos]
        embs = self.embedder.encode(texts)
        qv = embs[0:1, :]
        pv = embs[1:, :]
        sims = (qv @ pv.T)[0]
        order = np.argsort(-sims)[: max(0, int(top_k))]
        out: List[Dict[str, float]] = []
        for idx in order:
            i = int(idx)
            sim = float(sims[i])
            out.append({"tag": protos[i].tag, "score": sim, "confidence": self.similarity_to_conf(sim)})
        return out


def build_prototypes() -> Dict[str, List[TagProto]]:
    mood = [
        TagProto("治愈", "治愈、疗愈、抚慰、放松、助眠。"),
        TagProto("温暖", "温暖、温柔、善意、陪伴、亲情友情。"),
        TagProto("平静", "平静、安静、宁静、舒缓、从容。"),
        TagProto("热血", "热血、冒险、战斗、燃、强情节。"),
        TagProto("沉重", "沉重、压抑、残酷现实、创伤。"),
        TagProto("悲伤", "悲伤、哀伤、离别、失去。"),
        TagProto("哲思", "哲思、思辨、反思、意义、存在。"),
        TagProto("孤独", "孤独、寂寞、疏离、自我对话。"),
        TagProto("希望", "希望、救赎、重生、勇气。"),
        TagProto("幽默", "幽默、搞笑、讽刺、荒诞。"),
        TagProto("浪漫", "浪漫、爱情、心动、甜。"),
    ]
    scene = [
        TagProto("睡前", "睡前阅读，助眠、放松、温柔治愈。"),
        TagProto("通勤", "通勤阅读，节奏快、短小、可中断继续。"),
        TagProto("周末下午", "周末下午，沉浸式阅读，中长篇。"),
        TagProto("长期阅读", "长期阅读，长篇大部头、经典、信息量大。"),
        TagProto("旅行途中", "旅行途中，故事性强、轻松或冒险。"),
        TagProto("碎片阅读", "碎片阅读，短篇、随笔散文、随时读一点。"),
    ]
    style = [
        TagProto("文学", "文学性强、严肃叙事或经典作品等。"),
        TagProto("小说", "以故事叙事为主的虚构作品。"),
        TagProto("社会观察", "社会观察，现实题材、阶层与制度。"),
        TagProto("悬疑", "悬疑推理，命案、谜团、反转。"),
        TagProto("科幻", "科幻，宇宙、未来、外星、科技文明。"),
        TagProto("散文", "散文随笔，片段式、抒情。"),
        TagProto("历史", "历史题材，王朝、时代变迁。"),
        TagProto("爱情", "爱情题材，恋爱、婚姻、浪漫。"),
        TagProto("人物传记", "人物传记，自传回忆录、真实人物。"),
        TagProto("哲学思考", "哲学思考，存在与意义、伦理思想。"),
    ]
    return {"mood_tags": mood, "scene_tags": scene, "style_tags": style}


# -----------------------
# Rule matchers (keyword)
# -----------------------


MOOD_RULES: List[Tuple[str, str]] = [
    ("治愈", r"(治愈|疗愈|安慰|抚慰|放松|舒缓|助眠|温柔)"),
    ("温暖", r"(温暖|陪伴|亲情|友情|善意)"),
    ("平静", r"(平静|安静|宁静|从容|淡然)"),
    ("热血", r"(热血|燃|战斗|冒险|爽|刺激)"),
    ("沉重", r"(沉重|压抑|黑暗|残酷|创伤)"),
    ("悲伤", r"(悲伤|难过|哀伤|虐|眼泪|离别|失去)"),
    ("哲思", r"(哲思|思辨|反思|意义|存在|真相)"),
    ("孤独", r"(孤独|寂寞|疏离|无人|一个人)"),
    ("希望", r"(希望|救赎|重生|勇气)"),
    ("幽默", r"(幽默|搞笑|好笑|讽刺|荒诞)"),
    ("浪漫", r"(浪漫|爱情|恋爱|心动|甜)"),
]


SCENE_RULES: List[Tuple[str, str]] = [
    ("睡前", r"(睡前|入睡|助眠|床上|临睡)"),
    ("通勤", r"(通勤|地铁|公交|路上|上下班)"),
    ("周末下午", r"(周末|下午|休息日)"),
    ("长期阅读", r"(长期|大部头|慢慢看|系列|全集|上中下|全[一二三四五六七八九十]册)"),
    ("旅行途中", r"(旅行|旅途|路途|出差)"),
    ("碎片阅读", r"(碎片|短篇|随笔|一小段|随手读|短小)"),
]


STYLE_RULES: List[Tuple[str, str]] = [
    ("科幻", r"(科幻|宇宙|外星|星际|未来|太空|三体|基地)"),
    ("悬疑", r"(悬疑|推理|命案|凶手|侦探|谜|犯罪|追凶)"),
    ("散文", r"(散文|随笔|杂文|书信|札记|小品文)"),
    ("历史", r"(历史|王朝|帝国|时代|战争|革命)"),
    ("爱情", r"(爱情|恋爱|婚姻|相爱|浪漫)"),
    ("人物传记", r"(传记|回忆录|自传|口述史)"),
    ("哲学思考", r"(哲学|思考|存在|意义|伦理|信仰|理性)"),
    ("社会观察", r"(社会|阶层|现实|底层|制度|贫穷|城市|农村)"),
    ("小说", r"(小说|长篇|中篇|短篇|故事|情节|叙事)"),
    ("文学", r"(名著|经典|诺贝尔|文学史|文学)"),
]


def rule_match(text: str, rules: List[Tuple[str, str]], allowed: List[str], *, limit: int = 3) -> List[Dict[str, Any]]:
    q = _clean_text(text)
    out: List[Dict[str, Any]] = []
    for tag, pat in rules:
        if tag in allowed and re.search(pat, q, flags=re.IGNORECASE):
            # rule confidence: 0.6 ~ 0.9 (deterministic)
            # stronger if exact tag word appears in query.
            strong = 1.0 if tag in q else 0.0
            conf = 0.75 + 0.10 * strong  # 0.75 or 0.85
            out.append({"tag": tag, "confidence": _clip(conf, 0.6, 0.9), "source": "rule"})
        if len(out) >= limit:
            break
    return out


def merge_tag_results(
    rule_res: List[Dict[str, Any]],
    emb_res: List[Dict[str, Any]],
    *,
    allowed: List[str],
    limit: int = 3,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in sorted(rule_res, key=lambda x: float(x.get("confidence", 0.0)), reverse=True):
        t = item.get("tag")
        if t in allowed and t not in seen:
            out.append({"tag": t, "confidence": float(item["confidence"])})
            seen.add(t)
        if len(out) >= limit:
            return out
    for item in sorted(emb_res, key=lambda x: float(x.get("confidence", 0.0)), reverse=True):
        t = item.get("tag")
        if t in allowed and t not in seen:
            out.append({"tag": t, "confidence": float(item["confidence"])})
            seen.add(t)
        if len(out) >= limit:
            break
    # sort final by confidence desc
    out.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    return out


# -----------------------
# Modules (testable)
# -----------------------


class MoodTaggerV2:
    def __init__(self, *, matcher: Optional[EmbeddingTagMatcher] = None, threshold: float = 0.5):
        self.matcher = matcher
        self.threshold = float(threshold)
        self.protos = build_prototypes()["mood_tags"]

    def tag(self, query: str) -> List[Dict[str, Any]]:
        rule_res = rule_match(query, MOOD_RULES, MOOD_CANDIDATES, limit=3)
        emb_res: List[Dict[str, Any]] = []
        if self.matcher:
            for r in self.matcher.match(query, self.protos, top_k=3):
                emb_res.append({"tag": r["tag"], "confidence": float(r["confidence"]), "score": float(r["score"])})
        out = merge_tag_results(rule_res, emb_res, allowed=MOOD_CANDIDATES, limit=3)
        if not out or float(out[0]["confidence"]) < self.threshold:
            tag, conf = DEFAULT_FALLBACK["mood_tag"]
            return [{"tag": tag, "confidence": float(conf)}]
        return out


class SceneTaggerV2:
    def __init__(self, *, matcher: Optional[EmbeddingTagMatcher] = None, threshold: float = 0.5):
        self.matcher = matcher
        self.threshold = float(threshold)
        self.protos = build_prototypes()["scene_tags"]

    def tag(self, query: str) -> List[Dict[str, Any]]:
        rule_res = rule_match(query, SCENE_RULES, SCENE_CANDIDATES, limit=3)
        emb_res: List[Dict[str, Any]] = []
        if self.matcher:
            for r in self.matcher.match(query, self.protos, top_k=3):
                emb_res.append({"tag": r["tag"], "confidence": float(r["confidence"]), "score": float(r["score"])})
        out = merge_tag_results(rule_res, emb_res, allowed=SCENE_CANDIDATES, limit=3)
        if not out or float(out[0]["confidence"]) < self.threshold:
            tag, conf = DEFAULT_FALLBACK["scene_tag"]
            return [{"tag": tag, "confidence": float(conf)}]
        return out


class StyleTaggerV2:
    def __init__(self, *, matcher: Optional[EmbeddingTagMatcher] = None, threshold: float = 0.5):
        self.matcher = matcher
        self.threshold = float(threshold)
        self.protos = build_prototypes()["style_tags"]

    def tag(self, query: str) -> List[Dict[str, Any]]:
        rule_res = rule_match(query, STYLE_RULES, STYLE_CANDIDATES, limit=3)
        emb_res: List[Dict[str, Any]] = []
        if self.matcher:
            for r in self.matcher.match(query, self.protos, top_k=3):
                emb_res.append({"tag": r["tag"], "confidence": float(r["confidence"]), "score": float(r["score"])})
        out = merge_tag_results(rule_res, emb_res, allowed=STYLE_CANDIDATES, limit=3)
        if not out or float(out[0]["confidence"]) < self.threshold:
            tag, conf = DEFAULT_FALLBACK["style_tag"]
            return [{"tag": tag, "confidence": float(conf)}]
        return out


class LengthClassifierV2:
    def classify(self, query: str) -> Dict[str, Any]:
        q = _clean_text(query)
        # rule-based
        if re.search(r"(超长|非常长|大部头|全集|系列|上中下|全[一二三四五六七八九十]册)", q):
            return {"value": "超长篇", "confidence": 0.8}
        if re.search(r"(长篇|长一点|长的)", q):
            return {"value": "长篇", "confidence": 0.75}
        if re.search(r"(短篇|很短|短一点|小故事|短的)", q):
            return {"value": "短篇", "confidence": 0.75}
        if re.search(r"(中篇|适中|不太长)", q):
            return {"value": "中篇", "confidence": 0.7}
        # fallback per spec
        v, conf = DEFAULT_FALLBACK["length_type"]
        return {"value": v, "confidence": float(conf)}


class DifficultyEstimatorV2:
    def estimate(self, query: str, style_tags: List[str]) -> Dict[str, Any]:
        q = _clean_text(query)
        score = 3
        conf = 0.5
        if re.search(r"(非常轻松|轻松|不费脑|简单|入门|漫画|短故事)", q):
            score = 1
            conf = 0.75
        elif re.search(r"(烧脑|复杂|高密度|晦涩|难|硬核)", q):
            score = 5
            conf = 0.8
        else:
            if "哲学思考" in style_tags:
                score = 5
                conf = 0.7
            elif any(t in style_tags for t in ["文学", "历史"]):
                score = 4
                conf = 0.65
        score = _clip_int(score, 1, 5)
        # fallback trigger if conf < 0.5 (spec)
        if conf < 0.5:
            v, c = DEFAULT_FALLBACK["difficulty"]
            return {"value": int(v), "confidence": float(c)}
        return {"value": int(score), "confidence": float(_clip(conf, 0.2, 0.9))}


class PaceEstimatorV2:
    def estimate(self, query: str, style_tags: List[str]) -> Dict[str, Any]:
        q = _clean_text(query)
        score = 3
        conf = 0.5
        if re.search(r"(停不下来|上头|强情节|爽|节奏快|追读|反转)", q):
            score = 5
            conf = 0.8
        elif re.search(r"(慢热|抒情|细腻|平静|舒缓)", q):
            score = 2
            conf = 0.7
        else:
            if "悬疑" in style_tags:
                score = 4
                conf = 0.65
            if "散文" in style_tags:
                score = 2
                conf = max(conf, 0.6)
        score = _clip_int(score, 1, 5)
        if conf < 0.5:
            v, c = DEFAULT_FALLBACK["pace"]
            return {"value": int(v), "confidence": float(c)}
        return {"value": int(score), "confidence": float(_clip(conf, 0.2, 0.9))}


class TagEnricherV2:
    def __init__(
        self,
        *,
        embedder: Optional[Embedder] = None,
        embedding_threshold: float = 0.5,
    ):
        self.matcher = EmbeddingTagMatcher(embedder) if embedder is not None else None
        self.mood_tagger = MoodTaggerV2(matcher=self.matcher, threshold=embedding_threshold)
        self.scene_tagger = SceneTaggerV2(matcher=self.matcher, threshold=embedding_threshold)
        self.style_tagger = StyleTaggerV2(matcher=self.matcher, threshold=embedding_threshold)
        self.length_classifier = LengthClassifierV2()
        self.difficulty_estimator = DifficultyEstimatorV2()
        self.pace_estimator = PaceEstimatorV2()

    def parse(self, query: str) -> Dict[str, Any]:
        q = _clean_text(query)

        mood = self.mood_tagger.tag(q)
        scene = self.scene_tagger.tag(q)
        style = self.style_tagger.tag(q)

        # Only treat style tags as reliable signals for numeric estimators when confidence >= 0.5.
        style_tags_reliable = [
            x["tag"]
            for x in style
            if x.get("tag") in STYLE_CANDIDATES and float(x.get("confidence", 0.0)) >= 0.5
        ]
        difficulty = self.difficulty_estimator.estimate(q, style_tags_reliable)
        pace = self.pace_estimator.estimate(q, style_tags_reliable)
        length_type = self.length_classifier.classify(q)

        # enforce constraints: lists max 3, sorted by confidence desc, allowed only
        mood = [x for x in mood if x.get("tag") in MOOD_CANDIDATES]
        scene = [x for x in scene if x.get("tag") in SCENE_CANDIDATES]
        style = [x for x in style if x.get("tag") in STYLE_CANDIDATES]
        mood.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        scene.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        style.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        mood = mood[:3]
        scene = scene[:3]
        style = style[:3]

        # Ensure strict fallback when top confidence < 0.5 for that dimension.
        if not mood or float(mood[0].get("confidence", 0.0)) < 0.5:
            t, c = DEFAULT_FALLBACK["mood_tag"]
            mood = [{"tag": t, "confidence": float(c)}]
        if not scene or float(scene[0].get("confidence", 0.0)) < 0.5:
            t, c = DEFAULT_FALLBACK["scene_tag"]
            scene = [{"tag": t, "confidence": float(c)}]
        if not style or float(style[0].get("confidence", 0.0)) < 0.5:
            t, c = DEFAULT_FALLBACK["style_tag"]
            style = [{"tag": t, "confidence": float(c)}]

        # Strict output schema, no extra fields
        out: Dict[str, Any] = {
            "mood_tags": [{"tag": str(x["tag"]), "confidence": float(_clip(x["confidence"], 0.0, 1.0))} for x in mood],
            "scene_tags": [{"tag": str(x["tag"]), "confidence": float(_clip(x["confidence"], 0.0, 1.0))} for x in scene],
            "difficulty": {"value": int(_clip_int(int(difficulty["value"]), 1, 5)), "confidence": float(_clip(difficulty["confidence"], 0.0, 1.0))},
            "pace": {"value": int(_clip_int(int(pace["value"]), 1, 5)), "confidence": float(_clip(pace["confidence"], 0.0, 1.0))},
            "length_type": {"value": str(length_type["value"]) if length_type.get("value") in LENGTH_TYPES else DEFAULT_FALLBACK["length_type"][0], "confidence": float(_clip(length_type["confidence"], 0.0, 1.0))},
            "style_tags": [{"tag": str(x["tag"]), "confidence": float(_clip(x["confidence"], 0.0, 1.0))} for x in style],
        }
        return out

    def parse_json(self, query: str) -> str:
        return json.dumps(self.parse(query), ensure_ascii=False)


def build_default_enricher(enable_embedding: bool = True) -> TagEnricherV2:
    embedder = SentenceTransformerEmbedder(MODEL_NAME) if enable_embedding else None
    return TagEnricherV2(embedder=embedder, embedding_threshold=0.5)


def demo() -> None:
    enricher = build_default_enricher(enable_embedding=True)
    samples = [
        "想看睡前能让人平静下来的书",
        "通勤路上适合看的轻松小说",
        "类似百年孤独那种风格",
        "想看很烧脑的推理悬疑",
        "随便推荐点书",
        "最近很悲伤，想读点治愈的",
    ]
    for s in samples:
        print(s)
        print(enricher.parse_json(s))
        print()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    demo()

