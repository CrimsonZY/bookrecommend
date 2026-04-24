from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-zh-v1.5"


MOOD_CANDIDATES = ["治愈", "温暖", "平静", "热血", "沉重", "悲伤", "哲思", "孤独", "希望", "幽默", "浪漫"]
SCENE_CANDIDATES = ["睡前", "通勤", "周末下午", "长期阅读", "送礼", "旅行途中", "碎片阅读"]
STYLE_CANDIDATES = ["经典文学", "成长小说", "社会观察", "悬疑", "科幻", "散文", "历史", "爱情", "人物传记", "哲学思考"]
LENGTH_TYPES = ["短篇", "中篇", "长篇", "超长篇"]


DEFAULT_OUTPUT: Dict[str, Any] = {
    "mood_tags": ["平静"],
    "scene_tags": ["碎片阅读"],
    "difficulty": 3,
    "pace": 3,
    "length_type": "中篇",
    "style_tags": ["经典文学"],
}


def _clean_text(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return x / denom


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = MODEL_NAME):
        # SentenceTransformer will reuse HF cache if already downloaded.
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        emb = emb.astype(np.float32, copy=False)
        return _l2_normalize(emb)


@dataclass(frozen=True)
class TagProto:
    tag: str
    text: str


def _select_top_tags(scores: Dict[str, float], *, top_n: int, allowed: List[str]) -> List[str]:
    items = [(k, float(v)) for k, v in scores.items() if k in allowed]
    items.sort(key=lambda x: x[1], reverse=True)
    picked = [k for k, _ in items[:top_n]]
    out: List[str] = []
    for t in picked:
        if t not in out:
            out.append(t)
    return out


def _merge_limit(base: List[str], extra: List[str], *, limit: int, allowed: List[str]) -> List[str]:
    out: List[str] = []
    for t in base + extra:
        if t in allowed and t not in out:
            out.append(t)
        if len(out) >= limit:
            break
    return out


def _clip_int(x: int, lo: int = 1, hi: int = 5) -> int:
    return max(lo, min(hi, int(x)))


# -----------------------
# Prototypes for embedding
# -----------------------


def build_prototypes() -> Dict[str, List[TagProto]]:
    mood = [
        TagProto("治愈", "治愈、疗愈、被安慰、情绪低落时也能被抚慰。"),
        TagProto("温暖", "温暖、温柔、善意、陪伴、亲情友情。"),
        TagProto("平静", "平静、安静、舒缓、助眠、放松。"),
        TagProto("热血", "热血、冒险、战斗、燃、强行动。"),
        TagProto("沉重", "沉重、压抑、残酷现实、创伤。"),
        TagProto("悲伤", "悲伤、哀伤、离别、失去。"),
        TagProto("哲思", "哲思、思辨、反思、人生意义。"),
        TagProto("孤独", "孤独、寂寞、疏离、自我对话。"),
        TagProto("希望", "希望、救赎、重生、向前。"),
        TagProto("幽默", "幽默、搞笑、讽刺、轻松。"),
        TagProto("浪漫", "浪漫、爱情、心动。"),
    ]
    scene = [
        TagProto("睡前", "睡前读，放松、温柔治愈、助眠。"),
        TagProto("通勤", "通勤路上读，节奏快、短小、易中断继续。"),
        TagProto("周末下午", "周末下午读，沉浸式，中长篇，故事性强。"),
        TagProto("长期阅读", "需要长期阅读，篇幅长，经典或信息量大。"),
        TagProto("送礼", "适合送礼，口碑好、温暖经典、普适。"),
        TagProto("旅行途中", "旅行途中读，轻松或冒险，故事推进强。"),
        TagProto("碎片阅读", "碎片阅读，短篇、随笔散文，随时读一点。"),
    ]
    style = [
        TagProto("经典文学", "经典文学名著、文学性强。"),
        TagProto("成长小说", "成长小说，青春、少年少女成长。"),
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


def embedding_scores(embedder: Embedder, query: str, protos: List[TagProto]) -> Dict[str, float]:
    texts = [query] + [p.text for p in protos]
    embs = embedder.encode(texts)
    q = embs[0:1, :]
    p = embs[1:, :]
    sims = (q @ p.T)[0]
    return {protos[i].tag: float(sims[i]) for i in range(len(protos))}


# -----------------------
# Rule matchers (priority)
# -----------------------


MOOD_RULES: List[Tuple[str, str]] = [
    ("治愈", r"(治愈|疗愈|安慰|抚慰|温柔|放松|舒缓|助眠)"),
    ("温暖", r"(温暖|温柔|陪伴|亲情|友情|善意)"),
    ("平静", r"(平静|安静|宁静|从容|淡然|舒缓|助眠)"),
    ("热血", r"(热血|燃|战斗|冒险|爽|刺激)"),
    ("沉重", r"(沉重|压抑|黑暗|残酷|创伤)"),
    ("悲伤", r"(悲伤|难过|哀伤|虐|眼泪|离别|失去)"),
    ("哲思", r"(哲思|思辨|反思|意义|存在|真相)"),
    ("孤独", r"(孤独|寂寞|疏离|自我对话)"),
    ("希望", r"(希望|救赎|重生|治好|振作)"),
    ("幽默", r"(幽默|搞笑|好笑|讽刺|荒诞)"),
    ("浪漫", r"(浪漫|爱情|恋爱|心动|甜)"),
]


SCENE_RULES: List[Tuple[str, str]] = [
    ("睡前", r"(睡前|入睡|助眠|床上|临睡)"),
    ("通勤", r"(通勤|地铁|公交|路上|上下班)"),
    ("周末下午", r"(周末|下午|休息日)"),
    ("长期阅读", r"(长期|大部头|慢慢看|系列|全集|上中下|全[一二三四五六七八九十]册)"),
    ("送礼", r"(送礼|礼物|送人|送朋友)"),
    ("旅行途中", r"(旅行|旅途|路途|出差)"),
    ("碎片阅读", r"(碎片|短篇|随笔|一小段|随手读|通勤.*短|短小)"),
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
    ("成长小说", r"(成长|青春|少年|少女|校园|蜕变)"),
    ("经典文学", r"(名著|经典|诺贝尔|文学史|百年孤独|红楼梦|围城)"),
]


def rule_pick(text: str, rules: List[Tuple[str, str]], allowed: List[str], *, limit: int) -> List[str]:
    out: List[str] = []
    for tag, pat in rules:
        if tag in allowed and re.search(pat, text, flags=re.IGNORECASE):
            if tag not in out:
                out.append(tag)
        if len(out) >= limit:
            break
    return out


# -----------------------
# Modules (each testable)
# -----------------------


class MoodParser:
    def __init__(self, *, embedder: Optional[Embedder] = None, threshold: float = 0.35):
        self.embedder = embedder
        self.threshold = float(threshold)
        self.protos = build_prototypes()["mood_tags"]

    def parse(self, query: str) -> List[str]:
        q = _clean_text(query)
        rule_tags = rule_pick(q, MOOD_RULES, MOOD_CANDIDATES, limit=2)
        if len(rule_tags) >= 2 or not self.embedder:
            return rule_tags
        scores = embedding_scores(self.embedder, q, self.protos)
        picked = [t for t in _select_top_tags(scores, top_n=2, allowed=MOOD_CANDIDATES) if scores.get(t, 0.0) >= self.threshold]
        return _merge_limit(rule_tags, picked, limit=2, allowed=MOOD_CANDIDATES)


class SceneParser:
    def __init__(self, *, embedder: Optional[Embedder] = None, threshold: float = 0.35):
        self.embedder = embedder
        self.threshold = float(threshold)
        self.protos = build_prototypes()["scene_tags"]

    def parse(self, query: str) -> List[str]:
        q = _clean_text(query)
        rule_tags = rule_pick(q, SCENE_RULES, SCENE_CANDIDATES, limit=2)
        if len(rule_tags) >= 2 or not self.embedder:
            return rule_tags
        scores = embedding_scores(self.embedder, q, self.protos)
        picked = [t for t in _select_top_tags(scores, top_n=2, allowed=SCENE_CANDIDATES) if scores.get(t, 0.0) >= self.threshold]
        return _merge_limit(rule_tags, picked, limit=2, allowed=SCENE_CANDIDATES)


class StyleParser:
    def __init__(self, *, embedder: Optional[Embedder] = None, threshold: float = 0.35):
        self.embedder = embedder
        self.threshold = float(threshold)
        self.protos = build_prototypes()["style_tags"]

    def parse(self, query: str) -> List[str]:
        q = _clean_text(query)
        rule_tags = rule_pick(q, STYLE_RULES, STYLE_CANDIDATES, limit=2)
        if len(rule_tags) >= 2 or not self.embedder:
            return rule_tags
        scores = embedding_scores(self.embedder, q, self.protos)
        picked = [t for t in _select_top_tags(scores, top_n=2, allowed=STYLE_CANDIDATES) if scores.get(t, 0.0) >= self.threshold]
        return _merge_limit(rule_tags, picked, limit=2, allowed=STYLE_CANDIDATES)


class LengthClassifier:
    def classify(self, query: str) -> str:
        q = _clean_text(query)
        if re.search(r"(短篇|很短|小故事|短一点|快餐)", q):
            return "短篇"
        if re.search(r"(中篇|适中|不太长)", q):
            return "中篇"
        if re.search(r"(长篇|长一点|大部头)", q):
            return "长篇"
        if re.search(r"(超长|非常长|全集|系列|上中下|全[一二三四五六七八九十]册)", q):
            return "超长篇"
        return ""


class DifficultyEstimator:
    def estimate(self, query: str, style_tags: List[str], length_type: str) -> int:
        q = _clean_text(query)
        score = 3
        if re.search(r"(轻松|不费脑|简单|入门|漫画|短故事)", q):
            score -= 2
        if re.search(r"(烧脑|复杂|高密度|晦涩|难|硬核)", q):
            score += 2
        if "哲学思考" in style_tags:
            score += 2
        if "经典文学" in style_tags or "历史" in style_tags:
            score += 1
        if length_type in ("长篇", "超长篇"):
            score += 1
        if length_type == "短篇":
            score -= 1
        return _clip_int(score, 1, 5)


class PaceEstimator:
    def estimate(self, query: str, style_tags: List[str]) -> int:
        q = _clean_text(query)
        score = 3
        if re.search(r"(停不下来|上头|强情节|爽|节奏快|追读|反转)", q):
            score += 2
        if re.search(r"(慢热|抒情|细腻|平静|舒缓)", q):
            score -= 1
        if "悬疑" in style_tags:
            score += 1
        if "散文" in style_tags or "哲学思考" in style_tags:
            score -= 1
        return _clip_int(score, 1, 5)


class TagEnricherV1:
    """
    Intent Tag Enrichment Stage V1
    - rule-first
    - embedding-second (optional)
    - strict controlled tag set
    - fallback defaults if still empty/invalid
    """

    def __init__(
        self,
        *,
        embedder: Optional[Embedder] = None,
        embedding_threshold: float = 0.35,
    ):
        self.mood_parser = MoodParser(embedder=embedder, threshold=embedding_threshold)
        self.scene_parser = SceneParser(embedder=embedder, threshold=embedding_threshold)
        self.style_parser = StyleParser(embedder=embedder, threshold=embedding_threshold)
        self.difficulty_estimator = DifficultyEstimator()
        self.pace_estimator = PaceEstimator()
        self.length_classifier = LengthClassifier()

    def parse(self, query: str) -> Dict[str, Any]:
        q = _clean_text(query)
        mood = self.mood_parser.parse(q)
        scene = self.scene_parser.parse(q)
        style = self.style_parser.parse(q)
        length_type = self.length_classifier.classify(q)
        difficulty = self.difficulty_estimator.estimate(q, style, length_type or "中篇")
        pace = self.pace_estimator.estimate(q, style)

        out: Dict[str, Any] = {
            "mood_tags": [t for t in mood if t in MOOD_CANDIDATES][:2],
            "scene_tags": [t for t in scene if t in SCENE_CANDIDATES][:2],
            "difficulty": int(_clip_int(difficulty, 1, 5)),
            "pace": int(_clip_int(pace, 1, 5)),
            "length_type": length_type if length_type in LENGTH_TYPES else "",
            "style_tags": [t for t in style if t in STYLE_CANDIDATES][:2],
        }

        # If cannot 판단: apply strict defaults.
        if not out["mood_tags"]:
            out["mood_tags"] = list(DEFAULT_OUTPUT["mood_tags"])
        if not out["scene_tags"]:
            out["scene_tags"] = list(DEFAULT_OUTPUT["scene_tags"])
        if not out["style_tags"]:
            out["style_tags"] = list(DEFAULT_OUTPUT["style_tags"])
        if not out["length_type"]:
            out["length_type"] = DEFAULT_OUTPUT["length_type"]

        # difficulty/pace default only when query extremely vague.
        if not q or re.fullmatch(r"[。！？!?，,.;；\s]+", q):
            out["difficulty"] = DEFAULT_OUTPUT["difficulty"]
            out["pace"] = DEFAULT_OUTPUT["pace"]

        return out

    def parse_json(self, query: str) -> str:
        """Strict JSON string output."""
        return json.dumps(self.parse(query), ensure_ascii=False)


def build_default_enricher(enable_embedding: bool = True, threshold: float = 0.35) -> TagEnricherV1:
    embedder = SentenceTransformerEmbedder(MODEL_NAME) if enable_embedding else None
    return TagEnricherV1(embedder=embedder, embedding_threshold=threshold)


def demo() -> None:
    enricher = build_default_enricher(enable_embedding=True)
    samples = [
        "想看睡前能让人平静下来的书",
        "通勤路上适合看的轻松小说",
        "类似百年孤独那种风格",
        "想看很烧脑的推理悬疑",
        "随便推荐点书",
    ]
    for s in samples:
        print(s)
        print(enricher.parse_json(s))
        print()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    demo()

