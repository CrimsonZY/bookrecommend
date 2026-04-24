from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_PATH = Path("data/douban_top100_clean_refilled.json")
OUTPUT_PATH = Path("data/books_enriched_v2.json")
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


MOOD_CANDIDATES = ["治愈", "温暖", "平静", "热血", "沉重", "悲伤", "哲思", "孤独", "希望", "幽默", "浪漫"]
SCENE_CANDIDATES = ["睡前", "通勤", "周末下午", "长期阅读", "旅行途中", "碎片阅读"]
STYLE_CANDIDATES = ["文学", "小说", "社会观察", "悬疑", "科幻", "散文", "历史", "爱情", "人物传记", "哲学思考"]
LENGTH_TYPES = ["短篇", "中篇", "长篇", "超长篇"]


def _clean_text(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def _clean_multiline(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def build_book_text(book: Dict[str, Any]) -> str:
    title = _clean_text(book.get("title"))
    author = _clean_text(book.get("author"))
    era_nat = _clean_text(book.get("author_era_nationality"))
    intro = _clean_multiline(book.get("content_intro"))
    pages = book.get("pages")
    pages_s = str(pages).strip() if pages is not None else ""
    return "\n".join(
        [
            f"书名：{title}",
            f"作者：{author}",
            f"作者国籍/年代：{era_nat}",
            f"内容简介：{intro}",
            f"页数：{pages_s}",
        ]
    ).strip()


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def _pages_to_length_type(pages: Any) -> str:
    try:
        p = int(pages)
    except Exception:
        p = 0
    if p and p < 200:
        return "短篇"
    if 200 <= p < 400:
        return "中篇"
    if 400 <= p < 800:
        return "长篇"
    if p >= 800:
        return "超长篇"
    return ""


# -------------------------
# Rule-based taggers
# -------------------------


STYLE_RULES: List[Tuple[str, str]] = [
    ("科幻", r"(科幻|宇宙|外星|星际|未来|太空|三体|基地|机器人|时间旅行|文明)"),
    ("悬疑", r"(悬疑|推理|命案|凶手|侦探|谜|疑云|犯罪|作案|谋杀|追凶)"),
    ("散文", r"(散文|随笔|杂文|札记|书信|日记|小品文)"),
    ("人物传记", r"(传记|回忆录|自传|口述史|时代.*(人物|中国)|人物.*传)"),
    ("历史", r"(历史|王朝|帝国|革命|时代|史记|通鉴|朝代|政变|战争)"),
    ("爱情", r"(爱情|相爱|恋爱|婚姻|情人|浪漫|心动)"),
    ("小说", r"(小说|长篇|中篇|短篇|故事|情节|叙事)"),
    ("社会观察", r"(社会|阶层|现实|农村|城市|制度|底层|贫穷|资本|权力)"),
    ("哲学思考", r"(哲学|思考|存在|意义|虚无|自由|信仰|灵魂|伦理|理性)"),
    ("文学", r"(名著|经典|诺贝尔|文学史|四大名著|文学)"),
]


MOOD_RULES: List[Tuple[str, str]] = [
    ("治愈", r"(治愈|疗愈|抚慰|安慰|松弛|拥抱|柔软)"),
    ("温暖", r"(温暖|温柔|善意|陪伴|亲情|友情|小确幸)"),
    ("平静", r"(平静|安静|宁静|从容|淡然|舒缓)"),
    ("热血", r"(热血|冒险|战斗|燃|激情|英雄|奋起)"),
    ("沉重", r"(沉重|压抑|黑暗|绝望|残酷|创伤|暴力)"),
    ("悲伤", r"(悲伤|悲凉|痛苦|哀伤|眼泪|失去|离别)"),
    ("哲思", r"(哲思|思辨|反思|存在|意义|真相)"),
    ("孤独", r"(孤独|寂寞|疏离|一人|无助)"),
    ("希望", r"(希望|重生|救赎|勇气|向前|光)"),
    ("幽默", r"(幽默|搞笑|讽刺|荒诞|笑)"),
    ("浪漫", r"(浪漫|心动|爱情|恋爱|甜蜜)"),
]


def rule_match_tags(text: str, rules: List[Tuple[str, str]]) -> List[str]:
    out: List[str] = []
    for tag, pattern in rules:
        if re.search(pattern, text, flags=re.IGNORECASE):
            out.append(tag)
    # Stable order, unique
    uniq: List[str] = []
    for t in out:
        if t not in uniq:
            uniq.append(t)
    return uniq


def rule_scene_tags(book: Dict[str, Any], mood_tags: List[str], style_tags: List[str], length_type: str) -> List[str]:
    intro = _clean_multiline(book.get("content_intro"))
    title = _clean_text(book.get("title"))
    text = f"{title}\n{intro}"

    scenes: List[str] = []

    if any(t in mood_tags for t in ["治愈", "温暖", "平静"]) and length_type in ("短篇", "中篇", ""):
        scenes.append("睡前")
    if length_type in ("短篇", "中篇") and (re.search(r"(短篇|随笔|散文|碎片|片段|一则|小品)", text) is not None):
        scenes.append("碎片阅读")
    if "悬疑" in style_tags and length_type in ("短篇", "中篇", "长篇"):
        scenes.append("通勤")
    if length_type in ("长篇", "超长篇") or "文学" in style_tags:
        scenes.append("长期阅读")
    if length_type in ("中篇", "长篇") and ("科幻" in style_tags or "悬疑" in style_tags or "文学" in style_tags):
        scenes.append("周末下午")
    if any(t in style_tags for t in ["科幻", "悬疑", "小说"]) and length_type in ("中篇", "长篇"):
        scenes.append("旅行途中")

    # stable unique
    uniq: List[str] = []
    for s in scenes:
        if s not in uniq:
            uniq.append(s)
    return uniq


def rule_difficulty(book: Dict[str, Any], style_tags: List[str]) -> int:
    pages = book.get("pages", 0)
    try:
        p = int(pages)
    except Exception:
        p = 0
    intro = _clean_multiline(book.get("content_intro"))

    # Base on length
    if p <= 200:
        score = 2
    elif p <= 400:
        score = 3
    elif p <= 800:
        score = 4
    else:
        score = 5

    # Topic adjustments
    if any(t in style_tags for t in ["哲学思考", "历史", "文学"]):
        score += 1
    if "散文" in style_tags:
        score -= 1
    if "悬疑" in style_tags:
        score -= 0  # can be easy to read but keep neutral

    # Sentence complexity: approximate by avg sentence length
    sents = re.split(r"[。！？!?]", intro)
    sents = [s.strip() for s in sents if s.strip()]
    if sents:
        avg_len = sum(len(s) for s in sents) / max(1, len(sents))
        if avg_len > 45:
            score += 1
        elif avg_len < 18:
            score -= 1

    return _clamp_int(int(round(score)), 1, 5)


def rule_pace(book: Dict[str, Any], style_tags: List[str], mood_tags: List[str]) -> int:
    intro = _clean_multiline(book.get("content_intro"))
    text = intro

    score = 3
    if re.search(r"(反转|一口气|停不下来|追更|紧张|节奏快|步步惊心|扣人心弦)", text):
        score += 2
    if "悬疑" in style_tags:
        score += 1
    if "科幻" in style_tags and re.search(r"(危机|末日|战争|追逐)", text):
        score += 1
    if "散文" in style_tags or "哲学思考" in style_tags or "哲思" in mood_tags:
        score -= 1
    if "平静" in mood_tags:
        score -= 1
    if re.search(r"(慢热|缓慢|细腻|铺陈|冗长)", text):
        score -= 1

    return _clamp_int(score, 1, 5)


# -------------------------
# Embedding fallback
# -------------------------


@dataclass(frozen=True)
class TagPrototype:
    tag: str
    text: str


def build_prototypes() -> Dict[str, List[TagPrototype]]:
    mood = [
        TagPrototype("治愈", "治愈、抚慰、让人放松的故事，适合情绪低落时阅读。"),
        TagPrototype("温暖", "温暖、温柔、充满善意与陪伴的叙事。"),
        TagPrototype("平静", "平静、安静、舒缓的文字与氛围。"),
        TagPrototype("热血", "热血、冒险、战斗、激情与成长。"),
        TagPrototype("沉重", "沉重、压抑、残酷现实与创伤经历。"),
        TagPrototype("悲伤", "悲伤、离别、失去与哀伤。"),
        TagPrototype("哲思", "思辨、反思人生意义与真相。"),
        TagPrototype("孤独", "孤独、寂寞、疏离与自我对话。"),
        TagPrototype("希望", "希望、重生、救赎、向前的力量。"),
        TagPrototype("幽默", "幽默、讽刺、荒诞与轻松搞笑。"),
        TagPrototype("浪漫", "浪漫、心动、爱情与关系。"),
    ]
    scene = [
        TagPrototype("睡前", "适合睡前阅读：温柔治愈、放松、短篇或节奏舒缓。"),
        TagPrototype("通勤", "适合通勤阅读：节奏快、章节短、易读停靠也能继续。"),
        TagPrototype("周末下午", "适合周末下午：沉浸式阅读，中长篇或故事性强。"),
        TagPrototype("长期阅读", "适合长期阅读：长篇、经典、信息量大，需要投入时间。"),
        TagPrototype("旅行途中", "适合旅行途中：故事性强、冒险或轻快，可一路读下去。"),
        TagPrototype("碎片阅读", "适合碎片阅读：短篇、散文、随笔，随时读一段。"),
    ]
    style = [
        TagPrototype("文学", "文学性强、严肃叙事或经典作品等。"),
        TagPrototype("小说", "以故事叙事为主的虚构作品。"),
        TagPrototype("社会观察", "社会观察：现实题材、阶层、制度、人性与社会问题。"),
        TagPrototype("悬疑", "悬疑推理：命案、谜团、追凶、反转、紧张。"),
        TagPrototype("科幻", "科幻：未来、外星、宇宙、科技、文明。"),
        TagPrototype("散文", "散文随笔：片段式、抒情或随笔体裁。"),
        TagPrototype("历史", "历史：人物、王朝、时代变迁、史实叙事。"),
        TagPrototype("爱情", "爱情：亲密关系、恋爱、婚姻、浪漫。"),
        TagPrototype("人物传记", "人物传记：真实人物生平、回忆录、自传。"),
        TagPrototype("哲学思考", "哲学思考：存在、意义、伦理、思想讨论。"),
    ]
    return {"mood_tags": mood, "scene_tags": scene, "style_tags": style}


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Expect normalized
    return a @ b.T


# -------------------------
# Optional LLM call
# -------------------------


def _env_or(cli_value: str, env_key: str) -> str:
    v = (cli_value or "").strip()
    if v:
        return v
    return (os.environ.get(env_key) or "").strip()


def call_llm_json(
    *,
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
    book: Dict[str, Any],
    book_text: str,
) -> Optional[Dict[str, Any]]:
    """
    Best-effort deterministic classifier.
    Must return a dict with keys:
      mood_scores, scene_scores, style_scores: {tag: score(0-1)}
      difficulty, pace: int 1-5
      length_type: one of LENGTH_TYPES
    """
    provider = (provider or "none").strip().lower()
    if provider in ("none", ""):
        return None

    if not base_url or not model:
        return None

    system = (
        "你是一个图书标签生成器。请严格在给定候选集合中选择，不要发明新标签。"
        "必须输出纯 JSON，不要包含 markdown 代码块。"
    )
    user = {
        "task": "为图书生成标签与评分（确定性）",
        "candidates": {
            "mood_tags": MOOD_CANDIDATES,
            "scene_tags": SCENE_CANDIDATES,
            "style_tags": STYLE_CANDIDATES,
            "length_type": LENGTH_TYPES,
            "difficulty_range": [1, 5],
            "pace_range": [1, 5],
        },
        "book": {
            "subject_id": book.get("subject_id"),
            "title": book.get("title"),
            "author": book.get("author"),
            "author_era_nationality": book.get("author_era_nationality"),
            "pages": book.get("pages"),
        },
        "book_text": book_text[:4000],
        "output_schema": {
            "mood_scores": {"治愈": 0.0},
            "scene_scores": {"睡前": 0.0},
            "style_scores": {"科幻": 0.0},
            "difficulty": 3,
            "pace": 3,
            "length_type": "中篇",
        },
        "rules": [
            "分数范围 0-1，保留两位小数即可",
            "可给 0 分，表示不符合",
            "difficulty/pace 为 1-5 的整数",
            "length_type 必须为候选集合之一",
        ],
    }

    if provider == "ollama":
        # POST {base_url}/api/chat
        url = base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
        }
        return _http_json(url, payload, timeout_seconds=timeout_seconds, api_key="")

    if provider == "openai_compat":
        # POST {base_url}/v1/chat/completions
        url = base_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }
        resp = _http_json(url, payload, timeout_seconds=timeout_seconds, api_key=api_key)
        if not resp:
            return None
        # OpenAI compat schema: choices[0].message.content contains JSON.
        try:
            content = resp["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            return None

    return None


def _http_json(url: str, payload: Dict[str, Any], *, timeout_seconds: float, api_key: str) -> Optional[Dict[str, Any]]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


# -------------------------
# Merge + selection
# -------------------------


def select_by_threshold(scores: Dict[str, float], *, threshold: float, max_tags: int) -> List[str]:
    items = [(k, float(v)) for k, v in scores.items() if k in scores]
    items.sort(key=lambda x: x[1], reverse=True)
    picked = [k for k, v in items if v >= threshold]
    return picked[:max_tags]


def merge_unique(base: List[str], extra: List[str]) -> List[str]:
    out: List[str] = []
    for t in base + extra:
        if t and t not in out:
            out.append(t)
    return out


def compute_embedding_scores(
    model: SentenceTransformer,
    book_text: str,
    prototypes: List[TagPrototype],
) -> Dict[str, float]:
    # Encode book and prototypes in one batch for speed.
    texts = [book_text] + [p.text for p in prototypes]
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embs = embs.astype(np.float32, copy=False)
    # L2 normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embs = embs / norms
    book_emb = embs[0:1, :]
    proto_emb = embs[1:, :]
    sims = cosine_sim_matrix(book_emb, proto_emb)[0]
    return {prototypes[i].tag: float(sims[i]) for i in range(len(prototypes))}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Enrich Douban books with tags (rules + optional LLM + embedding fallback).")
    ap.add_argument("--in", dest="in_path", default=str(INPUT_PATH))
    ap.add_argument("--out", dest="out_path", default=str(OUTPUT_PATH))
    ap.add_argument("--llm-provider", choices=["openai_compat", "ollama", "none"], default="none")
    ap.add_argument("--llm-base-url", default="")
    ap.add_argument("--llm-model", default="")
    ap.add_argument("--llm-api-key", default="")
    ap.add_argument("--timeout-seconds", type=float, default=30.0)
    ap.add_argument("--sleep-ms", type=int, default=0, help="Sleep between LLM calls for rate limiting.")
    ap.add_argument("--fallback-embedding", type=str, default="true", help="true/false")
    ap.add_argument("--tag-threshold", type=float, default=0.35)
    ap.add_argument("--max-tags-per-type", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    books = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(books, list):
        raise SystemExit("input json must be a list")

    llm_provider = args.llm_provider
    llm_base_url = _env_or(args.llm_base_url, "LLM_BASE_URL")
    llm_model = _env_or(args.llm_model, "LLM_MODEL")
    llm_api_key = _env_or(args.llm_api_key, "LLM_API_KEY")

    fallback_embedding = str(args.fallback_embedding).strip().lower() in ("1", "true", "yes", "y")
    threshold = float(args.tag_threshold)
    max_tags = _clamp_int(int(args.max_tags_per_type), 1, 3)

    prototypes_map = build_prototypes()

    model: Optional[SentenceTransformer] = None
    if fallback_embedding:
        model = SentenceTransformer(MODEL_NAME)

    stats = {
        "llm_used": 0,
        "llm_success": 0,
        "fallback_used": 0,
        "mood_nonempty": 0,
        "scene_nonempty": 0,
        "style_nonempty": 0,
        "difficulty_sum": 0,
        "pace_sum": 0,
    }

    enriched: List[Dict[str, Any]] = []

    total = len(books)
    for i, b in enumerate(books, start=1):
        book = dict(b) if isinstance(b, dict) else {}
        text = build_book_text(book)

        title = _clean_text(book.get("title"))
        sid = _clean_text(book.get("subject_id"))

        # 1) rule tags
        style_rule = rule_match_tags(text, STYLE_RULES)
        mood_rule = rule_match_tags(text, MOOD_RULES)
        length_type = _pages_to_length_type(book.get("pages"))
        if not length_type:
            # infer from title keywords
            t = f"{title} {_clean_text(book.get('content_intro'))}"
            if re.search(r"(全集|套装|上中下|全[一二三四五六七八九十]册)", t):
                length_type = "超长篇"
        scene_rule = rule_scene_tags(book, mood_rule, style_rule, length_type or "")

        difficulty_rule = rule_difficulty(book, style_rule)
        pace_rule = rule_pace(book, style_rule, mood_rule)

        # 2) optional LLM scores
        llm_out = call_llm_json(
            provider=llm_provider,
            base_url=llm_base_url,
            model=llm_model,
            api_key=llm_api_key,
            timeout_seconds=float(args.timeout_seconds),
            book=book,
            book_text=text,
        )
        if llm_provider != "none" and llm_base_url and llm_model:
            stats["llm_used"] += 1

        mood_scores: Dict[str, float] = {}
        scene_scores: Dict[str, float] = {}
        style_scores: Dict[str, float] = {}
        difficulty_ai: Optional[int] = None
        pace_ai: Optional[int] = None
        length_ai: Optional[str] = None

        if isinstance(llm_out, dict):
            try:
                mood_scores = {k: float(v) for k, v in (llm_out.get("mood_scores") or {}).items() if k in MOOD_CANDIDATES}
                scene_scores = {k: float(v) for k, v in (llm_out.get("scene_scores") or {}).items() if k in SCENE_CANDIDATES}
                style_scores = {k: float(v) for k, v in (llm_out.get("style_scores") or {}).items() if k in STYLE_CANDIDATES}
                difficulty_ai = int(llm_out.get("difficulty")) if llm_out.get("difficulty") is not None else None
                pace_ai = int(llm_out.get("pace")) if llm_out.get("pace") is not None else None
                length_ai = str(llm_out.get("length_type") or "").strip()
                if length_ai and length_ai not in LENGTH_TYPES:
                    length_ai = ""
                stats["llm_success"] += 1
            except Exception:
                mood_scores = scene_scores = style_scores = {}
                difficulty_ai = pace_ai = None
                length_ai = None

        # 3) embedding fallback if needed
        if fallback_embedding and (not mood_scores or not scene_scores or not style_scores):
            stats["fallback_used"] += 1
            assert model is not None
            if not mood_scores:
                mood_scores = compute_embedding_scores(model, text, prototypes_map["mood_tags"])
            if not scene_scores:
                scene_scores = compute_embedding_scores(model, text, prototypes_map["scene_tags"])
            if not style_scores:
                style_scores = compute_embedding_scores(model, text, prototypes_map["style_tags"])

        mood_ai = select_by_threshold(mood_scores, threshold=threshold, max_tags=max_tags)
        scene_ai = select_by_threshold(scene_scores, threshold=threshold, max_tags=max_tags)
        style_ai = select_by_threshold(style_scores, threshold=threshold, max_tags=max_tags)

        mood_tags = merge_unique(mood_rule, mood_ai)
        scene_tags = merge_unique(scene_rule, scene_ai)
        style_tags = merge_unique(style_rule, style_ai)

        difficulty = _clamp_int(difficulty_ai if difficulty_ai is not None else difficulty_rule, 1, 5)
        pace = _clamp_int(pace_ai if pace_ai is not None else pace_rule, 1, 5)

        length_type_final = length_ai or length_type or _pages_to_length_type(book.get("pages")) or "中篇"

        book["mood_tags"] = mood_tags
        book["scene_tags"] = scene_tags
        book["difficulty"] = int(difficulty)
        book["pace"] = int(pace)
        book["length_type"] = length_type_final
        book["style_tags"] = style_tags

        enriched.append(book)

        # stats
        if mood_tags:
            stats["mood_nonempty"] += 1
        if scene_tags:
            stats["scene_nonempty"] += 1
        if style_tags:
            stats["style_nonempty"] += 1
        stats["difficulty_sum"] += int(difficulty)
        stats["pace_sum"] += int(pace)

        if not args.quiet:
            print(f"[{i:03d}/{total}] {sid} {title}")

        if args.sleep_ms and llm_provider != "none":
            time.sleep(max(0.0, args.sleep_ms / 1000.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary
    avg_diff = stats["difficulty_sum"] / max(1, total)
    avg_pace = stats["pace_sum"] / max(1, total)
    print("\n=== Summary ===")
    print(f"input_books: {total}")
    print(f"output: {out_path}")
    print(f"llm_used: {stats['llm_used']}, llm_success: {stats['llm_success']}")
    print(f"fallback_used: {stats['fallback_used']} (embedding)")
    print(f"coverage_mood: {stats['mood_nonempty']}/{total}")
    print(f"coverage_scene: {stats['scene_nonempty']}/{total}")
    print(f"coverage_style: {stats['style_nonempty']}/{total}")
    print(f"avg_difficulty: {avg_diff:.2f}")
    print(f"avg_pace: {avg_pace:.2f}")
    return 0


if __name__ == "__main__":
    # Make relative paths stable when invoked from anywhere.
    os.chdir(Path(__file__).resolve().parents[1])
    raise SystemExit(main())

