from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ml.system_config import load_system_config

_CFG = load_system_config()
_LAM = _CFG.diversity_lambda_by_intent()
_THR = _CFG.diversity_threshold_by_intent()

WEAK_LAMBDA = float(_LAM.get("weak", 0.60))
WEAK_SCORE_GATE_RATIO = float(_THR.get("weak", 0.76))

MEDIUM_LAMBDA = float(_LAM.get("medium", 0.68))
MEDIUM_SCORE_GATE_RATIO = float(_THR.get("medium", 0.82))

STRONG_LAMBDA = float(_LAM.get("strong", 0.74))
STRONG_SCORE_GATE_RATIO = float(_THR.get("strong", 0.86))

DEFAULT_FALLBACK = {
    "difficulty": 3,
    "pace": 3,
    "length_type": "中篇",
}

FALLBACK_TOP_N = int(_CFG.diversity_fallback_top_n() or 7)


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _to_tag_set(x: Any) -> set[str]:
    return set([str(t).strip() for t in _as_list(x) if str(t).strip()])


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    u = a.union(b)
    if not u:
        return 0.0
    return float(len(a.intersection(b)) / len(u))


def _intent_tag_non_empty(items: Any) -> bool:
    for it in _as_list(items):
        if isinstance(it, dict):
            t = str(it.get("tag") or "").strip()
            if t:
                return True
        else:
            t = str(it or "").strip()
            if t:
                return True
    return False


def _intent_value(intent: Dict[str, Any], key: str) -> Any:
    v = (intent or {}).get(key)
    if isinstance(v, dict) and "value" in v:
        return v.get("value")
    return None


def _intent_confidence(intent: Dict[str, Any], key: str) -> Optional[float]:
    v = (intent or {}).get(key)
    if not isinstance(v, dict):
        return None
    c = v.get("confidence", None)
    if c is None:
        return None
    try:
        return float(c)
    except Exception:
        return None


def _is_explicit_value(intent: Dict[str, Any], key: str) -> bool:
    vv = _intent_value(intent, key)
    if vv is None:
        return False
    if isinstance(vv, (int, float)):
        return int(vv) != int(DEFAULT_FALLBACK.get(key, 0))
    return str(vv).strip() != str(DEFAULT_FALLBACK.get(key, "")).strip()


def _any_tag_conf_ge(intent: Dict[str, Any], thr: float = 0.7) -> bool:
    for key in ["mood_tags", "scene_tags", "style_tags"]:
        for it in _as_list((intent or {}).get(key)):
            if not isinstance(it, dict):
                continue
            c = it.get("confidence", None)
            if c is None:
                continue
            try:
                if float(c) >= float(thr):
                    return True
            except Exception:
                continue
    return False


def _max_tag_conf(intent: Dict[str, Any], key: str) -> Optional[float]:
    best: Optional[float] = None
    for it in _as_list((intent or {}).get(key)):
        if not isinstance(it, dict):
            continue
        c = it.get("confidence", None)
        if c is None:
            continue
        try:
            cf = float(c)
        except Exception:
            continue
        best = cf if best is None else max(best, cf)
    return best


def intent_strength_score(query: str, intent: Dict[str, Any]) -> int:
    it = intent or {}
    score = 0

    scene_conf = _max_tag_conf(it, "scene_tags")
    if _intent_tag_non_empty(it.get("scene_tags")) and scene_conf is not None and scene_conf >= 0.7:
        score += 2
    style_conf = _max_tag_conf(it, "style_tags")
    if _intent_tag_non_empty(it.get("style_tags")) and style_conf is not None and style_conf >= 0.7:
        score += 2
    if _is_explicit_value(it, "difficulty") and (_intent_confidence(it, "difficulty") or 0.0) >= 0.7:
        score += 2
    if _is_explicit_value(it, "pace") and (_intent_confidence(it, "pace") or 0.0) >= 0.7:
        score += 2
    if _is_explicit_value(it, "length_type") and (_intent_confidence(it, "length_type") or 0.0) >= 0.7:
        score += 2

    if _intent_tag_non_empty(it.get("mood_tags")):
        score += 1
    if _any_tag_conf_ge(it, 0.7):
        score += 1
    if len((query or "").strip()) >= 10:
        score += 1

    return int(score)


def intent_strength_level(score: int) -> str:
    s = int(score)
    if s <= 1:
        return "weak"
    if 2 <= s <= 3:
        return "medium"
    return "strong"


class EmbeddingSimilarityCalculator:
    def cosine_similarity(self, v1: Sequence[float], v2: Sequence[float]) -> float:
        a = np.asarray(v1, dtype=np.float32)
        b = np.asarray(v2, dtype=np.float32)
        if a.size == 0 or b.size == 0:
            return 0.0
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


class TagSimilarityCalculator:
    def tag_similarity(self, meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> float:
        mood_a, mood_b = _to_tag_set(meta_a.get("mood_tags")), _to_tag_set(meta_b.get("mood_tags"))
        style_a, style_b = _to_tag_set(meta_a.get("style_tags")), _to_tag_set(meta_b.get("style_tags"))
        scene_a, scene_b = _to_tag_set(meta_a.get("scene_tags")), _to_tag_set(meta_b.get("scene_tags"))

        mood = _jaccard(mood_a, mood_b)
        style = _jaccard(style_a, style_b)
        scene = _jaccard(scene_a, scene_b)
        return float(0.4 * mood + 0.4 * style + 0.2 * scene)


@dataclass
class FinalSimilarity:
    emb_calc: EmbeddingSimilarityCalculator
    tag_calc: TagSimilarityCalculator

    def similarity(self, cand_a: Dict[str, Any], cand_b: Dict[str, Any]) -> float:
        emb_a = cand_a.get("embedding_vector") if isinstance(cand_a.get("embedding_vector"), list) else []
        emb_b = cand_b.get("embedding_vector") if isinstance(cand_b.get("embedding_vector"), list) else []
        meta_a = cand_a.get("metadata") if isinstance(cand_a.get("metadata"), dict) else {}
        meta_b = cand_b.get("metadata") if isinstance(cand_b.get("metadata"), dict) else {}

        emb_sim = self.emb_calc.cosine_similarity(emb_a, emb_b)
        tag_sim = self.tag_calc.tag_similarity(meta_a, meta_b)
        return float(0.7 * emb_sim + 0.3 * tag_sim)


class DiversityValidator:
    def __init__(self):
        self.selected: List[Dict[str, Any]] = []

    def can_add(self, cand: Dict[str, Any]) -> bool:
        if not self.selected:
            return True
        meta = cand.get("metadata") if isinstance(cand.get("metadata"), dict) else {}
        cand_style = _to_tag_set(meta.get("style_tags"))
        cand_mood = _to_tag_set(meta.get("mood_tags"))

        for s in self.selected:
            smeta = s.get("metadata") if isinstance(s.get("metadata"), dict) else {}
            s_style = _to_tag_set(smeta.get("style_tags"))
            s_mood = _to_tag_set(smeta.get("mood_tags"))
            if cand_style and s_style and cand_style == s_style:
                return False
            if cand_mood and s_mood and cand_mood == s_mood:
                return False
        return True

    def violation(self, cand: Dict[str, Any]) -> List[str]:
        """
        Return a list of violated constraint names if cand were added.
        """
        if not self.selected:
            return []
        meta = cand.get("metadata") if isinstance(cand.get("metadata"), dict) else {}
        cand_style = _to_tag_set(meta.get("style_tags"))
        cand_mood = _to_tag_set(meta.get("mood_tags"))
        violated: List[str] = []
        for s in self.selected:
            smeta = s.get("metadata") if isinstance(s.get("metadata"), dict) else {}
            s_style = _to_tag_set(smeta.get("style_tags"))
            s_mood = _to_tag_set(smeta.get("mood_tags"))
            if cand_style and s_style and cand_style == s_style:
                violated.append("style_tags_identical")
            if cand_mood and s_mood and cand_mood == s_mood:
                violated.append("mood_tags_identical")
        return sorted(list(set(violated)))

    def add(self, cand: Dict[str, Any]) -> None:
        self.selected.append(cand)


class MMRSelector:
    def __init__(self, sim: FinalSimilarity, lam: float):
        self.sim = sim
        self.lam = float(lam)

    def mmr_score(self, relevance: float, similarity_to_selected: float) -> float:
        return float(self.lam * relevance - (1.0 - self.lam) * similarity_to_selected)

    def select_topk(self, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        k = max(0, min(int(top_k), len(candidates)))
        if k == 0:
            return []

        # Step1: Top1 by relevance
        remaining = list(candidates)
        remaining.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
        selected: List[Dict[str, Any]] = [remaining.pop(0)]

        validator = DiversityValidator()
        validator.add(selected[0])

        while len(selected) < k and remaining:
            best_idx = None
            best_mmr = None

            for i, c in enumerate(remaining):
                if not validator.can_add(c):
                    continue
                rel = float(c.get("final_score") or 0.0)
                sim_to_sel = 0.0
                for s in selected:
                    sim_to_sel = max(sim_to_sel, self.sim.similarity(c, s))
                mmr = self.mmr_score(rel, sim_to_sel)
                if best_mmr is None or mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            if best_idx is None:
                # No candidate can satisfy constraints.
                # As required: keep higher relevance first, then optimize diversity.
                # Fallback to highest relevance remaining to ensure we can output TopK.
                remaining.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
                chosen = remaining.pop(0)
                selected.append(chosen)
                validator.add(chosen)
                continue

            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            validator.add(chosen)

        return selected


class DiversitySelectorV1:
    def __init__(
        self,
        *,
        emb_calc: Optional[EmbeddingSimilarityCalculator] = None,
        tag_calc: Optional[TagSimilarityCalculator] = None,
    ):
        self.emb_calc = emb_calc or EmbeddingSimilarityCalculator()
        self.tag_calc = tag_calc or TagSimilarityCalculator()
        self.sim = FinalSimilarity(self.emb_calc, self.tag_calc)

    def select(self, user_query: str, intent: Dict[str, Any], candidates_top10: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        # user_query / intent reserved for reason generation (no new inference here)
        _ = (user_query or "").strip()
        _intent = intent or {}

        cands = list(candidates_top10 or [])
        cands.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)

        if not cands:
            return []

        score = intent_strength_score(_, _intent)
        level = intent_strength_level(score)
        if level == "strong":
            lam = STRONG_LAMBDA
            score_gate_ratio = STRONG_SCORE_GATE_RATIO
        elif level == "medium":
            lam = MEDIUM_LAMBDA
            score_gate_ratio = MEDIUM_SCORE_GATE_RATIO
        else:
            lam = WEAK_LAMBDA
            score_gate_ratio = WEAK_SCORE_GATE_RATIO
        mmr = MMRSelector(self.sim, lam=lam)

        top1_score = float(cands[0].get("final_score") or 0.0)
        gate = float(top1_score * score_gate_ratio)
        eligible = [c for c in cands if float(c.get("final_score") or 0.0) >= gate]

        if len(eligible) < int(top_k):
            eligible = cands[: min(FALLBACK_TOP_N, len(cands))]

        selected = mmr.select_topk(eligible, top_k=top_k)
        out: List[Dict[str, Any]] = []

        for i, c in enumerate(selected):
            rel = float(c.get("final_score") or 0.0)
            if i == 0:
                div = 0.0
                sim_to_prev = 0.0
            else:
                sims = [self.sim.similarity(c, s) for s in selected[:i]]
                sim_to_prev = max(sims) if sims else 0.0
                div = float(1.0 - sim_to_prev)

            reason_parts = []
            reason_parts.append("reranker得分靠前")
            if i > 0:
                reason_parts.append(f"与已选书相似度{sim_to_prev:.2f}，形成差异化补充")
            out.append(
                {
                    "book_id": str(c.get("book_id") or ""),
                    "final_score": rel,
                    "diversity_score": div,
                    "selection_reason": "；".join(reason_parts),
                }
            )

        return out

