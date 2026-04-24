from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ml.system_config import load_system_config

def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _intent_tags(intent_items: Any) -> List[str]:
    """
    intent_items: [{"tag": "...", "confidence": 0.x}, ...]
    """
    out: List[Tuple[str, float]] = []
    for it in _as_list(intent_items):
        if not isinstance(it, dict):
            continue
        tag = str(it.get("tag") or "").strip()
        if not tag:
            continue
        conf = float(it.get("confidence") or 0.0)
        out.append((tag, conf))
    out.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in out]


def _intent_value(intent: Dict[str, Any], key: str, default: Any) -> Any:
    v = (intent or {}).get(key, {})
    if isinstance(v, dict) and "value" in v:
        return v.get("value", default)
    return default


def _intent_tag_set_with_conf(intent_items: Any, min_conf: float = 0.7) -> set[str]:
    out: set[str] = set()
    for it in _as_list(intent_items):
        if not isinstance(it, dict):
            continue
        tag = str(it.get("tag") or "").strip()
        if not tag:
            continue
        try:
            conf = float(it.get("confidence") or 0.0)
        except Exception:
            conf = 0.0
        if conf >= float(min_conf):
            out.add(tag)
    return out


def _intent_value_conf(intent: Dict[str, Any], key: str) -> float:
    v = (intent or {}).get(key, {})
    if not isinstance(v, dict):
        return 0.0
    try:
        return float(v.get("confidence") or 0.0)
    except Exception:
        return 0.0


def _apply_hard_filters(
    candidates: List[Dict[str, Any]],
    intent: Dict[str, Any],
    *,
    top_k: int,
    min_conf: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Strong-intent hard filters (style-first):
    - If intent has high-conf style tags: prioritize keeping candidates that intersect them.
      Scene/length are NOT allowed to filter out style-matching items (small dataset).
    - Scene/length are used only as secondary constraints when style constraint is absent.
    """
    it = intent or {}
    want_style = _intent_tag_set_with_conf(it.get("style_tags"), min_conf=min_conf)
    want_scene = _intent_tag_set_with_conf(it.get("scene_tags"), min_conf=min_conf)
    want_len = str(_intent_value(it, "length_type", "") or "").strip()
    want_len_conf = _intent_value_conf(it, "length_type")
    use_len = bool(want_len) and want_len_conf >= float(min_conf)

    # Only enable hard filtering in strong-intent mode:
    # we treat presence of high-confidence style tags as the strong trigger.
    if not want_style:
        return candidates

    def meta(c: Dict[str, Any]) -> Dict[str, Any]:
        return c.get("metadata") if isinstance(c.get("metadata"), dict) else {}

    def has_intersection(have: Any, want: set[str]) -> bool:
        if not want:
            return True
        have_set = set([str(x).strip() for x in _as_list(have) if str(x).strip()])
        return bool(have_set.intersection(want))

    # style-first pool: keep all style-matching candidates; if not enough, append others.
    style_pool = [c for c in candidates if has_intersection(meta(c).get("style_tags"), want_style)]
    if len(style_pool) >= int(top_k):
        return style_pool

    others = [c for c in candidates if c not in style_pool]
    return style_pool + others


class SemanticScorer:
    def score(self, candidate: Dict[str, Any]) -> float:
        # No embedding recompute allowed; just reuse existing score.
        return float(candidate.get("embedding_score") or 0.0)


class TagScorer:
    """
    mood/style overlap -> tag_match_score in [0, 0.5]
    """

    def score(self, intent: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        want_mood = set(_intent_tags(intent.get("mood_tags")))
        want_style = set(_intent_tags(intent.get("style_tags")))

        have_mood = set([str(x).strip() for x in _as_list(metadata.get("mood_tags")) if str(x).strip()])
        have_style = set([str(x).strip() for x in _as_list(metadata.get("style_tags")) if str(x).strip()])

        want = want_mood.union(want_style)
        if not want:
            return 0.0

        overlap = len(want.intersection(have_mood.union(have_style)))
        if overlap <= 0:
            return 0.0

        max_possible = len(want)
        if overlap >= max_possible:
            return 0.5

        ratio = overlap / max_possible
        # partial match -> [0.1, 0.3]
        return float(0.1 + 0.2 * ratio)


class SceneScorer:
    """
    scene overlap -> scene_match_score in [0, 0.3]
    """

    def score(self, intent: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        want_scene = set(_intent_tags(intent.get("scene_tags")))
        have_scene = set([str(x).strip() for x in _as_list(metadata.get("scene_tags")) if str(x).strip()])

        if not want_scene:
            return 0.0

        overlap = len(want_scene.intersection(have_scene))
        if overlap <= 0:
            return 0.0

        max_possible = len(want_scene)
        if overlap >= max_possible:
            return 0.3

        ratio = overlap / max_possible
        # partial match -> [0.1, 0.2]
        return float(0.1 + 0.1 * ratio)


class PenaltyCalculator:
    """
    Return positive penalty_score.
    """

    def __init__(self) -> None:
        # debug hook: populated on each penalty() call
        self.last_breakdown: Dict[str, float] = {"difficulty": 0.0, "pace": 0.0, "length_type": 0.0}

    def penalty(self, intent: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        want_diff = int(_intent_value(intent, "difficulty", 3) or 3)
        want_pace = int(_intent_value(intent, "pace", 3) or 3)
        want_len = str(_intent_value(intent, "length_type", "") or "")

        have_diff = int(metadata.get("difficulty", 0) or 0)
        have_pace = int(metadata.get("pace", 0) or 0)
        have_len = str(metadata.get("length_type", "") or "")

        p_diff = 0.0
        p_pace = 0.0
        p_len = 0.0

        # difficulty penalty
        if have_diff and abs(want_diff - have_diff) >= 1:
            p_diff = 0.15

        # pace penalty
        if have_pace and abs(want_pace - have_pace) >= 1:
            p_pace = 0.1

        # length_type penalty
        if want_len and have_len and have_len != want_len:
            p_len = 0.1

        self.last_breakdown = {"difficulty": float(p_diff), "pace": float(p_pace), "length_type": float(p_len)}
        p = float(p_diff + p_pace + p_len)
        return max(0.0, p)


@dataclass
class ScoreAggregator:
    w_semantic: float = 0.5
    w_tag: float = 0.3
    w_scene: float = 0.2

    def aggregate(
        self,
        semantic_score: float,
        tag_match_score: float,
        scene_match_score: float,
        penalty_score: float,
    ) -> float:
        return float(
            self.w_semantic * float(semantic_score)
            + self.w_tag * float(tag_match_score)
            + self.w_scene * float(scene_match_score)
            - float(penalty_score)
        )


class RerankerV1:
    def __init__(
        self,
        *,
        semantic_scorer: Optional[SemanticScorer] = None,
        tag_scorer: Optional[TagScorer] = None,
        scene_scorer: Optional[SceneScorer] = None,
        penalty_calculator: Optional[PenaltyCalculator] = None,
        aggregator: Optional[ScoreAggregator] = None,
    ):
        if aggregator is None:
            w = load_system_config().rerank_weights()
            aggregator = ScoreAggregator(
                w_semantic=float(w.get("semantic", 0.5)),
                w_tag=float(w.get("tag", 0.3)),
                w_scene=float(w.get("scene", 0.2)),
            )
        self.semantic_scorer = semantic_scorer or SemanticScorer()
        self.tag_scorer = tag_scorer or TagScorer()
        self.scene_scorer = scene_scorer or SceneScorer()
        self.penalty_calculator = penalty_calculator or PenaltyCalculator()
        self.aggregator = aggregator or ScoreAggregator()

    def rerank(
        self,
        user_query: str,
        intent: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        # user_query currently unused; reserved for future LLM/feature scorer injection.
        _ = (user_query or "").strip()

        # Strong-intent hard filtering before scoring (minimal, most direct improvement).
        # We treat presence of any high-confidence constraint as strong.
        filtered_candidates = _apply_hard_filters(candidates or [], intent or {}, top_k=int(top_k), min_conf=0.7)

        scored: List[Dict[str, Any]] = []
        for c in filtered_candidates or []:
            if not isinstance(c, dict):
                continue
            book_id = str(c.get("book_id") or "").strip()
            metadata = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
            embedding_vector = c.get("embedding_vector") if isinstance(c.get("embedding_vector"), list) else []

            semantic = float(self.semantic_scorer.score(c))
            tag_score = float(self.tag_scorer.score(intent or {}, metadata))
            scene_score = float(self.scene_scorer.score(intent or {}, metadata))
            penalty = float(self.penalty_calculator.penalty(intent or {}, metadata))
            penalty_breakdown = getattr(self.penalty_calculator, "last_breakdown", {})
            final = float(self.aggregator.aggregate(semantic, tag_score, scene_score, penalty))

            scored.append(
                {
                    "book_id": book_id,
                    "final_score": final,
                    "metadata": metadata,
                    "embedding_vector": embedding_vector,
                    "rerank_breakdown": {
                        "semantic_score": semantic,
                        "tag_match_score": tag_score,
                        "scene_match_score": scene_score,
                        "penalty_score": penalty,
                        "penalty_breakdown": {
                            "difficulty": float(penalty_breakdown.get("difficulty", 0.0) or 0.0),
                            "pace": float(penalty_breakdown.get("pace", 0.0) or 0.0),
                            "length_type": float(penalty_breakdown.get("length_type", 0.0) or 0.0),
                        },
                        "final_score": final,
                        "similarity_to_previous_ranked_items": [],
                    },
                }
            )

        scored.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
        k = max(0, min(int(top_k), len(scored)))
        return scored[:k]

