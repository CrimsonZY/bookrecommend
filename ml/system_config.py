from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "system_config.json"


def _deep_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        v = x.strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
        if v in ("0", "false", "no", "n", "off"):
            return False
    if isinstance(x, (int, float)):
        return bool(x)
    return bool(default)


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


@dataclass(frozen=True)
class SystemConfig:
    raw: Dict[str, Any]
    path: Path

    @property
    def system_version(self) -> str:
        return str(_deep_get(self.raw, "system_version", "") or "")

    @property
    def debug_mode(self) -> bool:
        return _as_bool(_deep_get(self.raw, "pipeline.debug_mode", False), False)

    def retrieval_weights(self) -> Dict[str, float]:
        base = _deep_get(self.raw, "retrieval.weights", {}) or {}
        return {
            "query_score": _as_float(base.get("query_score"), 0.4),
            "mood_score": _as_float(base.get("mood_score"), 0.2),
            "scene_score": _as_float(base.get("scene_score"), 0.2),
            "style_score": _as_float(base.get("style_score"), 0.2),
        }

    def rerank_weights(self) -> Dict[str, float]:
        base = _deep_get(self.raw, "rerank.weights", {}) or {}
        return {
            "semantic": _as_float(base.get("semantic"), 0.5),
            "tag": _as_float(base.get("tag"), 0.3),
            "scene": _as_float(base.get("scene"), 0.2),
        }

    def diversity_lambda_by_intent(self) -> Dict[str, float]:
        base = _deep_get(self.raw, "diversity.lambda_by_intent", {}) or {}
        return {
            "weak": _as_float(base.get("weak"), 0.60),
            "medium": _as_float(base.get("medium"), 0.68),
            "strong": _as_float(base.get("strong"), 0.74),
        }

    def diversity_threshold_by_intent(self) -> Dict[str, float]:
        base = _deep_get(self.raw, "diversity.threshold_by_intent", {}) or {}
        return {
            "weak": _as_float(base.get("weak"), 0.76),
            "medium": _as_float(base.get("medium"), 0.82),
            "strong": _as_float(base.get("strong"), 0.86),
        }

    def diversity_fallback_top_n(self) -> int:
        return _as_int(_deep_get(self.raw, "diversity.fallback_top_n", 7), 7)

    def frozen_asset_path(self, key: str) -> str:
        # key in {"embedding_path","index_path","books_path","label_policy_path"}
        return str(_deep_get(self.raw, f"frozen_assets.{key}", "") or "")

    def debug_truncate_intro_chars(self) -> int:
        return _as_int(_deep_get(self.raw, "pipeline.debug.truncate_content_intro_chars", 120), 120)

    def debug_emit_stdout_ndjson(self) -> bool:
        return _as_bool(_deep_get(self.raw, "pipeline.debug.emit_stdout_ndjson", True), True)

    def refresh_min_final_score(self) -> float:
        return _as_float(_deep_get(self.raw, "pipeline.refresh.min_final_score", 0.18), 0.18)


_CACHED: Optional[SystemConfig] = None


def load_system_config(path: Optional[str | Path] = None, *, use_cache: bool = True) -> SystemConfig:
    global _CACHED
    if use_cache and _CACHED is not None:
        return _CACHED

    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    raw = json.loads(p.read_text(encoding="utf-8"))
    cfg = SystemConfig(raw=raw if isinstance(raw, dict) else {}, path=p)
    if use_cache:
        _CACHED = cfg
    return cfg

