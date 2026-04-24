from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.recommender_service import RecommendService


EVAL_PATH = ROOT / "data" / "evaluation_set_v1.jsonl"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        items.append(json.loads(s))
    return items


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not EVAL_PATH.exists():
        print(f"missing evaluation set: {EVAL_PATH}")
        return 2

    svc = RecommendService.get()
    evals = _read_jsonl(EVAL_PATH)
    by_strength = Counter()

    t0 = time.perf_counter()
    for i, it in enumerate(evals, start=1):
        strength = str(it.get("strength") or "unknown")
        query = str(it.get("query") or "").strip()
        if not query:
            continue
        by_strength[strength] += 1
        res = svc.recommend(query, top_k=3)
        top_ids = [r.book_id for r in res]
        print(json.dumps({"i": i, "strength": strength, "query": query, "top_ids": top_ids}, ensure_ascii=False))

    ms = (time.perf_counter() - t0) * 1000.0
    print(json.dumps({"summary": {"n": sum(by_strength.values()), "by_strength": dict(by_strength), "elapsed_ms": ms}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

