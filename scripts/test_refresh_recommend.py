from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import requests


API = os.getenv("BOOKRECOMMEND_API", "http://127.0.0.1:8001/recommend")


def post(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(API, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def ids_from(resp: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for b in resp.get("books") or []:
        bid = str(b.get("book_id") or "").strip()
        if bid:
            out.append(bid)
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    query = "周末下午 想读点科幻"
    exclude_ids: List[str] = []
    seen: set[str] = set()

    for i in range(1, 41):
        payload = {"query": query, "top_k": 3, "exclude_book_ids": exclude_ids}
        try:
            resp = post(payload)
        except requests.HTTPError as e:
            r = e.response
            if r is not None and r.status_code == 409:
                try:
                    detail = r.json().get("detail")
                except Exception:
                    detail = r.text
                print(json.dumps({"step": f"refresh_{i}", "status": 409, "detail": detail}, ensure_ascii=False))
                print("[test_refresh_recommend] OK (exhausted)")
                return 0
            raise

        ids = ids_from(resp)
        print(json.dumps({"step": f"refresh_{i}", "exclude_count": len(exclude_ids), "ids": ids}, ensure_ascii=False))

        inter = set(ids).intersection(seen)
        assert not inter, f"refresh returned previously seen ids: {sorted(list(inter))}"

        for x in ids:
            if x not in seen:
                seen.add(x)
                exclude_ids.append(x)

    print("[test_refresh_recommend] WARNING: no exhaustion within 40 refreshes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

