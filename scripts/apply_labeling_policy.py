from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
A_PATH = ROOT / "data" / "books_enriched.json"
B_PATH = ROOT / "data" / "books_enriched_example.json"
OUT_PATH = ROOT / "data" / "books_enriched_v2.json"


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    return [str(x).strip()] if str(x).strip() else []


def _dedup_keep_order(xs: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in xs:
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _intersect_allowed(xs: List[str], allowed: Set[str]) -> List[str]:
    return [x for x in xs if x in allowed]


def _trim_to_max(xs: List[str], max_n: int) -> List[str]:
    if len(xs) <= max_n:
        return xs
    return xs[:max_n]


def normalize_mood_tags(tags: Any, *, allowed: Set[str]) -> List[str]:
    xs = _dedup_keep_order(_as_list(tags))

    # Conflict rule: "治愈" vs ("悲伤" or "沉重") => remove "治愈"
    if "治愈" in xs and ("悲伤" in xs or "沉重" in xs):
        xs = [t for t in xs if t != "治愈"]

    xs = _intersect_allowed(xs, allowed)

    # Count constraint: default 3, upper 4
    if len(xs) > 4:
        # keep more specific tags first (policy-derived heuristic)
        priority = [
            "悲伤",
            "沉重",
            "热血",
            "哲思",
            "幽默",
            "浪漫",
            "希望",
            "平静",
            "温暖",
            "治愈",
            "孤独",
        ]
        rank = {t: i for i, t in enumerate(priority)}
        xs = sorted(xs, key=lambda t: (rank.get(t, 10_000),))[:4]
    # if still >3, keep 3 unless removing would drop to 0 (keep at least 1)
    if len(xs) > 3:
        xs = xs[:3]
    return xs


def normalize_scene_tags(tags: Any, *, allowed: Set[str]) -> List[str]:
    xs = _dedup_keep_order(_as_list(tags))

    # Policy cleanup: remove "送礼" (strong signal exception not automated here)
    xs = [t for t in xs if t != "送礼"]

    xs = _intersect_allowed(xs, allowed)

    # Count constraint: default 3, upper 4 (keep order)
    xs = _trim_to_max(xs, 4)
    if len(xs) > 3:
        xs = xs[:3]
    return xs


def normalize_style_tags(tags: Any, *, allowed: Set[str]) -> List[str]:
    xs = _dedup_keep_order(_as_list(tags))

    # Normalize mapping: 经典文学 -> 文学
    xs = ["文学" if t == "经典文学" else t for t in xs]

    # Remove: 成长小说
    xs = [t for t in xs if t != "成长小说"]

    xs = _dedup_keep_order(xs)
    xs = _intersect_allowed(xs, allowed)

    # Genre slot uniqueness among these
    genre_order = ["人物传记", "科幻", "小说", "散文"]
    genre_set = set(genre_order)
    present_genres = [t for t in xs if t in genre_set]
    if len(present_genres) > 1:
        keep = next((g for g in genre_order if g in present_genres), present_genres[0])
        xs = [t for t in xs if (t not in genre_set) or (t == keep)]

    # Count constraint: default 3-4, upper 4
    if len(xs) > 4:
        theme_pref = ["社会观察", "哲学思考", "爱情", "历史"]
        theme_rank = {t: i for i, t in enumerate(theme_pref)}

        kept: List[str] = []
        # 1) keep the genre (if any)
        for g in genre_order:
            if g in xs:
                kept.append(g)
                break
        # 2) keep up to 2 themes
        themes = [t for t in xs if t in theme_rank]
        themes = sorted(themes, key=lambda t: theme_rank[t])
        for t in themes:
            if t not in kept and len([x for x in kept if x in theme_rank]) < 2:
                kept.append(t)
        # 3) keep 文学 if present
        if "文学" in xs and "文学" not in kept:
            kept.append("文学")
        # 4) fill remaining slots from original order
        for t in xs:
            if len(kept) >= 4:
                break
            if t not in kept:
                kept.append(t)
        xs = kept[:4]

    xs = _dedup_keep_order(xs)[:4]
    return xs


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Apply labeling-policy normalization to books_enriched.json")
    parser.add_argument(
        "--include-sample",
        action="store_true",
        help="Also apply the policy to subject_ids present in books_enriched_example.json (default: skip sample).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUT_PATH),
        help="Output json path (default: data/books_enriched_v2.json).",
    )
    args = parser.parse_args(argv)

    a = json.loads(A_PATH.read_text(encoding="utf-8"))
    b = json.loads(B_PATH.read_text(encoding="utf-8"))

    sample_subject_ids: Set[str] = set()
    for x in b:
        sid = str(x.get("subject_id") or "").strip()
        if sid:
            sample_subject_ids.add(sid)

    def vocab(field: str) -> Set[str]:
        s: Set[str] = set()
        for arr in (a, b):
            for x in arr:
                for t in _as_list(x.get(field)):
                    if t:
                        s.add(t)
        # also include normalized target tags explicitly used by policy
        if field == "style_tags":
            s.add("文学")
        return s

    allowed_mood = vocab("mood_tags")
    allowed_scene = vocab("scene_tags")
    allowed_style = vocab("style_tags")

    changed_books = 0
    changed_fields = Counter()
    add_counts = {f: Counter() for f in ("mood_tags", "scene_tags", "style_tags")}
    rem_counts = {f: Counter() for f in ("mood_tags", "scene_tags", "style_tags")}
    violations = Counter()

    out: List[Dict[str, Any]] = []
    for x in a:
        sid = str(x.get("subject_id") or "").strip()
        if (not args.include_sample) and sid and sid in sample_subject_ids:
            out.append(x)
            continue

        before = {
            "mood_tags": _dedup_keep_order(_as_list(x.get("mood_tags"))),
            "scene_tags": _dedup_keep_order(_as_list(x.get("scene_tags"))),
            "style_tags": _dedup_keep_order(_as_list(x.get("style_tags"))),
        }
        after = {
            "mood_tags": normalize_mood_tags(before["mood_tags"], allowed=allowed_mood),
            "scene_tags": normalize_scene_tags(before["scene_tags"], allowed=allowed_scene),
            "style_tags": normalize_style_tags(before["style_tags"], allowed=allowed_style),
        }

        y = dict(x)
        for f in ("mood_tags", "scene_tags", "style_tags"):
            if before[f] != after[f]:
                changed_fields[f] += 1
                add = set(after[f]) - set(before[f])
                rem = set(before[f]) - set(after[f])
                for t in add:
                    add_counts[f][t] += 1
                for t in rem:
                    rem_counts[f][t] += 1
                y[f] = after[f]
        if before != after:
            changed_books += 1

        # constraint checks: only enforce upper bound (policy defines max, but some entries may lack tags)
        if len(after["mood_tags"]) > 4:
            violations["mood_tags"] += 1
        if len(after["scene_tags"]) > 4:
            violations["scene_tags"] += 1
        if len(after["style_tags"]) > 4:
            violations["style_tags"] += 1

        out.append(y)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote: {out_path}")
    print(f"total_A={len(a)} sample_subject_ids={len(sample_subject_ids)}")
    print(f"changed_books={changed_books}")
    print("changed_fields=", dict(changed_fields))
    print("violations=", dict(violations))

    for f in ("mood_tags", "scene_tags", "style_tags"):
        print(f"\n[{f}] top_added:", add_counts[f].most_common(10))
        print(f"[{f}] top_removed:", rem_counts[f].most_common(10))


if __name__ == "__main__":
    main()

