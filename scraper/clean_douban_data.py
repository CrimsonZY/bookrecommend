from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_COVER = "https://example.com/default_cover.jpg"
DEFAULT_INTRO = "暂无简介"
DEFAULT_TRANSLATOR = "无译者"


OUTPUT_FIELDS: List[str] = [
    "subject_id",
    "title",
    "author",
    "author_era_nationality",
    "translator",
    "publisher",
    "pages",
    "content_intro",
    "cover_image",
    "details_url",
]


AUTHOR_SPLIT_RE = re.compile(r"\s*(?:/|／|、|;|；|&|＆|\||，)\s*")


def _clean_text(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def _clean_multiline(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def _parse_pages_to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return max(0, value)
    s = _clean_text(value)
    if not s:
        return 0
    m = re.search(r"(\d+)", s)
    if not m:
        return 0
    try:
        return max(0, int(m.group(1)))
    except Exception:
        return 0


def _normalize_author(author: Any) -> str:
    s = _clean_text(author)
    if not s:
        return ""
    parts = [p.strip() for p in AUTHOR_SPLIT_RE.split(s) if p.strip()]
    # Keep stable order; remove duplicates while preserving order.
    uniq: List[str] = []
    for p in parts:
        if p not in uniq:
            uniq.append(p)
    return ", ".join(uniq)


def _normalize_title(title: Any) -> str:
    return _clean_text(title)


def _normalize_date_if_present(rec: Dict[str, Any]) -> None:
    """
    If a date-like field exists (e.g. pub_date/publication_date), normalize to YYYY-MM-DD when possible.
    Current dataset typically doesn't contain such fields; this is future-proofing.
    """
    for key in ("pub_date", "publication_date", "publish_date", "出版日期"):
        if key not in rec:
            continue
        raw = _clean_text(rec.get(key))
        if not raw:
            rec[key] = ""
            continue
        # Try parse formats: YYYY-M-D, YYYY.MM.DD, YYYY/MM/DD, YYYY-MM
        m = re.match(r"^(\d{4})[-./](\d{1,2})(?:[-./](\d{1,2}))?$", raw)
        if not m:
            # If only year
            m2 = re.match(r"^(\d{4})$", raw)
            rec[key] = raw if not m2 else f"{m2.group(1)}-01-01"
            continue
        y = int(m.group(1))
        mo = int(m.group(2))
        d = int(m.group(3) or 1)
        rec[key] = f"{y:04d}-{mo:02d}-{d:02d}"


def _project_and_fill(raw: Dict[str, Any], stats: Counter) -> Dict[str, Any]:
    out: Dict[str, Any] = {k: "" for k in OUTPUT_FIELDS}

    for k in OUTPUT_FIELDS:
        if k in raw:
            out[k] = raw.get(k)

    out["subject_id"] = _clean_text(out.get("subject_id"))
    out["details_url"] = _clean_text(out.get("details_url"))

    out["title"] = _normalize_title(out.get("title"))
    out["author"] = _normalize_author(out.get("author"))
    out["author_era_nationality"] = _clean_text(out.get("author_era_nationality"))
    out["publisher"] = _clean_text(out.get("publisher"))

    intro = _clean_multiline(out.get("content_intro"))
    if not intro:
        intro = DEFAULT_INTRO
        stats["fill_content_intro"] += 1
    out["content_intro"] = intro

    cover = _clean_text(out.get("cover_image"))
    if not cover:
        cover = DEFAULT_COVER
        stats["fill_cover_image"] += 1
    out["cover_image"] = cover

    translator = _normalize_author(out.get("translator"))
    if not translator:
        translator = DEFAULT_TRANSLATOR
        stats["fill_translator"] += 1
    out["translator"] = translator

    pages_int = _parse_pages_to_int(out.get("pages"))
    if pages_int == 0:
        # Count as fill when original was missing or non-numeric.
        if not str(out.get("pages") or "").strip() or not re.search(r"\d+", str(out.get("pages") or "")):
            stats["fill_pages"] += 1
    out["pages"] = pages_int

    _normalize_date_if_present(out)
    return out


def _dedupe_by_title_author(records: Iterable[Dict[str, Any]], stats: Counter) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for r in records:
        title_key = _clean_text(r.get("title"))
        author_key = _clean_text(r.get("author"))
        key = (title_key, author_key)
        if key in seen:
            stats["dedupe_removed"] += 1
            continue
        seen.add(key)
        out.append(r)
    return out


def _save_json(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _save_csv(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = dict(r)
            # CSV expects scalars; ensure pages is stringified.
            row["pages"] = str(row.get("pages", 0))
            w.writerow({k: row.get(k, "") for k in OUTPUT_FIELDS})


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Clean and preprocess Douban book dataset for recommender usage.")
    p.add_argument("--in", dest="in_path", default="data/douban_top100_filled.json", help="Input JSON path")
    p.add_argument("--out-json", dest="out_json", default="data/douban_top100_clean.json", help="Output JSON path")
    p.add_argument("--out-csv", dest="out_csv", default="data/douban_top100_clean.csv", help="Output CSV path")
    args = p.parse_args(argv)

    in_path = Path(args.in_path)
    raw_records = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise SystemExit("Input JSON must be a list of objects.")

    stats: Counter = Counter()
    projected = [_project_and_fill(r if isinstance(r, dict) else {}, stats) for r in raw_records]
    before = len(projected)
    deduped = _dedupe_by_title_author(projected, stats)
    after = len(deduped)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    _save_json(deduped, out_json)
    _save_csv(deduped, out_csv)

    print("done")
    print(f"- input_records: {before}")
    print(f"- output_records: {after}")
    print(f"- dedupe_removed: {stats.get('dedupe_removed', 0)}")
    print(
        "- filled: "
        + json.dumps(
            {
                "cover_image": stats.get("fill_cover_image", 0),
                "content_intro": stats.get("fill_content_intro", 0),
                "translator": stats.get("fill_translator", 0),
                "pages": stats.get("fill_pages", 0),
            },
            ensure_ascii=False,
        )
    )
    print(f"- out_json: {out_json}")
    print(f"- out_csv: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

