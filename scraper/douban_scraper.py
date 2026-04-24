from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://book.douban.com/top250"


FIELDS: List[str] = [
    "subject_id",
    "title",
    "author",
    "author_era_nationality",
    "publisher",
    "content_intro",
    "translator",
    "pages",
    "cover_image",
    "details_url",
]


def _normalize_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        # Keep query for paging URLs; drop fragment.
        return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))
    except Exception:
        return url.strip()


def _normalize_url_for_key(url: str) -> str:
    """Normalize for dedupe keys: drop query + fragment."""
    try:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except Exception:
        return url.strip()


def _extract_subject_id(details_url: str) -> str:
    m = re.search(r"/subject/(\d+)/", details_url or "")
    return m.group(1) if m else ""


def _clean_text(s: str) -> str:
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_multiline(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def _parse_int_maybe(s: str) -> str:
    if not s:
        return ""
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else _clean_text(s)


def _looks_like_blocked_page(html: str) -> bool:
    if not html:
        return True
    # Common patterns when Douban triggers verification or access restrictions.
    needles = [
        "sec.douban.com",  # security / captcha domain
        "验证码",
        "访问受限",
        "检测到有异常请求",
        "异常请求",
        "验证",
        "robot",
        "deny",
    ]
    low = html.lower()
    return any(n.lower() in low for n in needles)


def _split_author_era_nationality(author: str) -> Tuple[str, str]:
    """
    Examples:
      "[清] 曹雪芹 著" -> ("曹雪芹 著", "清")
      "［英］ 乔治·奥威尔" -> ("乔治·奥威尔", "英")
      "加西亚·马尔克斯" -> ("加西亚·马尔克斯", "")
    """
    if not author:
        return "", ""
    s = author.strip()
    m = re.match(r"^[\[\(（【［](.+?)[\]\)）】］]\s*(.*)$", s)
    if not m:
        return _clean_text(s), ""
    era = _clean_text(m.group(1))
    rest = _clean_text(m.group(2))
    return rest, era


@dataclass
class FetchConfig:
    timeout_connect: float = 5.0
    timeout_read: float = 15.0
    max_retries: int = 4
    sleep_min: float = 1.0
    sleep_max: float = 3.0


class Fetcher:
    def __init__(self, cfg: FetchConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0 Safari/537.36"
                ),
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Connection": "keep-alive",
            }
        )

    def get_html(self, url: str, *, referer: Optional[str] = None) -> str:
        url = _normalize_url(url)
        headers = {"Referer": referer} if referer else None

        last_exc: Optional[BaseException] = None
        for attempt in range(self.cfg.max_retries + 1):
            if attempt > 0:
                backoff = min(30.0, (2**attempt) + random.uniform(0, 1.0))
                time.sleep(backoff)

            try:
                resp = self.session.get(
                    url,
                    headers=headers,
                    timeout=(self.cfg.timeout_connect, self.cfg.timeout_read),
                )
                status = resp.status_code
                if status == 200:
                    resp.encoding = resp.apparent_encoding or "utf-8"
                    html = resp.text
                    # Douban sometimes returns verification/blocked pages with 200.
                    if _looks_like_blocked_page(html):
                        last_exc = RuntimeError("blocked/captcha page detected")
                        continue
                    self._polite_sleep()
                    return html

                if status in (429, 500, 502, 503, 504):
                    last_exc = RuntimeError(f"retryable http {status}")
                    continue

                resp.raise_for_status()
            except BaseException as e:
                last_exc = e
                continue

        raise RuntimeError(f"failed to fetch {url}: {last_exc!r}")

    def _polite_sleep(self) -> None:
        if self.cfg.sleep_max <= 0:
            return
        lo = max(0.0, self.cfg.sleep_min)
        hi = max(lo, self.cfg.sleep_max)
        time.sleep(random.uniform(lo, hi))


def parse_top250_list(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    out: List[Dict[str, str]] = []

    for tr in soup.select("tr.item"):
        a = tr.select_one("div.pl2 a")
        if not a or not a.get("href"):
            continue
        details_url = _normalize_url_for_key(a["href"])
        title = _clean_text(a.get_text(" ", strip=True)).replace("...", "").strip()

        img = tr.select_one("a.nbg img")
        cover = _normalize_url(img.get("src", "")) if img else ""

        subject_id = _extract_subject_id(details_url)
        out.append(
            {
                "subject_id": subject_id,
                "title": title,
                "details_url": details_url,
                "cover_image": cover,
            }
        )

    return out


def _info_table_kv(info_node) -> Dict[str, str]:
    """
    Parse Douban #info block, which typically is a sequence:
      <span class="pl">作者</span>: <a>...</a><br/>
    We traverse nodes and collect text until <br/>.
    """
    if not info_node:
        return {}

    kv: Dict[str, str] = {}
    children = list(info_node.children)
    i = 0
    while i < len(children):
        child = children[i]

        if getattr(child, "name", None) == "span" and "pl" in (child.get("class") or []):
            key = _clean_text(child.get_text(" ", strip=True)).rstrip(":：")
            i += 1

            val_parts: List[str] = []
            while i < len(children):
                node = children[i]
                if getattr(node, "name", None) == "br":
                    break
                text = ""
                if isinstance(getattr(node, "name", None), str):
                    text = node.get_text(" ", strip=True)
                else:
                    text = str(node).strip()
                text = _clean_text(text).lstrip(":：").strip()
                if text:
                    val_parts.append(text)
                i += 1

            value = _clean_text(" ".join(val_parts))
            if key and value and key not in kv:
                kv[key] = value

        i += 1

    return kv


def parse_subject_detail(html: str, *, fallback_cover: str = "") -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    og = soup.find("meta", attrs={"property": "og:image"})
    cover = ""
    if og and og.has_attr("content"):
        cover = _normalize_url(str(og["content"]))
    if not cover:
        mp = soup.select_one("#mainpic img")
        cover = _normalize_url(mp.get("src", "")) if mp else ""
    if not cover:
        cover = fallback_cover or ""

    info = soup.find(id="info")
    kv = _info_table_kv(info)

    author = kv.get("作者", "")
    author_clean, era_nat = _split_author_era_nationality(author)

    publisher = kv.get("出版社", "")
    translator = kv.get("译者", "")
    pages = _parse_int_maybe(kv.get("页数", ""))

    intro_nodes = soup.select("#link-report .intro")
    intro = ""
    if intro_nodes:
        intro = _clean_multiline(intro_nodes[-1].get_text("\n", strip=True))
    if not intro:
        # fallback: pick the longest intro-like block under related_info
        cand = soup.select(".related_info .intro")
        if cand:
            texts = [_clean_multiline(x.get_text("\n", strip=True)) for x in cand]
            texts = [t for t in texts if t]
            intro = max(texts, key=len) if texts else ""

    return {
        "author": author_clean,
        "author_era_nationality": era_nat,
        "publisher": _clean_text(publisher),
        "translator": _clean_text(translator),
        "pages": pages,
        "content_intro": intro,
        "cover_image": cover,
    }


def normalize_record(base: Dict[str, str], detail: Dict[str, str]) -> Dict[str, str]:
    rec: Dict[str, str] = {k: "" for k in FIELDS}
    rec.update({k: (v or "") for k, v in (base or {}).items() if k in rec})
    rec.update({k: (v or "") for k, v in (detail or {}).items() if k in rec})

    rec["details_url"] = _normalize_url_for_key(rec.get("details_url", ""))
    rec["subject_id"] = rec.get("subject_id") or _extract_subject_id(rec["details_url"])

    rec["title"] = _clean_text(rec.get("title", ""))
    rec["author"] = _clean_text(rec.get("author", ""))
    rec["author_era_nationality"] = _clean_text(rec.get("author_era_nationality", ""))
    rec["publisher"] = _clean_text(rec.get("publisher", ""))
    rec["translator"] = _clean_text(rec.get("translator", ""))
    rec["pages"] = _parse_int_maybe(rec.get("pages", ""))
    rec["cover_image"] = _normalize_url(rec.get("cover_image", ""))
    rec["content_intro"] = _clean_multiline(rec.get("content_intro", ""))

    if rec["author"] and not rec["author_era_nationality"]:
        a2, era2 = _split_author_era_nationality(rec["author"])
        rec["author"] = a2
        rec["author_era_nationality"] = era2

    return rec


def dedupe(records: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set[str] = set()
    out: List[Dict[str, str]] = []
    for r in records:
        sid = (r.get("subject_id") or "").strip()
        url = _normalize_url_for_key((r.get("details_url") or "").strip())
        key = sid or url
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def save_csv(records: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = {k: (r.get(k, "") or "") for k in FIELDS}
            w.writerow(row)


def save_json(records: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def crawl_top_books(
    fetcher: Fetcher,
    *,
    limit: int = 100,
    page_size: int = 25,
    max_items: int = 250,
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    seen: set[str] = set()

    start = 0
    while start < max_items and len(results) < limit:
        list_url = f"{BASE_URL}?start={start}"
        html = fetcher.get_html(list_url, referer=BASE_URL)
        items = parse_top250_list(html)
        for it in items:
            sid = it.get("subject_id") or _extract_subject_id(it.get("details_url", ""))
            key = sid or it.get("details_url", "")
            if not key or key in seen:
                continue
            seen.add(key)
            results.append(it)
            if len(results) >= limit:
                break
        start += page_size

    return results[:limit]


def enrich_with_details(fetcher: Fetcher, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for base in items:
        url = base.get("details_url", "")
        if not url:
            continue
        try:
            detail_html = fetcher.get_html(url, referer=BASE_URL)
            detail = parse_subject_detail(detail_html, fallback_cover=base.get("cover_image", ""))
            out.append(normalize_record(base, detail))
        except Exception:
            out.append(normalize_record(base, {}))
    return dedupe(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Douban Top250 scraper (non-UGC fields only).")
    parser.add_argument("--limit", type=int, default=100, help="How many books to crawl (default: 100)")
    parser.add_argument("--out", type=str, default="data", help="Output directory (default: data)")
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["csv", "json", "both"],
        help="Output format",
    )
    parser.add_argument("--page-size", type=int, default=25, help="Pagination size (default: 25)")
    parser.add_argument("--sleep-min", type=float, default=1.0, help="Min delay seconds per request")
    parser.add_argument("--sleep-max", type=float, default=3.0, help="Max delay seconds per request")
    parser.add_argument("--timeout-connect", type=float, default=5.0, help="Connect timeout seconds")
    parser.add_argument("--timeout-read", type=float, default=15.0, help="Read timeout seconds")
    parser.add_argument("--retries", type=int, default=4, help="Retry times for 429/5xx")

    args = parser.parse_args(argv)

    limit = max(1, min(int(args.limit), 250))
    cfg = FetchConfig(
        timeout_connect=float(args.timeout_connect),
        timeout_read=float(args.timeout_read),
        max_retries=max(0, int(args.retries)),
        sleep_min=max(0.0, float(args.sleep_min)),
        sleep_max=max(0.0, float(args.sleep_max)),
    )

    fetcher = Fetcher(cfg)

    items = crawl_top_books(fetcher, limit=limit, page_size=int(args.page_size))
    records = enrich_with_details(fetcher, items)

    # If some detail pages failed and dedupe reduced count, try to fill from further list pages.
    if len(records) < limit:
        more = crawl_top_books(fetcher, limit=250, page_size=int(args.page_size))
        merged = dedupe(records + enrich_with_details(fetcher, more))
        records = merged[:limit]

    out_dir = Path(args.out)
    csv_path = out_dir / "douban_top100.csv"
    json_path = out_dir / "douban_top100.json"

    if args.format in ("csv", "both"):
        save_csv(records, csv_path)
    if args.format in ("json", "both"):
        save_json(records, json_path)

    print(f"saved {len(records)} records")
    if args.format in ("csv", "both"):
        print(f"- {csv_path}")
    if args.format in ("json", "both"):
        print(f"- {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
