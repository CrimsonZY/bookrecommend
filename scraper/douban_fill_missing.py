from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup
import requests

BASE_URL = "https://book.douban.com/top250"
DETAIL_BASE = "https://book.douban.com/subject/"

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
        return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))
    except Exception:
        return (url or "").strip()


def _normalize_url_for_key(url: str) -> str:
    try:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except Exception:
        return (url or "").strip()


def _extract_subject_id(details_url: str) -> str:
    m = re.search(r"/subject/(\d+)/", details_url or "")
    return m.group(1) if m else ""


def _clean_text(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def _parse_int_maybe(s: str) -> str:
    if not s:
        return ""
    m = re.search(r"(\d+)", str(s))
    return m.group(1) if m else _clean_text(str(s))


def _looks_like_blocked_page(html: str) -> bool:
    if not html:
        return True
    low = html.lower()
    # Be conservative: avoid false-positives on normal pages.
    strong_needles = [
        "sec.douban.com",  # security / captcha domain
        "captcha",
        "请输入验证码",
        "验证码错误",
        "访问受限",
        "检测到有异常请求",
        "异常请求",
        "request blocked",
        "访问频率过快",
    ]
    return any(n.lower() in low for n in strong_needles)


def _split_author_era_nationality(author: str) -> Tuple[str, str]:
    if not author:
        return "", ""
    s = author.strip()
    # Build pattern with real unicode chars (avoid raw-string \u ambiguity).
    openers = "[\\[\\(\uFF08\u3010\uFF3B]"  # [, (, （, 【, ［
    closers = "[\\]\\)\uFF09\u3011\uFF3D]"  # ], ), ）, 】, ］
    pattern = "^" + openers + r"(.+?)" + closers + r"\s*(.*)$"
    m = re.match(pattern, s)
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
    sleep_min: float = 0.4
    sleep_max: float = 1.2


class Fetcher:
    def __init__(self, cfg: FetchConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self._apply_default_headers()
        self._warmed = False

    def _apply_default_headers(self) -> None:
        user_agents = [
            # A tiny rotation helps when repeatedly retrying a small set of pages.
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0 Safari/537.36"
            ),
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0 Safari/537.36"
            ),
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0 Safari/537.36"
            ),
        ]
        self.session.headers.update(
            {
                "User-Agent": random.choice(user_agents),
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Connection": "keep-alive",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }
        )

    def reset_session(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass
        self.session = requests.Session()
        self._apply_default_headers()
        self._warmed = False

    def warmup(self) -> None:
        """
        Visit Top250 once to acquire basic cookies before hitting detail pages.
        This can reduce immediate block/captcha rate.
        """
        if self._warmed:
            return
        try:
            resp = self.session.get(
                BASE_URL,
                timeout=(self.cfg.timeout_connect, self.cfg.timeout_read),
            )
            if resp.status_code == 200:
                self._warmed = True
        except Exception:
            # Warmup is best-effort.
            pass

    def get_html(self, url: str, *, referer: Optional[str] = None) -> str:
        url = _normalize_url(url)
        headers = {"Referer": referer} if referer else None

        last_exc: Optional[BaseException] = None
        for attempt in range(self.cfg.max_retries + 1):
            if attempt > 0:
                backoff = min(30.0, (2**attempt) + random.uniform(0, 1.0))
                time.sleep(backoff)
            try:
                self.warmup()
                resp = self.session.get(
                    url,
                    headers=headers,
                    timeout=(self.cfg.timeout_connect, self.cfg.timeout_read),
                )
                if resp.status_code == 200:
                    # Douban pages are utf-8; relying on apparent_encoding can mis-detect
                    # and produce mojibake in extracted fields.
                    resp.encoding = "utf-8"
                    html = resp.text
                    if _looks_like_blocked_page(html):
                        last_exc = RuntimeError("blocked/captcha page detected")
                        continue
                    self._polite_sleep()
                    return html
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_exc = RuntimeError(f"retryable http {resp.status_code}")
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


TARGET_FIELDS = (
    "author",
    "author_era_nationality",
    "translator",
    "pages",
    "publisher",
    "content_intro",
)


def _clean_multiline(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def _iter_info_pairs_by_span_pl(soup: BeautifulSoup) -> Dict[str, str]:
    """
    More robust parsing of Douban #info:
    iterate each <span class="pl">Key:</span> and collect its value by reading
    its next_siblings until <br>.
    """
    info = soup.find(id="info")
    if not info:
        return {}

    kv: Dict[str, str] = {}
    for sp in info.find_all("span", class_="pl"):
        key = _clean_text(sp.get_text(" ", strip=True)).rstrip(":：")
        if not key or key in kv:
            continue

        parts: List[str] = []
        for sib in sp.next_siblings:
            # Stop at line break boundary for this field.
            if getattr(sib, "name", None) == "br":
                break
            if getattr(sib, "name", None):
                text = sib.get_text(" ", strip=True)
            else:
                text = str(sib)
            text = _clean_text(text).lstrip(":：").strip()
            if text:
                parts.append(text)

        val = _clean_text(" ".join(parts))
        if val:
            kv[key] = val

    return kv


def _parse_detail_fields_for_fill(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    kv = _iter_info_pairs_by_span_pl(soup)

    author_raw = kv.get("作者", "")
    author, era_nat = _split_author_era_nationality(author_raw)

    translator = kv.get("译者", "")
    pages = _parse_int_maybe(kv.get("页数", ""))
    if not pages:
        # Some entries omit "页数" but include it in binding (e.g. "精装16开").
        pages = _parse_int_maybe(kv.get("装帧", ""))

    publisher = _clean_text(kv.get("出版社", ""))

    intro_nodes = soup.select("#link-report .intro")
    intro = ""
    if intro_nodes:
        intro = _clean_multiline(intro_nodes[-1].get_text("\n", strip=True))
    if not intro:
        cand = soup.select(".related_info .intro")
        if cand:
            texts = [_clean_multiline(x.get_text("\n", strip=True)) for x in cand]
            texts = [t for t in texts if t]
            intro = max(texts, key=len) if texts else ""

    return {
        "author": _clean_text(author),
        "author_era_nationality": _clean_text(era_nat),
        "translator": _clean_text(translator),
        "pages": pages,
        "publisher": publisher,
        "content_intro": intro,
    }


def _needs_fill(rec: Dict[str, str]) -> bool:
    for k in TARGET_FIELDS:
        if not (rec.get(k) or "").strip():
            return True
    return False


def _dedupe_records(records: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set[str] = set()
    out: List[Dict[str, str]] = []
    for r in records:
        sid = (r.get("subject_id") or "").strip()
        url = _normalize_url_for_key((r.get("details_url") or "").strip())
        key = sid or url
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _save_csv(records: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: (r.get(k, "") or "") for k in FIELDS})


def _save_json(records: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _count_filled_delta(before: Dict[str, str], after: Dict[str, str]) -> int:
    c = 0
    for k in TARGET_FIELDS:
        b = (before.get(k) or "").strip()
        a = (after.get(k) or "").strip()
        if (not b) and a:
            c += 1
    return c


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fill missing fields in Douban Top100 JSON incrementally.")
    p.add_argument(
        "--in",
        dest="in_path",
        default="data/douban_top100_filled.json",
        help="Input JSON path",
    )
    p.add_argument(
        "--out-json",
        dest="out_json",
        default="data/douban_top100_refilled.json",
        help="Output JSON path",
    )
    p.add_argument(
        "--out-csv",
        dest="out_csv",
        default="data/douban_top100_refilled.csv",
        help="Output CSV path",
    )
    p.add_argument("--only-missing", action="store_true", default=True, help="Only fill missing fields")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit how many records to fill (0 means no limit; useful for debugging).",
    )
    p.add_argument(
        "--subject-ids",
        type=str,
        default="",
        help="Comma-separated subject_ids to retry only (optional).",
    )
    p.add_argument(
        "--blocked-wait",
        type=float,
        default=20.0,
        help="Extra wait seconds when blocked/captcha detected before resetting session.",
    )

    # Default to a safer pacing to avoid triggering anti-bot on detail pages.
    p.add_argument("--sleep-min", type=float, default=1.0)
    p.add_argument("--sleep-max", type=float, default=3.0)
    p.add_argument("--timeout-connect", type=float, default=5.0)
    p.add_argument("--timeout-read", type=float, default=15.0)
    p.add_argument("--retries", type=int, default=4)

    args = p.parse_args(argv)

    in_path = Path(args.in_path)
    records: List[Dict[str, str]] = json.loads(in_path.read_text(encoding="utf-8"))
    records = _dedupe_records(records)

    cfg = FetchConfig(
        timeout_connect=float(args.timeout_connect),
        timeout_read=float(args.timeout_read),
        max_retries=max(0, int(args.retries)),
        sleep_min=max(0.0, float(args.sleep_min)),
        sleep_max=max(0.0, float(args.sleep_max)),
    )
    fetcher = Fetcher(cfg)

    need = [r for r in records if _needs_fill(r)]
    total = len(records)
    todo = len(need)
    limit = max(0, int(args.limit))
    if limit and todo > limit:
        todo = limit

    filled_fields = 0
    ok_pages = 0
    fail_pages = 0
    err_kinds: Dict[str, int] = {}
    err_samples: Dict[str, str] = {}
    blocked_pages = 0
    subject_filter: Optional[set[str]] = None
    if str(args.subject_ids).strip():
        subject_filter = {s.strip() for s in str(args.subject_ids).split(",") if s.strip()}

    processed = 0
    for rec in records:
        if not _needs_fill(rec):
            continue
        if limit and processed >= limit:
            break
        processed += 1

        url = _normalize_url_for_key(rec.get("details_url", ""))
        if not url:
            continue

        # Ensure subject_id exists for consistency (no extra requests).
        rec["subject_id"] = (rec.get("subject_id") or "").strip() or _extract_subject_id(url)
        rec["details_url"] = url
        if subject_filter is not None and rec["subject_id"] not in subject_filter:
            continue

        try:
            html = fetcher.get_html(url, referer=BASE_URL)
            got = _parse_detail_fields_for_fill(html)
            before = dict(rec)

            for k in TARGET_FIELDS:
                if (rec.get(k) or "").strip():
                    continue
                val = (got.get(k) or "").strip()
                if val:
                    rec[k] = val

            filled_fields += _count_filled_delta(before, rec)
            ok_pages += 1
        except Exception as e:
            fail_pages += 1
            kind = type(e).__name__
            msg = str(e)[:160]
            if "blocked/captcha" in msg:
                kind = "BlockedOrCaptcha"
                blocked_pages += 1
                # When blocked, wait a bit more and reset session for next attempt.
                time.sleep(max(0.0, float(args.blocked_wait)) + random.uniform(0, 3.0))
                fetcher.reset_session()
            err_kinds[kind] = err_kinds.get(kind, 0) + 1
            if kind not in err_samples:
                err_samples[kind] = msg
            continue

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    _save_json(records, out_json)
    _save_csv(records, out_csv)

    print("done")
    print(f"- total_records: {total}")
    print(f"- need_fill_records: {todo}")
    print(f"- fetched_ok_pages: {ok_pages}")
    print(f"- fetched_fail_pages: {fail_pages}")
    if blocked_pages:
        print(f"- blocked_pages: {blocked_pages}")
    print(f"- filled_fields_count: {filled_fields}")
    if err_kinds:
        print(f"- error_kinds: {err_kinds}")
        print(f"- error_samples: {err_samples}")
    print(f"- out_json: {out_json}")
    print(f"- out_csv: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

