from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.recommender_service import ExhaustedRecommendationError, RecommendResult, timed_recommend


app = FastAPI(title="BookRecommend API", version="0.1.0")

# Allow the static dev server (http.server on :8000) to call this API (uvicorn on :8001).
_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
_allowed_origins = [o.strip() for o in _origins_env.split(",") if o.strip()] or [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=3)
    intent_v2: Dict[str, Any] | None = None
    custom_text: str | None = None
    exclude_book_ids: Optional[List[str]] = None


class BookOut(BaseModel):
    # canonical (snake_case) — prefer these in clients
    subject_id: str
    book_id: str
    title: str
    author: str
    cover_image: str
    content_intro: str
    details_url: str
    style_tags: List[str] = []
    mood_tags: List[str] = []
    scene_tags: List[str] = []
    difficulty: int | None = None
    pace: int | None = None
    length_type: str | None = None

    # legacy/derived — keep for compatibility
    cover_url: str
    summary: str
    book_enriched: Dict[str, Any]
    score: float


class RecommendResponse(BaseModel):
    success: bool
    query: str
    books: List[BookOut]


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    q = (req.query or "").strip()
    if req.custom_text:
        ct = str(req.custom_text).strip()
        if ct:
            q = f"{q}\n补充需求：{ct}"
    if not q:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    try:
        exclude_ids = [str(x).strip() for x in (req.exclude_book_ids or []) if str(x).strip()]
        res = timed_recommend(q, int(req.top_k), exclude_book_ids=exclude_ids)
        books: List[RecommendResult] = res["books"]
        elapsed_ms = float(res["elapsed_ms"])
    except ExhaustedRecommendationError:
        raise HTTPException(status_code=409, detail="抱歉，由于书籍库限制，无法再推荐新书")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # don't leak stacktrace to client
        raise HTTPException(status_code=500, detail=f"recommendation failed: {e.__class__.__name__}")

    # simple log
    book_ids = [b.book_id for b in books]
    print(f"[recommend] ms={elapsed_ms:.1f} top_k={req.top_k} query={q[:200]!r} result_ids={book_ids}")

    return RecommendResponse(
        success=True,
        query=q,
        books=[
            BookOut(
                subject_id=str(b.book_enriched.get("subject_id") or b.book_id),
                book_id=b.book_id,
                title=b.title,
                author=b.author,
                cover_image=str(b.book_enriched.get("cover_image") or b.cover_url or ""),
                content_intro=str(b.book_enriched.get("content_intro") or ""),
                details_url=str(b.book_enriched.get("details_url") or ""),
                style_tags=list(b.book_enriched.get("style_tags") or []),
                mood_tags=list(b.book_enriched.get("mood_tags") or []),
                scene_tags=list(b.book_enriched.get("scene_tags") or []),
                difficulty=(int(b.book_enriched.get("difficulty")) if b.book_enriched.get("difficulty") is not None else None),
                pace=(int(b.book_enriched.get("pace")) if b.book_enriched.get("pace") is not None else None),
                length_type=(str(b.book_enriched.get("length_type")) if b.book_enriched.get("length_type") is not None else None),
                cover_url=b.cover_url,
                summary=b.summary,
                book_enriched=b.book_enriched,
                score=b.score,
            )
            for b in books
        ],
    )

