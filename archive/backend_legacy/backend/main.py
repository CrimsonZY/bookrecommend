from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10_000)


class Book(BaseModel):
    id: str
    title: str
    author: str
    cover_image: str
    tags: List[str]
    recommend_reason: str
    details_url: str
    ai_reason: str


class RecommendResponse(BaseModel):
    status: Literal["success", "error"]
    message: str = ""
    data: List[Book] = []


app = FastAPI(title="BookRecommend API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _tags_from_prompt(prompt: str) -> List[str]:
    p = prompt.lower()
    tags: List[str] = []
    if any(k in p for k in ["通勤", "地铁", "公交", "路上", "commute"]):
        tags += ["通勤", "轻松"]
    if any(k in p for k in ["睡前", "入睡", "放松", "bedtime"]):
        tags += ["睡前", "治愈"]
    if any(k in p for k in ["快速", "简短", "碎片", "quick"]):
        tags += ["短篇", "高效"]
    if any(k in p for k in ["沉浸", "周末", "深度", "weekend"]):
        tags += ["沉浸", "深度阅读"]
    if any(k in p for k in ["送", "礼物", "ta", "gift"]):
        tags += ["送礼", "口碑"]
    if any(k in p for k in ["新书", "探索", "discover"]):
        tags += ["新书", "探索"]

    # 去重并限制数量，避免 UI 过长
    uniq = []
    for t in tags:
        if t not in uniq:
            uniq.append(t)
    return uniq[:6] or ["精选", "适合你"]


def _mock_books(prompt: str) -> List[Book]:
    tags = _tags_from_prompt(prompt)
    reason = "根据你的提示词，优先选择节奏匹配、阅读负担合适且口碑稳定的书籍。"
    ai_reason = f"我参考了你的提示词重点（{ '、'.join(tags) }），并综合可读性与场景适配度给出推荐。"

    # 使用稳定的占位封面（无需本地资源）
    covers = [
        "https://picsum.photos/seed/book1/240/320",
        "https://picsum.photos/seed/book2/240/320",
        "https://picsum.photos/seed/book3/240/320",
        "https://picsum.photos/seed/book4/240/320",
        "https://picsum.photos/seed/book5/240/320",
    ]

    base = [
        ("在细雨中呼吸", "林见"),
        ("短篇的温柔", "周轻"),
        ("地铁到站之前", "许行"),
        ("周末沉浸清单", "沈叙"),
        ("把时间留给阅读", "顾然"),
    ]

    out: List[Book] = []
    for i, (title, author) in enumerate(base, start=1):
        out.append(
            Book(
                id=str(i),
                title=title,
                author=author,
                cover_image=covers[(i - 1) % len(covers)],
                tags=tags,
                recommend_reason=reason,
                details_url="https://example.com/book/" + str(i),
                ai_reason=ai_reason,
            )
        )
    return out


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    prompt = (req.prompt or "").strip()
    if not prompt:
        return RecommendResponse(status="error", message="prompt 不能为空", data=[])
    return RecommendResponse(status="success", message="", data=_mock_books(prompt))

