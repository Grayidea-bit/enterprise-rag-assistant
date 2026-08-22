"""檢索與問答端點。

/search  只做向量檢索,不呼叫 LLM —— 用來單獨檢查檢索品質。
/chat    完整 RAG:agent 自行決定何時呼叫檢索工具,回傳答案與引用來源。
/chat/stream 同上但以 SSE 串流,先送來源再逐段送文字。
"""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.deps import resolve_tenant
from core.agent import get_agent
from core.embedding import embed_query
from core.tools import RagDeps, Retrieved
from database.func import search_chunks

router = APIRouter(tags=["chat"])

EXCERPT_CHARS = 300


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=50)
    max_distance: float | None = Field(default=None, ge=0)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=50)
    max_distance: float | None = Field(default=None, ge=0)


class Source(BaseModel):
    source: str
    title: str | None
    distance: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


def _to_source(hit: Retrieved) -> Source:
    excerpt = hit.content[:EXCERPT_CHARS]
    if len(hit.content) > EXCERPT_CHARS:
        excerpt += "…"
    return Source(
        source=hit.source, title=hit.title, distance=hit.distance, excerpt=excerpt
    )


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/search", response_model=list[Source])
async def search(
    body: SearchRequest,
    tenant_id: str = Depends(resolve_tenant),
) -> list[Source]:
    """純向量檢索,不經過 LLM。"""
    embedding = await embed_query(body.query)
    rows = await search_chunks(
        tenant_id, embedding, limit=body.limit, max_distance=body.max_distance
    )
    return [
        _to_source(Retrieved(content=r[0], distance=float(r[1]), title=r[2], source=r[3]))
        for r in rows
    ]


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    tenant_id: str = Depends(resolve_tenant),
) -> ChatResponse:
    deps = RagDeps(
        tenant_id=tenant_id, limit=body.limit, max_distance=body.max_distance
    )
    result = await get_agent().run(body.question, deps=deps)
    return ChatResponse(
        answer=result.output,
        sources=[_to_source(h) for h in deps.unique_sources()],
    )


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    tenant_id: str = Depends(resolve_tenant),
) -> StreamingResponse:
    """SSE 串流。

    事件順序:sources(檢索完成,可能是空陣列)→ 多個 delta → done。
    途中出錯會送一個 error 事件 —— HTTP 狀態碼此時已經送出去了,不能再改。
    """

    async def events() -> AsyncIterator[str]:
        deps = RagDeps(
            tenant_id=tenant_id, limit=body.limit, max_distance=body.max_distance
        )
        try:
            async with get_agent().run_stream(body.question, deps=deps) as result:
                # 進到這裡時工具已經跑完了,所以來源可以先送出去給前端顯示
                yield _sse(
                    "sources",
                    {"sources": [_to_source(h).model_dump() for h in deps.unique_sources()]},
                )
                async for delta in result.stream_text(delta=True):
                    yield _sse("delta", {"text": delta})
            yield _sse("done", {})
        except Exception as e:  # noqa: BLE001 - 串流已開始,只能用事件回報
            yield _sse("error", {"type": type(e).__name__, "detail": str(e)[:500]})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
