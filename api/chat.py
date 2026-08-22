"""檢索與問答端點。

/search       只做向量檢索,不呼叫 LLM —— 用來單獨檢查檢索品質。
/chat         完整 RAG:agent 自行決定何時呼叫檢索工具,回傳答案與引用來源。
/chat/stream  同上但以 SSE 串流,先送對話 id 與來源,再逐段送文字。

兩個 chat 端點都會把回合寫進 conversations / messages,帶 conversation_id
就接續前一輪對話,不帶就開一個新的。
"""

import json
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.conversations import require_conversation, to_history
from api.deps import resolve_tenant
from core.agent import get_agent
from core.embedding import embed_query
from core.tools import RagDeps, Retrieved
from database.func import (
    append_message,
    create_conversation,
    list_messages,
    retrieve,
    set_conversation_title,
)

router = APIRouter(tags=["chat"])

EXCERPT_CHARS = 300
TITLE_CHARS = 40


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=50)
    max_distance: float | None = Field(default=None, ge=0)
    # 不給就用 env 的 RETRIEVAL_MODE;給了可以直接比較兩種模式
    mode: Literal["vector", "hybrid"] | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None
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
    conversation_id: int


def _to_source(hit: Retrieved) -> Source:
    excerpt = hit.content[:EXCERPT_CHARS]
    if len(hit.content) > EXCERPT_CHARS:
        excerpt += "…"
    return Source(
        source=hit.source, title=hit.title, distance=hit.distance, excerpt=excerpt
    )


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _prepare(tenant_id: str, body: ChatRequest):
    """解析對話 id 並取回歷史。回傳 (conversation_id, history, deps)。"""
    if body.conversation_id is None:
        conversation_id = await create_conversation(tenant_id)
        history = []
    else:
        conversation_id = body.conversation_id
        await require_conversation(tenant_id, conversation_id)
        history = to_history(await list_messages(tenant_id, conversation_id))

    deps = RagDeps(
        tenant_id=tenant_id, limit=body.limit, max_distance=body.max_distance
    )
    return conversation_id, history, deps


async def _persist(
    tenant_id: str,
    conversation_id: int,
    question: str,
    answer: str,
    sources: list[Source],
) -> None:
    await append_message(tenant_id, conversation_id, "user", question)
    await append_message(
        tenant_id,
        conversation_id,
        "assistant",
        answer,
        [s.model_dump() for s in sources],
    )
    # 第一輪問題當標題(set_conversation_title 只在 title 還是 NULL 時才寫)
    await set_conversation_title(tenant_id, conversation_id, question[:TITLE_CHARS])


@router.post("/search", response_model=list[Source])
async def search(
    body: SearchRequest,
    tenant_id: str = Depends(resolve_tenant),
) -> list[Source]:
    """純向量檢索,不經過 LLM。"""
    embedding = await embed_query(body.query)
    rows = await retrieve(
        tenant_id,
        embedding,
        body.query,
        limit=body.limit,
        max_distance=body.max_distance,
        mode=body.mode,
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
    conversation_id, history, deps = await _prepare(tenant_id, body)
    result = await get_agent().run(
        body.question, deps=deps, message_history=history or None
    )
    sources = [_to_source(h) for h in deps.unique_sources()]
    await _persist(tenant_id, conversation_id, body.question, result.output, sources)
    return ChatResponse(
        answer=result.output, sources=sources, conversation_id=conversation_id
    )


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    tenant_id: str = Depends(resolve_tenant),
) -> StreamingResponse:
    """SSE 串流。

    事件順序:conversation → sources → 多個 delta → done。
    途中出錯會送一個 error 事件 —— HTTP 狀態碼此時已經送出去了,不能再改。
    """
    # 這一段要在串流開始前做,404 之類的錯誤才能用正常的狀態碼回報
    conversation_id, history, deps = await _prepare(tenant_id, body)

    async def events() -> AsyncIterator[str]:
        try:
            yield _sse("conversation", {"conversation_id": conversation_id})
            chunks: list[str] = []
            async with get_agent().run_stream(
                body.question, deps=deps, message_history=history or None
            ) as result:
                # 進到這裡時工具已經跑完了,所以來源可以先送出去給前端顯示
                sources = [_to_source(h) for h in deps.unique_sources()]
                yield _sse(
                    "sources", {"sources": [s.model_dump() for s in sources]}
                )
                async for delta in result.stream_text(delta=True):
                    chunks.append(delta)
                    yield _sse("delta", {"text": delta})
            await _persist(
                tenant_id, conversation_id, body.question, "".join(chunks), sources
            )
            yield _sse("done", {})
        except Exception as e:  # noqa: BLE001 - 串流已開始,只能用事件回報
            yield _sse("error", {"type": type(e).__name__, "detail": str(e)[:500]})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
