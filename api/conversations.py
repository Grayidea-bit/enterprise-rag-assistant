"""對話管理端點。"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from api.deps import resolve_tenant
from database.func import (
    conversation_exists,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    list_messages,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])

# 帶回模型的歷史回合上限。不設限的話 context 會隨對話無限膨脹。
MAX_HISTORY_MESSAGES = 20


class ConversationSummary(BaseModel):
    id: int
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int


class Message(BaseModel):
    id: int
    role: str
    content: str
    sources: list[dict[str, Any]]
    created_at: datetime


class CreateConversation(BaseModel):
    title: str | None = Field(default=None, max_length=200)


async def require_conversation(tenant_id: str, conversation_id: int) -> None:
    """別的租戶的對話一律當成不存在,不洩漏它存在的事實。"""
    if not await conversation_exists(tenant_id, conversation_id):
        raise HTTPException(status_code=404, detail="找不到這個對話")


def to_history(rows: list[dict[str, Any]]) -> list[ModelMessage]:
    """把儲存的文字回合還原成模型看得懂的歷史。

    只還原 user / assistant 的文字,不重播舊的工具呼叫 —— 模型不需要看到
    上一輪撈了哪些段落,那只會佔 context。
    """
    history: list[ModelMessage] = []
    for row in rows[-MAX_HISTORY_MESSAGES:]:
        if row["role"] == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=row["content"])]))
        else:
            history.append(ModelResponse(parts=[TextPart(content=row["content"])]))
    return history


@router.post("", status_code=201, response_model=ConversationSummary)
async def new_conversation(
    body: CreateConversation | None = None,
    tenant_id: str = Depends(resolve_tenant),
) -> ConversationSummary:
    conversation_id = await create_conversation(tenant_id, body.title if body else None)
    # 一定要用剛拿到的 id 讀回來。之前是取「這個租戶最新的一筆」,
    # 兩個請求同時進來時會回到對方建立的對話。
    row = await get_conversation(tenant_id, conversation_id)
    if row is None:
        raise HTTPException(status_code=500, detail="對話建立後卻讀不回來")
    return ConversationSummary(**row)


@router.get("", response_model=list[ConversationSummary])
async def get_conversations(
    limit: int = 50,
    tenant_id: str = Depends(resolve_tenant),
) -> list[ConversationSummary]:
    return [ConversationSummary(**r) for r in await list_conversations(tenant_id, limit)]


@router.get("/{conversation_id}", response_model=list[Message])
async def get_messages(
    conversation_id: int,
    tenant_id: str = Depends(resolve_tenant),
) -> list[Message]:
    await require_conversation(tenant_id, conversation_id)
    return [Message(**r) for r in await list_messages(tenant_id, conversation_id)]


@router.delete("/{conversation_id}", status_code=204)
async def remove_conversation(
    conversation_id: int,
    tenant_id: str = Depends(resolve_tenant),
) -> None:
    if not await delete_conversation(tenant_id, conversation_id):
        raise HTTPException(status_code=404, detail="找不到這個對話")
