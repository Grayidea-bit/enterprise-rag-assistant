from collections.abc import Sequence
from functools import cache

from pydantic_ai.embeddings import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel

from config import env_settings
from core.llm import build_provider


@cache
def get_embedder() -> Embedder:
    """延遲建構 embedder;與 chat 共用同一套 OpenAI 相容 provider（之後快取）

    刻意不設定 EmbeddingSettings.dimensions:bge-m3 這類模型維度是固定的,
    有些相容端點也不接受該參數;改用回傳後的執行期驗證。
    """
    base_url, api_key = env_settings.embedding_target
    return Embedder(
        OpenAIEmbeddingModel(
            env_settings.EMBEDDING_MODEL,
            provider=build_provider(base_url, api_key),
        )
    )


def _validate(vectors: list[list[float]]) -> list[list[float]]:
    """檢查維度是否與 EMBEDDING_DIM 一致。

    維度不合若放行,會等到寫入 pgvector 時才炸,而且錯誤訊息完全看不出原因。
    """
    expected = env_settings.EMBEDDING_DIM
    for i, v in enumerate(vectors):
        if len(v) != expected:
            raise ValueError(
                f"Embedding 維度不符:模型 {env_settings.EMBEDDING_MODEL} "
                f"第 {i} 筆回傳 {len(v)} 維,但 EMBEDDING_DIM 設定為 {expected}。"
                f"請確認 .env 的 EMBEDDING_DIM 與 schema.sql 的 VECTOR(n) 一致。"
            )
    return vectors


async def embed_query(text: str) -> list[float]:
    """把查詢字串轉成向量。"""
    result = await get_embedder().embed_query(text)
    return _validate(result.embeddings)[0]


async def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    """把一批文件切塊轉成向量,順序與輸入一致。"""
    if not texts:
        return []
    result = await get_embedder().embed_documents(list(texts))
    return _validate(result.embeddings)
