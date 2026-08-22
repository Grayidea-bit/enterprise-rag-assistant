"""資料庫存取函式。全部要求傳入 tenant_id,租戶隔離在這一層強制執行。"""

from collections.abc import Sequence
from typing import Any

import psycopg
from pgvector import Vector
from psycopg.types.json import Jsonb

from config import env_settings
from database.conn import pool

# pgvector 0.8 才有的 iterative scan;探測一次後快取結果
_iterative_scan_supported: bool | None = None


async def _supports_iterative_scan() -> bool:
    """HNSW 是先搜向量圖再過濾,租戶資料稀疏時會回傳不滿 k 筆。

    pgvector 0.8 的 hnsw.iterative_scan 會自動擴大搜尋直到湊滿,
    舊版沒有這個參數,只能接受 recall 損失。
    """
    global _iterative_scan_supported
    if _iterative_scan_supported is None:
        try:
            async with pool.connection() as conn:
                # 必須先碰一次 vector 型別。pgvector 的 GUC 是在 library 載入時
                # 才註冊的,在還沒用過 vector 的連線上直接 SHOW 會報
                # "unrecognized configuration parameter",誤判成不支援。
                await conn.execute("SELECT '[1]'::vector")
                await conn.execute("SHOW hnsw.iterative_scan")
            _iterative_scan_supported = True
        except psycopg.Error:
            _iterative_scan_supported = False
    return _iterative_scan_supported


async def upsert_document(
    tenant_id: str,
    title: str | None,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[int, bool]:
    """建立或更新一份文件,回傳 (document_id, 是否為既有文件)。

    同一租戶內 source 唯一,所以重複上傳同一個檔名會取代原本那份,
    而不是在庫裡長出兩套 chunk。
    """
    async with pool.connection() as conn:
        row = await (
            await conn.execute(
                """
                INSERT INTO documents (tenant_id, title, source, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (tenant_id, source) DO UPDATE
                    SET title      = EXCLUDED.title,
                        metadata   = EXCLUDED.metadata,
                        updated_at = now()
                RETURNING id, (xmax <> 0) AS existed
                """,
                (tenant_id, title, source, Jsonb(metadata) if metadata else None),
            )
        ).fetchone()
        if not row:
            raise RuntimeError(f"Upsert document failed, source: {source}")
        return row[0], row[1]


async def delete_chunks(tenant_id: str, document_id: int) -> int:
    """清掉一份文件既有的 chunk,回傳刪除筆數(重新 ingest 時用)。"""
    async with pool.connection() as conn:
        cur = await conn.execute(
            "DELETE FROM chunks WHERE tenant_id = %s AND document_id = %s",
            (tenant_id, document_id),
        )
        return cur.rowcount


async def insert_chunks(
    tenant_id: str,
    document_id: int,
    contents: Sequence[str],
    embeddings: Sequence[Sequence[float]],
) -> int:
    """批次寫入 chunk。contents 與 embeddings 必須等長且順序一致。"""
    if len(contents) != len(embeddings):
        raise ValueError(
            f"contents({len(contents)}) 與 embeddings({len(embeddings)}) 長度不一致"
        )
    if not contents:
        return 0

    rows = [
        (document_id, tenant_id, content, Vector(list(embedding)), index)
        for index, (content, embedding) in enumerate(zip(contents, embeddings))
    ]
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.executemany(
                """
                INSERT INTO chunks (document_id, tenant_id, content, embedding, chunk_index)
                VALUES (%s, %s, %s, %s, %s)
                """,
                rows,
            )
    return len(rows)


async def search_chunks(
    tenant_id: str,
    query_embedding: Sequence[float],
    limit: int = 5,
    max_distance: float | None = None,
) -> list[tuple[str, float, str | None, str]]:
    """租戶內的向量檢索,回傳 [(content, distance, title, source)],距離由近到遠。

    max_distance 是可選的相似度門檻:不設的話永遠回傳 limit 筆,
    即使全部都不相關 —— 那是 RAG 幻覺最常見的來源。
    """
    sql = """
        SELECT c.content, c.embedding <=> %(q)s AS distance, d.title, d.source
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.tenant_id = %(tenant)s
    """
    params: dict[str, Any] = {
        # 一定要包成 Vector:純 list 會被 psycopg 當成 double precision[],
        # 而 <=> 運算子沒有 vector <-> float8[] 的多載(INSERT 有隱式轉型所以看不出來)
        "q": Vector(list(query_embedding)),
        "tenant": tenant_id,
        "limit": limit,
    }
    if max_distance is not None:
        sql += " AND c.embedding <=> %(q)s <= %(max_distance)s"
        params["max_distance"] = max_distance
    sql += " ORDER BY distance LIMIT %(limit)s"

    use_iterative = await _supports_iterative_scan()
    async with pool.connection() as conn:
        if use_iterative:
            # SET LOCAL 只在這個交易內生效,不會汙染連線池裡的其他使用者
            await conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
        return await (await conn.execute(sql, params)).fetchall()


# RRF 的平滑常數。60 是原論文(Cormack et al. 2009)的建議值,
# 作用是壓低頭部名次的權重差距,讓兩路排名不會被其中一路的第一名獨占。
RRF_K = 60

# 詞彙那一路的相似度門檻。pg_trgm 預設 0.6 對中文太嚴 —— 實測「住宿費上限」
# 對命中的段落只有 0.50,會被整個濾掉。
WORD_SIMILARITY_THRESHOLD = 0.25


async def search_chunks_hybrid(
    tenant_id: str,
    query_embedding: Sequence[float],
    query_text: str,
    limit: int = 5,
    max_distance: float | None = None,
) -> list[tuple[str, float, str | None, str]]:
    """向量 + trigram 詞彙的混合檢索,用 RRF 融合兩路排名。

    兩路是互補的:向量處理語意(「特休天數」→「特別休假」),
    trigram 處理字面(金額、編號、專有名詞這些 embedding 會糊掉的東西)。
    融合只看名次不看分數,所以兩路分數尺度天差地遠也無所謂。

    注意 max_distance 只套用在向量那一路 —— 純靠字面命中的段落本來就
    可能離查詢向量很遠,拿距離去濾會把詞彙那一路的貢獻全部砍掉。
    """
    vec_filter = ""
    params: dict[str, Any] = {
        "q": Vector(list(query_embedding)),
        "text": query_text,
        "tenant": tenant_id,
        "limit": limit,
        # 每一路各取這麼多候選再融合。取太少會讓只在單一路名列前茅的段落進不了決選。
        "pool": max(limit * 4, 20),
        "k": RRF_K,
    }
    if max_distance is not None:
        vec_filter = " AND c.embedding <=> %(q)s <= %(max_distance)s"
        params["max_distance"] = max_distance

    sql = f"""
        WITH vec AS (
            SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(q)s) AS rank
            FROM chunks c
            WHERE c.tenant_id = %(tenant)s{vec_filter}
            ORDER BY c.embedding <=> %(q)s
            LIMIT %(pool)s
        ),
        lex AS (
            SELECT c.id,
                   ROW_NUMBER() OVER (
                       ORDER BY word_similarity(%(text)s, c.content) DESC, c.id
                   ) AS rank
            FROM chunks c
            WHERE c.tenant_id = %(tenant)s AND %(text)s <%% c.content
            ORDER BY word_similarity(%(text)s, c.content) DESC, c.id
            LIMIT %(pool)s
        ),
        fused AS (
            SELECT id, SUM(1.0 / (%(k)s + rank)) AS score
            FROM (SELECT id, rank FROM vec UNION ALL SELECT id, rank FROM lex) u
            GROUP BY id
        )
        SELECT c.content, c.embedding <=> %(q)s AS distance, d.title, d.source
        FROM fused f
        JOIN chunks c ON c.id = f.id
        JOIN documents d ON d.id = c.document_id
        ORDER BY f.score DESC, distance
        LIMIT %(limit)s
    """

    use_iterative = await _supports_iterative_scan()
    async with pool.connection() as conn:
        if use_iterative:
            await conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
        # SET LOCAL 是 utility 語句,不接受參數化的 %s;值是模組常數,
        # 用 float() 強制轉型確保內插進去的一定是數字
        await conn.execute(
            "SET LOCAL pg_trgm.word_similarity_threshold = "
            f"{float(WORD_SIMILARITY_THRESHOLD)}"
        )
        return await (await conn.execute(sql, params)).fetchall()


async def retrieve(
    tenant_id: str,
    query_embedding: Sequence[float],
    query_text: str,
    limit: int = 5,
    max_distance: float | None = None,
    mode: str | None = None,
) -> list[tuple[str, float, str | None, str]]:
    """依設定的檢索模式取回段落。mode 不給就用 env 的 RETRIEVAL_MODE。"""
    if (mode or env_settings.RETRIEVAL_MODE) == "vector":
        return await search_chunks(tenant_id, query_embedding, limit, max_distance)
    return await search_chunks_hybrid(
        tenant_id, query_embedding, query_text, limit, max_distance
    )


async def list_documents(tenant_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """列出租戶的文件與各自的 chunk 數(給前端 / 驗證用)。"""
    async with pool.connection() as conn:
        rows = await (
            await conn.execute(
                """
                SELECT d.id, d.title, d.source, d.created_at, d.updated_at,
                       COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.tenant_id = %s
                GROUP BY d.id
                ORDER BY d.updated_at DESC
                LIMIT %s
                """,
                (tenant_id, limit),
            )
        ).fetchall()
    return [
        {
            "id": r[0],
            "title": r[1],
            "source": r[2],
            "created_at": r[3],
            "updated_at": r[4],
            "chunk_count": r[5],
        }
        for r in rows
    ]


# ── 對話 ────────────────────────────────────────────────────────


async def create_conversation(tenant_id: str, title: str | None = None) -> int:
    async with pool.connection() as conn:
        row = await (
            await conn.execute(
                "INSERT INTO conversations (tenant_id, title) VALUES (%s, %s) RETURNING id",
                (tenant_id, title),
            )
        ).fetchone()
        if not row:
            raise RuntimeError("Create conversation failed")
        return row[0]


async def conversation_exists(tenant_id: str, conversation_id: int) -> bool:
    """順便當成租戶授權檢查:別的租戶的 conversation 一律視為不存在。"""
    async with pool.connection() as conn:
        row = await (
            await conn.execute(
                "SELECT 1 FROM conversations WHERE id = %s AND tenant_id = %s",
                (conversation_id, tenant_id),
            )
        ).fetchone()
        return row is not None


async def list_conversations(tenant_id: str, limit: int = 50) -> list[dict[str, Any]]:
    async with pool.connection() as conn:
        rows = await (
            await conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id)
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE c.tenant_id = %s
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT %s
                """,
                (tenant_id, limit),
            )
        ).fetchall()
    return [
        {
            "id": r[0],
            "title": r[1],
            "created_at": r[2],
            "updated_at": r[3],
            "message_count": r[4],
        }
        for r in rows
    ]


async def list_messages(
    tenant_id: str, conversation_id: int, limit: int = 200
) -> list[dict[str, Any]]:
    async with pool.connection() as conn:
        rows = await (
            await conn.execute(
                """
                SELECT id, role, content, sources, created_at
                FROM messages
                WHERE tenant_id = %s AND conversation_id = %s
                ORDER BY id
                LIMIT %s
                """,
                (tenant_id, conversation_id, limit),
            )
        ).fetchall()
    return [
        {
            "id": r[0],
            "role": r[1],
            "content": r[2],
            "sources": r[3] or [],
            "created_at": r[4],
        }
        for r in rows
    ]


async def append_message(
    tenant_id: str,
    conversation_id: int,
    role: str,
    content: str,
    sources: list[dict[str, Any]] | None = None,
) -> int:
    """寫入一則訊息並更新對話的 updated_at(讓列表照最近使用排序)。"""
    async with pool.connection() as conn:
        row = await (
            await conn.execute(
                """
                INSERT INTO messages (conversation_id, tenant_id, role, content, sources)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    conversation_id,
                    tenant_id,
                    role,
                    content,
                    Jsonb(sources) if sources else None,
                ),
            )
        ).fetchone()
        await conn.execute(
            "UPDATE conversations SET updated_at = now() WHERE id = %s AND tenant_id = %s",
            (conversation_id, tenant_id),
        )
        if not row:
            raise RuntimeError("Append message failed")
        return row[0]


async def set_conversation_title(
    tenant_id: str, conversation_id: int, title: str
) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE conversations SET title = %s WHERE id = %s AND tenant_id = %s AND title IS NULL",
            (title, conversation_id, tenant_id),
        )


async def delete_conversation(tenant_id: str, conversation_id: int) -> bool:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "DELETE FROM conversations WHERE id = %s AND tenant_id = %s",
            (conversation_id, tenant_id),
        )
        return cur.rowcount > 0
