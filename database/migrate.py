"""極簡的 SQL migration 執行器。

沒有用 Alembic:這個專案完全是手寫 SQL、沒有 ORM,Alembic 的自動產生與
model 比對能力全都用不上,只會多一層抽象。編號 SQL + 一張紀錄表就夠了,
這也是 dbmate / golang-migrate 這類工具的做法。

每個檔案在自己的交易裡執行,成功才寫入版本紀錄 —— 失敗的 migration
不會留下半套結果,也不會被誤記成已套用。
"""

from contextlib import asynccontextmanager
from pathlib import Path

import psycopg

from config import env_settings

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

# 多個實例同時啟動時,用 advisory lock 確保只有一個在跑 migration。
# 數字本身沒有意義,只要全專案一致即可。
ADVISORY_LOCK_ID = 8317465


@asynccontextmanager
async def _connection():
    """自己開一條連線,刻意不走 database.conn 的連線池。

    池子的 configure=register_vector_async 會在每條連線上查 vector 型別的 OID,
    而全新的資料庫還沒有 vector extension —— 於是「建立 extension」的那支
    migration 永遠跑不起來:池子連不上,migration 就跑不了;migration 沒跑,
    extension 就不存在。schema 層的工作本來就不需要向量型別轉接器。
    """
    conn = await psycopg.AsyncConnection.connect(env_settings.DATABASE_URL, autocommit=True)
    try:
        yield conn
    finally:
        await conn.close()


async def _ensure_table(conn) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def available() -> list[tuple[str, Path]]:
    """回傳 [(版本, 檔案)],依檔名排序。版本就是檔名去掉副檔名。"""
    return sorted((path.stem, path) for path in MIGRATIONS_DIR.glob("*.sql") if path.is_file())


async def applied() -> set[str]:
    async with _connection() as conn:
        await _ensure_table(conn)
        rows = await (await conn.execute("SELECT version FROM schema_migrations")).fetchall()
    return {r[0] for r in rows}


async def pending() -> list[tuple[str, Path]]:
    done = await applied()
    return [(v, p) for v, p in available() if v not in done]


async def upgrade() -> list[str]:
    """套用所有未執行的 migration,回傳這次實際套用的版本清單。"""
    async with _connection() as conn:
        # 這個鎖跟著連線走,離開 with 區塊歸還連線時自動釋放
        await conn.execute("SELECT pg_advisory_lock(%s)", (ADVISORY_LOCK_ID,))
        try:
            await _ensure_table(conn)
            rows = await (await conn.execute("SELECT version FROM schema_migrations")).fetchall()
            done = {r[0] for r in rows}

            applied_now: list[str] = []
            for version, path in available():
                if version in done:
                    continue
                sql = path.read_text(encoding="utf-8")
                async with conn.transaction():
                    await conn.execute(sql)
                    await conn.execute(
                        "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
                    )
                applied_now.append(version)
            return applied_now
        finally:
            await conn.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_ID,))
