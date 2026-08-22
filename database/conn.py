from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from config import env_settings

# 非同步連線池:每條連線拿出來時自動註冊 vector type。
# 用 async 是因為每個請求都要等 embedding / chat 的網路 I/O,
# 同步池會讓一個慢查詢卡住整個 event loop。
pool = AsyncConnectionPool(
    env_settings.DATABASE_URL,
    min_size=2,
    max_size=10,
    configure=register_vector_async,
    open=False,
)
