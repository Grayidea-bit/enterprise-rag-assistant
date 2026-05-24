from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from config import env_settings

# 連線池:每條連線拿出來時自動註冊 vector type
pool = ConnectionPool(
    env_settings.DATABASE_URL,
    min_size=2,
    max_size=10,
    configure=register_vector,
    open=False,
)
