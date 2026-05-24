-- createdb enterprise_rag
-- psql enterprise_rag -f database/sql/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- 文件層級:原始檔、metadata
CREATE TABLE IF NOT EXISTS documents  (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT,
    source      TEXT,
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- 切塊層級:每個 chunk 的文字 + 向量
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT REFERENCES documents(id) ON DELETE CASCADE,
    content     TEXT NOT NULL,
    embedding   VECTOR(1024),
    chunk_index INT
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops);
