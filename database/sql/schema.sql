-- createdb enterprise_rag
-- psql enterprise_rag -f database/sql/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- 文件層級:原始檔、metadata
-- tenant_id 是租戶隔離的依據。目前沒有身分驗證,值由 X-Tenant-Id header 帶入,
-- 這裡只保證「資料模型與檢索路徑」是隔離的。
CREATE TABLE IF NOT EXISTS documents  (
    id          BIGSERIAL PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    title       TEXT,
    source      TEXT NOT NULL,
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now(),
    -- 同一租戶內 source 唯一,重複上傳走 upsert 取代而不是長出第二份
    CONSTRAINT uq_documents_tenant_source UNIQUE (tenant_id, source)
);

-- 切塊層級:每個 chunk 的文字 + 向量
-- tenant_id 刻意反正規化,讓向量檢索可以直接過濾,不必為了拿租戶而 JOIN documents
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id   TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   VECTOR(1024),
    chunk_index INT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- 租戶過濾用。HNSW 是「先搜向量圖再過濾」,租戶資料稀疏時會撈不滿 k 筆;
-- search_chunks() 會嘗試開 pgvector 0.8 的 hnsw.iterative_scan 來補救。
CREATE INDEX IF NOT EXISTS idx_chunks_tenant
    ON chunks (tenant_id);

-- CASCADE 刪除與依序取回 chunk 都會用到
CREATE INDEX IF NOT EXISTS idx_chunks_document
    ON chunks (document_id, chunk_index);
