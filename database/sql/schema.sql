-- createdb enterprise_rag
-- psql enterprise_rag -f database/sql/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;
-- 混合檢索的詞彙那一路。之所以用 trigram 而不是 tsvector,是因為
-- to_tsvector 對中文不分詞 —— 整句話會變成單一 token,全文檢索完全失效。
CREATE EXTENSION IF NOT EXISTS pg_trgm;

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

-- 詞彙檢索用。支援 <% (word_similarity) 運算子
CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm
    ON chunks USING gin (content gin_trgm_ops);

-- 對話層級:一個 conversation 就是一串問答
CREATE TABLE IF NOT EXISTS conversations (
    id          BIGSERIAL PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    title       TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversations_tenant
    ON conversations (tenant_id, updated_at DESC);

-- 訊息層級:刻意存成 user / assistant 的文字回合,而不是 pydantic-ai 的內部訊息格式。
-- 那個格式是函式庫內部結構,存進 DB 等於把 schema 綁在函式庫版本上;
-- 而且把舊的工具呼叫與檢索結果原封不動塞回 context 只是在燒 token。
CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    tenant_id       TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    sources         JSONB,          -- assistant 訊息才有,存當時引用的來源
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (conversation_id, id);

-- API 金鑰。只存 SHA-256 雜湊,明文金鑰產生後就不再留存。
-- 這裡用 sha256 而不是 bcrypt/argon2 是刻意的:金鑰是 32 bytes 的高熵隨機值,
-- 沒有字典攻擊的空間,慢雜湊只會讓每個請求都付出不必要的成本。
-- (低熵的「使用者密碼」則相反,那種一定要用慢雜湊。)
CREATE TABLE IF NOT EXISTS api_keys (
    id           BIGSERIAL PRIMARY KEY,
    key_hash     TEXT NOT NULL UNIQUE,
    tenant_id    TEXT NOT NULL,
    name         TEXT,
    prefix       TEXT NOT NULL,          -- 金鑰前綴,只為了讓人在列表裡認得出來
    created_at   TIMESTAMPTZ DEFAULT now(),
    last_used_at TIMESTAMPTZ,
    revoked_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_api_keys_tenant ON api_keys (tenant_id);
