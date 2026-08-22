"""文件上傳與列表端點。

ingest 流程:上傳 → 解碼 → 切塊 → embed → 寫入 pgvector。
刻意先 embed 再寫 DB,這樣 embedding 失敗時不會留下半份文件。
"""

from datetime import datetime
from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from config import env_settings
from core.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, split_text
from core.embedding import embed_documents
from database.func import (
    delete_chunks,
    insert_chunks,
    list_documents,
    upsert_document,
)

router = APIRouter(prefix="/documents", tags=["documents"])

# 這一刀只做純文字;PDF / docx 之後當成獨立的 parser 加進來
ALLOWED_SUFFIXES = {".txt", ".md", ".markdown", ".text"}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024


def resolve_tenant(x_tenant_id: str | None = Header(default=None)) -> str:
    """租戶來自 X-Tenant-Id header。

    注意這裡沒有任何身分驗證 —— 隔離的是資料模型與檢索路徑,不是身分。
    要正式對外服務必須在這層之前補上認證。
    """
    tenant_id = (x_tenant_id or env_settings.DEFAULT_TENANT_ID).strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-Id 不能是空字串")
    return tenant_id


class IngestResponse(BaseModel):
    document_id: int
    tenant_id: str
    source: str
    title: str | None
    chunks: int
    replaced: bool  # 是否覆蓋了同租戶下同名的既有文件


class DocumentSummary(BaseModel):
    id: int
    title: str | None
    source: str
    created_at: datetime
    updated_at: datetime
    chunk_count: int


@router.post("", status_code=201, response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="純文字或 Markdown 檔"),
    title: str | None = Form(default=None),
    chunk_size: int = Form(default=DEFAULT_CHUNK_SIZE),
    overlap: int = Form(default=DEFAULT_OVERLAP),
    tenant_id: str = Depends(resolve_tenant),
) -> IngestResponse:
    source = PurePosixPath(file.filename or "").name
    if not source:
        raise HTTPException(status_code=400, detail="缺少檔名")

    suffix = PurePosixPath(source).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"目前只支援 {sorted(ALLOWED_SUFFIXES)},收到的是 '{suffix or '(無副檔名)'}'",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"檔案 {len(raw)} bytes 超過上限 {MAX_UPLOAD_BYTES} bytes",
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"檔案不是有效的 UTF-8 文字:{e}"
        ) from e

    try:
        contents = split_text(text, chunk_size=chunk_size, overlap=overlap)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    if not contents:
        raise HTTPException(status_code=400, detail="檔案沒有可切塊的內容")

    # 先算 embedding:這步失敗就整個中止,DB 保持乾淨
    embeddings = await embed_documents(contents)

    document_id, existed = await upsert_document(
        tenant_id=tenant_id,
        title=title or source,
        source=source,
        metadata={
            "bytes": len(raw),
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
    )
    if existed:
        await delete_chunks(tenant_id, document_id)
    written = await insert_chunks(tenant_id, document_id, contents, embeddings)

    return IngestResponse(
        document_id=document_id,
        tenant_id=tenant_id,
        source=source,
        title=title or source,
        chunks=written,
        replaced=existed,
    )


@router.get("", response_model=list[DocumentSummary])
async def get_documents(
    limit: int = 50,
    tenant_id: str = Depends(resolve_tenant),
) -> list[DocumentSummary]:
    """列出該租戶的文件。用來確認 ingest 真的寫進去了。"""
    rows = await list_documents(tenant_id, limit=limit)
    return [DocumentSummary(**row) for row in rows]
