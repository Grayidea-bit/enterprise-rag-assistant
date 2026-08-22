"""端點共用的相依項。"""

from fastapi import Header, HTTPException

from config import env_settings


def resolve_tenant(x_tenant_id: str | None = Header(default=None)) -> str:
    """租戶來自 X-Tenant-Id header。

    注意這裡沒有任何身分驗證 —— 隔離的是資料模型與檢索路徑,不是身分。
    要正式對外服務必須在這層之前補上認證。
    """
    tenant_id = (x_tenant_id or env_settings.DEFAULT_TENANT_ID).strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-Id 不能是空字串")
    return tenant_id
