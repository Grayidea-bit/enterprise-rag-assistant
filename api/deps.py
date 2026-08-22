"""端點共用的相依項。租戶解析與認證的唯一入口。"""

from fastapi import Header, HTTPException

from config import env_settings
from core.auth import extract_key, hash_key
from database.func import tenant_for_key

UNAUTHENTICATED = {"WWW-Authenticate": "Bearer"}


async def resolve_tenant(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None),
) -> str:
    """決定這個請求屬於哪個租戶。

    AUTH_MODE=api_key(預設):租戶完全由金鑰決定,X-Tenant-Id 一律忽略 ——
      否則帶了合法金鑰的人就能宣稱任意租戶,等於沒有隔離。
    AUTH_MODE=disabled:開發用,直接採信 X-Tenant-Id。
    """
    if env_settings.AUTH_MODE == "api_key":
        key = extract_key(authorization, x_api_key)
        if not key:
            raise HTTPException(
                status_code=401,
                detail="缺少 API 金鑰。請帶 Authorization: Bearer <key> 或 X-API-Key。",
                headers=UNAUTHENTICATED,
            )
        tenant_id = await tenant_for_key(hash_key(key))
        if not tenant_id:
            # 不區分「不存在」與「已撤銷」,避免洩漏金鑰是否曾經有效
            raise HTTPException(
                status_code=401,
                detail="API 金鑰無效或已撤銷。",
                headers=UNAUTHENTICATED,
            )
        return tenant_id

    tenant_id = (x_tenant_id or env_settings.DEFAULT_TENANT_ID).strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-Id 不能是空字串")
    return tenant_id
