"""smoke script 共用:替測試租戶臨時鑄一把金鑰,測完清掉。

讓每個 smoke script 都走真正的認證路徑,而不是把 AUTH_MODE 關掉繞過去。
"""

from core.auth import generate_key, hash_key
from database.conn import pool
from database.func import insert_api_key


async def mint(tenant_id: str) -> dict[str, str]:
    """建立一把金鑰並回傳可直接用的 headers。"""
    key, key_hash, prefix = generate_key()
    await insert_api_key(key_hash, tenant_id, "smoke-test", prefix)
    return {"Authorization": f"Bearer {key}"}


async def drop_keys(*tenant_ids: str) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            "DELETE FROM api_keys WHERE tenant_id = ANY(%s)", (list(tenant_ids),)
        )


def bogus() -> dict[str, str]:
    """一把格式正確但從未註冊的金鑰。"""
    key, _, _ = generate_key()
    return {"Authorization": f"Bearer {key}"}


__all__ = ["mint", "drop_keys", "bogus", "hash_key"]
