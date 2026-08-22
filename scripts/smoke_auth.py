"""API 金鑰認證 smoke test。

    python scripts/smoke_auth.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from config import env_settings  # noqa: E402
from core.auth import generate_key, hash_key  # noqa: E402
from database import db_shutdown, db_startup  # noqa: E402
from database.conn import pool  # noqa: E402
from database.func import insert_api_key, list_api_keys, revoke_api_key  # noqa: E402
from scripts._smoke_auth import bogus, drop_keys, mint  # noqa: E402
from server import app  # noqa: E402

LINE = "─" * 62
TENANT_A = "smoke-auth-a"
TENANT_B = "smoke-auth-b"
results: list[tuple[str, bool, str]] = []


def ok(n, m, t):
    print(f"  \033[32m✓\033[0m {m}  \033[2m({t:.1f}s)\033[0m")
    results.append((n, True, m))


def fail(n, e, t):
    print(f"  \033[31m✗\033[0m {e}  \033[2m({t:.1f}s)\033[0m")
    results.append((n, False, e))


async def check(n, title, fn):
    print(f"\n{title}")
    t = time.perf_counter()
    try:
        ok(n, await fn(), time.perf_counter() - t)
    except Exception as e:
        fail(n, f"{type(e).__name__}: {e}", time.perf_counter() - t)


async def cleanup():
    await drop_keys(TENANT_A, TENANT_B)
    async with pool.connection() as conn:
        await conn.execute(
            "DELETE FROM documents WHERE tenant_id = ANY(%s)", ([TENANT_A, TENANT_B],)
        )


async def main() -> int:
    print(LINE)
    print(" API 金鑰認證 smoke test")
    print(LINE)
    print(f" AUTH_MODE = {env_settings.AUTH_MODE}")
    if env_settings.AUTH_MODE != "api_key":
        print(" ⚠ 這支測試需要 AUTH_MODE=api_key,先暫時切過去")
        env_settings.AUTH_MODE = "api_key"

    await db_startup()
    await cleanup()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        a = await mint(TENANT_A)
        b = await mint(TENANT_B)

        async def t_no_key():
            r = await client.get("/documents")
            if r.status_code != 401:
                raise RuntimeError(f"沒帶金鑰應回 401,實際 {r.status_code}")
            if "bearer" not in r.headers.get("www-authenticate", "").lower():
                raise RuntimeError("401 應帶 WWW-Authenticate: Bearer")
            return "401 + WWW-Authenticate: Bearer"

        async def t_bad_key():
            for hdr, label in [
                (bogus(), "沒註冊過的金鑰"),
                # header 只能是 latin-1,所以垃圾值也得用 ASCII
                ({"Authorization": "Bearer not-a-real-key"}, "格式不對的金鑰"),
                ({"Authorization": "Bearer "}, "空的 Bearer 值"),
                ({"Authorization": "Basic abc"}, "非 Bearer scheme"),
            ]:
                r = await client.get("/documents", headers=hdr)
                if r.status_code != 401:
                    raise RuntimeError(f"{label} 應回 401,實際 {r.status_code}")
            return "未註冊 / 亂打 / 非 Bearer 皆回 401"

        async def t_valid_key():
            r = await client.post(
                "/documents",
                headers=a,
                files={"file": ("政策.md", "本公司的休假政策如下。".encode(), "text/markdown")},
            )
            if r.status_code != 201:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")
            if r.json()["tenant_id"] != TENANT_A:
                raise RuntimeError(f"租戶錯了: {r.json()['tenant_id']}")
            seen = (await client.get("/documents", headers=b)).json()
            if seen:
                raise RuntimeError(f"B 的金鑰看到了 A 的文件: {seen}")
            return f"金鑰決定租戶({TENANT_A}),另一把金鑰看不到"

        async def t_header_cannot_override():
            """最關鍵的一條:帶了合法金鑰還想用 X-Tenant-Id 宣稱別的租戶。"""
            r = await client.get(
                "/documents", headers={**b, "X-Tenant-Id": TENANT_A}
            )
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            if r.json():
                raise RuntimeError(
                    f"X-Tenant-Id 竟然蓋過了金鑰,B 看到了 A 的資料: {r.json()}"
                )
            return "X-Tenant-Id 被完全忽略,租戶只由金鑰決定"

        async def t_x_api_key_header():
            key, key_hash, prefix = generate_key()
            await insert_api_key(key_hash, TENANT_A, "smoke-test", prefix)
            r = await client.get("/documents", headers={"X-API-Key": key})
            if r.status_code != 200 or len(r.json()) != 1:
                raise RuntimeError(f"X-API-Key 不能用:HTTP {r.status_code}")
            return "X-API-Key header 也可以用"

        async def t_revoke():
            key, key_hash, prefix = generate_key()
            key_id = await insert_api_key(key_hash, TENANT_A, "will-revoke", prefix)
            hdr = {"Authorization": f"Bearer {key}"}
            if (await client.get("/documents", headers=hdr)).status_code != 200:
                raise RuntimeError("新金鑰應該可用")
            if not await revoke_api_key(key_id):
                raise RuntimeError("撤銷失敗")
            r = await client.get("/documents", headers=hdr)
            if r.status_code != 401:
                raise RuntimeError(f"撤銷後應回 401,實際 {r.status_code}")
            if await revoke_api_key(key_id):
                raise RuntimeError("重複撤銷不該回報成功")
            return "撤銷後立即失效,重複撤銷回報 False"

        async def t_last_used():
            rows = await list_api_keys(TENANT_A)
            used = [r for r in rows if r["last_used_at"]]
            if not used:
                raise RuntimeError("last_used_at 沒有被更新")
            if any(r["prefix"] not in r["prefix"] or len(r["prefix"]) > 12 for r in rows):
                raise RuntimeError("prefix 不該是完整金鑰")
            return f"{len(used)}/{len(rows)} 把金鑰有 last_used_at,列表只露出前綴"

        async def t_hash_not_reversible():
            key, key_hash, _ = generate_key()
            if key in key_hash or len(key_hash) != 64:
                raise RuntimeError("雜湊看起來不對")
            async with pool.connection() as conn:
                rows = await (
                    await conn.execute(
                        "SELECT key_hash FROM api_keys WHERE tenant_id = %s", (TENANT_A,)
                    )
                ).fetchall()
            if any(r[0].startswith("erag_") for r in rows):
                raise RuntimeError("資料庫裡存到了明文金鑰!")
            return f"DB 內 {len(rows)} 筆皆為 64 字元雜湊,無明文"

        async def t_disabled_mode():
            env_settings.AUTH_MODE = "disabled"
            try:
                r = await client.get("/documents", headers={"X-Tenant-Id": TENANT_A})
                if r.status_code != 200 or len(r.json()) != 1:
                    raise RuntimeError(f"開發模式失效:HTTP {r.status_code}")
                r2 = await client.get("/documents")
                if r2.status_code != 200:
                    raise RuntimeError("開發模式沒帶 header 應落到預設租戶")
            finally:
                env_settings.AUTH_MODE = "api_key"
            return "AUTH_MODE=disabled 時採信 X-Tenant-Id,未帶則用預設租戶"

        await check("缺金鑰", "[1/9] 沒帶金鑰", t_no_key)
        await check("壞金鑰", "[2/9] 無效金鑰", t_bad_key)
        await check("正常流程", "[3/9] 有效金鑰決定租戶", t_valid_key)
        await check("無法冒充", "[4/9] X-Tenant-Id 不能蓋過金鑰", t_header_cannot_override)
        await check("替代 header", "[5/9] X-API-Key", t_x_api_key_header)
        await check("撤銷", "[6/9] 撤銷後立即失效", t_revoke)
        await check("使用紀錄", "[7/9] last_used_at 與前綴", t_last_used)
        await check("不可還原", "[8/9] DB 只存雜湊", t_hash_not_reversible)
        await check("開發模式", "[9/9] AUTH_MODE=disabled", t_disabled_mode)

    await cleanup()
    await db_shutdown()

    print(f"\n{LINE}")
    passed = sum(1 for _, g, _ in results if g)
    for n, g, m in results:
        print(f" {'\033[32m✓\033[0m' if g else '\033[31m✗\033[0m'} {n:10} {m[:64]}")
    print(f"{LINE}\n {passed}/{len(results)} 通過")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
