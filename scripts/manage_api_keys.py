"""API 金鑰管理。

    python scripts/manage_api_keys.py create --tenant acme --name "行銷部"
    python scripts/manage_api_keys.py list
    python scripts/manage_api_keys.py revoke 3

明文金鑰只在 create 當下印出一次,資料庫裡只留 SHA-256 雜湊,事後無法還原。
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.auth import generate_key  # noqa: E402
from database import db_shutdown, db_startup  # noqa: E402
from database.func import (  # noqa: E402
    insert_api_key,
    list_api_keys,
    revoke_api_key,
)


async def cmd_create(args) -> int:
    key, key_hash, prefix = generate_key()
    key_id = await insert_api_key(key_hash, args.tenant, args.name, prefix)
    print(f"\n  已建立金鑰 #{key_id}(租戶 {args.tenant})")
    print(f"\n    {key}\n")
    print("  這是唯一一次看到明文的機會,請立刻存起來。\n")
    return 0


async def cmd_list(args) -> int:
    rows = await list_api_keys(args.tenant)
    if not rows:
        print("  (沒有任何金鑰)")
        return 0
    print(f"\n  {'ID':<5}{'租戶':<16}{'前綴':<14}{'名稱':<16}{'狀態':<10}{'最後使用'}")
    print("  " + "-" * 78)
    for r in rows:
        status = "已撤銷" if r["revoked_at"] else "有效"
        used = r["last_used_at"].strftime("%Y-%m-%d %H:%M") if r["last_used_at"] else "從未"
        print(
            f"  {r['id']:<5}{r['tenant_id']:<16}{r['prefix'] + '…':<14}"
            f"{(r['name'] or '-'):<16}{status:<10}{used}"
        )
    print()
    return 0


async def cmd_revoke(args) -> int:
    if await revoke_api_key(args.id):
        print(f"  金鑰 #{args.id} 已撤銷")
        return 0
    print(f"  金鑰 #{args.id} 不存在或早已撤銷")
    return 1


async def main() -> int:
    ap = argparse.ArgumentParser(description="API 金鑰管理")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="建立新金鑰")
    c.add_argument("--tenant", required=True, help="這把金鑰對應的租戶")
    c.add_argument("--name", help="給人看的名稱")
    c.set_defaults(fn=cmd_create)

    ls = sub.add_parser("list", help="列出金鑰")
    ls.add_argument("--tenant", help="只看某個租戶")
    ls.set_defaults(fn=cmd_list)

    rv = sub.add_parser("revoke", help="撤銷金鑰")
    rv.add_argument("id", type=int)
    rv.set_defaults(fn=cmd_revoke)

    args = ap.parse_args()
    await db_startup()
    try:
        return await args.fn(args)
    finally:
        await db_shutdown()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
