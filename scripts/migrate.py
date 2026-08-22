"""Schema migration CLI。

python scripts/migrate.py status
python scripts/migrate.py up
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.migrate import applied, available, upgrade


async def cmd_status(_args) -> int:
    done = await applied()
    rows = available()
    if not rows:
        print("  (找不到任何 migration)")
        return 0
    print()
    for version, _path in rows:
        mark = "\033[32m✓\033[0m 已套用" if version in done else "\033[33m·\033[0m 待套用"
        print(f"  {mark}  {version}")
    left = sum(1 for v, _ in rows if v not in done)
    print(f"\n  共 {len(rows)} 支,{left} 支待套用\n")
    return 0


async def cmd_up(_args) -> int:
    applied_now = await upgrade()
    if not applied_now:
        print("  已是最新,沒有待套用的 migration")
    else:
        for version in applied_now:
            print(f"  \033[32m✓\033[0m 已套用 {version}")
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser(description="Schema migration")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="列出已套用與待套用").set_defaults(fn=cmd_status)
    sub.add_parser("up", help="套用所有待執行的 migration").set_defaults(fn=cmd_up)
    args = ap.parse_args()
    # 刻意不開連線池:migration 走自己的連線,在 extension 還不存在的
    # 全新資料庫上池子根本連不起來
    return await args.fn(args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
