from config import env_settings
from database.conn import pool
from database.migrate import pending, upgrade


async def db_startup():
    # migration 必須在開池之前跑完。池子每條連線都要註冊 pgvector 的型別,
    # 而全新資料庫要等 migration 建好 extension 之後才有那個型別。
    if env_settings.AUTO_MIGRATE:
        for version in await upgrade():
            print(f"migration applied: {version}")
    else:
        outstanding = await pending()
        if outstanding:
            names = ", ".join(v for v, _ in outstanding)
            print(
                f"⚠  有 {len(outstanding)} 支 migration 尚未套用({names})。"
                "執行 `python scripts/migrate.py up`,或設 AUTO_MIGRATE=true。"
            )

    await pool.open()
    await pool.wait()


async def db_shutdown():
    await pool.close()
