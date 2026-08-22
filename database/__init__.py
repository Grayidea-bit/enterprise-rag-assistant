from config import env_settings
from database.conn import pool
from database.migrate import pending, upgrade


async def db_startup():
    await pool.open()
    await pool.wait()

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


async def db_shutdown():
    await pool.close()
