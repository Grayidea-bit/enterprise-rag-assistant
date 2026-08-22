from database.conn import pool


async def db_startup():
    await pool.open()
    await pool.wait()


async def db_shutdown():
    await pool.close()
