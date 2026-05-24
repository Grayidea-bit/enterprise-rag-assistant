from database.conn import pool


def db_startup():
    pool.open()
    pool.wait()


def db_shutdown():
    pool.close()
