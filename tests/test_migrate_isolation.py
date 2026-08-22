"""migration 模組必須與 pgvector 連線池脫鉤。

這條測試守的是一個實際發生過、而且只在全新資料庫上才會出現的先有雞還是先有蛋:
連線池的 configure=register_vector_async 會在每條連線上查 vector 型別的 OID,
但全新資料庫要等 migration 建好 extension 之後才有那個型別。
migration 一旦改回走池子,全新環境就再也起不來 —— 而既有環境完全測不出來。
"""

import ast
import inspect
from pathlib import Path

import database.migrate as migrate_module

SOURCE = Path(inspect.getfile(migrate_module)).read_text(encoding="utf-8")


def imported_names() -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_does_not_import_the_connection_pool():
    names = imported_names()
    assert "database.conn" not in names
    assert "pool" not in names


def test_opens_its_own_connection():
    assert "psycopg.AsyncConnection.connect" in SOURCE


def test_db_startup_migrates_before_opening_the_pool():
    import database

    source = Path(inspect.getfile(database)).read_text(encoding="utf-8")
    startup = source[source.index("async def db_startup") :]
    assert startup.index("upgrade(") < startup.index("pool.open("), "migration 必須在開池之前跑完"
