from database.conn import pool


def insert_document(title: str, source: str) -> int:
    """建立一份文件,回傳 document_id"""
    with pool.connection() as conn:
        row = conn.execute(
            """
            INSERT INTO documents (title, source)
            VALUES (%s, %s)
            RETURNING id
            """,
            (title, source),
        ).fetchone()
        if not row:
            raise Exception("Insert data fail title: ", title)
        return row[0]


def insert_chunk(document_id: int, content: str, embedding, chunk_index: int):
    """寫入一個 chunk(embedding 直接傳 list 或 numpy array)"""
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO chunks (document_id, content, embedding, chunk_index)
            VALUES (%s, %s, %s, %s)
            """,
            (document_id, content, embedding, chunk_index),
        )


def search_chunks(query_embedding, limit: int = 5):
    """vector search return (content, distance, title, source)"""
    with pool.connection() as conn:
        sql = """
            SELECT c.content, c.embedding <=> %s AS distance,
                    d.title, d.source
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY distance
            LIMIT %s
        """
        params = (query_embedding, limit)

        return conn.execute(sql, params).fetchall()
