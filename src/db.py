from psycopg import Cursor

def vector_search(
    cursor: Cursor,
    vector: list[float],
    limit: int = 5
) -> list[dict]:
    """Returns the most similar records to the given vector."""
    
    # NOTE: Casting for vector type https://github.com/pgvector/pgvector-python/issues/4
    query = """ 
    SELECT text
    FROM records
    ORDER BY embedding <=> %s::vector DESC
    LIMIT %s
    """
    cursor.execute(query, (vector, limit))
    result = cursor.fetchall()
    cursor.close()
    return result
