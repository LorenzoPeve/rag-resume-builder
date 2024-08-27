from dotenv import load_dotenv
import os
import psycopg

from src import embeddings, db

load_dotenv(override=True)

def _get_cursor():
    conn = psycopg.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )
    return conn.cursor()

def init_db():
    conn = psycopg.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )

    with conn as conn:
        cur = conn.cursor()
        cur.execute("""
            DROP SCHEMA public CASCADE;
            CREATE SCHEMA public;
            
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE records (
                id bigserial PRIMARY KEY,
                text VARCHAR,
                embedding vector(1536),
                search_text tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED           
            );
                    
            CREATE INDEX vector_hnsw_index ON records USING hnsw (embedding vector_cosine_ops);
            SET maintenance_work_mem = '1GB';
        """)

        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_files",
            "shakespeare.txt"
        )
        with open(path, encoding='utf-8') as f:
            text = f.read()
        e = embeddings.Embeddings(text, chunk_size=500, chunk_overlap=50)
        vectors: list[tuple] = e.get_embeddings()

        insert_query = """
        INSERT INTO records (text, embedding) VALUES (%s, %s)
        """
        cur.executemany(insert_query, vectors)

def test_vector_search():

    text = 'This is medicine'
    e = embeddings.Embeddings(text)
    vector = e.get_embeddings()[0]
    cursor = _get_cursor()
    result = db.vector_search(cursor, vector, limit=5)
    assert len(result) == 5


