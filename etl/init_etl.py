from dotenv import load_dotenv
import os
import psycopg
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src import embeddings

load_dotenv()

# Get embeddings
with open(os.path.join(os.path.dirname(__file__), "shakespeare.txt")) as f:
    text = f.read()
e = embeddings.Embeddings(text, chunk_size=500, chunk_overlap=50)
vectors = e.get_embeddings()

# Connect to the database
conn = psycopg.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
)

# Create tables and load records
with conn as conn:
    cur = conn.cursor()
    cur.execute("""
        DROP SCHEMA public CASCADE;
        CREATE SCHEMA public;
        
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE records (
            id bigserial PRIMARY KEY,
            text VARCHAR,
            embedding vector(1536)
        );
    """)
    insert_query = """
    INSERT INTO records (text, embedding) VALUES (%s, %s)
    """
    cur.executemany(insert_query, vectors)
    