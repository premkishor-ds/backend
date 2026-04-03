import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

# Database Connection Parameters
DB_HOST = os.getenv("DB_HOST", "192.168.1.29")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "ai-based-maxol-rag-search")
DB_USER = os.getenv("DB_USER", "ai-based-maxol-rag-search")
DB_PASS = os.getenv("DB_PASS", "Pg4cD8kdFr8vQwn7Mr4zjW")

def initialize_database():
    """
    Initializes the PostgreSQL database by:
    1. Enabling the pgvector extension.
    2. Creating the 'products' table for structured SQL queries.
    3. Creating the 'documents' table for vector embeddings.
    """
    try:
        # Establish connection to the server
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        conn.autocommit = True
        cur = conn.cursor()

        print("--- Connecting to Database & Initializing ---")

        # 1. Enable pgvector extension
        print("Enabling pgvector extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 2. Create Products table (Structured Data)
        print("Creating 'products' table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                category VARCHAR(100),
                price DECIMAL(10, 2),
                stock INTEGER,
                location VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 3. Create Documents table (Unstructured Data + Vectors)
        # We use dimensions=1536 for OpenAI's 'text-embedding-3-small' or 'text-embedding-ada-002'
        print("Creating 'documents' table with pgvector...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding vector(1536),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        print("--- Database Setup Complete! ---")
        cur.close()
        conn.close()

    except Exception as e:
        print(f"Error initializing database: {e}")

if __name__ == "__main__":
    initialize_database()
