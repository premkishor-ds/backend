import json
import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load environment variables (API Key and DB details)
load_dotenv(dotenv_path="../backend/.env")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database credentials
DB_HOST = os.getenv("DB_HOST", "192.168.1.29")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "ai-based-maxol-rag-search")
DB_USER = os.getenv("DB_USER", "ai-based-maxol-rag-search")
DB_PASS = os.getenv("DB_PASS", "Pg4cD8kdFr8vQwn7Mr4zjW")

def get_embedding(text):
    """
    Calls OpenAI to turn a string of text into a list of 1536 numbers (a vector).
    This vector captures the 'meaning' of the text.
    """
    text = text.replace("\n", " ") # Clean up text
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def ingest_json_data(file_path):
    """
    Reads a JSON file, cleans it, and stores it in BOTH SQL and Vector tables.
    """
    try:
        # Load the JSON data
        with open(file_path, 'r', encoding='utf-8') as f:
            products = json.load(f)

        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        cur = conn.cursor()

        print(f"--- Starting Ingestion for {file_path} ---")

        for item in products:
            # Safely get values with defaults
            name = item.get("name") or item.get("title") or item.get("question") or "Untitled"
            category = item.get("category") or "General"
            price = item.get("price") or 0.0
            stock = item.get("stock") or 0
            location = item.get("location") or "Unknown"
            description = item.get("description") or item.get("content") or item.get("answer") or ""

            # Only insert into products if there are sensible values
            if name != "Untitled" or price != 0.0:
                cur.execute("""
                    INSERT INTO products (name, category, price, stock, location)
                    VALUES (%s, %s, %s, %s, %s);
                """, (name, category, price, stock, location))

            # Always insert into documents for vector search
            # Combine all string values for embedding
            searchable_text = f"{name}. {category}. {description}. {location}."
            print(f"  -> Generating vector for: {name[:50]}...")
            
            try:
                vector = get_embedding(searchable_text)
                cur.execute("""
                    INSERT INTO documents (content, metadata, embedding)
                    VALUES (%s, %s, %s);
                """, (searchable_text, json.dumps(item), vector))
            except Exception as e:
                print(f"  -> Error embedding {name}: {e}")

        conn.commit()
        cur.close()
        conn.close()
        print("--- Ingestion Complete! ---")

    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    # Point to the data directory inside backend
    data_dir = "../data"
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(data_dir, filename)
                ingest_json_data(file_path)
    else:
        print(f"Directory not found: {data_dir}. Please ensure files are in backend/data/.")
