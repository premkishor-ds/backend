import os
import json
import psycopg2
import requests
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from prompts import INTENT_CLASSIFICATION_PROMPT, SQL_GENERATION_PROMPT, FINAL_ANSWER_PROMPT

# 1. Load environment variables
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), ".env"),
    override=True,  # Ensure the repo's .env wins over any pre-existing env vars
)

# 2. Initialize FastAPI
app = FastAPI(title="Maxol AI RAG Search")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Database connection parameters
DB_HOST = os.getenv("DB_HOST", "192.168.1.29")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "ai-based-maxol-rag-search")
DB_USER = os.getenv("DB_USER", "ai-based-maxol-rag-search")
DB_PASS = os.getenv("DB_PASS", "Pg4cD8kdFr8vQwn7Mr4zjW")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI model config.
# Default matches `testapi.html` which works in the browser.
OPENAI_RESPONSES_MODEL = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-4.1-mini")
OPENAI_CHAT_COMPLETIONS_MODEL = os.getenv(
    "OPENAI_CHAT_COMPLETIONS_MODEL",
    OPENAI_RESPONSES_MODEL,
)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Max items returned in `retrieved` (Sources panel + LLM context). Vector path uses this as diversify limit; SQL is sliced to this.
MAX_RETRIEVED_SOURCES = int(os.getenv("SEARCH_RETRIEVED_MAX", "5"))

# Initialize OpenAI client once
client = OpenAI(api_key=OPENAI_API_KEY)

class SearchQuery(BaseModel):
    query: str

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,port=DB_PORT,dbname=DB_NAME,user=DB_USER,password=DB_PASS
    )

def call_chat_completion(prompt: str) -> str:
    """Call OpenAI chat completions with a safe model.
    Uses a configurable model (defaults to `OPENAI_RESPONSES_MODEL`).
    """
    response = client.chat.completions.create(
        model=OPENAI_CHAT_COMPLETIONS_MODEL,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Keep the original call_openai_responses for fallback (optional)
def call_openai_responses(prompt: str) -> str:
    """Experimental v1/responses endpoint – kept for compatibility.
    Falls back to chat completions if the model is unavailable.
    """
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "model": OPENAI_RESPONSES_MODEL,
        "input": prompt
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        # Fallback to chat completions
        return call_chat_completion(prompt)
    result = response.json()
    try:
        return result['output'][0]['content'][0]['text']
    except (KeyError, IndexError):
        return str(result)

def _parse_metadata(meta):
    """Normalize JSONB / str metadata to dict."""
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except Exception:
            return {}
    return {}


def _source_file_from_metadata(meta) -> str:
    d = _parse_metadata(meta)
    sf = d.get("source_file")
    return sf if isinstance(sf, str) else ""


def diversify_vector_hits(rows, limit: int = 3):
    """
    Prefer diverse sources in retrieved context.

    Pure top-k vector search often returns multiple rows from the same dataset
    (e.g. FAQ) because Q&A text is semantically similar to many questions.
    We take a larger candidate pool, then pick up to `limit` rows: first pass
    prefers at most one row per `metadata.source_file`, second pass fills the rest
    by nearest-neighbor order.
    """
    if not rows:
        return []

    picked = []
    picked_keys = set()
    used_sources = set()

    def key_for(content, meta):
        return (content[:200] if isinstance(content, str) else str(content), str(meta))

    # Pass 1: one per source_file (preserving global distance order)
    for content, meta in rows:
        if len(picked) >= limit:
            break
        sf = _source_file_from_metadata(meta)
        k = key_for(content, meta)
        if k in picked_keys:
            continue
        if sf and sf in used_sources:
            continue
        picked.append({"content": content, "metadata": meta})
        picked_keys.add(k)
        if sf:
            used_sources.add(sf)

    # Pass 2: fill remaining slots in distance order
    for content, meta in rows:
        if len(picked) >= limit:
            break
        k = key_for(content, meta)
        if k in picked_keys:
            continue
        picked.append({"content": content, "metadata": meta})
        picked_keys.add(k)

    return picked[:limit]


def get_embedding(text):
    """
    Experimental embedding logic or fallback
    """
    # Note: If v1/responses doesn't do embeddings, we might still need a standard model
    # but let's try to keep it simple for now. 
    # For now, I'll keep the standard embedding call but use ada-002 which is safe.
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=[text], model=OPENAI_EMBEDDING_MODEL)
    return response.data[0].embedding

@app.post("/search")
async def search(query_data: SearchQuery):
    user_query = query_data.query

    try:
        # Step A: Intent Understanding
        intent_prompt = INTENT_CLASSIFICATION_PROMPT.format(user_query=user_query)
        intent_raw = call_openai_responses(intent_prompt)
        intent = "SQL" if "SQL" in intent_raw.upper() else "VECTOR"
        print(f"--- Detected Intent: {intent} ---")

        retrieved_data = []
        conn = get_db_connection()
        cur = conn.cursor()

        if intent == "SQL":
            sql_prompt = SQL_GENERATION_PROMPT.format(user_query=user_query)
            generated_sql = call_openai_responses(sql_prompt).strip()
            print(f"Generated SQL: {generated_sql}")

            try:
                cur.execute(generated_sql)
                columns = [desc[0] for desc in cur.description]
                retrieved_data = [dict(zip(columns, row)) for row in cur.fetchall()]
                retrieved_data = retrieved_data[:MAX_RETRIEVED_SOURCES]
            except Exception as sql_err:
                print(f"SQL Error: {sql_err}")
                intent = "VECTOR"

        if intent == "VECTOR" or not retrieved_data:
            query_embedding = get_embedding(user_query)
            # Fetch more neighbors, then diversify so FAQ doesn't dominate every answer.
            pool_limit = int(os.getenv("SEARCH_VECTOR_POOL", "40"))
            final_limit = MAX_RETRIEVED_SOURCES
            cur.execute(
                """
                SELECT content, metadata
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_embedding, pool_limit),
            )
            rows = cur.fetchall()
            retrieved_data = diversify_vector_hits(rows, limit=final_limit)

        cur.close()
        conn.close()

        # If retrieval turned up nothing, don't let the LLM hallucinate an answer.
        if not retrieved_data:
            return {
                "answer": "I could not find the answer in the available data.",
                "retrieved": [],
                "intent": intent,
            }

        # Step D: Final Response Generation
        # psycopg2 may return DECIMAL values for numeric columns (e.g. products.price).
        # Convert them to JSON-safe values so we can build the final prompt.
        final_prompt = FINAL_ANSWER_PROMPT.format(
            retrieved_data=json.dumps(retrieved_data, indent=2, default=str),
            user_query=user_query,
        )
        answer = call_openai_responses(final_prompt)

        return {
            "answer": answer,
            "retrieved": retrieved_data,
            "intent": intent
        }

    except Exception as e:
        print(f"Backend Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "Maxol AI Backend is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
