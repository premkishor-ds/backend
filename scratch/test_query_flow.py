import os
import json
import psycopg2
import requests
from dotenv import load_dotenv

import sys
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DB_HOST = os.getenv("DB_HOST", "192.168.1.29")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "ai-based-maxol-rag-search")
DB_USER = os.getenv("DB_USER", "ai-based-maxol-rag-search")
DB_PASS = os.getenv("DB_PASS", "Pg4cD8kdFr8vQwn7Mr4zjW")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://192.168.10.148:11434/api/generate")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://192.168.10.148:11434/api/embed")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

user_query = "What are the fuel prices at McBride's Enniskillen?"

def get_embedding(text):
    payload = {"model": OLLAMA_EMBED_MODEL}
    if "api/embed" in OLLAMA_EMBED_URL:
        payload["input"] = text
    else:
        payload["prompt"] = text
    response = requests.post(OLLAMA_EMBED_URL, json=payload)
    res_data = response.json()
    if "embedding" in res_data:
        return res_data["embedding"]
    elif "embeddings" in res_data:
        return res_data["embeddings"][0]
    return []

# Connect to database and retrieve
conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
cur = conn.cursor()

query_embedding = get_embedding(user_query)
cur.execute(
    """
    SELECT content, metadata
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 40;
    """,
    (query_embedding,),
)
rows = cur.fetchall()

# Diversify
picked = []
picked_keys = set()
used_sources = set()
for content, meta in rows:
    if len(picked) >= 5: break
    sf = meta.get("source_file", "")
    k = (content[:200], str(meta))
    if k in picked_keys or (sf and sf in used_sources): continue
    picked.append({"content": content, "metadata": meta})
    picked_keys.add(k)
    if sf: used_sources.add(sf)
for content, meta in rows:
    if len(picked) >= 5: break
    k = (content[:200], str(meta))
    if k in picked_keys: continue
    picked.append({"content": content, "metadata": meta})
    picked_keys.add(k)

cur.close()
conn.close()

# Format context with intent = "SQL" (since it wasn't reset)
intent = "SQL"
formatted_context = ""
if intent == "SQL":
    for idx, row in enumerate(picked):
        row_str = ", ".join(f"{k}: {v}" for k, v in row.items() if v is not None)
        formatted_context += f"Product {idx+1}: {row_str}\n"

print("Formatted Context (Length:", len(formatted_context), "):")
print(formatted_context[:1000])

from prompts import FINAL_ANSWER_PROMPT
final_prompt = FINAL_ANSWER_PROMPT.format(
    retrieved_data=formatted_context.strip(),
    user_query=user_query,
)

payload = {
    "model": OLLAMA_MODEL,
    "prompt": final_prompt,
    "stream": False,
    "format": "json"
}

print("\nCalling Ollama...")
response = requests.post(OLLAMA_API_URL, json=payload)
print("Response status:", response.status_code)
print("Ollama Response:", response.json()["response"])
