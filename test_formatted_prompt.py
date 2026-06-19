import os
import json
import psycopg2
import requests
from dotenv import load_dotenv
from prompts import FINAL_ANSWER_PROMPT

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "192.168.1.29")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "ai-based-maxol-rag-search")
DB_USER = os.getenv("DB_USER", "ai-based-maxol-rag-search")
DB_PASS = os.getenv("DB_PASS", "Pg4cD8kdFr8vQwn7Mr4zjW")

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://192.168.10.148:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://192.168.10.148:11434/api/embed")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

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

def _source_file_from_metadata(meta) -> str:
    if not meta: return ""
    return meta.get("source_file", "")

def diversify_vector_hits(rows, limit=5):
    picked = []
    picked_keys = set()
    used_sources = set()
    for content, meta in rows:
        if len(picked) >= limit: break
        sf = _source_file_from_metadata(meta)
        k = (content[:200], str(meta))
        if k in picked_keys or (sf and sf in used_sources): continue
        picked.append({"content": content, "metadata": meta})
        picked_keys.add(k)
        if sf: used_sources.add(sf)
    for content, meta in rows:
        if len(picked) >= limit: break
        k = (content[:200], str(meta))
        if k in picked_keys: continue
        picked.append({"content": content, "metadata": meta})
        picked_keys.add(k)
    return picked[:limit]

conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
cur = conn.cursor()
emb = get_embedding("Engine Oil")
cur.execute(
    "SELECT content, metadata FROM documents ORDER BY embedding <=> %s::vector LIMIT 40;",
    (emb,)
)
rows = cur.fetchall()
retrieved_data = diversify_vector_hits(rows, limit=5)
cur.close()
conn.close()

# Format context as clean text instead of JSON
formatted_context = ""
for idx, item in enumerate(retrieved_data):
    source_file = _source_file_from_metadata(item.get("metadata"))
    content = item.get("content", "")
    formatted_context += f"Document {idx+1} (Source: {source_file}):\n{content}\n\n"

# Update final prompt to have a structured output template
TEST_PROMPT = """You are a helpful customer service AI for Maxol.
Given the following context (retrieved from a database) and a user's original question, provide a natural, friendly answer and 3 relevant follow-up questions that the user might want to ask next.

Rules:
- Return the response in JSON format with exactly two keys: "answer" (string) and "suggestions" (a list of exactly 3 strings).
- "answer" MUST contain the actual friendly response text. Do NOT return the raw data or list.
- Ensure the JSON is valid and can be parsed.

Expected JSON Structure:
{{
  "answer": "Your actual friendly customer service response content here.",
  "suggestions": [
    "Suggested question 1?",
    "Suggested question 2?",
    "Suggested question 3?"
  ]
}}

Context:
{context}

User Question:
{user_query}

Response (JSON):"""

prompt = TEST_PROMPT.format(
    context=formatted_context,
    user_query="Engine Oil"
)

payload = {
    "model": OLLAMA_MODEL,
    "prompt": prompt,
    "stream": False,
    "format": "json"
}

print("Querying Ollama with formatted context...")
response = requests.post(OLLAMA_API_URL, json=payload)
print(response.json()["response"])
