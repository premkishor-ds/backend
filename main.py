import os
import json
import psycopg2
import requests
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
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://192.168.10.148:11434/api/generate")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://192.168.10.148:11434/api/embeddings")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen2.5:14b")

# Max items returned in `retrieved` (Sources panel + LLM context). Vector path uses this as diversify limit; SQL is sliced to this.
MAX_RETRIEVED_SOURCES = int(os.getenv("SEARCH_RETRIEVED_MAX", "5"))

class SearchQuery(BaseModel):
    query: str

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,port=DB_PORT,dbname=DB_NAME,user=DB_USER,password=DB_PASS
    )

def call_ollama_generate(prompt: str, response_format: str = None) -> str:
    """Call local Ollama generate endpoint."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    if response_format:
        payload["format"] = response_format
        
    response = requests.post(
        OLLAMA_API_URL,
        json=payload
    )
    response.raise_for_status()
    return response.json()["response"].strip()

def call_chat_completion(prompt: str, response_format: str = None) -> str:
    """Call Ollama generate endpoint as a replacement for chat completions."""
    return call_ollama_generate(prompt, response_format)

def call_openai_responses(prompt: str, response_format: str = None) -> str:
    """Call Ollama generate endpoint as a replacement for responses."""
    return call_ollama_generate(prompt, response_format)

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
    Generate embedding using local Ollama instance
    """
    payload = {"model": OLLAMA_EMBED_MODEL}
    if "api/embed" in OLLAMA_EMBED_URL:
        payload["input"] = text
    else:
        payload["prompt"] = text
        
    response = requests.post(OLLAMA_EMBED_URL, json=payload)
    response.raise_for_status()
    res_data = response.json()
    if "embedding" in res_data:
        return res_data["embedding"]
    elif "embeddings" in res_data:
        return res_data["embeddings"][0]
    else:
        raise ValueError(f"Unexpected embedding response format from Ollama: {res_data}")

def format_retrieved_item(item: dict) -> str:
    """Helper to cleanly format retrieved records based on their source file type."""
    content = item.get("content", "")
    meta = item.get("metadata") or {}
    source_file = meta.get("source_file", "unknown")
    record = meta.get("record", {})
    
    # Get the inner 'data' dictionary if it exists, otherwise use record
    data = record.get("data", record)
    if not isinstance(data, dict):
        return content
        
    if source_file == "product.json":
        name = data.get("c_productHeadings") or data.get("name") or "Untitled Product"
        desc = ""
        # Check description fields
        desc_sum = data.get("c_product_description_md_summary")
        if isinstance(desc_sum, list) and desc_sum:
            desc = desc_sum[0].get("product_description_md_summary", "")
        if not desc:
            desc = data.get("c_productDescriptionMdSummaryArrayMap_...") or data.get("description") or ""
        if not desc:
            for k, v in data.items():
                if ("description" in k.lower() or "desc" in k.lower()) and isinstance(v, str) and len(v) > 20:
                    desc = v
                    break
        
        category = ""
        if isinstance(data.get("dm_directoryParents"), list) and data["dm_directoryParents"]:
            category = " > ".join(str(p.get("name")) for p in data["dm_directoryParents"] if p.get("name"))
        
        price = data.get("price") or data.get("c_product_price")
        price_str = f"€{price:.2f}" if price else "Pricing available on request"
        
        item_str = f"Product Name: {name}\n"
        if category:
            item_str += f"Category: {category}\n"
        if price_str:
            item_str += f"Price: {price_str}\n"
        if desc:
            item_str += f"Description: {desc}\n"
        return item_str
        
    elif source_file == "faq.json":
        q = data.get("question") or data.get("name") or ""
        ans = ""
        ans_v2 = data.get("answerV2")
        if isinstance(ans_v2, dict) and "json" in ans_v2:
            try:
                def walk_lexical(x):
                    parts = []
                    if isinstance(x, dict):
                        if x.get("type") == "text" and isinstance(x.get("text"), str):
                            parts.append(x.get("text"))
                        elif x.get("type") == "link" and isinstance(x.get("url"), str):
                            link_text = "".join(walk_lexical(c) for c in x.get("children", []))
                            parts.append(f"[{link_text}]({x.get('url')})")
                        else:
                            for v in x.values():
                                parts.append(walk_lexical(v))
                    elif isinstance(x, list):
                        for v in x:
                            parts.append(walk_lexical(v))
                    return "".join(parts)
                ans = walk_lexical(ans_v2["json"]).strip()
            except Exception:
                pass
        if not ans:
            ans = data.get("answer") or ""
        return f"FAQ Question: {q}\nFAQ Answer: {ans}\n"
        
    elif source_file == "location.json":
        name = data.get("name") or ""
        address = ""
        addr_obj = data.get("address")
        if isinstance(addr_obj, dict):
            line1 = addr_obj.get("line1") or ""
            city = addr_obj.get("city") or ""
            postal = addr_obj.get("postalCode") or ""
            address = f"{line1}, {city} {postal}".strip(", ")
        services = []
        if isinstance(data.get("c_forecourtServices"), list):
            services.extend(data["c_forecourtServices"])
        if isinstance(data.get("c_inStoreServices"), list):
            services.extend(data["c_inStoreServices"])
        services_str = ", ".join(services) if services else "Fuel, Convenience Store"
        
        loc_str = f"Location Name: {name}\n"
        if address:
            loc_str += f"Address: {address}\n"
        loc_str += f"Services Available: {services_str}\n"
        return loc_str

    name = data.get("name") or data.get("title") or ""
    desc = data.get("description") or data.get("c_pagesAboutDescription") or ""
    if not desc:
        desc = content[:500]
    return f"Source: {source_file}\nTitle/Name: {name}\nDetails: {desc}\n"

@app.post("/search")
async def search(query_data: SearchQuery):
    user_query = query_data.query

    try:
        # Step A: Intent Understanding (Fast Heuristic Classifier)
        sql_keywords = [
            "price", "cost", "stock", "quantity", "how many", "how much", 
            "euro", "€", "aisle", "products in", "list of", "compare", "cheapest", "expensive"
        ]
        is_sql = any(k in user_query.lower() for k in sql_keywords)
        intent = "SQL" if is_sql else "VECTOR"
        print(f"--- Detected Intent (Heuristic): {intent} ---")

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
            if intent == "SQL":
                print("--- SQL query returned no results. Falling back to VECTOR search ---")
                intent = "VECTOR"
            try:
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
            except Exception as vector_err:
                print(f"Vector search failed: {vector_err}. Falling back to keyword text search.")
                words = [w for w in user_query.strip().split() if len(w) > 1]
                if not words:
                    words = [user_query]
                conditions = " OR ".join(["content ILIKE %s" for _ in words])
                params = [f"%{w}%" for w in words]
                cur.execute(
                    f"""
                    SELECT content, metadata
                    FROM documents
                    WHERE {conditions}
                    LIMIT %s;
                    """,
                    (*params, MAX_RETRIEVED_SOURCES)
                )
                rows = cur.fetchall()
                retrieved_data = [{"content": r[0], "metadata": r[1]} for r in rows]

        cur.close()
        conn.close()

        # If retrieval turned up nothing, don't let the LLM hallucinate an answer.
        if not retrieved_data:
            return {
                "answer": "I could not find the answer in the available data.",
                "retrieved": [],
                "intent": intent,
            }

        # Clean and format retrieved data into a non-JSON, plain text block to prevent LLM from getting confused about output structure
        formatted_context = ""
        if intent == "SQL":
            for idx, row in enumerate(retrieved_data):
                row_str = ", ".join(f"{k}: {v}" for k, v in row.items() if v is not None)
                formatted_context += f"Product {idx+1}: {row_str}\n"
        else:
            for idx, item in enumerate(retrieved_data):
                source_file = (item.get("metadata") or {}).get("source_file", "unknown")
                formatted_context += f"Document {idx+1} (Source: {source_file}):\n{format_retrieved_item(item)}\n"

        # Step D: Final Response Generation
        final_prompt = FINAL_ANSWER_PROMPT.format(
            retrieved_data=formatted_context.strip(),
            user_query=user_query,
        )
        llm_response = call_openai_responses(final_prompt, response_format="json")
        print(f"DEBUG LLM RESPONSE: {llm_response}")
        
        # Parse JSON response from LLM
        answer = llm_response
        suggestions = ["Tell me more", "Where can I find this?", "Opening hours"]
        
        try:
            # Clean up the response if it contains markdown code blocks
            cleaned_response = llm_response.strip()
            print(f"DEBUG CLEANED RESPONSE: {cleaned_response}")
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:-3].strip()
            
            parsed = None
            try:
                parsed = json.loads(cleaned_response)
                print(f"DEBUG PARSED JSON: {parsed}")
            except Exception:
                try:
                    import ast
                    parsed = ast.literal_eval(cleaned_response)
                    print(f"DEBUG PARSED AST: {parsed}")
                except Exception as ast_err:
                    print(f"AST Parse Error: {ast_err}")
            
            if isinstance(parsed, dict):
                print(f"DEBUG PARSED KEYS: {list(parsed.keys())}")
                answer = parsed.get("answer") or parsed.get("message") or parsed.get("content") or parsed.get("response") or parsed.get("text")
                if not answer or answer.strip() in ("{}", "{ }", ""):
                    answer = "I could not find the answer in the available data. Please check in-store or consult a Maxol team member for details."
                if isinstance(answer, dict):
                    answer = answer.get("content") or answer.get("text") or answer.get("answer") or str(answer)
                
                # Resilient suggestions fallback (Instant context-aware fallback to save time)
                raw_sug = parsed.get("suggestions")
                if isinstance(raw_sug, list) and len(raw_sug) > 0:
                    suggestions = [str(s) for s in raw_sug]
                else:
                    low_query = user_query.lower()
                    if "fuel" in low_query or "price" in low_query or "diesel" in low_query or "unleaded" in low_query:
                        suggestions = ["Do you accept Fuel Cards?", "What is Premium Fuel?", "How can I pay for fuel?"]
                    elif "station" in low_query or "location" in low_query or "where" in low_query or "near" in low_query:
                        suggestions = ["What are the opening hours?", "What services are available?", "How do I find a station?"]
                    elif "oil" in low_query or "product" in low_query or "grease" in low_query:
                        suggestions = ["What engine oil do I need?", "Where can I buy lubricants?", "What is AdBlue?"]
                    else:
                        suggestions = ["What products do you sell?", "Where is Maxol located?", "What is Maxol Loyalty?"]
            else:
                # Robust regex extraction if dictionary parsing failed
                import re
                extracted = ""
                for key in ["answer", "message", "content", "response", "text"]:
                    pattern = rf'["\']{key}["\']\s*:\s*(["\'])(.*?)\1'
                    match = re.search(pattern, cleaned_response, re.DOTALL)
                    if match:
                        extracted = match.group(2).strip()
                        break
                if extracted:
                    answer = extracted
        except Exception as json_err:
            print(f"JSON Parse Error: {json_err}. Using raw response.")


        # Convert HTML links to Markdown links for ReactMarkdown to render properly
        if isinstance(answer, str):
            import re
            html_link_pattern = r'<a\s+(?:[^>]*?\s+)?href=["\']([^"\']+)["\'][^>]*>(.*?)</a>'
            answer = re.sub(html_link_pattern, r'[\2](\1)', answer)


        return {
            "answer": answer,
            "suggestions": suggestions,
            "retrieved": retrieved_data,
            "intent": intent
        }

    except Exception as e:
        print(f"Backend Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/initial-suggestions")
async def initial_suggestions():
    """
    Returns 4-5 relevant search suggestions based on actual database content.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        suggestions = []
        
        # 1. Get top product categories
        cur.execute("SELECT category FROM products GROUP BY category ORDER BY count(*) DESC LIMIT 2;")
        for row in cur.fetchall():
            if row[0]:
                suggestions.append({"label": row[0], "value": row[0]})
                
        # 2. Add evergreen/location context
        cur.execute("SELECT count(*) FROM documents WHERE metadata->>'source_file' = 'location.json';")
        if cur.fetchone()[0] > 0:
            suggestions.append({"label": "EV Charging", "value": "EV Charging"})
            
        # 3. Add Business or FAQ if available
        cur.execute("SELECT count(*) FROM documents WHERE metadata->>'source_file' = 'business.json';")
        if cur.fetchone()[0] > 0:
            suggestions.append({"label": "Business Fuel", "value": "Business Fuel"})

        cur.close()
        conn.close()
        
        # Fallback if DB is empty
        if not suggestions:
            suggestions = [
                {"label": "Engine Oil", "value": "Engine Oil"},
                {"label": "EV Charging", "value": "EV Charging"}
            ]
            
        return suggestions[:4]
    except Exception as e:
        print(f"Stats Error: {e}")
        return [
            {"label": "EV Charging", "value": "EV Charging"},
            {"label": "Fuel Prices", "value": "Fuel Prices"},
            {"label": "Engine Oil", "value": "Engine Oil"},
            {"label": "Business Fuel", "value": "Business Fuel"}
        ]

@app.get("/")
def home():
    return {"status": "Maxol AI Backend is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
