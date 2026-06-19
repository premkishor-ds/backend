import os
import json
import psycopg2
import requests
from dotenv import load_dotenv

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

def format_retrieved_item(item: dict) -> str:
    content = item.get("content", "")
    meta = item.get("metadata") or {}
    source_file = meta.get("source_file", "unknown")
    record = meta.get("record", {})
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

conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
cur = conn.cursor()
emb = get_embedding("Engine Oil")
cur.execute(
    "SELECT content, metadata FROM documents ORDER BY embedding <=> %s::vector LIMIT 40;",
    (emb,)
)
rows = cur.fetchall()

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

formatted_context = ""
for idx, item in enumerate(picked):
    formatted_context += f"Document {idx+1} (Source: {item['metadata'].get('source_file')}):\n{format_retrieved_item(item)}\n"

from prompts import FINAL_ANSWER_PROMPT

prompt = FINAL_ANSWER_PROMPT.format(
    retrieved_data=formatted_context,
    user_query="Engine Oil"
)

payload = {
    "model": OLLAMA_MODEL,
    "prompt": prompt,
    "stream": False,
    "format": "json"
}

print("Querying Ollama with clean formatted context...")
response = requests.post(OLLAMA_API_URL, json=payload)
print(response.json()["response"])
