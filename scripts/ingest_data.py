"""
Ingest all Maxol JSON exports into PostgreSQL + pgvector.

Each file is a JSON array of records. Records are nested (e.g. { "data": { ... } }).
We embed recursive text from the full record so search works for aboutus, faq, location, etc.

Usage:
  cd backend/scripts
  python -u ingest_data.py              # append to existing tables
  python -u ingest_data.py --fresh      # truncate documents + products first
"""

import argparse
import json
import os
import sys

import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

# Load backend/.env (this file lives in backend/scripts/)
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"),
    override=True,
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

DB_HOST = os.getenv("DB_HOST", "192.168.1.29")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "ai-based-maxol-rag-search")
DB_USER = os.getenv("DB_USER", "ai-based-maxol-rag-search")
DB_PASS = os.getenv("DB_PASS", "Pg4cD8kdFr8vQwn7Mr4zjW")

# Process in a stable order (all site content you listed).
DATA_FILES = [
    "aboutus.json",
    "business.json",
    "faq.json",
    "forecourt.json",
    "instore.json",
    "location.json",
    "product.json",
]

MAX_EMBED_CHARS = 12000


def get_embedding(text: str) -> list[float]:
    text = text.replace("\n", " ").strip()
    if not text:
        text = " "
    response = client.embeddings.create(
        input=[text],
        model=OPENAI_EMBEDDING_MODEL,
    )
    return response.data[0].embedding


def build_searchable_text(item: object, max_chars: int = MAX_EMBED_CHARS) -> str:
    """
    Pull human-readable text from arbitrarily nested JSON (Yext-style exports).
    """
    parts: list[str] = []

    def walk(x: object) -> None:
        if len(" ".join(parts)) >= max_chars:
            return
        if isinstance(x, str):
            s = " ".join(x.split())
            if len(s) > 1:
                parts.append(s)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(item)
    text = " ".join(parts).strip()
    if len(text) < 40:
        # Fallback: structured dump still embeds better than nothing
        try:
            text = json.dumps(item, ensure_ascii=False)
        except Exception:
            text = str(item)
    return text[:max_chars]


def label_for_log(item: object) -> str:
    if isinstance(item, dict):
        data = item.get("data")
        if isinstance(data, dict):
            for key in (
                "c_productHeadings",
                "name",
                "question",
                "c_pagesAboutTitle",
                "id",
            ):
                v = data.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()[:80]
        for key in ("name", "title", "question"):
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()[:80]
    return "record"


def insert_product_row(cur, item: dict) -> None:
    """Only real catalogue rows belong in `products` (SQL path)."""
    data = item.get("data")
    if not isinstance(data, dict):
        return
    name = (
        data.get("c_productHeadings")
        or data.get("name")
        or "Untitled"
    )
    if name == "Untitled":
        return
    category = "General"
    if isinstance(data.get("dm_directoryParents"), list) and data["dm_directoryParents"]:
        try:
            last = data["dm_directoryParents"][-1]
            if isinstance(last, dict) and last.get("name"):
                category = str(last["name"])[:100]
        except Exception:
            pass
    price = float(data.get("price") or 0.0)
    stock = int(data.get("stock") or 0)
    location = "Unknown"
    cur.execute(
        """
        INSERT INTO products (name, category, price, stock, location)
        VALUES (%s, %s, %s, %s, %s);
        """,
        (name[:255], category[:100], price, stock, location[:255]),
    )


def truncate_tables(cur) -> None:
    cur.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
    cur.execute("TRUNCATE TABLE products RESTART IDENTITY;")


def ingest_json_data(conn, file_path: str, source_name: str, ingest_products: bool) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{file_path} must be a JSON array")

    cur = conn.cursor()
    print(f"--- Starting ingestion: {source_name} ({len(records)} records) ---", flush=True)

    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            continue

        meta = {"source_file": source_name, "index": idx, "record": item}
        label = label_for_log(item)
        searchable_text = build_searchable_text(item)

        if ingest_products:
            try:
                insert_product_row(cur, item)
            except Exception as e:
                print(f"  -> product insert error [{label}]: {e}", flush=True)

        print(f"  -> [{idx + 1}/{len(records)}] embed: {label[:60]}...", flush=True)
        try:
            vector = get_embedding(searchable_text)
            cur.execute(
                """
                INSERT INTO documents (content, metadata, embedding)
                VALUES (%s, %s, %s);
                """,
                (searchable_text, json.dumps(meta, ensure_ascii=False), vector),
            )
        except Exception as e:
            print(f"  -> Error embedding [{label}]: {e}", flush=True)

    conn.commit()
    cur.close()
    print(f"--- Finished: {source_name} ---", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Truncate documents + products before ingesting (recommended after code fixes)",
    )
    args = parser.parse_args()

    data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
    if not os.path.isdir(data_dir):
        print(f"Directory not found: {data_dir}", file=sys.stderr)
        return 1

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )
    conn.autocommit = False

    cur = conn.cursor()
    if args.fresh:
        print("--- TRUNCATE documents + products (fresh ingest) ---", flush=True)
        truncate_tables(cur)
        conn.commit()
    cur.close()

    missing = [fn for fn in DATA_FILES if not os.path.isfile(os.path.join(data_dir, fn))]
    if missing:
        print("Missing expected files:", ", ".join(missing), file=sys.stderr)

    for fn in DATA_FILES:
        path = os.path.join(data_dir, fn)
        if not os.path.isfile(path):
            print(f"SKIP (not found): {fn}", flush=True)
            continue
        ingest_products = fn == "product.json"
        ingest_json_data(conn, path, fn, ingest_products=ingest_products)

    conn.close()
    print("--- All listed files processed ---", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
