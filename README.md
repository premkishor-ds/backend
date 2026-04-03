# Maxol AI RAG Search — Backend

FastAPI service that routes natural-language queries to SQL or vector search (pgvector), then generates an answer with OpenAI.

## Requirements

- Python 3.10+
- PostgreSQL with the `pgvector` extension and a `documents` table (and optional `products` schema for SQL intent)

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
```

## Environment variables

Copy the example file and fill in real values:

```bash
copy .env.example .env
```

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required) |
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASS` | PostgreSQL connection |
| `OPENAI_RESPONSES_MODEL` | Model for `/v1/responses` calls (default: `gpt-4.1-mini`) |
| `OPENAI_CHAT_COMPLETIONS_MODEL` | Model for chat fallback (defaults to `OPENAI_RESPONSES_MODEL`) |
| `OPENAI_EMBEDDING_MODEL` | Embedding model (default: `text-embedding-3-small`, matches `vector(1536)` in `init_db.py`) |

The app loads `backend/.env` next to `main.py` and overrides any existing process environment for those keys.

## Ingest site data (required for vector search)

JSON exports must be loaded into the `documents` table (with embeddings). The ingest script processes these files in order:

`aboutus.json`, `business.json`, `faq.json`, `forecourt.json`, `instore.json`, `location.json`, `product.json`

```bash
cd backend\scripts
python -u ingest_data.py --fresh
```

- `--fresh` truncates `documents` and `products` first (use after changing ingest logic or to avoid duplicates).
- `product.json` rows also populate the `products` table (for SQL-style questions).

## Run

```bash
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/`
- Search: `POST http://localhost:8000/search` with JSON body `{ "query": "your question" }`

Point the frontend at this URL via `NEXT_PUBLIC_BACKEND_URL` (see `frontend/.env.example`).
