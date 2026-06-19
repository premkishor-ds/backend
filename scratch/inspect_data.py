import psycopg2
import json

conn = psycopg2.connect('postgresql://ai-based-maxol-rag-search:Pg4cD8kdFr8vQwn7Mr4zjW@192.168.1.29:5433/ai-based-maxol-rag-search')
cur = conn.cursor()

cur.execute("SELECT DISTINCT metadata->>'source_file' FROM documents")
files = [r[0] for r in cur.fetchall()]
print('Source files:', files)
print('---')

for f in files:
    cur.execute("SELECT metadata FROM documents WHERE metadata->>'source_file' = %s LIMIT 1", (f,))
    r = cur.fetchone()
    if r:
        meta = r[0]
        record = meta.get('record', {})
        data = record.get('data', record)
        print(f"File: {f}")
        # Print first few fields and nested structures
        print(json.dumps({k: v for k, v in list(data.items())[:5]}, indent=2))
        print('---')

cur.close()
conn.close()
