import psycopg2
import json

conn = psycopg2.connect('postgresql://ai-based-maxol-rag-search:Pg4cD8kdFr8vQwn7Mr4zjW@192.168.1.29:5433/ai-based-maxol-rag-search')
cur = conn.cursor()

print("Searching for 'Enniskillen' in documents:")
cur.execute("SELECT content, metadata FROM documents WHERE content ILIKE '%enniskillen%' OR content ILIKE '%mcbride%' LIMIT 5")
for r in cur.fetchall():
    print("Content preview:", r[0][:200])
    print("Metadata:", json.dumps(r[1], indent=2))
    print("---")

print("Searching for 'Enniskillen' in products:")
cur.execute("SELECT * FROM products WHERE name ILIKE '%enniskillen%' OR location ILIKE '%enniskillen%' LIMIT 5")
for r in cur.fetchall():
    print(r)
    print("---")

cur.close()
conn.close()
