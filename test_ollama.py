import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://192.168.10.148:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

prompt = """You are a helpful customer service AI for Maxol.
Given the following context (retrieved from a database) and a user's original question, provide a natural, friendly answer and 3 relevant follow-up questions.

Rules:
- Return the response in JSON format with exactly two keys: "answer" (string) and "suggestions" (list of strings).

Context:
[{"content": "Maxol Supersynth FE 0W20 is a modern synthetic motor oil.", "metadata": {}}]

User Question:
Engine Oil

Response (JSON):"""

payload = {
    "model": OLLAMA_MODEL,
    "prompt": prompt,
    "stream": False,
    "format": "json"
}

response = requests.post(OLLAMA_API_URL, json=payload)
print(response.json()["response"])
