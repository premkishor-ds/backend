import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://192.168.10.148:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

context = """Document 1 (Source: product.json):
{"name": "Maxol Performance 10W/40", "description": "A universal oil for cars and vans, suitable for petrol and diesel engines, with or without turbo-charging."}

Document 2 (Source: product.json):
{"name": "Maxol Supersynth FE 0W20", "description": "A modern, synthetic, fuel economy motor oil based on special selected synthetic base oils."}"""

# Use ChatML format
prompt = f"""<|im_start|>system
You are a helpful customer service AI for Maxol.
Given the context (retrieved from a database) and the user's question, provide a natural, friendly answer and 3 relevant follow-up questions.

Rules:
- Return the response in JSON format with exactly two keys: "answer" (string) and "suggestions" (a list of exactly 3 strings).
- "answer" MUST contain the actual friendly response text in markdown format. Do NOT return the raw database query output.
- Ensure the JSON is valid and can be parsed.

Example of expected output structure:
{{
  "answer": "We have several engine oils available, including **Maxol Performance 10W/40** and **Maxol Supersynth FE 0W20**...",
  "suggestions": [
    "What are the opening hours?",
    "Where is the nearest Maxol station?",
    "Do you have business fuel cards?"
  ]
}}
<|im_end|>
<|im_start|>user
Context:
{context}

User Question:
Engine Oil
<|im_end|>
<|im_start|>assistant
"""

payload = {
    "model": OLLAMA_MODEL,
    "prompt": prompt,
    "stream": False,
    "format": "json"
}

response = requests.post(OLLAMA_API_URL, json=payload)
print(response.json()["response"])
