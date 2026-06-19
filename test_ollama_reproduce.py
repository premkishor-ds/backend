import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://192.168.10.148:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

from prompts import FINAL_ANSWER_PROMPT

retrieved_data = [
  {
    "content": "Maxol Supersynth FE 0W20 - The Maxol Group",
    "metadata": {
      "source_file": "product.json",
      "record": {
        "id": "14121558",
        "name": "Maxol Supersynth FE 0W20",
        "description": "A modern, synthetic, fuel economy motor oil based on special selected synthetic base oils..."
      }
    }
  }
]

prompt = FINAL_ANSWER_PROMPT.format(
    retrieved_data=json.dumps(retrieved_data, indent=2),
    user_query="Engine Oil"
)

payload = {
    "model": OLLAMA_MODEL,
    "prompt": prompt,
    "stream": False,
    "format": "json"
}

response = requests.post(OLLAMA_API_URL, json=payload)
print(response.json()["response"])
