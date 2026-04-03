import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../backend/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    models = client.models.list()
    for model in models:
        print(model.id)
except Exception as e:
    print(f"Error: {e}")
