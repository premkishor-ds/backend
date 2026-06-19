import requests

url = "http://127.0.0.1:8000/search"
payload = {"query": "What are the fuel prices at McBride's Enniskillen?"}

response = requests.post(url, json=payload)
print("Status Code:", response.status_code)
res_json = response.json()
print("Answer:", res_json.get("answer"))
print("Suggestions:", res_json.get("suggestions"))
print("Intent:", res_json.get("intent"))
