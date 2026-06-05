import requests

url = "https://generativelanguage.googleapis.com/v1/models?key=YOUR_GEMINI_KEY_HERE"
for m in requests.get(url).json().get("models", []):
    print(m["name"], m.get("supportedGenerationMethods", []))
