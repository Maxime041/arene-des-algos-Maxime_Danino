import requests

url = "http://127.0.0.1:5000/predict"

# 1. Cas normal (30 valeurs numériques)
payload_normal = {"features": [1.0] * 30}
response = requests.post(url, json=payload_normal)
print("Cas normal :", response.json())

# 2. Cas limite (clé manquante)
payload_limite = {}
response = requests.post(url, json=payload_limite)
print("Cas limite :", response.status_code, response.json())

# 3. Cas adversarial (texte)
payload_adv = {"features": ["erreur"] * 30}
response = requests.post(url, json=payload_adv)
print("Cas adversarial :", response.status_code, response.json())
