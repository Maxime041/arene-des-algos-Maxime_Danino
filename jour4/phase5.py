import joblib
import numpy as np
from flask import Flask, request, jsonify

def sauvegarder_modele(modele, scaler, chemin="modele.joblib"):
    """Sauvegarde le modèle ET le scaler ensemble (sinon les prédictions
    en prod seront fausses : il faut normaliser pareil qu'à l'entraînement).
    """
    # TODO : joblib.dump({"modele": modele, "scaler": scaler}, chemin)
    joblib.dump({"modele": modele, "scaler": scaler}, chemin)


app = Flask(__name__)
# Désactive le tri alphabétique automatique des clés dans le JSON
app.json.sort_keys = False

# TODO : au démarrage, recharger le dict avec joblib.load(...)
try:
    dict_modele = joblib.load("modele.joblib")
    modele = dict_modele["modele"]
    scaler = dict_modele["scaler"]
except FileNotFoundError:
    modele = None
    scaler = None


@app.route("/predict", methods=["POST"])
def predict():
    """Reçoit un JSON {"features": [...]}, renvoie {"prediction": ..., "proba": ...}.
    Doit : lire le JSON, normaliser avec le scaler, prédire, renvoyer du JSON.
    """
    # TODO : data = request.get_json()
    data = request.get_json()
    
    # TODO : valider que "features" existe et a la bonne longueur
    if not data or "features" not in data or len(data["features"]) != 30:
        return jsonify({"error": "La clé 'features' est manquante ou invalide (doit contenir 30 valeurs)."}), 400
        
    try:
        features = [float(x) for x in data["features"]]
    except (ValueError, TypeError):
        return jsonify({"error": "Toutes les valeurs de features doivent être numériques."}), 400
        
    # TODO : normaliser, prédire, renvoyer jsonify({...})
    if modele is None or scaler is None:
        return jsonify({"error": "Modèle ou scaler non chargé sur le serveur."}), 500
        
    features_scaled = scaler.transform(np.array(features).reshape(1, -1))
    pred = int(modele.predict(features_scaled)[0])
    proba = float(modele.predict_proba(features_scaled)[0][pred])
    label = "benigne" if pred == 1 else "maligne"
    
    return jsonify({
        "prediction": pred,
        "probabilite": round(proba, 2),
        "label": label
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)