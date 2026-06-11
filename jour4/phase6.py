import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

def lancer_webapp():
    """Charge le modèle, affiche des champs de saisie pour les features,
    un bouton 'Prédire', et le résultat avec sa probabilité.
    Lancement : streamlit run app.py
    """
    # TODO : charger le dict modèle+scaler avec joblib.load
    try:
        dict_modele = joblib.load("modele.joblib")
        modele = dict_modele["modele"]
        scaler = dict_modele["scaler"]
    except FileNotFoundError:
        st.error("Fichier 'modele.joblib' manquant. Veuillez d'abord sauvegarder le modèle.")
        return

    data_bc = load_breast_cancer()
    mean_features = data_bc.data.mean(axis=0)
    min_features = data_bc.data.min(axis=0)
    max_features = data_bc.data.max(axis=0)

    # TODO : st.title(...) et des st.number_input(...) pour les features clés
    st.title("Diagnostic de Tumeur Mammaire")
    st.write("Saisissez les caractéristiques clés de la tumeur.")

    val_radius = st.number_input("Rayon moyen", min_value=0.0, value=float(mean_features[0]))
    val_texture = st.number_input("Texture moyenne", min_value=0.0, value=float(mean_features[1]))
    val_perimeter = st.number_input("Périmètre moyen", min_value=0.0, value=float(mean_features[2]))
    val_area = st.number_input("Aire moyenne", min_value=0.0, value=float(mean_features[3]))
    val_smoothness = st.number_input("Lissage moyen", min_value=0.0, value=float(mean_features[4]))

    # TODO : au clic sur st.button("Prédire"), normaliser, prédire, st.write le résultat
    if st.button("Prédire"):
        inputs = [val_radius, val_texture, val_perimeter, val_area, val_smoothness]
        if any(v is None for v in inputs):
            st.warning("Veuillez remplir tous les champs.")
            return

        # Cas adversarial : vérification des valeurs hors limites
        if any(inputs[i] < min_features[i] or inputs[i] > max_features[i] for i in range(5)):
            st.warning("Attention : Valeurs hors plage d'entraînement. La prédiction peut être moins fiable.")

        # Reconstitution du vecteur complet (30 caractéristiques)
        features_vector = mean_features.copy()
        features_vector[:5] = inputs

        features_scaled = scaler.transform(features_vector.reshape(1, -1))
        pred = int(modele.predict(features_scaled)[0])
        proba = float(modele.predict_proba(features_scaled)[0][pred])
        
        # TODO : afficher la probabilité (st.metric ou st.progress)
        if pred == 1:
            st.success(f"Tumeur bénigne ({proba * 100:.1f}%)")
        else:
            st.error(f"Tumeur maligne ({proba * 100:.1f}%)")
            
        st.metric(label="Confiance", value=f"{proba * 100:.1f}%")
        st.progress(proba)

if __name__ == "__main__":
    lancer_webapp()
