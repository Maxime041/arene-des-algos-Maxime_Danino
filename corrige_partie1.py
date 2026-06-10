import matplotlib
matplotlib.use("Agg")  # backend sans fenêtre : on écrit les figures en PNG, pas d'affichage

import matplotlib.pyplot as plt
from collections import Counter

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix


# ---------------------------------------------------------------------------
# Le casting des algos. random_state figé partout : on veut un classement
# reproductible, sinon impossible de comparer brut vs scalé honnêtement.
# ---------------------------------------------------------------------------
def get_modeles():
    return {
        "Régression logistique": LogisticRegression(max_iter=10000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Arbre de décision": DecisionTreeClassifier(random_state=0),
    }


# ---------------------------------------------------------------------------
# Phase 1 : Charger et explorer
# ---------------------------------------------------------------------------
def explorer_dataset(X, y, noms_classes=None):
    """Affiche les caractéristiques de base d'un dataset de classification.
    Lignes/colonnes, classes présentes, effectif ET pourcentage par classe.
    Le pourcentage est volontaire : c'est lui qui rend un déséquilibre visible.
    """
    n_lignes, n_colonnes = X.shape
    print(f"Lignes, colonnes : ({n_lignes}, {n_colonnes})")

    effectifs = Counter(y)
    for classe in sorted(effectifs):
        n = effectifs[classe]
        pct = n / n_lignes
        # le libellé "maligne/bénigne" n'existe que si on le fournit, sinon on reste générique
        libelle = ""
        if noms_classes is not None and classe < len(noms_classes):
            libelle = f" ({noms_classes[classe]})"
        print(f"Classe {classe}{libelle} : {n} cas ({pct:.1%})")
    return effectifs


def checkpoint_phase1(X, y):
    """Les 3 situations du checkpoint qualité, pour montrer que l'explo ne ment pas."""
    print("--- cas normal : dataset complet ---")
    explorer_dataset(X, y, noms_classes=["maligne", "bénigne"])

    print("\n--- cas limite : une seule classe (y == 0) ---")
    # on filtre une classe : le décompte doit rester honnête (une seule ligne, 100%)
    masque = y == 0
    explorer_dataset(X[masque], y[masque], noms_classes=["maligne", "bénigne"])

    print("\n--- cas adversarial : un faux dataset à 95 % d'une classe ---")
    # 95 % de classe 1, 5 % de classe 0 : le pourcentage doit crever les yeux.
    n = 200
    y_desequilibre = [1] * 190 + [0] * 10
    X_bidon = X[:n]
    explorer_dataset(X_bidon, y_desequilibre)
    print("Le pourcentage rend le déséquilibre évident : 95 % d'une classe, on le voit tout de suite.")


# ---------------------------------------------------------------------------
# Phase 2 : Le pipeline supervisé complet
# ---------------------------------------------------------------------------
def entrainer_et_evaluer(modele, X_train, X_test, y_train, y_test):
    """Entraîne, prédit, renvoie l'accuracy (float entre 0 et 1)."""
    modele.fit(X_train, y_train)
    return accuracy_score(y_test, modele.predict(X_test))


# ---------------------------------------------------------------------------
# Phase 3 : L'Arène (premier classement)
# ---------------------------------------------------------------------------
def arene(modeles, X_train, X_test, y_train, y_test, titre="Classement"):
    """Fait s'affronter tous les modèles sur le MÊME split. Trie par accuracy."""
    resultats = []
    for nom, modele in modeles.items():
        # on ré-instancie pour repartir d'un modèle vierge (utile quand on rejoue l'arène)
        frais = modele.__class__(**modele.get_params())
        acc = entrainer_et_evaluer(frais, X_train, X_test, y_train, y_test)
        resultats.append((nom, acc))

    resultats.sort(key=lambda r: r[1], reverse=True)

    print(f"{titre} :")
    for rang, (nom, acc) in enumerate(resultats, start=1):
        print(f"{rang}. {nom:<24}: {acc:.1%}")
    return resultats


# ---------------------------------------------------------------------------
# Phase 4 : Bascule non-supervisé
# ---------------------------------------------------------------------------
def clustering_aveugle(X, n_clusters=2):
    """Regroupe les données SANS jamais voir y. Renvoie le cluster de chaque point.
    fit_predict fait l'entraînement ET l'attribution d'un coup.
    """
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    return km.fit_predict(X)


def accord_clusters_vs_classes(clusters, y):
    """Mesure à quel point les clusters retrouvent les vraies classes.

    KMeans numérote ses groupes arbitrairement : le cluster 0 peut correspondre
    à la classe 1. On teste donc l'appariement direct ET l'appariement inversé,
    et on garde le meilleur. C'est une mesure d'accord, pas une "accuracy" au
    sens supervisé : le modèle n'a jamais vu y.
    """
    direct = accuracy_score(y, clusters)
    inverse = accuracy_score(y, 1 - clusters)
    return max(direct, inverse)


# ---------------------------------------------------------------------------
# Phase 6 : Voir pour comprendre
# ---------------------------------------------------------------------------
def graphe_classement(resultats, chemin_png, titre):
    """Diagramme en barres des accuracies. Lisible sans le code : titre + axes nommés."""
    noms = [nom for nom, _ in resultats]
    accs = [acc * 100 for _, acc in resultats]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(noms, accs, color="#4C72B0")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(titre)
    ax.set_ylim(0, 100)
    for i, v in enumerate(accs):
        ax.text(i, v + 1, f"{v:.1f}", ha="center")
    fig.tight_layout()
    fig.savefig(chemin_png, dpi=120)
    plt.close(fig)
    print(f"Figure écrite : {chemin_png}")


def graphe_matrice_confusion(modele, X_train, X_test, y_train, y_test, chemin_png, titre):
    """Matrice de confusion du champion : pas seulement combien d'erreurs, mais lesquelles."""
    modele.fit(X_train, y_train)
    cm = confusion_matrix(y_test, modele.predict(X_test))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Vraie classe")
    ax.set_title(titre)
    # on écrit le compte dans chaque case : la diagonale = bonnes réponses
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            couleur = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=couleur)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(chemin_png, dpi=120)
    plt.close(fig)
    print(f"Figure écrite : {chemin_png}")
    return cm


# ---------------------------------------------------------------------------
# Phase 7, Manche 1 : le buff scaling
# ---------------------------------------------------------------------------
def comparer_scaling(modeles, X_train, X_test, y_train, y_test):
    """Pour chaque algo : accuracy brute vs scalée. Trie par gain décroissant.
    Le scaler s'ajuste sur le TRAIN seul (version honnête : le test reste invisible).
    """
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    lignes = []
    for nom, modele in modeles.items():
        brut = entrainer_et_evaluer(modele.__class__(**modele.get_params()),
                                    X_train, X_test, y_train, y_test)
        scale = entrainer_et_evaluer(modele.__class__(**modele.get_params()),
                                     X_train_s, X_test_s, y_train, y_test)
        lignes.append((nom, brut, scale, scale - brut))

    lignes.sort(key=lambda r: r[3], reverse=True)  # le classement des GAINS est le résultat

    print(f"{'Algo':<24}{'Brut':>8}{'Scalé':>9}{'Gain':>8}")
    for nom, brut, scale, gain in lignes:
        print(f"{nom:<24}{brut:>7.1%}{scale:>9.1%}{gain*100:>+7.1f}")
    return lignes


# ---------------------------------------------------------------------------
# Phase 7, Manche 2 : la triche (data leakage)
# ---------------------------------------------------------------------------
def demo_leakage(champion_cls, champion_kwargs, X, y):
    """Quantifie le mensonge : accuracy honnête (scaler sur train) vs trichée (scaler sur tout X)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # HONNÊTE : le scaler ne voit que le train.
    scaler_ok = StandardScaler().fit(X_train)
    acc_honnete = entrainer_et_evaluer(champion_cls(**champion_kwargs),
                                       scaler_ok.transform(X_train), scaler_ok.transform(X_test),
                                       y_train, y_test)

    # TRICHE : le scaler s'ajuste sur TOUT X (train + test). Le test fuite dans
    # la moyenne/écart-type : le modèle "connaît" déjà la distribution de test.
    scaler_triche = StandardScaler().fit(X)
    acc_triche = entrainer_et_evaluer(champion_cls(**champion_kwargs),
                                      scaler_triche.transform(X_train), scaler_triche.transform(X_test),
                                      y_train, y_test)

    print(f"Honnête (scaler sur train seul) : {acc_honnete:.1%}")
    print(f"Triche  (scaler sur tout X)     : {acc_triche:.1%}")
    print(f"Points gagnés EN TRICHANT       : {(acc_triche - acc_honnete)*100:+.1f}")
    print("Ce delta, c'est le mensonge qu'on se raconterait en prod. "
          "On ajuste TOUJOURS le scaler sur le train seul.")
    return acc_honnete, acc_triche


# ---------------------------------------------------------------------------
# Phase 8 : Raconter votre Arène (note de synthèse générée)
# ---------------------------------------------------------------------------
def note_de_synthese(classement_cancer, classement_wine, champion, gain_triche):
    """Génère le squelette de note de synthèse à coller dans le README.
    Le but : un texte qu'une personne qui ne code pas comprend en une minute.
    """
    podium = "\n".join(f"  {r}. {nom} : {acc:.1%}"
                       for r, (nom, acc) in enumerate(classement_cancer, start=1))
    print("===== Note de synthèse (à adapter dans le README) =====")
    print(f"""
Problème : classer des tumeurs (bénigne / maligne) à partir de 30 mesures.

Algos comparés (même découpage train/test) :
{podium}

Champion retenu : {champion}.
Pourquoi lui : meilleure accuracy ET il profite franchement de la mise à l'échelle,
ce qui en fait un choix robuste une fois le pipeline propre. À nuancer selon le coût
des erreurs : sur du médical, rater une tumeur maligne (faux négatif) est bien plus
grave qu'une fausse alerte. Un arbre de décision serait plus explicable si on doit
justifier chaque prédiction à un médecin.

Garde-fou appris : ajuster le scaler sur tout le dataset gonfle artificiellement le
score d'environ {gain_triche:+.1f} point(s) ici. C'est de la fuite de données. Toujours
fit le scaler sur le train seul (ou un make_pipeline qui rend la triche impossible).

Deuxième terrain (vin, 3 classes) : les mêmes fonctions ont tourné sans modification,
preuve que l'API uniforme tient. Classement vin :
""" + "\n".join(f"  {r}. {nom} : {acc:.1%}"
                 for r, (nom, acc) in enumerate(classement_wine, start=1)))


# ---------------------------------------------------------------------------
# Déroulé complet
# ---------------------------------------------------------------------------
def main():
    # ===== Phase 1 : charger et explorer =====
    print("=== Phase 1 : charger et explorer (breast cancer) ===")
    X, y = load_breast_cancer(return_X_y=True)
    checkpoint_phase1(X, y)

    # ===== Phase 2 : pipeline supervisé (un seul modèle pour valider la mécanique) =====
    print("\n=== Phase 2 : pipeline supervisé complet ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    acc_demo = entrainer_et_evaluer(LogisticRegression(max_iter=10000),
                                    X_train, X_test, y_train, y_test)
    print(f"Régression logistique seule : {acc_demo:.1%}")

    # ===== Phase 3 : l'Arène =====
    print("\n=== Phase 3 : l'Arène (breast cancer) ===")
    print("(Sans scaling, la régression logistique a besoin de max_iter élevé pour converger.)")
    classement_cancer = arene(get_modeles(), X_train, X_test, y_train, y_test,
                              titre="Classement breast cancer")
    champion_nom = classement_cancer[0][0]

    # ===== Phase 4 : bascule non-supervisé =====
    print("\n=== Phase 4 : KMeans aveugle vs vraies classes ===")
    clusters = clustering_aveugle(X, n_clusters=2)
    accord = accord_clusters_vs_classes(clusters, y)
    print(f"Accord clusters / vraies classes : {accord:.1%}")
    print("Le KMeans n'a jamais vu y, et retrouve quand même une bonne partie de la "
          "structure : signe que les deux groupes sont géométriquement séparables.")

    # ===== Phase 5 : changer de terrain (wine, 3 classes) =====
    print("\n=== Phase 5 : même Arène sur un autre terrain (wine, 3 classes) ===")
    Xw, yw = load_wine(return_X_y=True)
    explorer_dataset(Xw, yw)
    Xw_train, Xw_test, yw_train, yw_test = train_test_split(
        Xw, yw, test_size=0.2, random_state=42, stratify=yw)
    classement_wine = arene(get_modeles(), Xw_train, Xw_test, yw_train, yw_test,
                            titre="Classement wine")
    print("Mêmes fonctions, dataset à 3 classes encaissé sans une ligne de plus : "
          "c'est tout l'intérêt de l'API uniforme.")

    # ===== Phase 6 : voir pour comprendre =====
    print("\n=== Phase 6 : visualisations (PNG) ===")
    graphe_classement(classement_cancer, "phase6_classement_cancer.png",
                      "Accuracy par algo (breast cancer)")
    champion_modele = get_modeles()[champion_nom]
    cm = graphe_matrice_confusion(champion_modele, X_train, X_test, y_train, y_test,
                                  "phase6_confusion_champion.png",
                                  f"Matrice de confusion : {champion_nom}")
    # cm[1,0] = vraies bénignes prédites malignes ? l'ordre dépend des labels ;
    # ce qui compte pédagogiquement : un faux négatif (maligne ratée) est le plus grave.
    print(f"Matrice de confusion (lignes = vraie classe, colonnes = prédite) :\n{cm}")

    # ===== Phase 7 : buff scaling + triche =====
    print("\n=== Phase 7, Manche 1 : le buff scaling ===")
    classement_gains = comparer_scaling(get_modeles(), X_train, X_test, y_train, y_test)

    print("\n=== Phase 7, Manche 2 : la triche (data leakage) ===")
    # le champion de la triche = celui qui profite le plus du buff (tête du classement des gains)
    champion_triche = classement_gains[0][0]
    print(f"Champion testé : {champion_triche}")
    champ = get_modeles()[champion_triche]
    acc_honnete, acc_triche = demo_leakage(champ.__class__, champ.get_params(), X, y)
    # delta honnête/triche en points, pour la note de synthèse (mêmes chiffres que ci-dessus)
    gain_triche = (acc_triche - acc_honnete) * 100

    # ===== Phase 8 : note de synthèse =====
    print("\n=== Phase 8 : note de synthèse ===")
    note_de_synthese(classement_cancer, classement_wine, champion_nom, gain_triche)


if __name__ == "__main__":
    main()

