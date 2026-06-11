# Synthèse de l'Arène des Algorithmes - Jour 3

Cette note de synthèse présente les résultats et conclusions de la journée du Jour 3.

## Phase A : Prédire les prix immobiliers (régression)

### Cas limite : entraînement sur 100 lignes
Oui, le **R² peut fortement baisser**. Avec seulement 100 lignes, le modèle a moins d'exemples pour apprendre et ses prédictions sont moins fiables.

**Pourquoi ?**
Un modèle a besoin de beaucoup de données pour comprendre les relations entre les variables et faire de bonnes prédictions.

### Cas adversarial : revenu médian = 0 et 9000 habitants
Le modèle peut renvoyer une valeur peu réaliste, car ces données sont très différentes de celles utilisées pour l'entraînement.

**En production :**
Il faut vérifier que les données d'entrée sont valides et refuser ou signaler les valeurs hors plage avant de faire la prédiction.

---

## Phase B : Segmenter les clients d'AirBnB (non supervisé)

### Cas normal
5 segments ont été identifiés. Chaque segment correspond à un groupe d'annonces ayant des caractéristiques similaires.

### Cas limite
Sans standardisation, les clusters ont moins de sens car une variable avec de grandes valeurs peut dominer les autres et influencer le résultat.

### Cas adversarial
Une annonce à 100 000 € déplace les centres des clusters et fausse le regroupement. C'est pourquoi le nettoyage des valeurs aberrantes est indispensable avant d'appliquer KMeans.

---

## Phase C : Courriel vs spam (texte)

### Happy path
Le modèle détecte correctement la majorité des spams. Le **recall spam est élevé**, donc il repère bien les messages indésirables.

### Edge case (message vide)
Un message vide est accepté par le vectorizer et est généralement classé comme **ham** (non spam).

### Adversarial (spam déguisé)
Le spam déguisé est mal classé en **ham**. Le modèle se fait donc piéger par des messages trop proches du langage normal.

---

## Phase D : Décrypter les signaux d'un sonar (classification binaire)

### Cas normal
Le **SVM (rbf)** est le plus performant avec **0.93 d'accuracy**. Il est bien adapté à ce type de données avec beaucoup de variables et peu d'exemples.

### Cas limite (sans standardisation)
Sans standardisation, les performances du **SVM** et de la **régression logistique** baissent.
Le **Random Forest** est moins impacté car il dépend peu de l'échelle des variables.

### Cas adversarial (écho à zéro)
Un signal avec toutes les valeurs à zéro est quand même classé par le modèle avec une certaine confiance, même s'il n'a aucun sens réel.

En pratique, ce type de donnée devrait être détecté en amont comme une **erreur de capteur**, et ne jamais être envoyé au modèle.

---

## Phase E : Le Fight des IA (ouverte)
Implémentation d'un leaderboard global chronométré et sélection du champion de l'arène.
