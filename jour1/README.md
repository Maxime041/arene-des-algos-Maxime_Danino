# Synthèse de l'Arène des Algorithmes - Jour 1 

Cette note de synthèse présente les résultats et conclusions de mon arène de comparaison d'algorithmes d'apprentissage automatique (Machine Learning).

## Le Problème
L'objectif est d'évaluer et de comparer plusieurs modèles d'apprentissage automatique pour résoudre des problèmes de classification (répartir des données dans des catégories). J'ai testé mes algorithmes sur deux cas concrets :
1. **Le diagnostic médical** : Prédire si une tumeur mammaire est bénigne (sans danger) ou maligne (cancéreuse) à partir de mesures médicales.
2. **La classification de vins** : Identifier le cépage d'origine d'un vin à partir de ses caractéristiques chimiques.

## Les Algorithmes Comparés
J'ai fait s'affronter trois algorithmes classiques :
- **Régression logistique** : Calcule la probabilité qu'un élément appartienne à une catégorie.
- **Arbre de décision** : Fonctionne comme un arbre de choix avec des règles de décision simples (ex: "si la taille > 15 alors...").
- **KNN (K-Plus Proches Voisins)** : Compare le nouvel élément aux exemples existants les plus proches pour décider.

## Tableau de Classement Final
Le tableau ci-dessous regroupe les scores de précision (le pourcentage de bonnes prédictions) obtenus par chaque modèle :

| Algorithme | Diagnostic Cancer (Données brutes) | Classification Vin (Données brutes) | Classification Vin (Données standardisées*) |
| :--- | :---: | :---: | :---: |
| **Régression logistique** | **95,6%** | **100,0%** | **100,0%** |
| **Arbre de décision** | 94,7% | 94,4% | 94,4% |
| **KNN (K-Plus Proches Voisins)** | 95,6% | 72,2% | 94,4% |

*\* La standardisation (scaling) consiste à ramener toutes les caractéristiques à une échelle commune pour éviter qu'une variable avec de grands chiffres n'écrase les autres. Le KNN y est extrêmement sensible (gain de +22,2% après standardisation).*

## Le Champion Retenu : La Régression Logistique
Bien que la régression logistique et le KNN obtiennent des scores de précision similaires sur les données préparées, je choisis la **Régression Logistique** pour une utilisation dans le monde réel.

### Pourquoi ce choix ?

1. **Facilité d'explication (Explicabilité)**
La régression logistique ne donne pas seulement une réponse brute (malade ou sain). Elle fournit une probabilité précise (ex: "92% de chances que la tumeur soit bénigne"). C'est une information cruciale pour un médecin ou un utilisateur final, car elle indique le degré de certitude du modèle.

2. **Gestion fine des erreurs**
Dans le domaine médical, toutes les erreurs n'ont pas la même gravité. Manquer un cancer (faux négatif) est critique, alors que s'inquiéter à tort pour une tumeur bénigne (faux positif) mènera simplement à des examens de contrôle. Avec la régression logistique, on peut facilement abaisser le seuil de décision (ex: alerter dès 30% de probabilité de cancer) pour minimiser au maximum les erreurs graves.

3. **Vitesse et légèreté**
Cet algorithme est mathématiquement simple. Il s'entraîne en une fraction de seconde et consomme très peu d'énergie et de mémoire. Il peut être déployé facilement sur n'importe quel appareil sans matériel coûteux.
