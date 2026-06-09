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

# Jour 2  
## Choix Phase 2 : Imputation par la médiane
J'ai choisi d'imputer les trous plutôt que de supprimer les lignes pour ne pas perdre les informations contenues dans les autres colonnes. J'ai utilisé la médiane plutôt que la moyenne car elle résiste mieux aux valeurs extrêmes

## Phase 3 

### Question : Edge case : Contract a 3 modalités (Month-to-month, One year, Two year). Est-ce nominal ou ordinal ? Argumentez votre choix d'encodage dans le README.

La variable est en réalité ordinale, elle suit une logique de progression (1 mois < 1 an < 2 ans). Un encodage ordinal aurait été plus adapté, mais le One-Hot Encoding fonctionne aussi.

### Question : adversarial : que fait votre One-Hot si la colonne contient une catégorie ultra-rare présente sur 1 seul client ? Et le customerID , si vous l'encodiez par erreur en One-Hot, combien de colonnes ça créerait ? (Comptez. C'est la leçon de l'explosion de dimensions, en vrai.)

Encoder customerID aurait créé une colonne par client (7043 colonnes).
C’est inutile car chaque valeur est unique et cela rendrait le modèle inefficace.

## Phase 4

### Les colonnes numériques ont-elles des outliers ?

La méthode IQR n'a détecté aucun outlier dans les colonnes **tenure**, **MonthlyCharges** et **TotalCharges**.

**Stratégie choisie :**

* **tenure** : conserver les valeurs.
* **MonthlyCharges** : conserver les valeurs.
* **TotalCharges** : conserver les valeurs.

Comme aucun outlier n'a été trouvé, aucune correction n'est nécessaire. De plus, dans un contexte télécom, des montants élevés peuvent représenter un comportement client réel et utile pour le modèle.

---

### Vérifications réalisées

#### 1. Comportement normal

Le test sur **MonthlyCharges** a montré que les calculs des quartiles et des bornes IQR fonctionnent correctement.

#### 2. Colonne sans outlier

Une colonne de test contenant uniquement la valeur 50 a été utilisée. La fonction a correctement détecté **0 outlier** sans erreur.

#### 3. Impact sur les clients churn

Aucun outlier n'ayant été détecté, aucune ligne n'a été supprimée. La perte de clients ayant résilié (**churn**) est donc de **0 %**, bien en dessous du seuil d'alerte de **5 %**.

## Phase 5

### Question : Happy path : la heatmap s'affiche, les VIF sont calculés.

Le calcul fonctionne bien. Les résultats montrent que `TotalCharges` (VIF de 8.07) et `tenure` (VIF de 6.32) ont un VIF supérieur à 5. Cela confirme que ces variables sont très corrélées entre elles (elles portent la même information).

### Question : Edge case : deux colonnes parfaitement identiques (dupliquez-en une volontairement). Que vaut le VIF ?

En dupliquant la colonne `TotalCharges`, le calcul du VIF explose et affiche **inf** (infini) pour les deux colonnes identiques. L'outil a même déclenché un avertissement mathématique (division par zéro), ce qui est le comportement normal quand une variable est redondante à 100%.

### Question : Adversarial : si vous supprimez TotalCharges pour cause de VIF élevé, recalculez les VIF après suppression : le problème a-t-il disparu ?

Oui, le problème a totalement disparu ! En supprimant `TotalCharges`, les VIF de `tenure` et `MonthlyCharges` sont retombés à **2.61** (donc bien en dessous du seuil critique de 5). Nous n'avons pas perdu d'information précieuse, car le total payé n'était finalement qu'une combinaison du prix mensuel multiplié par le temps passé. Le dataset est maintenant assaini.


## Phase 6 : Les variables qui prédisent le churn

### Happy Path : Les deux méthodes sont-elles d'accord ?

Oui. Les deux méthodes mettent en avant plusieurs variables importantes comme **tenure**, **TotalCharges**, **MonthlyCharges**, **Contract_Two year** et **PaymentMethod_Electronic check**.

**Interprétation métier :**

* Les clients récents ont davantage tendance à quitter l'entreprise.
* Les contrats de longue durée réduisent le risque de churn.
* Les montants facturés et le mode de paiement ont également un impact sur le départ des clients.

---

### Edge Case : Une colonne quasi constante finit-elle en bas du classement ?

Oui. Une colonne qui contient presque toujours la même valeur n'apporte aucune information utile. Elle est donc classée parmi les variables les moins importantes et n'apparaît pas dans le Top 10.

---

### Adversarial : Forte importance mais faible corrélation

Si une variable est importante pour le modèle Random Forest mais a une faible corrélation, cela signifie que son effet sur le churn est plus complexe qu'une simple relation linéaire.

La corrélation mesure uniquement les relations directes, tandis que Random Forest peut détecter des interactions et des comportements plus complexes entre les variables.
