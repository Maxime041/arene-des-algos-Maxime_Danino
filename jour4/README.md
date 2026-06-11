# Plan de l'Après-midi - Jour 4 : Évaluation Rigoureuse & Mise en Production

Ce document présente le plan de travail de l'après-midi pour le Jour 4, décrivant les différentes phases à implémenter.

## Phase 0 : Mise en route

## Phase 1 : Séparer les données proprement (Train / Validation / Test)

## Phase 2 : Bootstrap et bagging, comprendre le rééchantillonnage

### Réponses aux questions :
- **Que se passe-t-il si une itération produit un échantillon out-of-bag (OOB) vide ?**  
  S'il n'y a pas de données OOB, je ne peux pas tester le modèle. J'ai géré cela en ignorant l'itération (`continue`) pour ne pas fausser la moyenne.
- **Impact de l'oubli de `replace=True` (sans remise) :**  
  L'échantillon est identique au dataset d'origine. L'OOB est toujours vide et je teste sur les mêmes données, ce qui donne un écart-type de 0.
- **Test avec `n_iterations=1` :**  
  J'obtiens un seul score et aucun écart-type. Mon code s'adapte et affiche uniquement le score obtenu sans écart-type.

## Phase 3 : La validation croisée k-fold

### Réponses aux questions :
- **Pourquoi le Leave-One-Out (LOO) est-il lent ?**  
  Parce que le modèle doit se ré-entraîner autant de fois qu'il y a de lignes dans le dataset (569 fois ici). J'ai mesuré environ 4.5 secondes d'exécution.
- **KFold standard vs StratifiedKFold sur dataset déséquilibré (95/5) :**  
  Avec `KFold` standard, la répartition est aléatoire : un fold peut ne contenir aucun exemple de la classe rare (classe 1), faussant la moyenne. `StratifiedKFold` préserve la proportion de 5% de classe 1 dans chaque fold, ce qui stabilise l'évaluation.

## Phase 4 : Choisir la bonne métrique selon le coût métier

### Réponses aux questions :
- **Comparaison accuracy vs coût métier :**  
  La meilleure accuracy ne garantit pas le plus petit coût. Par exemple, le modèle paresseux (toujours 0) a 99% d'accuracy mais laisse passer toutes les fraudes, ce qui engendre des pertes financières maximales.
- **Gestion du cas limite (division par zéro) :**  
  J'ai configuré `zero_division=0` dans les fonctions de scoring (`precision_score`, etc.) pour éviter les plantages si le modèle ne prédit aucune fraude.
- **Démonstration chiffrée pour un décideur :**  
  - **Modèle du collègue (99% d'accuracy)** : 100 fraudes ratées (FN) $\times$ 10 = **1000 €** de pertes.
  - **Mon modèle pro (98.1% d'accuracy, 95% recall, 50% precision)** : 5 fraudes ratées $\times$ 10 + 95 fausses alertes (FP) $\times$ 1 = **145 €** de pertes.
  - **Verdict** : Mon modèle est **6.9 fois plus économique** malgré une accuracy plus basse.

## Phase 5 : Sérialiser le modèle et le servir derrière une API

## Phase 6 : Déployer une WebApp de prédiction

## Phase 7 : L'arbitrage final (phase ouverte)
