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

## Phase 4 : Choisir la bonne métrique selon le coût métier

## Phase 5 : Sérialiser le modèle et le servir derrière une API

## Phase 6 : Déployer une WebApp de prédiction

## Phase 7 : L'arbitrage final (phase ouverte)
