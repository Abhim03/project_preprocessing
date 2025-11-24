# Checklist Cleaning

## 1. Valeurs manquantes
- Calcul du pourcentage de NaN
- Décisions :
  - supprimer la colonne
  - imputer numérique
  - imputer catégoriel
  - imputer datetime
- Méthode choisie selon configuration

## 2. Outliers
- Détection automatique
- Méthode : IQR / Z-score / Isolation Forest
- Action : clip / winsorize / suppression

## 3. Doublons
- Détection des lignes dupliquées
- Suppression si activée

## 4. Incohérences
- Valeurs impossibles (ex : âge < 0)
- Correction si possible
- Suppression sinon
