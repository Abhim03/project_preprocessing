# Feature Selection Plan

## 1. Critères de suppression
- Variance = 0
- Corrélation > 0.95
- Faible variance

## 2. Sélection intelligente
- Mutual Information
- Chi² pour colonnes catégorielles
- ANOVA pour numériques

## 3. PCA (optionnel)
- Activée pour datasets larges
- variance_threshold = 95 %
