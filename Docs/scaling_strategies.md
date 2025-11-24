# Stratégies de scaling

## 1. Détection automatique
- skewness > 1 → PowerTransform
- skewness entre 0.5 et 1 → MinMaxScaler
- skewness < 0.5 → StandardScaler

## 2. Cas spécifiques
- Données positives → log transform possible
- Distributions très asymétriques → Yeo-Johnson

## 3. Objectifs
- Éviter les features dominantes
- Améliorer la convergence des modèles
