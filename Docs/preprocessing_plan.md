# Plan global du préprocessing

## 1. Ingestion des données
- Détection automatique du format (csv, excel, parquet, json).
- Chargement du DataFrame.
- Détection de colonnes ID.
- Inférence des types (numérique, catégoriel, datetime).
- Validation du schéma initial.

## 2. Analyse exploratoire automatique (EDA)
- Statistiques descriptives.
- Distribution des variables numériques.
- Distribution des variables catégorielles.
- Heatmap de corrélation.
- Détection des colonnes quasi constantes.
- Détection des corrélations fortes.
- Rapport sur la qualité des données.

## 3. Gestion des valeurs manquantes
- Analyse du pourcentage de données manquantes.
- Détection du type MAR / MCAR / MNAR.
- Stratégies d'imputation selon le type :
  - Numérique : mean / median / KNN.
  - Catégoriel : most_frequent / "missing".
  - Datetime : interpolation.
- Suppression des colonnes dépassant un seuil de NaN.

## 4. Gestion des outliers
- Détection automatique : IQR, Z-score, Isolation Forest.
- Méthodes d’action : clipping, winsorizing, suppression.

## 5. Encodage des variables catégorielles
- Détection de la cardinalité.
- Règles :
  - faible cardinalité → One Hot.
  - moyenne cardinalité → Target Encoding.
  - haute cardinalité → Hashing.
  - ordinal → OrdinalEncoder automatique.

## 6. Scaling
- Détection automatique basée sur la distribution.
- StandardScaler, MinMaxScaler ou PowerTransform selon skewness.

## 7. Feature Engineering
- Extraction de features temporelles.
- Groupement des catégories rares.
- Transformations log.
- Interactions (optionnel).

## 8. Feature Selection
- Suppression des colonnes constantes.
- Suppression des colonnes très corrélées.
- Variance Threshold.
- Mutual Information.
- PCA (optionnel).

## 9. Train/Test Split
- Test_size configurable.
- Stratification automatique pour target catégorielle.
- Export : X_train, X_test, y_train, y_test.

## 10. Export final
- Data nettoyée complète.
- Pipeline sérialisé (pickle).
- Rapports JSON.
- Graphiques (EDA).
