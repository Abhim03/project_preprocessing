# EDA Checklist (Analyse exploratoire automatique)

## 1. Informations générales
- Taille du dataset (nombre de lignes et colonnes)
- Types des colonnes (numérique, catégorielle, datetime, booléen, texte)
- Détection des colonnes ID
- Détection des colonnes constantes (unique value)
- Détection des colonnes quasi-constantes
- Détection des colonnes à forte cardinalité

## 2. Statistiques descriptives
- Moyenne, médiane, variance, écart-type
- Min, max
- Skewness (asymétrie)
- Kurtosis (aplatissement)

## 3. Visualisations automatiques
- Histogrammes pour toutes les colonnes numériques
- Boxplots numériques pour détecter les outliers
- Countplots (barplots) pour les colonnes catégorielles
- Heatmap des corrélations numériques
- Pairplot (optionnel)
- Matrice de valeurs manquantes
- Graphique du pourcentage de NaN par colonne

## 4. Analyse des valeurs manquantes
- Calcul du pourcentage de valeurs manquantes par colonne
- Identification des colonnes avec trop de NaN
- Détection des patterns de données manquantes (MAR, MCAR, MNAR)
- Recommandation d’imputation selon la configuration générale

## 5. Analyse des outliers
- Détection via IQR
- Détection via Z-score
- Détection via Isolation Forest (si nécessaire)
- Nombre d’outliers par colonne
- Colonnes les plus touchées par les valeurs aberrantes

## 6. Analyse des relations entre variables
- Corrélation > 0.95 (à supprimer éventuellement)
- Colonnes fortement corrélées à la target (si target fournie)
- Redondance potentielle entre variables

## 7. Rapport final EDA
Ce rapport doit synthétiser :
- Colonnes problématiques
- Colonnes à supprimer
- Colonnes à imputer
- Colonnes nécessitant un encodage particulier
- Colonnes à scaler
- Colonnes à transformer
- Colonnes avec outliers importants
