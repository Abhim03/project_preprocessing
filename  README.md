# Projet : Pipeline de Préprocessing Automatisé

## Objectif du projet
Ce projet vise à construire un pipeline de préprocessing entièrement automatisé capable de :
- analyser n’importe quel dataset,
- réaliser une exploration automatique,
- nettoyer les données,
- gérer les valeurs manquantes,
- traiter les outliers,
- encoder les variables catégorielles,
- appliquer le scaling adapté,
- générer des features supplémentaires,
- sélectionner les meilleures caractéristiques,
- effectuer un train/test split,
- exporter un dataset prêt pour la modélisation.

Le pipeline est configurable via un fichier YAML et structuré de manière modulaire selon les bonnes pratiques d’ingénierie logicielle.

---

## Architecture du projet

```
project_preprocessing/
│
├── README.md
│
├── config/
│   └── settings.yaml
│
├── docs/
│   ├── preprocessing_plan.md
│   ├── EDA_checklist.md
│   ├── cleaning_checklist.md
│   ├── encoding_strategies.md
│   ├── scaling_strategies.md
│   ├── feature_engineering_plan.md
│   ├── feature_selection_plan.md
│   ├── train_test_split_rules.md
│   └── export_rules.md
│
├── preprocessing/
│   ├── __init__.py
│   ├── utils.py
│   ├── pipeline.py
│   │
│   ├── reader/
│   │   ├── file_loader.py
│   │   └── schema_inference.py
│   │
│   ├── analyzer/
│   │   ├── data_summary.py
│   │   ├── type_detector.py
│   │   ├── missing_values_analyzer.py
│   │   └── outlier_detector.py
│   │
│   ├── cleaner/
│   │   ├── missing_values_handler.py
│   │   ├── outlier_handler.py
│   │   ├── duplicates_handler.py
│   │   └── inconsistent_values_handler.py
│   │
│   ├── transformer/
│   │   ├── encoders.py
│   │   ├── scalers.py
│   │   ├── feature_generator.py
│   │   └── feature_selector.py
│   │
│   ├── visualizer/
│   │   ├── distribution_plots.py
│   │   ├── correlations_plots.py
│   │   └── missing_values_plots.py
│   │
│   └── validator/
│       └── data_validation.py
│
└── reports/
    └── (dossier destiné aux rapports générés)
```

---

## Fonctionnement général
Le pipeline suit les étapes suivantes :

1. Chargement des données (reader)
2. Inférence du schéma et détection des types
3. Analyse exploratoire automatique
4. Traitement des valeurs manquantes
5. Gestion des outliers
6. Nettoyage des incohérences et doublons
7. Encodage des variables catégorielles
8. Scaling des variables numériques
9. Feature engineering
10. Feature selection
11. Train/Test split
12. Export final des données et du pipeline

Toutes les décisions (méthodes, seuils, comportements automatiques) sont définies dans `config/settings.yaml`.

---

## Configuration
La configuration du pipeline est centralisée dans :

```
config/settings.yaml
```

Elle permet de contrôler :
- les stratégies de nettoyage,
- les méthodes d’imputation,
- les méthodes d’encodage,
- les règles de scaling,
- les stratégies de sélection de features,
- le pourcentage du test split,
- le comportement du pipeline lors de l’export.

---

## Conditions préalables
Bibliothèques nécessaires :
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib (pour les visualisations)
- seaborn (optionnel)
- PyYAML

---

## Auteurs
Projet réalisé dans le cadre d’un hackathon par trois membres :
- Abderrahim Sadegh
- Thomas Boucas
- Junior Fetmi

