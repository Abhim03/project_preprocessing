# Règles de Train/Test Split

## 1. Paramètres
- test_size : 0.2
- shuffle : true
- stratify : auto

## 2. Règles
- stratify activé si target catégorielle
- split uniquement après preprocessing complet
- export :
  - X_train.csv
  - X_test.csv
  - y_train.csv
  - y_test.csv
