# Stratégies d’encodage

## 1. Détection du type de variable
- Nominale
- Ordinale
- Cardinalité faible/moyenne/haute

## 2. Règles d'encodage
### Faible cardinalité (< 10)
One Hot Encoding

### Moyenne cardinalité (10–50)
Target Encoding avec protection contre le leakage

### Haute cardinalité (> 50)
Hashing Encoder

### Variables ordinales
OrdinalEncoder basé sur :
- fréquence
- moyenne de la target si disponible

## 3. Règles supplémentaires
- Gestion des catégories inconnues
- Conservation d’un mapping pour les transformations inverses
