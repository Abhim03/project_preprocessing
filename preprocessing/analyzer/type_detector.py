import pandas as pd
from preprocessing.utils import log


class TypeDetector:
    """
    Valide, corrige et améliore l'inférence des types détectés dans schema_inference.py.
    - Vérifie si les types détectés sont cohérents
    - Corrige les colonnes mal typées
    - Convertit explicitement les colonnes vers les bons types
    """

    def __init__(self):
        pass

    def refine_types(self, df, initial_schema):
        """
        Prend en entrée :
        - df : le DataFrame brut
        - initial_schema : dict {col: type_detecté}

        Retourne :
        - refined_schema : dict amélioré
        """

        log("Affinement du schéma détecté...")

        refined_schema = {}

        for col, col_type in initial_schema.items():

            series = df[col]

            # 1. Vérifier si la colonne datetime est vraiment convertible
            if col_type == "datetime":
                try:
                    df[col] = pd.to_datetime(series, errors="raise")
                    refined_schema[col] = "datetime"
                    continue
                except Exception:
                    # Si échec → la classe est probablement catégorielle
                    refined_schema[col] = "categorical"
                    continue

            # 2. Vérifier si une colonne catégorielle a trop de valeurs uniques
            if col_type == "categorical":
                nunique = series.nunique()

                if nunique == len(series):
                    # colonne identifiant masquée
                    refined_schema[col] = "id"
                    continue

                if nunique > 100:
                    # probable texte libre
                    refined_schema[col] = "text"
                    continue

                refined_schema[col] = "categorical"
                continue

            # 3. Vérification numérique
            if col_type == "numerical":
                # si numérique mais contient beaucoup de texte → mauvaise classification
                non_num = series.apply(lambda x: isinstance(x, str)).sum()
                if non_num > 0:
                    refined_schema[col] = "categorical"
                else:
                    refined_schema[col] = "numerical"
                continue

            # 4. Booléens
            if col_type == "boolean":
                refined_schema[col] = "boolean"
                continue

            # 5. ID
            if col_type == "id":
                refined_schema[col] = "id"
                continue

            # 6. Texte
            if col_type == "text":
                refined_schema[col] = "text"
                continue

            # 7. Par défaut
            refined_schema[col] = col_type

        # Afficher le schéma final
        log("Schéma final après amélioration :")
        for col, t in refined_schema.items():
            print(f" - {col}: {t}")

        return refined_schema
