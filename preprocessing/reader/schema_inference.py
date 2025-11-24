import pandas as pd
from preprocessing.utils import detect_id_columns, log


class SchemaInference:
    """
    Analyse le DataFrame et infère automatiquement le type de chaque colonne :
    - numerical
    - categorical
    - datetime
    - boolean
    - text
    - id
    """

    def __init__(self):
        pass

    def infer_types(self, df):
        """
        Retourne un dictionnaire {colonne: type_detecté}
        """

        log("Inférence automatique du schéma des données...")

        schema = {}

        # Détecter colonnes ID en premier
        id_cols = detect_id_columns(df)

        for col in df.columns:

            # Cas 1 : colonnes ID
            if col in id_cols:
                schema[col] = "id"
                continue

            series = df[col]

            # Cas 2 : booléens
            if series.dropna().nunique() == 2 and set(series.dropna().unique()).issubset({0, 1, True, False}):
                schema[col] = "boolean"
                continue

            # Cas 3 : datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                schema[col] = "datetime"
                continue
            else:
                # Essayer de convertir automatiquement en datetime
                try:
                    pd.to_datetime(series.dropna(), errors="raise")
                    schema[col] = "datetime"
                    continue
                except Exception:
                    pass

            # Cas 4 : numérique
            if pd.api.types.is_numeric_dtype(series):
                schema[col] = "numerical"
                continue

            # Cas 5 : texte (longueur moyenne > 30 caractères)
            if series.dropna().astype(str).str.len().mean() > 30:
                schema[col] = "text"
                continue

            # Cas 6 : catégorielle
            schema[col] = "categorical"

        log("Schéma détecté :")
        for col, t in schema.items():
            print(f" - {col}: {t}")

        return schema
