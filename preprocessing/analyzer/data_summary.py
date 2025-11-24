import pandas as pd
from preprocessing.utils import log


class DataSummary:
    """
    Génère un résumé structuré du DataFrame :
    - stats descriptives
    - comptage des valeurs uniques
    - colonnes quasi constantes
    - cardinalité
    - valeurs manquantes
    """

    def __init__(self):
        pass

    def summarize(self, df, schema):
        """
        df : pandas DataFrame
        schema : dict (type détecté pour chaque colonne)

        Retour :
        summary : dict contenant toutes les informations utiles
        """

        log("Création du résumé global du dataset...")

        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes_inferred": schema,
            "missing_values": {},
            "unique_counts": {},
            "quasi_constant_columns": [],
            "high_cardinality_columns": [],
            "stats_numerical": {},
        }

        for col in df.columns:

            series = df[col]
            col_type = schema[col]

            # -------------------------------------------------
            # Valeurs manquantes
            # -------------------------------------------------
            missing_ratio = series.isna().mean()
            summary["missing_values"][col] = float(missing_ratio)

            # -------------------------------------------------
            # Cardinalité
            # -------------------------------------------------
            unique_count = series.nunique()
            summary["unique_counts"][col] = int(unique_count)

            # Colonne quasi constante
            if unique_count == 1:
                summary["quasi_constant_columns"].append(col)

            # Haute cardinalité (configurable dans settings or > 100 by convention)
            if unique_count > 100 and col_type == "categorical":
                summary["high_cardinality_columns"].append(col)

            # -------------------------------------------------
            # Statistiques descriptives numériques
            # -------------------------------------------------
            if col_type == "numerical":
                try:
                    summary["stats_numerical"][col] = {
                        "mean": float(series.mean()),
                        "median": float(series.median()),
                        "std": float(series.std()),
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "skewness": float(series.skew()),
                        "kurtosis": float(series.kurtosis())
                    }
                except Exception:
                    summary["stats_numerical"][col] = "non-calculable"

        log("Résumé du dataset généré avec succès.")

        return summary
