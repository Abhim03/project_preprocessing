import pandas as pd
from preprocessing.utils import log


class MissingValuesAnalyzer:
    """
    Analyse avancée des valeurs manquantes dans le dataset.
    Objectifs :
    - calcul du pourcentage de NaN par colonne
    - classification des colonnes : no_missing, low_missing, medium_missing, high_missing
    - détection des colonnes candidates à la suppression
    - pré-choix de stratégies d'imputation selon le type et le taux de NaN
    """

    def __init__(self, config):
        self.config = config["missing_values"]

    def analyze(self, df, schema):
        """
        Analyse les NaN pour chaque colonne.
        Retourne un dictionnaire structuré :
        {
            colonne: {
                "missing_ratio": float,
                "category": "none" | "low" | "medium" | "high",
                "recommended_action": "keep" | "impute" | "drop",
                "recommended_method": "mean" | "median" | "most_frequent" | "missing" | None
            }
        }
        """

        log("Analyse des valeurs manquantes...")

        drop_threshold = self.config["drop_threshold"]
        results = {}

        for col in df.columns:
            series = df[col]
            col_type = schema[col]

            # Pourcentage de valeurs manquantes
            missing_ratio = series.isna().mean()

            # Classification
            if missing_ratio == 0:
                category = "none"
            elif missing_ratio < 0.05:
                category = "low"
            elif missing_ratio < 0.25:
                category = "medium"
            else:
                category = "high"

            # Recommandation : supprimer la colonne ?
            if missing_ratio >= drop_threshold:
                recommended_action = "drop"
                recommended_method = None
            else:
                recommended_action = "impute"

                # Recommandation de méthode selon le type
                if col_type == "numerical":
                    method = self.config["strategy_numerical"]
                elif col_type == "categorical":
                    method = self.config["strategy_categorical"]
                elif col_type == "datetime":
                    method = self.config["strategy_datetime"]
                else:
                    method = None

                recommended_method = method

            # Stockage des résultats
            results[col] = {
                "missing_ratio": float(missing_ratio),
                "category": category,
                "recommended_action": recommended_action,
                "recommended_method": recommended_method,
            }

        log("Analyse des NaN terminée.")

        return results
