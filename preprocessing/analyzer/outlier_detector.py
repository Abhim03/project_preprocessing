import numpy as np
from sklearn.ensemble import IsolationForest
from preprocessing.utils import log


class OutlierDetector:
    """
    Détecte les valeurs aberrantes (outliers) dans les colonnes numériques :
    - IQR
    - Z-score
    - Isolation Forest (pour distributions complexes)

    Ce module fournit uniquement l'analyse :
    - ratio d'outliers
    - méthode la plus adaptée
    """

    def __init__(self, config):
        self.config = config["outliers"]

    def detect_iqr(self, series):
        """Détection par IQR (Interquartile Range)."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - self.config["iqr_multiplier"] * iqr
        upper_bound = q3 + self.config["iqr_multiplier"] * iqr
        mask = (series < lower_bound) | (series > upper_bound)
        return mask.sum(), mask

    def detect_zscore(self, series):
        """Détection par Z-score."""
        mean = series.mean()
        std = series.std()
        if std == 0:
            return 0, series != series  # aucun outlier
        zscores = (series - mean) / std
        mask = np.abs(zscores) > self.config["zscore_threshold"]
        return mask.sum(), mask

    def detect_isolation_forest(self, series):
        """Détection par Isolation Forest."""
        model = IsolationForest(
            contamination=self.config["isolation_forest_contamination"],
            random_state=42
        )
        preds = model.fit_predict(series.values.reshape(-1, 1))
        mask = preds == -1
        return mask.sum(), mask

    def analyze(self, df, schema):
        """
        Analyse complète de toutes les colonnes numériques.
        Retourne un dictionnaire :
        {
            col: {
                "iqr_outliers": int,
                "zscore_outliers": int,
                "isoforest_outliers": int,
                "best_method": "iqr" | "zscore" | "isoforest" | "none",
                "outlier_ratio": float
            }
        }
        """

        log("Détection des outliers...")

        results = {}

        for col in df.columns:

            if schema[col] != "numerical":
                continue

            series = df[col].dropna()

            # Ignore colonnes sans variabilité
            if series.nunique() <= 1:
                results[col] = {
                    "iqr_outliers": 0,
                    "zscore_outliers": 0,
                    "isoforest_outliers": 0,
                    "best_method": "none",
                    "outlier_ratio": 0.0,
                }
                continue

            # Détections
            iqr_count, _ = self.detect_iqr(series)
            zscore_count, _ = self.detect_zscore(series)
            iso_count, _ = self.detect_isolation_forest(series)

            # Sélection meilleure méthode
            counts = {
                "iqr": iqr_count,
                "zscore": zscore_count,
                "isoforest": iso_count,
            }

            # Meilleure méthode = celle qui détecte le plus d'outliers cohérents
            best_method = max(counts, key=counts.get)

            outlier_ratio = counts[best_method] / len(series)

            results[col] = {
                "iqr_outliers": int(iqr_count),
                "zscore_outliers": int(zscore_count),
                "isoforest_outliers": int(iso_count),
                "best_method": best_method,
                "outlier_ratio": float(outlier_ratio),
            }

        log("Analyse des outliers terminée.")

        return results
