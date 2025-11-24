import pandas as pd
from sklearn.impute import KNNImputer
from preprocessing.utils import log


class MissingValuesHandler:
    """
    Impute les valeurs manquantes dans le dataset.
    
    Utilise les recommandations de missing_values_analyzer :
    - mean / median / KNN (numérique)
    - most_frequent / 'missing' (catégoriel)
    - interpolation (datetime)
    - suppression de colonnes si nécessaire
    """

    def __init__(self, config):
        self.config = config["missing_values"]

    # -------------------------------------------------------------------------
    # Méthodes d'imputation numérique
    # -------------------------------------------------------------------------
    def _impute_numerical(self, df, col, strategy):
        series = df[col]

        if strategy == "mean":
            df[col] = series.fillna(series.mean())

        elif strategy == "median":
            df[col] = series.fillna(series.median())

        elif strategy == "knn" or strategy == "auto":
            # Auto = KNN si colonne complexe ou beaucoup de NaN
            imputer = KNNImputer(n_neighbors=self.config["knn_neighbors"])
            df[[col]] = imputer.fit_transform(df[[col]])

        else:
            df[col] = series.fillna(series.mean())

    # -------------------------------------------------------------------------
    # Méthodes d'imputation catégorielle
    # -------------------------------------------------------------------------
    def _impute_categorical(self, df, col, strategy):
        series = df[col]

        if strategy == "most_frequent":
            df[col] = series.fillna(series.mode().iloc[0])
        elif strategy == "missing" or strategy == "auto":
            df[col] = series.fillna("missing")
        else:
            df[col] = series.fillna("missing")

    # -------------------------------------------------------------------------
    # Méthodes d'imputation datetime
    # -------------------------------------------------------------------------
    def _impute_datetime(self, df, col, strategy):
        series = df[col]

        if strategy == "interpolate" or strategy == "auto":
            df[col] = series.interpolate(method="time")
        else:
            df[col] = series.fillna(series.median())

    # -------------------------------------------------------------------------
    # Méthode principale
    # -------------------------------------------------------------------------
    def apply(self, df, schema, missing_report):
        """
        df : DataFrame
        schema : dictionnaire des types
        missing_report : sortie de MissingValuesAnalyzer.analyze()

        Retour :
        df imputé
        """

        log("Début de l'imputation des valeurs manquantes...")

        df_clean = df.copy()

        for col in df_clean.columns:

            info = missing_report[col]
            action = info["recommended_action"]

            # -----------------------------------------------------------------
            # 1. Suppression colonne (seulement si très mauvaise)
            # -----------------------------------------------------------------
            if action == "drop":
                log(f"Colonne supprimée (trop de NaN) : {col}")
                df_clean.drop(columns=[col], inplace=True)
                continue

            # Sinon → imputation
            strategy = info["recommended_method"]
            col_type = schema[col]

            # -----------------------------------------------------------------
            # 2. Imputation selon type
            # -----------------------------------------------------------------
            if col_type == "numerical":
                self._impute_numerical(df_clean, col, strategy)

            elif col_type == "categorical":
                self._impute_categorical(df_clean, col, strategy)

            elif col_type == "datetime":
                self._impute_datetime(df_clean, col, strategy)

            else:
                # type non géré
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])

        log("Imputation terminée.")

        return df_clean
