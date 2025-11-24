import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from preprocessing.utils import log


class Scalers:
    """
    Applique le scaling automatique des colonnes numériques.
    Méthodes possibles :
    - StandardScaler
    - MinMaxScaler
    - PowerTransformer (Yeo-Johnson)
    - Log transform (cas spécifiques)

    La méthode choisie dépend :
    - skewness de la colonne
    - paramètres dans settings.yaml
    """

    def __init__(self, config):
        self.config = config["scaling"]

        # stocker les scalers pour la transformation du test set
        self.scalers = {}

    # ----------------------------------------------------------------------
    # Sélection automatique selon la distribution
    # ----------------------------------------------------------------------
    def _select_method(self, skewness):
        if self.config["method"] != "auto":
            return self.config["method"]

        # Skewness forte -> PowerTransformer
        if abs(skewness) > 1:
            return "power"

        # Skewness légère -> MinMax
        if abs(skewness) > 0.5:
            return "minmax"

        return "standard"

    # ----------------------------------------------------------------------
    # Application scaling sur une colonne
    # ----------------------------------------------------------------------
    def _scale_column(self, df, col, method):
        series = df[col]

        if method == "log":
            df[col] = np.log1p(series.clip(lower=0))
            return None  # no scaler needed

        elif method == "standard":
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(series.values.reshape(-1, 1))
            return scaler

        elif method == "minmax":
            scaler = MinMaxScaler(feature_range=self.config["minmax_range"])
            df[col] = scaler.fit_transform(series.values.reshape(-1, 1))
            return scaler

        elif method == "power":
            scaler = PowerTransformer(method="yeo-johnson")
            df[col] = scaler.fit_transform(series.values.reshape(-1, 1))
            return scaler

        else:
            # fallback
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(series.values.reshape(-1, 1))
            return scaler

    # ----------------------------------------------------------------------
    # Méthode principale (train)
    # ----------------------------------------------------------------------
    def apply(self, df, schema):
        log("Début du scaling numérique...")

        df_scaled = df.copy()

        for col in df.columns:

            if schema[col] != "numerical":
                continue

            series = df[col].dropna()

            if series.nunique() <= 1:
                continue  # pas besoin de scaler

            skewness = float(series.skew())
            method = self._select_method(skewness)

            log(f"Colonne '{col}' -> method: {method} (skewness = {skewness:.3f})")

            scaler = self._scale_column(df_scaled, col, method)

            if scaler is not None:
                self.scalers[col] = scaler

        log("Scaling terminé.")
        return df_scaled

    # ----------------------------------------------------------------------
    # Méthode pour scaler test set
    # ----------------------------------------------------------------------
    def transform(self, df):
        log("Application du scaling sur test set...")

        df_scaled = df.copy()

        for col, scaler in self.scalers.items():
            if col in df_scaled:
                df_scaled[col] = scaler.transform(df_scaled[col].values.reshape(-1, 1))

        log("Scaling test set terminé.")
        return df_scaled
