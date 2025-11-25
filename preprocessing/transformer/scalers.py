import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
)
from preprocessing.utils import log


class Scalers:
    """
    Scaling intelligent basé sur :
    - skewness
    - présence d'outliers
    - variance / range
    - paramètres settings.yaml

    Méthodes possibles :
    - standard
    - minmax
    - robust
    - power (yeo-johnson)
    - log (cas skewness extrême)

    La logique auto sélectionne le meilleur scaler pour chaque feature.
    """

    def __init__(self, config):
        self.config = config["scaling"]

        # Pour réappliquer les bons scalers sur test
        self.scalers = {}

    # ----------------------------------------------------------------------
    # UTILITAIRES INTELLIGENTS
    # ----------------------------------------------------------------------

    def _has_outliers(self, series):
        """Détecte les outliers via règle IQR."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        return outliers > 0

    def _is_skewed(self, skewness):
        """Détecte si la variable est très skewed."""
        return abs(skewness) > 1.0

    # LOGIQUE DE CHOIX
    def _choose_method(self, series):
        if self.config["method"] != "auto":
            return self.config["method"]

        skew = float(series.skew())
        outliers = self._has_outliers(series)

        # Cas extrême → log transform
        if abs(skew) > 2.5:
            return "log"

        # Skewness importante → power transformer
        if self._is_skewed(skew):
            return "power"

        # Outliers → RobustScaler
        if outliers:
            return "robust"

        # Si plage très différente (minmax recommandé)
        if series.min() < 0 or series.max() > 1e3:
            return "minmax"

        # Default
        return "standard"

    # ----------------------------------------------------------------------
    # APPLY SUR TRAIN
    # ----------------------------------------------------------------------

    def _scale_column(self, df, col, method):
        values = df[col].astype(float)

        if method == "log":
            # log(1+x) pour positivité
            df[col] = np.log1p(values.clip(lower=0))
            return None

        elif method == "standard":
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(values.to_numpy().reshape(-1, 1))
            return scaler

        elif method == "minmax":
            scaler = MinMaxScaler(
                feature_range=self.config.get("minmax_range", (0, 1))
            )
            df[col] = scaler.fit_transform(values.to_numpy().reshape(-1, 1))
            return scaler

        elif method == "robust":
            scaler = RobustScaler()
            df[col] = scaler.fit_transform(values.to_numpy().reshape(-1, 1))
            return scaler

        elif method == "power":
            scaler = PowerTransformer(method="yeo-johnson")
            df[col] = scaler.fit_transform(values.to_numpy().reshape(-1, 1))
            return scaler

        # fallback
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(values.to_numpy().reshape(-1, 1))
        return scaler

    def apply(self, df, schema):
        log("=== Début du scaling intelligent ===")

        df_scaled = df.copy()

        for col in df.columns:
            if schema[col] != "numerical":
                continue

            series = df[col].dropna()
            if series.nunique() <= 1:
                continue

            method = self._choose_method(series)

            log(
                f"Colonne '{col}' | skew={series.skew():.3f} | "
                f"method={method}"
            )

            scaler = self._scale_column(df_scaled, col, method)

            if scaler is not None:
                self.scalers[col] = scaler

        log("=== Scaling terminé ===")
        return df_scaled

    # ----------------------------------------------------------------------
    # TRANSFORM SUR TEST
    # ----------------------------------------------------------------------

    def transform(self, df):
        log("Application scaling sur test set...")

        df_scaled = df.copy()

        for col, scaler in self.scalers.items():
            if col in df_scaled:
                df_scaled[col] = scaler.transform(
                    df_scaled[col].values.reshape(-1, 1)
                )

        log("Scaling test set terminé.")
        return df_scaled
