import pandas as pd
import numpy as np
from preprocessing.utils import log


class FeatureGenerator:
    """
    Génère automatiquement des features supplémentaires :
    - Variables temporelles
    - Log-transform
    - Binning
    - Regroupement de catégories rares
    - Interactions optionnelles
    """

    def __init__(self, config):
        self.config = config["feature_engineering"]

    # ----------------------------------------------------------------------
    # 1. Features temporelles
    # ----------------------------------------------------------------------
    def _generate_datetime_features(self, df, col):
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_weekday"] = df[col].dt.weekday
        df[f"{col}_quarter"] = df[col].dt.quarter

        # Saison
        df[f"{col}_season"] = df[col].dt.month % 12 // 3

        return df

    # ----------------------------------------------------------------------
    # 2. Regroupement catégories rares
    # ----------------------------------------------------------------------
    def _rare_category_grouping(self, df, col):
        threshold = self.config["rare_category_threshold"]
        freq = df[col].value_counts(normalize=True)
        rare_categories = freq[freq < threshold].index
        df[col] = df[col].replace(rare_categories, "OTHER")
        return df

    # ----------------------------------------------------------------------
    # 3. Log transform (si toutes les valeurs positives)
    # ----------------------------------------------------------------------
    def _log_transform(self, df, col):
        if (df[col] > 0).all():
            df[f"{col}_log"] = np.log1p(df[col])
        return df

    # ----------------------------------------------------------------------
    # 4. Binning automatique
    # ----------------------------------------------------------------------
    def _binning(self, df, col):
        try:
            df[f"{col}_bin"] = pd.qcut(df[col], q=4, duplicates="drop")
        except Exception:
            pass
        return df

    # ----------------------------------------------------------------------
    # 5. Interactions (optionnel)
    # ----------------------------------------------------------------------
    def _generate_interactions(self, df, numerical_cols):
        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                c1 = numerical_cols[i]
                c2 = numerical_cols[j]
                new_col = f"{c1}_x_{c2}"
                df[new_col] = df[c1] * df[c2]
        return df

    # ----------------------------------------------------------------------
    # MÉTHODE PRINCIPALE
    # ----------------------------------------------------------------------
    def apply(self, df, schema):
        log("Génération automatique de nouvelles features...")

        df_gen = df.copy()

        numerical_cols = [col for col, t in schema.items() if t == "numerical"]
        categorical_cols = [col for col, t in schema.items() if t == "categorical"]
        datetime_cols = [col for col, t in schema.items() if t == "datetime"]

        # ------------------------------------------------------------------
        # Features temporelles
        # ------------------------------------------------------------------
        if self.config["generate_datetime_features"]:
            for col in datetime_cols:
                df_gen = self._generate_datetime_features(df_gen, col)
                log(f"Features temporelles générées pour {col}")

        # ------------------------------------------------------------------
        # Regroupement catégories rares
        # ------------------------------------------------------------------
        for col in categorical_cols:
            df_gen = self._rare_category_grouping(df_gen, col)

        # ------------------------------------------------------------------
        # Log transform
        # ------------------------------------------------------------------
        for col in numerical_cols:
            df_gen = self._log_transform(df_gen, col)

        # ------------------------------------------------------------------
        # Binning
        # ------------------------------------------------------------------
        for col in numerical_cols:
            df_gen = self._binning(df_gen, col)

        # ------------------------------------------------------------------
        # Interactions
        # ------------------------------------------------------------------
        if self.config["generate_interactions"]:
            df_gen = self._generate_interactions(df_gen, numerical_cols)
            log("Interactions générées entre variables numériques.")

        log("Feature engineering terminé.")

        return df_gen
