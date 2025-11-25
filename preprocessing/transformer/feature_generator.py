import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from preprocessing.utils import log


class FeatureGenerator:
    """
    Feature Engineering avancé :
    - Extraction temporelle enrichie
    - Regroupement intelligent des catégories rares
    - Log transform, power transform
    - Binning intelligent
    - Interactions avancées : mult, ratio, diff, sum
    - Polynomial features
    - Crossed categorical features
    - Suppression colonnes quasi-constantes
    """

    def __init__(self, config):
        self.config = config["feature_engineering"]

    # ----------------------------------------------------------------------
    # 0. Détection automatique des colonnes quasi-constantes
    # ----------------------------------------------------------------------
    def _remove_near_constant(self, df):
        threshold = self.config.get("near_constant_threshold", 0.995)
        to_drop = []

        for col in df.columns:
            top_freq = df[col].value_counts(normalize=True, dropna=False).max()
            if top_freq > threshold:
                to_drop.append(col)

        if to_drop:
            log(f"Colonnes quasi-constantes supprimées : {to_drop}")
            df = df.drop(columns=to_drop)

        return df

    # ----------------------------------------------------------------------
    # 1. Features temporelles enrichies
    # ----------------------------------------------------------------------
    def _generate_datetime_features(self, df, col):
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_weekday"] = df[col].dt.weekday
        df[f"{col}_quarter"] = df[col].dt.quarter
        df[f"{col}_weekofyear"] = df[col].dt.isocalendar().week.astype(int)

        df[f"{col}_is_month_start"] = df[col].dt.is_month_start.astype(int)
        df[f"{col}_is_month_end"] = df[col].dt.is_month_end.astype(int)
        df[f"{col}_is_quarter_start"] = df[col].dt.is_quarter_start.astype(int)
        df[f"{col}_is_quarter_end"] = df[col].dt.is_quarter_end.astype(int)

        # Saison simple
        df[f"{col}_season"] = df[col].dt.month % 12 // 3

        # Weekend
        df[f"{col}_is_weekend"] = (df[col].dt.weekday >= 5).astype(int)

        return df

    # ----------------------------------------------------------------------
    # 2. Rare category grouping intelligent
    # ----------------------------------------------------------------------
    def _rare_category_grouping(self, df, col):
        threshold = self.config["rare_category_threshold"]
        freq = df[col].value_counts(normalize=True)

        rare = freq[freq < threshold].index
        if len(rare) > 0:
            df[col] = df[col].replace(rare, "OTHER")

        return df

    # ----------------------------------------------------------------------
    # 3. Log transform
    # ----------------------------------------------------------------------
    def _log_transform(self, df, col):
        if (df[col] > 0).all():
            df[f"{col}_log"] = np.log1p(df[col])
        return df

    # ----------------------------------------------------------------------
    # 4. Power transform optionnel
    # ----------------------------------------------------------------------
    def _power_transform(self, df, col):
        if not self.config.get("enable_power_transform", False):
            return df

        try:
            df[f"{col}_power"] = np.power(df[col], 0.5)
        except Exception:
            pass
        return df

    # ----------------------------------------------------------------------
    # 5. Binning avancé
    # ----------------------------------------------------------------------
    def _binning(self, df, col):
        # qcut
        try:
            df[f"{col}_bin_q"] = pd.qcut(df[col], q=4, duplicates="drop")
        except Exception:
            pass

        # cut uniforme
        try:
            df[f"{col}_bin_cut"] = pd.cut(df[col], bins=4)
        except Exception:
            pass

        return df

    # ----------------------------------------------------------------------
    # 6. Crossed categorical features
    # ----------------------------------------------------------------------
    def _cross_categorical(self, df, categorical_cols):
        if not self.config["generate_cross_categorical"]:
            return df

        for i in range(len(categorical_cols)):
            for j in range(i + 1, len(categorical_cols)):
                c1, c2 = categorical_cols[i], categorical_cols[j]
                df[f"{c1}_x_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)

        return df

    # ----------------------------------------------------------------------
    # 7. Interactions avancées
    # ----------------------------------------------------------------------
    def _generate_interactions(self, df, numerical_cols):
        if not self.config["generate_interactions"]:
            return df

        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                c1, c2 = numerical_cols[i], numerical_cols[j]

                df[f"{c1}_mul_{c2}"] = df[c1] * df[c2]
                df[f"{c1}_ratio_{c2}"] = df[c1] / (df[c2] + 1e-6)
                df[f"{c1}_diff_{c2}"] = df[c1] - df[c2]
                df[f"{c1}_sum_{c2}"] = df[c1] + df[c2]

        return df

    # ----------------------------------------------------------------------
    # 8. Polynomial features
    # ----------------------------------------------------------------------
    def _polynomial_features(self, df, numerical_cols):
        if not self.config["generate_polynomials"]:
            return df

        try:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            transformed = poly.fit_transform(df[numerical_cols])

            new_cols = poly.get_feature_names_out(numerical_cols)
            df_poly = pd.DataFrame(transformed, columns=new_cols, index=df.index)

            return pd.concat([df, df_poly], axis=1)
        except Exception:
            return df

    # ----------------------------------------------------------------------
    # MÉTHODE PRINCIPALE
    # ----------------------------------------------------------------------
    def apply(self, df, schema):
        log("=== Début Feature Engineering avancé ===")

        df_gen = df.copy()

        numerical_cols = [c for c, t in schema.items() if t == "numerical"]
        categorical_cols = [c for c, t in schema.items() if t == "categorical"]
        datetime_cols = [c for c, t in schema.items() if t == "datetime"]

        # Nettoyage colonnes quasi constantes
        if self.config["remove_near_constant"]:
            df_gen = self._remove_near_constant(df_gen)

        # ------------------- Features temporelles -----------------------
        if self.config["generate_datetime_features"]:
            for col in datetime_cols:
                df_gen = self._generate_datetime_features(df_gen, col)

        # ------------------- Rare categories ----------------------------
        for col in categorical_cols:
            df_gen = self._rare_category_grouping(df_gen, col)

        # ------------------- Numerical transforms -----------------------
        for col in numerical_cols:
            if self.config["enable_log_transform"]:
                df_gen = self._log_transform(df_gen, col)

            df_gen = self._power_transform(df_gen, col)

            if self.config["enable_binning"]:
                df_gen = self._binning(df_gen, col)

        # ------------------- Cross categorical --------------------------
        df_gen = self._cross_categorical(df_gen, categorical_cols)

        # ------------------- Interactions -------------------------------
        df_gen = self._generate_interactions(df_gen, numerical_cols)

        # ------------------- Polynomial features ------------------------
        df_gen = self._polynomial_features(df_gen, numerical_cols)

        log("=== Feature Engineering terminé ===")
        return df_gen
