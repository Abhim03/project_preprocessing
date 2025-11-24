import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, mutual_info_classif
from sklearn.decomposition import PCA
from preprocessing.utils import log


class FeatureSelector:
    """
    Sélection automatique de features :
    - suppression colonnes constantes / quasi-constantes
    - suppression colonnes très corrélées
    - seuil de variance
    - mutual information
    - PCA (optionnel)
    """

    def __init__(self, config):
        self.config = config["feature_selection"]
        self.pca_model = None
        self.selected_features = None

    # ----------------------------------------------------------------------
    # 1. Constantes
    # ----------------------------------------------------------------------
    def _remove_constant(self, df):
        constants = [col for col in df.columns if df[col].nunique() <= 1]
        if constants:
            log(f"Colonnes constantes supprimées : {constants}")
            df = df.drop(columns=constants)
        return df

    # ----------------------------------------------------------------------
    # 2. VarianceThreshold
    # ----------------------------------------------------------------------
    def _apply_variance_threshold(self, df):
        threshold = self.config["variance_threshold"]
        if threshold <= 0:
            return df

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df)

        kept = df.columns[selector.get_support()]
        removed = [c for c in df.columns if c not in kept]

        if removed:
            log(f"Colonnes supprimées (variance faible) : {removed}")

        return df[kept]

    # ----------------------------------------------------------------------
    # 3. Corrélation forte
    # ----------------------------------------------------------------------
    def _remove_high_correlation(self, df):
        threshold = self.config["correlation_threshold"]

        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

        if to_drop:
            log(f"Colonnes supprimées (corrélation > {threshold}) : {to_drop}")
            df = df.drop(columns=to_drop)

        return df

    # ----------------------------------------------------------------------
    # 4. Mutual Information (numérique + cat)
    # ----------------------------------------------------------------------
    def _apply_mutual_information(self, df, y):
        try:
            if y.nunique() < 20:
                mi = mutual_info_classif(df, y, discrete_features="auto")
            else:
                mi = mutual_info_regression(df, y)

            mi_series = pd.Series(mi, index=df.columns)
            threshold = mi_series.quantile(0.05)  # garder 95% des features informatives

            selected = mi_series[mi_series >= threshold].index
            removed = mi_series[mi_series < threshold].index

            if len(removed) > 0:
                log(f"Colonnes supprimées par mutual information : {list(removed)}")

            return df[selected]
        except Exception:
            log("Mutual information ignorée (erreur ou données inadaptées).")
            return df

    # ----------------------------------------------------------------------
    # 5. PCA optionnelle
    # ----------------------------------------------------------------------
    def _apply_pca(self, df):
        if not self.config["apply_pca"]:
            return df

        variance_ratio = self.config["pca_variance_ratio"]

        pca = PCA(n_components=variance_ratio)
        transformed = pca.fit_transform(df)

        self.pca_model = pca

        cols = [f"PC{i+1}" for i in range(transformed.shape[1])]
        df_pca = pd.DataFrame(transformed, columns=cols, index=df.index)

        log(f"PCA appliquée : {transformed.shape[1]} composants conservés")

        return df_pca

    # ----------------------------------------------------------------------
    # MÉTHODE PRINCIPALE
    # ----------------------------------------------------------------------
    def apply(self, df, schema, y=None):
        log("Sélection automatique des features...")

        df_fs = df.copy()

        numeric_df = df_fs.select_dtypes(include=[np.number])

        # Step 1 : remove constant columns
        numeric_df = self._remove_constant(numeric_df)

        # Step 2 : variance threshold
        numeric_df = self._apply_variance_threshold(numeric_df)

        # Step 3 : high correlation
        if self.config["remove_correlated"]:
            numeric_df = self._remove_high_correlation(numeric_df)

        # Step 4 : mutual information
        if self.config["use_mutual_information"] and y is not None:
            numeric_df = self._apply_mutual_information(numeric_df, y)

        # Step 5 : PCA (optional)
        numeric_df = self._apply_pca(numeric_df)

        # Conserver les colonnes sélectionnées
        self.selected_features = numeric_df.columns.tolist()

        log("Sélection des features terminée.")
        return numeric_df

    # ----------------------------------------------------------------------
    # Pour transformer un test set
    # ----------------------------------------------------------------------
    def transform(self, df):
        if self.pca_model is not None:
            transformed = self.pca_model.transform(df.values)
            cols = [f"PC{i+1}" for i in range(transformed.shape[1])]
            return pd.DataFrame(transformed, columns=cols, index=df.index)

        # Sinon simplement garder les mêmes colonnes
        return df[self.selected_features]
