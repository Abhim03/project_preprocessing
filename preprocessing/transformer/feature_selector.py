import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_regression,
    mutual_info_classif,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from preprocessing.utils import log


class FeatureSelector:
    """
    Sélection avancée de features :
    - constantes et quasi-constantes
    - colonnes dupliquées
    - variance threshold
    - corrélation forte intelligente
    - mutual information améliorée
    - sélection via modèles (RF)
    - PCA optionnelle
    """

    def __init__(self, config):
        self.config = config["feature_selection"]

        self.selected_features = None
        self.pca_model = None
        self.model_selector = None

    # -----------------------------------------------------------
    # 1. Colonnes constantes
    # -----------------------------------------------------------
    def _remove_constant(self, df):
        constants = [col for col in df if df[col].nunique() <= 1]
        if constants:
            log(f"Colonnes constantes supprimées : {constants}")
            df = df.drop(columns=constants)
        return df

    # -----------------------------------------------------------
    # 2. Colonnes quasi-constantes
    # -----------------------------------------------------------
    def _remove_near_constant(self, df):
        threshold = self.config.get("near_constant_threshold", 0.995)
        to_drop = []
        for col in df.columns:
            freq = df[col].value_counts(normalize=True, dropna=False).max()
            if freq >= threshold:
                to_drop.append(col)

        if to_drop:
            log(f"Colonnes quasi-constantes supprimées : {to_drop}")
            df = df.drop(columns=to_drop)

        return df

    # -----------------------------------------------------------
    # 3. Colonnes dupliquées
    # -----------------------------------------------------------
    def _remove_duplicates(self, df):
        duplicates = []
        seen = {}
        for col in df.columns:
            data = tuple(df[col].fillna(-999).values)
            if data in seen:
                duplicates.append(col)
            else:
                seen[data] = col

        if duplicates:
            log(f"Colonnes dupliquées supprimées : {duplicates}")
            df = df.drop(columns=duplicates)

        return df

    # -----------------------------------------------------------
    # 4. Variance threshold
    # -----------------------------------------------------------
    def _apply_variance_threshold(self, df):
        threshold = self.config["variance_threshold"]
        if threshold <= 0:
            return df

        selector = VarianceThreshold(threshold)
        selector.fit(df)

        kept = df.columns[selector.get_support()]
        drop = [c for c in df.columns if c not in kept]

        if drop:
            log(f"Variance faible -> supprimées : {drop}")

        return df[kept]

    # -----------------------------------------------------------
    # 5. Correlation forte
    # -----------------------------------------------------------
    def _remove_high_correlation(self, df):
        thr = self.config["correlation_threshold"]

        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))

        drop = []
        for col in upper.columns:
            if any(upper[col] > thr):
                drop.append(col)

        if drop:
            log(f"Colonnes supprimées (corrélation > {thr}) : {drop}")
            df = df.drop(columns=drop)

        return df

    # -----------------------------------------------------------
    # 6. Mutual Information améliorée
    # -----------------------------------------------------------
    def _apply_mutual_information(self, df, y):
        try:
            if y.nunique() < 20:
                mi = mutual_info_classif(df, y, discrete_features="auto")
            else:
                mi = mutual_info_regression(df, y)

            mi = pd.Series(mi, index=df.columns)

            threshold = mi.quantile(self.config["mi_quantile"])
            kept = mi[mi >= threshold].index
            drop = mi[mi < threshold].index

            log(f"MI -> gardés : {len(kept)}, supprimés : {list(drop)}")

            return df[kept]

        except Exception as e:
            log(f"MI ignorée (erreur={e})")
            return df

    # -----------------------------------------------------------
    # 7. Sélection via modèles
    # -----------------------------------------------------------
    def _model_based_selection(self, df, y):
        if not self.config.get("use_model_selection", False):
            return df

        log("Sélection par importance modèle…")

        # On utilise RandomForest pour obtenir importances
        if y.nunique() < 20:
            model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)

        model.fit(df, y)
        importances = pd.Series(model.feature_importances_, index=df.columns)

        thr = importances.quantile(self.config["model_importance_quantile"])

        kept = importances[importances >= thr].index
        drop = importances[importances < thr].index

        log(f"Model-selection -> gardés : {len(kept)}, supprimés : {list(drop[:10])}...")

        self.model_selector = model
        return df[kept]

    # -----------------------------------------------------------
    # 8. PCA
    # -----------------------------------------------------------
    def _apply_pca(self, df):
        if not self.config["apply_pca"]:
            return df

        ratio = self.config["pca_variance_ratio"]

        try:
            pca = PCA(n_components=ratio)
            transformed = pca.fit_transform(df)

            cols = [f"PC{i+1}" for i in range(transformed.shape[1])]
            df_pca = pd.DataFrame(transformed, index=df.index, columns=cols)

            self.pca_model = pca

            log(f"PCA appliquée -> {df_pca.shape[1]} composants")

            return df_pca
        except Exception as e:
            log(f"PCA ignorée (erreur={e})")
            return df

    # -----------------------------------------------------------
    # MAIN METHOD
    # -----------------------------------------------------------
    def apply(self, df, schema, y=None):
        log("=== Sélection des features ===")

        df_sel = df.copy()

        # On ne garde que les colonnes numériques
        df_num = df_sel.select_dtypes(include=[np.number])

        # 1. Constantes
        df_num = self._remove_constant(df_num)

        # 2. Quasi-constantes
        if self.config["remove_near_constant"]:
            df_num = self._remove_near_constant(df_num)

        # 3. Dupliquées
        if self.config["remove_duplicates"]:
            df_num = self._remove_duplicates(df_num)

        # 4. Variance
        df_num = self._apply_variance_threshold(df_num)

        # 5. Corrélation
        if self.config["remove_correlated"]:
            df_num = self._remove_high_correlation(df_num)

        # 6. Mutual Information
        if self.config["use_mutual_information"] and y is not None:
            df_num = self._apply_mutual_information(df_num, y)

        # 7. Model-based selection
        if self.config["use_model_selection"] and y is not None:
            df_num = self._model_based_selection(df_num, y)

        # 8. PCA
        df_num = self._apply_pca(df_num)

        # Sauvegarde des features
        self.selected_features = df_num.columns.tolist()

        log("=== Sélection terminée ===")
        return df_num

    # -----------------------------------------------------------
    # Transform test set
    # -----------------------------------------------------------
    def transform(self, df):
        # PCA
        if self.pca_model is not None:
            transformed = self.pca_model.transform(df.values)
            cols = [f"PC{i+1}" for i in range(transformed.shape[1])]
            return pd.DataFrame(transformed, index=df.index, columns=cols)

        # Sinon simplement garder les colonnes sélectionnées
        return df[self.selected_features]
