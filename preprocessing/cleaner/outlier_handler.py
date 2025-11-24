import numpy as np
from preprocessing.utils import log


class OutlierHandler:
    """
    Traite les outliers selon les recommandations de OutlierDetector.
    
    Méthodes prises en charge :
    - clip         : limitation au min/max autorisé
    - winsorize    : réduction plus douce
    - remove       : suppression des lignes concernées

    La méthode utilisée est définie dans settings.yaml :
    outliers.action = "clip" | "winsorize" | "remove"
    """

    def __init__(self, config):
        self.config = config["outliers"]
        self.action = self.config["action"]

    # ---------------------------------------------------------------------
    # Définition des bornes pour IQR
    # ---------------------------------------------------------------------
    def _iqr_bounds(self, series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.config["iqr_multiplier"] * iqr
        upper = q3 + self.config["iqr_multiplier"] * iqr
        return lower, upper

    # ---------------------------------------------------------------------
    # Application réelle du traitement
    # ---------------------------------------------------------------------
    def _apply_action(self, df, col, mask):
        """
        Applique la méthode choisie (clip, winsorize ou remove)
        sur la colonne 'col' selon le masque booléen 'mask'.
        """

        if self.action == "remove":
            before = len(df)
            df_clean = df.loc[~mask].copy()
            removed = before - len(df_clean)
            log(f"Lignes supprimées pour outliers ({col}) : {removed}")
            return df_clean

        elif self.action == "clip":
            # Limiter les valeurs aux bornes valides
            lower, upper = self._iqr_bounds(df[col])
            df[col] = np.clip(df[col], lower, upper)
            log(f"Clip appliqué sur la colonne : {col}")
            return df

        elif self.action == "winsorize":
            # Winsorizing à 5% / 95%
            lower = df[col].quantile(0.05)
            upper = df[col].quantile(0.95)
            df[col] = np.clip(df[col], lower, upper)
            log(f"Winsorizing appliqué sur la colonne : {col}")
            return df

        else:
            log("Aucune action valide définie. Aucun traitement appliqué.")
            return df

    # ---------------------------------------------------------------------
    # Méthode principale
    # ---------------------------------------------------------------------
    def apply(self, df, schema, outlier_report):
        """
        Paramètres :
        - df : DataFrame
        - schema : types des colonnes
        - outlier_report : sortie du OutlierDetector

        Retour :
        - df nettoyé des outliers
        """

        if not self.config["enabled"]:
            log("Traitement des outliers désactivé dans settings.yaml.")
            return df

        log("Début du traitement des outliers...")

        df_clean = df.copy()

        for col in df_clean.columns:

            if col not in outlier_report:
                continue  # colonne non numérique ou aucun risque

            series = df_clean[col]
            method = outlier_report[col]["best_method"]

            # Recalcul du masque d'outliers selon la méthode sélectionnée
            if method == "iqr":
                lower, upper = self._iqr_bounds(series)
                mask = (series < lower) | (series > upper)

            elif method == "zscore":
                mean = series.mean()
                std = series.std()
                if std == 0:
                    continue
                zscores = (series - mean) / std
                mask = np.abs(zscores) > self.config["zscore_threshold"]

            elif method == "isoforest":
                from sklearn.ensemble import IsolationForest
                model = IsolationForest(
                    contamination=self.config["isolation_forest_contamination"],
                    random_state=42
                )
                preds = model.fit_predict(series.values.reshape(-1, 1))
                mask = preds == -1

            else:
                continue

            # Appliquer la méthode d’action
            df_clean = self._apply_action(df_clean, col, mask)

        log("Traitement des outliers terminé.")

        return df_clean
