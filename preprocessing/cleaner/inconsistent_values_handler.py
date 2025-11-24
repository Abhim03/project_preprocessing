import pandas as pd
from preprocessing.utils import log


class InconsistentValuesHandler:
    """
    Détecte et corrige certaines valeurs incohérentes dans les données.
    
    Fonctions couvertes :
    - valeurs numériques impossibles (ex : âge < 0)
    - valeurs négatives non autorisées (optionnel)
    - dates invalides
    - colonnes numériques contenant du texte ou des symboles
    - valeurs aberrantes textuelles dans les colonnes catégorielles
    """

    def __init__(self):
        pass

    def detect_numeric_inconsistencies(self, series):
        """
        Détecte :
        - valeurs négatives si la colonne ne devrait pas en avoir
        - valeurs textuelles dans les colonnes numériques
        """
        problems = {}

        # Textes dans une colonne censée être numérique
        non_numeric = series.apply(lambda x: isinstance(x, str)).sum()
        if non_numeric > 0:
            problems["non_numeric_values"] = int(non_numeric)

        # Valeurs négatives (souvent incohérentes sauf contextes très spécifiques)
        negative_values = (series < 0).sum()
        if negative_values > 0:
            problems["negative_values"] = int(negative_values)

        return problems

    def detect_datetime_inconsistencies(self, series):
        """
        Détecte les dates invalides ou incohérentes.
        Par exemple :
        - dates dans le futur (optionnel)
        - dates hors intervalle réaliste
        """
        problems = {}

        # Dates impossibles (si conversion échoue -> déjà traité ailleurs)
        # Option : détecter les dates > année 2100 ou < 1900
        invalid_years = ((series.dt.year < 1900) | (series.dt.year > 2100)).sum()
        if invalid_years > 0:
            problems["invalid_years"] = int(invalid_years)

        return problems

    def detect_categorical_inconsistencies(self, series):
        """
        Détecte les valeurs anormales dans les colonnes catégorielles :
        - valeurs numériques hors contexte
        - chaînes vides
        - catégories non interprétables
        """
        problems = {}

        blank_values = (series.astype(str).str.strip() == "").sum()
        if blank_values > 0:
            problems["blank_strings"] = int(blank_values)

        return problems

    def analyze(self, df, schema):
        """
        Analyse globale de toutes les colonnes en fonction du type détecté.
        Retourne un dictionnaire des anomalies détectées.
        """

        log("Analyse des valeurs incohérentes...")

        inconsistencies = {}

        for col in df.columns:
            col_type = schema[col]
            series = df[col]

            if col_type == "numerical":
                issues = self.detect_numeric_inconsistencies(series)
            elif col_type == "datetime":
                issues = self.detect_datetime_inconsistencies(series)
            elif col_type == "categorical":
                issues = self.detect_categorical_inconsistencies(series)
            else:
                issues = {}

            if issues:
                inconsistencies[col] = issues

        if inconsistencies:
            log(f"Incohérences détectées dans {len(inconsistencies)} colonnes.")
        else:
            log("Aucune valeur incohérente détectée.")

        return inconsistencies

    def clean(self, df, schema):
        """
        Corrige automatiquement certaines incohérences simples :
        - Conversion numérique forcée
        - Remplacement des valeurs négatives par NaN (pour imputation ultérieure)
        - Nettoyage des chaînes vides dans les catégorielles
        - Remplacement des dates invalides par NaN
        """

        log("Correction des incohérences...")

        df_clean = df.copy()

        for col in df.columns:
            col_type = schema[col]
            series = df_clean[col]

            # Correction pour numériques
            if col_type == "numerical":
                # Convertir les textes numériques
                df_clean[col] = pd.to_numeric(series, errors="coerce")
                # Remplacer valeurs négatives par NaN
                df_clean.loc[df_clean[col] < 0, col] = pd.NA

            # Correction pour datetime
            elif col_type == "datetime":
                df_clean[col] = pd.to_datetime(series, errors="coerce")
                # Remplacer dates hors normes
                df_clean.loc[
                    (df_clean[col].dt.year < 1900) | (df_clean[col].dt.year > 2100),
                    col
                ] = pd.NaT

            # Correction pour catégorielles
            elif col_type == "categorical":
                df_clean.loc[df_clean[col].astype(str).str.strip() == "", col] = pd.NA

        log("Correction des incohérences terminée.")

        return df_clean
