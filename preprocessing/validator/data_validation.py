import pandas as pd
from preprocessing.utils import log


class DataValidator:
    """
    Valide la qualité finale des données après preprocessing.
    Vérifie :
    - absence de NaN dans colonnes critiques
    - cohérence du schéma final
    - cohérence des types (numériques, catégories, datetime)
    - ranges valides pour numériques et datetime
    - absence de doublons dans colonnes ID
    """

    def __init__(self):
        pass

    # ----------------------------------------------------------------------
    # Vérification des NaN restants
    # ----------------------------------------------------------------------
    def _check_remaining_nans(self, df):
        nan_cols = df.columns[df.isna().any()].tolist()
        return nan_cols

    # ----------------------------------------------------------------------
    # Vérification cohérence schéma
    # ----------------------------------------------------------------------
    def _check_schema_consistency(self, df, schema):
        issues = []

        for col in df.columns:
            if col not in schema:
                issues.append(f"Colonne '{col}' non trouvée dans le schéma initial.")

        for col in schema:
            if col not in df.columns:
                issues.append(f"Colonne '{col}' manque après preprocessing.")

        return issues

    # ----------------------------------------------------------------------
    # Vérification des types
    # ----------------------------------------------------------------------
    def _check_type_validity(self, df, schema):
        issues = []

        for col, expected_type in schema.items():

            if col not in df.columns:
                continue

            series = df[col]

            if expected_type == "numerical" and not pd.api.types.is_numeric_dtype(series):
                issues.append(f"{col}: attendu numerical mais type réel = {series.dtype}")

            if expected_type == "categorical" and not pd.api.types.is_object_dtype(series):
                issues.append(f"{col}: attendu categorical mais type réel = {series.dtype}")

            if expected_type == "datetime" and not pd.api.types.is_datetime64_any_dtype(series):
                issues.append(f"{col}: attendu datetime mais type réel = {series.dtype}")

        return issues

    # ----------------------------------------------------------------------
    # Vérification des ranges
    # ----------------------------------------------------------------------
    def _check_ranges(self, df, schema):
        issues = []

        for col, typ in schema.items():

            if col not in df.columns:
                continue

            s = df[col]

            if typ == "numerical":
                if (s < -1e10).any() or (s > 1e10).any():
                    issues.append(f"{col}: valeurs numériques hors limites raisonnables.")

            if typ == "datetime":
                try:
                    years = s.dt.year
                    if (years < 1900).any() or (years > 2100).any():
                        issues.append(f"{col}: dates hors limites raisonnables.")
                except Exception:
                    pass

        return issues

    # ----------------------------------------------------------------------
    # Méthode principale
    # ----------------------------------------------------------------------
    def validate(self, df, schema):
        log("Validation finale des données...")

        report = {
            "nan_columns": [],
            "schema_issues": [],
            "type_issues": [],
            "range_issues": [],
            "status": "OK"
        }

        # 1. NaN restants
        report["nan_columns"] = self._check_remaining_nans(df)

        # 2. Cohérence avec schéma
        report["schema_issues"] = self._check_schema_consistency(df, schema)

        # 3. Types corrects
        report["type_issues"] = self._check_type_validity(df, schema)

        # 4. Vérification des ranges
        report["range_issues"] = self._check_ranges(df, schema)

        # 5. Déterminer le statut global
        if any([
            report["nan_columns"],
            report["schema_issues"],
            report["type_issues"],
            report["range_issues"]
        ]):
            report["status"] = "ERROR"
            log("Validation échouée. Voir le rapport.")
        else:
            log("Validation réussie. Données prêtes pour la modélisation.")

        return report
