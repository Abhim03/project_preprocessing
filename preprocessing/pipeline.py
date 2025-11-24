import os
import json
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing.utils import (
    load_config,
    ensure_directory,
    log,
)

# Reader
from preprocessing.reader.file_loader import FileLoader
from preprocessing.reader.schema_inference import SchemaInference

# Analyzer
from preprocessing.analyzer.type_detector import TypeDetector
from preprocessing.analyzer.data_summary import DataSummary
from preprocessing.analyzer.missing_values_analyzer import MissingValuesAnalyzer
from preprocessing.analyzer.outlier_detector import OutlierDetector

# Cleaner
from preprocessing.cleaner.duplicates_handler import DuplicatesHandler
from preprocessing.cleaner.inconsistent_values_handler import InconsistentValuesHandler
from preprocessing.cleaner.missing_values_handler import MissingValuesHandler
from preprocessing.cleaner.outlier_handler import OutlierHandler

# Transformer
from preprocessing.transformer.encoders import Encoders
from preprocessing.transformer.scalers import Scalers
from preprocessing.transformer.feature_generator import FeatureGenerator
from preprocessing.transformer.feature_selector import FeatureSelector

# Visualizer
from preprocessing.visualizer.distribution_plots import DistributionPlots
from preprocessing.visualizer.correlations_plots import CorrelationsPlots
from preprocessing.visualizer.missing_values_plots import MissingValuesPlots

# Validator
from preprocessing.validator.data_validation import DataValidator


class PreprocessingPipeline:
    """
    Pipeline de préprocessing complet, de la lecture des données
    jusqu'au train/test split et à l'export.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        # Charger la configuration
        self.config = load_config(config_path)

        # Modules
        self.file_loader = FileLoader()
        self.schema_inference = SchemaInference()
        self.type_detector = TypeDetector()
        self.data_summary = DataSummary()
        self.missing_analyzer = MissingValuesAnalyzer(self.config)
        self.outlier_detector = OutlierDetector(self.config)

        self.duplicates_handler = DuplicatesHandler(self.config)
        self.inconsistent_handler = InconsistentValuesHandler()
        self.missing_handler = MissingValuesHandler(self.config)
        self.outlier_handler = OutlierHandler(self.config)

        self.encoders = Encoders(self.config)
        self.scalers = Scalers(self.config)
        self.feature_generator = FeatureGenerator(self.config)
        self.feature_selector = FeatureSelector(self.config)

        self.dist_plots = DistributionPlots()
        self.corr_plots = CorrelationsPlots()
        self.missing_plots = MissingValuesPlots()

        self.validator = DataValidator()

        # Sous-configs
        self.split_config = self.config.get("split", {})
        self.export_config = self.config.get("export", {})
        self.output_dir = self.export_config.get("output_dir", "reports/")
        ensure_directory(self.output_dir)

    # ------------------------------------------------------------------ #
    #   Étapes internes
    # ------------------------------------------------------------------ #

    def _run_eda_and_plots(self, df: pd.DataFrame, schema: dict) -> dict:
        """Lance EDA automatique + génération des plots et retourne un résumé."""
        summary = self.data_summary.summarize(df, schema)
        missing_report = self.missing_analyzer.analyze(df, schema)
        outlier_report = self.outlier_detector.analyze(df, schema)

        # Plots
        try:
            self.dist_plots.apply(df, schema)
            self.corr_plots.apply(df, schema)
            self.missing_plots.apply(df)
        except Exception as e:
            log(f"Erreur lors de la génération des plots (ignorée) : {str(e)}")

        return {
            "summary": summary,
            "missing_report": missing_report,
            "outlier_report": outlier_report,
        }

    def _clean_data(
        self,
        df: pd.DataFrame,
        schema: dict,
        missing_report: dict,
        outlier_report: dict,
    ) -> pd.DataFrame:
        """Applique toutes les étapes de cleaning."""
        log("=== Étape CLEANING ===")

        # 1. Doublons
        self.duplicates_handler.analyze(df)
        df = self.duplicates_handler.remove(df)

        # 2. Incohérences
        inconsistencies = self.inconsistent_handler.analyze(df, schema)
        if inconsistencies:
            log(f"Incohérences détectées : {inconsistencies}")
        df = self.inconsistent_handler.clean(df, schema)

        # 3. Valeurs manquantes
        df = self.missing_handler.apply(df, schema, missing_report)

        # 4. Outliers
        df = self.outlier_handler.apply(df, schema, outlier_report)

        return df

    def _split_train_test(self, X: pd.DataFrame, y: pd.Series):
        """Split train/test intelligent avec stratification automatique."""
        if not self.split_config.get("enabled", True) or y is None:
            log("Train/test split désactivé ou target absente.")
            return None

        test_size = self.split_config.get("test_size", 0.2)
        shuffle = self.split_config.get("shuffle", True)
        stratify_mode = self.split_config.get("stratify", "auto")

        if stratify_mode == "auto" and y.nunique() < 20:
            stratify = y
        else:
            stratify = None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=shuffle,
            stratify=stratify,
            random_state=42,
        )

        log(
            f"Split effectué : "
            f"X_train={X_train.shape}, X_test={X_test.shape}, "
            f"y_train={y_train.shape}, y_test={y_test.shape}"
        )

        return X_train, X_test, y_train, y_test

    def _build_scaling_schema(self, df: pd.DataFrame) -> dict:
        """Construit un schéma simplifié pour le scaling après encodage."""
        schema = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                schema[col] = "numerical"
            else:
                schema[col] = "categorical"
        return schema

    # ------------------------------------------------------------------ #
    #   Export
    # ------------------------------------------------------------------ #

    def _export_outputs(
        self,
        df_clean: pd.DataFrame,
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        eda_report: dict = None,
    ):
        """Gère tous les exports en CSV/JSON + pipeline pickle."""
        ensure_directory(self.output_dir)

        # 1. Data nettoyée complète
        if self.export_config.get("save_clean_data", True):
            clean_path = os.path.join(self.output_dir, "clean_data.csv")
            df_clean.to_csv(clean_path, index=False)
            log(f"Data nettoyée sauvegardée : {clean_path}")

        # 2. Train / test
        if (
            self.export_config.get("save_train_test", True)
            and X_train is not None
            and y_train is not None
        ):
            X_train.to_csv(os.path.join(self.output_dir, "X_train.csv"), index=False)
            X_test.to_csv(os.path.join(self.output_dir, "X_test.csv"), index=False)
            y_train.to_csv(os.path.join(self.output_dir, "y_train.csv"), index=False)
            y_test.to_csv(os.path.join(self.output_dir, "y_test.csv"), index=False)
            log("Train/test exportés dans le dossier reports/.")

        # 3. Rapport de preprocessing (EDA + NaN + outliers)
        if self.export_config.get("save_reports", True) and eda_report is not None:
            report_path = os.path.join(self.output_dir, "preprocessing_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(eda_report, f, indent=2, ensure_ascii=False)
            log(f"Rapport EDA sauvegardé : {report_path}")

        # 4. Sauvegarde du pipeline (encodeurs, scalers, selector)
        if self.export_config.get("save_pipeline", True):
            pipeline_state = {
                "encoders": self.encoders,
                "scalers": self.scalers,
                "feature_selector": self.feature_selector,
            }
            pipeline_path = os.path.join(self.output_dir, "preprocessing_pipeline.pkl")
            with open(pipeline_path, "wb") as f:
                pickle.dump(pipeline_state, f)
            log(f"Pipeline de preprocessing sauvegardé : {pipeline_path}")

    # ------------------------------------------------------------------ #
    #   Méthode principale
    # ------------------------------------------------------------------ #

    def run(self, data_path: str, target_col: str | None = None):
        """
        Lance tout le pipeline sur un fichier de données.

        Paramètres
        ----------
        data_path : str
            Chemin vers le fichier (csv/xlsx/parquet/json).
        target_col : str ou None
            Nom de la colonne cible (si connue).

        Retour :
        - Si target_col fourni :
            X_train, X_test, y_train, y_test
        - Sinon :
            df_final (DataFrame prêt pour modélisation non supervisée)
        """

        log("=== DÉBUT DU PIPELINE DE PREPROCESSING ===")

        # --------------------------------------------------------------
        # 1. Chargement du fichier
        # --------------------------------------------------------------
        df_raw = self.file_loader.load(data_path)

        # --------------------------------------------------------------
        # 2. Inférence + affinement du schéma
        # --------------------------------------------------------------
        initial_schema = self.schema_inference.infer_types(df_raw)
        refined_schema = self.type_detector.refine_types(df_raw, initial_schema)

        # --------------------------------------------------------------
        # 3. EDA + rapports + plots
        # --------------------------------------------------------------
        eda_report = self._run_eda_and_plots(df_raw, refined_schema)

        # --------------------------------------------------------------
        # 4. Cleaning (doublons, incohérences, NaN, outliers)
        # --------------------------------------------------------------
        df_clean = self._clean_data(
            df_raw,
            refined_schema,
            eda_report["missing_report"],
            eda_report["outlier_report"],
        )

        # --------------------------------------------------------------
        # 5. Feature engineering (nouvelles colonnes)
        # --------------------------------------------------------------
        df_features = self.feature_generator.apply(df_clean, refined_schema)

        # --------------------------------------------------------------
        # 6. Séparation X / y
        # --------------------------------------------------------------
        if target_col is not None and target_col in df_features.columns:
            y = df_features[target_col]
            X = df_features.drop(columns=[target_col])
        else:
            y = None
            X = df_features

        # --------------------------------------------------------------
        # 7. Encodage des variables catégorielles
        # --------------------------------------------------------------
        X_encoded = self.encoders.apply(X, refined_schema, y)

        # --------------------------------------------------------------
        # 8. Scaling
        # --------------------------------------------------------------
        scaling_schema = self._build_scaling_schema(X_encoded)
        X_scaled = self.scalers.apply(X_encoded, scaling_schema)

        # --------------------------------------------------------------
        # 9. Feature selection
        # --------------------------------------------------------------
        X_final = self.feature_selector.apply(X_scaled, scaling_schema, y)

        # --------------------------------------------------------------
        # 10. Validation finale des données
        # --------------------------------------------------------------
        to_validate = X_final.copy()
        if y is not None:
            to_validate[target_col] = y

        validation_report = self.validator.validate(to_validate, scaling_schema)
        eda_report["validation_report"] = validation_report

        # --------------------------------------------------------------
        # 11. Train/test split (si y dispo)
        # --------------------------------------------------------------
        split_result = None
        if y is not None:
            split_result = self._split_train_test(X_final, y)

        # --------------------------------------------------------------
        # 12. Exports
        # --------------------------------------------------------------
        if split_result is not None:
            X_train, X_test, y_train, y_test = split_result
            self._export_outputs(
                df_clean=df_clean,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                eda_report=eda_report,
            )
            log("=== FIN DU PIPELINE (avec target) ===")
            return X_train, X_test, y_train, y_test

        else:
            self._export_outputs(
                df_clean=df_clean,
                eda_report=eda_report,
            )
            log("=== FIN DU PIPELINE (sans target) ===")
            return X_final
