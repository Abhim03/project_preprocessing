import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient
import os
import json
from datetime import datetime


class NotebookGenerator:

    def __init__(self):
        pass

    def _add(self, cells, code=None, md=None):
        """Ajoute une cellule markdown ou code."""
        if md is not None:
            cells.append(new_markdown_cell(md))
        if code is not None:
            cells.append(new_code_cell(code))

    def generate_notebook(self, results: dict, output_path="reports/preprocessing_report.ipynb"):
        """
        Génère un notebook complet comprenant toutes les étapes
        du preprocessing basé sur les objets fournis par la pipeline.
        """

        df_raw = results.get("df_raw")
        df_clean = results.get("df_clean")
        df_features = results.get("df_features")
        X_final = results.get("X_final")
        y = results.get("y")
        schema = results.get("schema")
        eda_report = results.get("eda_report")
        split = results.get("split")

        nb = new_notebook()
        cells = []

        # --------------------------------------------------------
        # 1 - INTRODUCTION
        # --------------------------------------------------------
        self._add(cells, md=f"# Preprocessing Report\nGénéré automatiquement le {datetime.now()}.\n")

        # --------------------------------------------------------
        # 2 - RAW DATA
        # --------------------------------------------------------
        self._add(cells, md="## 1. Données brutes : aperçu")
        self._add(cells, code="df_raw.head()")

        # --------------------------------------------------------
        # 3 - SCHEMA
        # --------------------------------------------------------
        self._add(cells, md="## 2. Schéma inféré automatiquement")
        self._add(cells, code="schema")

        # --------------------------------------------------------
        # 4 - MISSING VALUES
        # --------------------------------------------------------
        self._add(cells, md="## 3. Valeurs manquantes détectées")
        self._add(cells, code="df_raw.isna().sum()")

        self._add(cells, md="Après nettoyage :")
        self._add(cells, code="df_clean.isna().sum()")

        # --------------------------------------------------------
        # 5 - OUTLIERS
        # --------------------------------------------------------
        self._add(cells, md="## 4. Outliers détectés (résumé)")
        self._add(cells, code="eda_report['outlier_report']")

        # --------------------------------------------------------
        #' 6 - VISUALISATION EDA (Plots déjà enregistrés)
        # --------------------------------------------------------
        self._add(cells, md="## 5. Visualisations EDA (voir images exportées)")
        self._add(cells, code="eda_report['summary']")

        # --------------------------------------------------------
        # 7 - FEATURE ENGINEERING
        # --------------------------------------------------------
        self._add(cells, md="## 6. Feature Engineering : aperçu des colonnes")
        self._add(cells, code="df_features.head()")

        # --------------------------------------------------------
        # 8 - ENCODING + SCALING + FEATURE SELECTION
        # --------------------------------------------------------
        self._add(cells, md="## 7. Données finales prêtes pour modélisation")
        self._add(cells, code="X_final.head()")

        # --------------------------------------------------------
        # 9 - TRAIN/TEST SPLIT
        # --------------------------------------------------------
        if split is not None:
            self._add(cells, md="## 8. Train/Test Split")
            self._add(cells, code=f"X_train.shape, X_test.shape")

        # --------------------------------------------------------
        # 10 - RÉSUMÉ FINAL
        # --------------------------------------------------------
        self._add(cells, md="## 9. Résumé final du preprocessing")
        self._add(cells, code="eda_report")

        # Structure finale
        nb["cells"] = cells

        # --------------------------------------------------------
        # Sauvegarde notebook
        # --------------------------------------------------------
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        # Exécuter le notebook
        client = NotebookClient(nb)
        client.execute()

        # Sauvegarde du notebook exécuté
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        print(f"Notebook généré : {output_path}")
