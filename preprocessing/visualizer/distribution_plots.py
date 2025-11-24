import os
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.utils import ensure_directory, log


class DistributionPlots:
    """
    Génère automatiquement :
    - histogrammes pour colonnes numériques
    - boxplots pour outliers
    - countplots pour colonnes catégorielles

    Les plots sont sauvegardés dans : reports/plots/distributions/
    """

    def __init__(self, output_dir="reports/plots/distributions/"):
        self.output_dir = output_dir
        ensure_directory(self.output_dir)

    # ----------------------------------------------------------------------
    # Numerical distributions (hist + boxplot)
    # ----------------------------------------------------------------------
    def plot_numerical(self, df, col):
        try:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution - {col}")
            plt.savefig(os.path.join(self.output_dir, f"{col}_hist.png"))
            plt.close()

            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col].dropna())
            plt.title(f"Boxplot - {col}")
            plt.savefig(os.path.join(self.output_dir, f"{col}_boxplot.png"))
            plt.close()
        except Exception as e:
            log(f"Plot ignoré pour {col} (erreur : {str(e)})")

    # ----------------------------------------------------------------------
    # Categorical distributions (countplot)
    # ----------------------------------------------------------------------
    def plot_categorical(self, df, col):
        try:
            plt.figure(figsize=(10, 5))
            df[col].value_counts(dropna=False).plot(kind="bar")
            plt.title(f"Countplot - {col}")
            plt.savefig(os.path.join(self.output_dir, f"{col}_count.png"))
            plt.close()
        except Exception as e:
            log(f"Plot ignoré pour {col} (erreur : {str(e)})")

    # ----------------------------------------------------------------------
    # Méthode principale
    # ----------------------------------------------------------------------
    def apply(self, df, schema):
        log("Génération des graphiques de distributions...")

        for col in df.columns:

            col_type = schema[col]

            if col_type == "numerical":
                self.plot_numerical(df, col)

            elif col_type == "categorical":
                self.plot_categorical(df, col)

        log("Graphiques de distribution générés.")
