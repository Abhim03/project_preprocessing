import os
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.utils import ensure_directory, log


class MissingValuesPlots:
    """
    Génère :
    - barplot du pourcentage de valeurs manquantes
    - heatmap des NaN
    """

    def __init__(self, output_dir="reports/plots/missing_values/"):
        self.output_dir = output_dir
        ensure_directory(self.output_dir)

    # ----------------------------------------------------------------------
    # Barplot du pourcentage de NaN
    # ----------------------------------------------------------------------
    def plot_missing_barplot(self, df):
        try:
            missing_ratio = df.isna().mean().sort_values(ascending=False)

            plt.figure(figsize=(12, 6))
            missing_ratio.plot(kind="bar", color="red")
            plt.title("Pourcentage de valeurs manquantes par colonne")
            plt.ylabel("Ratio de NaN")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "missing_ratio_barplot.png"))
            plt.close()
        except Exception as e:
            log(f"Barplot NaN ignoré (erreur : {str(e)})")

    # ----------------------------------------------------------------------
    # Heatmap des NaN
    # ----------------------------------------------------------------------
    def plot_missing_heatmap(self, df):
        try:
            plt.figure(figsize=(12, 8))
            sns.heatmap(df.isna(), cmap="viridis", cbar=False)
            plt.title("Heatmap des valeurs manquantes")
            plt.savefig(os.path.join(self.output_dir, "missing_heatmap.png"))
            plt.close()
        except Exception as e:
            log(f"Heatmap NaN ignorée (erreur : {str(e)})")

    # ----------------------------------------------------------------------
    # Méthode principale
    # ----------------------------------------------------------------------
    def apply(self, df):
        log("Génération des graphiques de valeurs manquantes...")

        # Plot 1 : barplot des pourcentages de NaN
        self.plot_missing_barplot(df)

        # Plot 2 : heatmap complète des NaN
        self.plot_missing_heatmap(df)

        log("Graphiques de valeurs manquantes générés.")
