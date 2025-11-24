import os
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.utils import ensure_directory, log


class CorrelationsPlots:
    """
    Génère les graphiques de corrélation :
    - heatmap des corrélations numériques
    - carte annotée
    """

    def __init__(self, output_dir="reports/plots/correlations/"):
        self.output_dir = output_dir
        ensure_directory(self.output_dir)

    # ----------------------------------------------------------------------
    # Heatmap principale
    # ----------------------------------------------------------------------
    def plot_correlation_heatmap(self, df):
        try:
            corr = df.corr()

            plt.figure(figsize=(12, 10))
            sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
            plt.title("Heatmap des corrélations")
            plt.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"))
            plt.close()
        except Exception as e:
            log(f"Heatmap ignorée (erreur : {str(e)})")

    # ----------------------------------------------------------------------
    # Heatmap annotée (optionnel)
    # ----------------------------------------------------------------------
    def plot_correlation_annotated(self, df):
        try:
            corr = df.corr()

            plt.figure(figsize=(14, 12))
            sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
            plt.title("Heatmap annotée des corrélations")
            plt.savefig(os.path.join(self.output_dir, "correlation_heatmap_annotated.png"))
            plt.close()
        except Exception as e:
            log(f"Heatmap annotée ignorée (erreur : {str(e)})")

    # ----------------------------------------------------------------------
    # Méthode principale
    # ----------------------------------------------------------------------
    def apply(self, df, schema):
        log("Génération des graphiques de corrélations...")

        # Conserver uniquement les numerical
        numerical_df = df.select_dtypes(include=["number"])

        if numerical_df.shape[1] <= 1:
            log("Trop peu de colonnes numériques pour la heatmap.")
            return

        self.plot_correlation_heatmap(numerical_df)
        self.plot_correlation_annotated(numerical_df)

        log("Graphiques de corrélation générés.")
