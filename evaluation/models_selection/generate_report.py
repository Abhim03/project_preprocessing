# evaluation/generate_report.py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import io
from .plots import plot_regression_results, plot_error_distribution, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from .shap_analysis import compute_shap_values, plot_shap_summary

def generate_model_report(results_df, models_saved, X_test, y_test, task_type, pdf_path="reports/model_report.pdf", top_n_features=20):
    """
    Génère un PDF complet détaillant :
      - métriques de chaque modèle
      - visualisations
      - comparaison et ranking
      - justification du choix
    """
    # Créer PDF
    pdf = PdfPages(pdf_path)

    # 1) Section par modèle
    for i, row in results_df.iterrows():
        model_name = row['model_name']
        model = models_saved[model_name]

        # --- titre ---
        plt.figure(figsize=(8,1))
        plt.axis('off')
        plt.text(0.5, 0.5, f"Modèle : {model_name}", ha='center', va='center', fontsize=16, weight='bold')
        pdf.savefig()
        plt.close()

        # --- Métriques ---
        metrics_text = "\n".join([f"{k}: {v}" for k,v in row.items() if k not in ['model', 'model_name', 'confusion_matrix', 'model_size_bytes']])
        plt.figure(figsize=(8,2))
        plt.axis('off')
        plt.text(0,0.5, metrics_text, fontsize=10)
        pdf.savefig()
        plt.close()

        # --- Graphiques principaux ---
        if task_type == 'regression':
            # scatter y_true vs y_pred
            y_pred = model.predict(X_test)
            fig, ax = plt.subplots(figsize=(6,6))
            plot_regression_results(y_test, y_pred, title=f"{model_name} - True vs Pred", ax=ax)
            pdf.savefig(fig)
            plt.close(fig)

            # erreur distribution
            errors = y_test - y_pred
            fig, ax = plt.subplots(figsize=(6,4))
            plot_error_distribution(errors, title=f"{model_name} - Error Distribution", ax=ax)
            pdf.savefig(fig)
            plt.close(fig)

        else: # classification
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

            # confusion matrix
            fig, ax = plt.subplots(figsize=(6,5))
            plot_confusion_matrix(row['confusion_matrix'], title=f"{model_name} - Confusion Matrix", ax=ax)
            pdf.savefig(fig)
            plt.close(fig)

            # ROC curve
            if y_proba is not None:
                fig, ax = plt.subplots(figsize=(6,6))
                plot_roc_curve(y_test, y_proba, title=f"{model_name} - ROC Curve", ax=ax)
                pdf.savefig(fig)
                plt.close(fig)

        # --- Feature importance / SHAP ---
        try:
            if hasattr(model, "feature_importances_"):
                fig, ax = plt.subplots(figsize=(8, max(3, 0.25*top_n_features)))
                plot_feature_importance(model.feature_importances_, X_test.columns, title=f"{model_name} - Feature Importance", ax=ax, top_n=top_n_features)
                pdf.savefig(fig)
                plt.close(fig)
            else:
                # SHAP
                X_sample = X_test.sample(min(200, len(X_test)), random_state=42)
                shap_values, _ = compute_shap_values(model, X_sample)
                # SHAP summary plot
                fig = plt.figure()
                plot_shap_summary(shap_values, X_sample, show=False)
                pdf.savefig(fig)
                plt.close(fig)
        except Exception as e:
            print(f"Skipping SHAP/feature importance for {model_name}: {e}")

    # 2) Section de comparaison
    plt.figure(figsize=(8,2))
    plt.axis('off')
    plt.text(0.5, 0.5, "Comparaison et Ranking des modèles", ha='center', va='center', fontsize=16, weight='bold')
    pdf.savefig()
    plt.close()

    # Tableau comparatif
    comp_cols = [c for c in results_df.columns if c not in ['model','model_size_bytes','model']]
    fig, ax = plt.subplots(figsize=(10, len(results_df)*0.4 + 1))
    ax.axis('off')
    table = ax.table(cellText=results_df[comp_cols].round(3).values,
                     colLabels=comp_cols,
                     rowLabels=results_df['model_name'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    pdf.savefig(fig)
    plt.close(fig)

    # 3) Section choix final
    best_model_name = results_df.sort_values('global_score', ascending=False).iloc[0]['model_name']
    plt.figure(figsize=(8,1))
    plt.axis('off')
    plt.text(0.5, 0.5, f"Meilleur modèle sélectionné : {best_model_name}", ha='center', va='center', fontsize=14, weight='bold')
    pdf.savefig()
    plt.close()

    pdf.close()
    print(f"PDF généré avec succès : {pdf_path}")
