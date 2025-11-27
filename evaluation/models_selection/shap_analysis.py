# shap_analysis.py
import shap
import numpy as np


def compute_shap_values(model, X_sample):
    """Renvoie shap_values et explainer. Choisit TreeExplainer si possible, sinon KernelExplainer (lent)."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, 100))
        shap_values = explainer.shap_values(X_sample)
    return shap_values, explainer


def plot_shap_summary(shap_values, X_sample, show=True):
    shap.summary_plot(shap_values, X_sample, show=show)


def plot_shap_dependence(shap_values, X_sample, feature_name, show=True):
    shap.dependence_plot(feature_name, shap_values, X_sample, show=show)
