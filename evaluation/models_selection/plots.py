# plots.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay


def plot_regression_results(y_true, y_pred, title=None, ax=None):
    """Scatter vrai vs prédit avec ligne y=x."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_true, y_pred, alpha=0.6)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    ax.plot([mn, mx], [mn, mx], linestyle='--')
    ax.set_xlabel('y_true')
    ax.set_ylabel('y_pred')
    if title:
        ax.set_title(title)
    return ax


def plot_error_distribution(errors, title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(errors, bins=30, alpha=0.7)
    ax.set_xlabel('Error')
    ax.set_ylabel('Count')
    if title:
        ax.set_title(title)
    return ax


def plot_confusion_matrix(conf_matrix, labels=None, title=None, ax=None):
    import seaborn as sns
    if isinstance(conf_matrix, (list, np.ndarray)):
        cm = np.array(conf_matrix)
    else:
        cm = conf_matrix.values if hasattr(conf_matrix, 'values') else np.array(conf_matrix)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
    if labels is not None:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    if title:
        ax.set_title(title)
    return ax


def plot_roc_curve(y_true, y_proba, pos_label=1, title=None, ax=None):
    # y_proba: either vector of probabilities for positive class or 2D array
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    if title:
        ax.set_title(title)
    return ax


def plot_feature_importance(importances, feature_names, title=None, ax=None, top_n=25):
    # importances: array-like
    importances = np.array(importances)
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, 0.25*len(names))))
    ax.barh(range(len(names))[::-1], vals)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names)
    ax.set_xlabel('Importance')
    if title:
        ax.set_title(title)
    return ax