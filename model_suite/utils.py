# utils.py
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt

# -----------------------------
# Classification Metrics
# -----------------------------
def classification_metrics(y_true, y_pred):
    """
    Returns a dictionary of common classification metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }
    return metrics

# -----------------------------
# Regression Metrics
# -----------------------------
def regression_metrics(y_true, y_pred):
    """
    Returns a dictionary of common regression metrics
    """
    metrics = {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }
    return metrics

# -----------------------------
# Plot Comparison
# -----------------------------
def plot_model_comparison(models, metric_name="accuracy"):
    """
    Plots a bar chart comparing models based on a chosen metric
    :param models: list of BaseModel instances with metrics populated
    :param metric_name: str, metric to compare (e.g., 'accuracy', 'RMSE')
    """
    names = [m.model_name for m in models]
    values = [m.metrics.get(metric_name, None) for m in models]

    plt.figure(figsize=(8, 5))
    plt.bar(names, values, color='skyblue')
    plt.ylabel(metric_name)
    plt.title(f'Model Comparison by {metric_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
