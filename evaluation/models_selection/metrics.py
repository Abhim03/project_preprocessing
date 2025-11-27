# metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)


def regression_metrics(y_true, y_pred, include_mape=False):
    """Retourne un dict contenant R2, MAE, MSE, RMSE et optionnellement MAPE."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    res = {}
    res['r2'] = float(r2_score(y_true, y_pred))
    res['mae'] = float(mean_absolute_error(y_true, y_pred))
    res['mse'] = float(mean_squared_error(y_true, y_pred))
    res['rmse'] = float(np.sqrt(res['mse']))
    if include_mape:
        # évitez la division par zéro
        eps = 1e-8
        res['mape'] = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)
    return res


def classification_metrics(y_true, y_pred, y_proba=None, labels=None):
    """Retourne un dict contenant accuracy, precision, recall, f1, confusion_matrix, et opc. roc_auc si y_proba fourni."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    res = {}
    res['accuracy'] = float(accuracy_score(y_true, y_pred))
    # micro/weighted choices: ici on calcule weighted pour être robuste aux classes déséquilibrées
    res['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    res['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    res['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    res['confusion_matrix'] = pd.DataFrame(confusion_matrix(y_true, y_pred),
                                           index=labels or None, columns=labels or None)
    if y_proba is not None:
        try:
            # si multi-classe, y_proba doit être shape (n_samples, n_classes)
            if y_proba.ndim == 2:
                # calculer macro AUC si possible
                res['roc_auc'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr'))
            else:
                res['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            res['roc_auc'] = None
    return res