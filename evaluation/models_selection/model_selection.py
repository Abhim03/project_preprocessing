# model_selection.py
import time
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from .metrics import regression_metrics, classification_metrics
from .utils import timer
import joblib


def evaluate_model(model, X_train, y_train, X_test, y_test, task_type='regression', return_model=False, fit=True):
    """Fit (optionnel) le modèle, prédit, calcule métriques, chronomètre.
    Renvoie un dict de résultats.
    """
    res = {'model_name': type(model).__name__}
    m = clone(model)

    # entraînement
    t0 = time.time()
    if fit:
        m.fit(X_train, y_train)
    train_time = time.time() - t0
    res['training_time'] = train_time

    # prédictions
    try:
        y_pred = m.predict(X_test)
    except NotFittedError:
        raise

    # probabilités si classification
    y_proba = None
    if task_type == 'classification':
        if hasattr(m, 'predict_proba'):
            try:
                y_proba = m.predict_proba(X_test)
            except Exception:
                y_proba = None
        elif hasattr(m, 'decision_function'):
            try:
                dec = m.decision_function(X_test)
                # si binaire, convert to two-col prob approx
                if dec.ndim == 1:
                    from sklearn.preprocessing import minmax_scale
                    probs = minmax_scale(dec)
                    y_proba = np.vstack([1-probs, probs]).T
                else:
                    y_proba = dec
            except Exception:
                y_proba = None

    # metrics
    if task_type == 'regression':
        metrics = regression_metrics(y_test, y_pred)
        res.update(metrics)
    else:
        metrics = classification_metrics(y_test, y_pred, y_proba)
        # confusion matrix est un DataFrame -> on l'ajoute séparément
        cm = metrics.pop('confusion_matrix', None)
        res.update(metrics)
        res['confusion_matrix'] = cm

    # taille du modèle (sérialisation)
    try:
        # stocker en mémoire puis mesurer
        import io
        buf = io.BytesIO()
        joblib.dump(m, buf)
        res['model_size_bytes'] = buf.getbuffer().nbytes
    except Exception:
        res['model_size_bytes'] = None

    if return_model:
        res['model'] = m
    return res


def evaluate_models(models_list, X_train, y_train, X_test, y_test, task_type='regression', parallel=False):
    """Boucle sur la liste de modèles (list d'instances). Retourne DataFrame résultats et dict d'instances si demandé."""
    results = []
    models_saved = {}
    for model in models_list:
        print(f"Evaluating {type(model).__name__}...")
        r = evaluate_model(model, X_train, y_train, X_test, y_test, task_type=task_type, return_model=True)
        # extraire modèle
        models_saved[r['model_name']] = r.pop('model')
        results.append(r)
    df = pd.DataFrame(results)
    # organiser colonnes : model_name, metrics..., training_time, model_size_bytes
    return df, models_saved


def _normalize_series(s, higher_is_better=True):
    s = s.astype(float)
    if s.isnull().all():
        return s
    if higher_is_better:
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    else:
        # lower is better (ex: rmse, mae, training_time, model_size_bytes)
        inv = (s.max() - s) / (s.max() - s.min() + 1e-9)
        return inv


def rank_models(results_df, task_type='regression', custom_weights=None):
    """Normalise les métriques et calcule un score global.
    custom_weights: dict metric->weight
    Retourne DataFrame trié avec colonne 'global_score'.
    """
    df = results_df.copy()
    # choisir métriques pertinentes
    if task_type == 'regression':
        metrics = ['r2', 'rmse', 'mae', 'training_time', 'model_size_bytes']
        higher_is_better = {'r2': True, 'rmse': False, 'mae': False, 'training_time': False, 'model_size_bytes': False}
    else:
        metrics = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc', 'training_time', 'model_size_bytes']
        higher_is_better = {m: True for m in metrics}
        # metrics that are actually 'lower is better'
        higher_is_better['training_time'] = False
        higher_is_better['model_size_bytes'] = False

    # apply default weights
    weights = {m: 1.0 for m in metrics}
    if custom_weights:
        weights.update(custom_weights)

    # normalized columns
    norm_cols = {}
    for m in metrics:
        if m in df.columns:
            norm_cols[m] = _normalize_series(df[m].fillna(df[m].median()), higher_is_better=higher_is_better[m])
        else:
            # absent -> zeros
            norm_cols[m] = pd.Series(0, index=df.index)

    norm_df = pd.DataFrame(norm_cols)

    # compute global score
    weighted = sum(norm_df[m] * weights.get(m, 1.0) for m in norm_df.columns)
    # scale to 0-1
    global_score = _normalize_series(weighted, higher_is_better=True)
    df['global_score'] = global_score
    df_sorted = df.sort_values('global_score', ascending=False).reset_index(drop=True)
    return df_sorted


def select_best_model(results_df, task_type='regression', custom_rules=None):
    """Règles simples pour sélectionner le meilleur modèle.
    Exemples:
      - si classification et F1>0.8 => preferer f1
      - sinon utiliser global_score
    """
    df = results_df.copy()
    if 'global_score' not in df.columns:
        df = rank_models(df, task_type=task_type)

    # exemples de règle simple
    if task_type == 'classification' and custom_rules is None:
        # si un modèle a f1 >= 0.8 et auc >=0.8 -> choose highest f1 among them
        cand = df[(df.get('f1', 0) >= 0.8) & ((df.get('roc_auc', 0) >= 0.8) | df.get('roc_auc', 0).isnull())]
        if len(cand) > 0:
            return cand.sort_values('f1', ascending=False).iloc[0]['model_name']

    # fallback: best global_score
    return df.sort_values('global_score', ascending=False).iloc[0]['model_name']