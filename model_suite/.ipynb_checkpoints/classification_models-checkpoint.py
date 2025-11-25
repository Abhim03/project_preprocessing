# classification_models.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from base_model import BaseModel

# Optional: import XGBoost
try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False
    print("XGBoost not installed. Skipping XGBClassifier.")

class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        model = LogisticRegression(**kwargs)
        super().__init__(model, model_name="LogisticRegression")

class RandomForestClassifierModel(BaseModel):
    def __init__(self, **kwargs):
        model = RandomForestClassifier(**kwargs)
        super().__init__(model, model_name="RandomForestClassifier")

class SVMModel(BaseModel):
    def __init__(self, **kwargs):
        model = SVC(probability=True, **kwargs)
        super().__init__(model, model_name="SVMClassifier")

if xgboost_available:
    class XGBoostClassifierModel(BaseModel):
        def __init__(self, **kwargs):
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)
            super().__init__(model, model_name="XGBoostClassifier")
