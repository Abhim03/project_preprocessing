# regression_models.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from base_model import BaseModel

# Optional: import XGBoost
try:
    from xgboost import XGBRegressor
    xgboost_available = True
except ImportError:
    xgboost_available = False
    print("XGBoost not installed. Skipping XGBRegressor.")

class LinearRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        model = LinearRegression(**kwargs)
        super().__init__(model, model_name="LinearRegression")

class RandomForestRegressorModel(BaseModel):
    def __init__(self, **kwargs):
        model = RandomForestRegressor(**kwargs)
        super().__init__(model, model_name="RandomForestRegressor")

if xgboost_available:
    class XGBoostRegressorModel(BaseModel):
        def __init__(self, **kwargs):
            model = XGBRegressor(**kwargs)
            super().__init__(model, model_name="XGBoostRegressor")
