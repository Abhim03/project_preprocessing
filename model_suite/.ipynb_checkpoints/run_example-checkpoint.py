# run_example.py
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Import models
from classification_models import LogisticRegressionModel, RandomForestClassifierModel, SVMModel
try:
    from classification_models import XGBoostClassifierModel
    xgb_cls_available = True
except ImportError:
    xgb_cls_available = False

from regression_models import LinearRegressionModel, RandomForestRegressorModel
try:
    from regression_models import XGBoostRegressorModel
    xgb_reg_available = True
except ImportError:
    xgb_reg_available = False

# Import utils
from utils import classification_metrics, regression_metrics, plot_model_comparison

# -----------------------------
# Classification Demo
# -----------------------------
print("=== Classification Demo ===")
X_cls, y_cls = make_classification(n_samples=500, n_features=10, n_informative=7, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

cls_models = [
    LogisticRegressionModel(max_iter=500),
    RandomForestClassifierModel(n_estimators=100),
    SVMModel()
]

if xgb_cls_available:
    cls_models.append(XGBoostClassifierModel())

# Train, validate and show performance
for model in cls_models:
    model.train(X_train_cls, y_train_cls)
    model.validate(X_test_cls, y_test_cls, classification_metrics)
    model.show_performance()
    model.save_model(save_path="saved_models/classification")

# Plot comparison (accuracy)
plot_model_comparison(cls_models, metric_name="accuracy")

# -----------------------------
# Regression Demo
# -----------------------------
print("\n=== Regression Demo ===")
X_reg, y_reg = make_regression(n_samples=500, n_features=10, n_informative=7, noise=0.1, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_models = [
    LinearRegressionModel(),
    RandomForestRegressorModel(n_estimators=100)
]

if xgb_reg_available:
    reg_models.append(XGBoostRegressorModel())

# Train, validate and show performance
for model in reg_models:
    model.train(X_train_reg, y_train_reg)
    model.validate(X_test_reg, y_test_reg, regression_metrics)
    model.show_performance()
    model.save_model(save_path="saved_models/regression")

# Plot comparison (R2)
plot_model_comparison(reg_models, metric_name="R2")
