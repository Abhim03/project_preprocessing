# base_model.py
import pickle
import time
from pathlib import Path

class BaseModel:
    def __init__(self, model, model_name="BaseModel"):
        """
        Base class for all models.
        :param model: an instantiated sklearn-like model
        :param model_name: string identifier for the model
        """
        self.model = model
        self.model_name = model_name
        self.metrics = {}
        self.train_time = None

    def train(self, X_train, y_train):
        """
        Train the model on the training data
        """
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time
        print(f"[{self.model_name}] Training completed in {self.train_time:.2f} seconds.")

    def validate(self, X_test, y_test, metric_func):
        """
        Validate the model on test data.
        :param metric_func: function that returns metrics dict given y_test, y_pred
        """
        y_pred = self.model.predict(X_test)
        self.metrics = metric_func(y_test, y_pred)
        print(f"[{self.model_name}] Validation metrics: {self.metrics}")
        return self.metrics

    def show_performance(self):
        """
        Print or summarize performance
        """
        if not self.metrics:
            print(f"[{self.model_name}] No metrics available. Please run validate() first.")
        else:
            print(f"Performance of {self.model_name}:")
            for k, v in self.metrics.items():
                print(f" - {k}: {v}")

    def save_model(self, save_path="models"):
        """
        Save model using pickle
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        file_path = Path(save_path) / f"{self.model_name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[{self.model_name}] Model saved at {file_path}")

    def load_model(self, load_path):
        """
        Load model from a pickle file
        """
        with open(load_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"[{self.model_name}] Model loaded from {load_path}")
