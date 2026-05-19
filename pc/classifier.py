import os
import pickle
import numpy as np

# Make ML imports optional
try:
    from sklearn.ensemble import RandomForestClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

MODEL_FILE = 'sheer_rf_model.pkl'

class OptionalClassifier:
    """
    Handles ML prediction. This module is optional. If scikit-learn is not installed
    or the model is not trained, it gracefully returns None or default values.
    """
    def __init__(self, use_ml=True):
        self.use_ml = use_ml and ML_AVAILABLE
        self.clf = None
        self.is_trained = False

        if self.use_ml:
            self.clf = RandomForestClassifier(n_estimators=100, random_state=42)

    def load_model(self, filepath=MODEL_FILE):
        if not self.use_ml:
            return False

        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found. Operating without ML.")
            return False

        with open(filepath, 'rb') as f:
            self.clf = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}.")
        return True

    def train_dummy(self):
        """Trains a dummy model for testing if no model file exists."""
        if not self.use_ml:
            return False

        print("Training dummy model for demonstration...")
        X = np.random.rand(200, 14) * 2000
        y = np.random.randint(0, 4, 200)
        self.clf.fit(X, y)
        self.is_trained = True
        return True

    def predict_proba(self, features):
        """
        Predicts probabilities for each class.
        Returns: 1D array of probabilities, or None if ML is disabled/untrained.
        """
        if not self.use_ml or not self.is_trained:
            return None

        X = features.reshape(1, -1)
        proba = self.clf.predict_proba(X)
        return proba[0]
