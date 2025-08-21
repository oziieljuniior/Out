import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class Modelo:
    def __init__(self, C=0.1, max_iter=50000):
        self.model = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=max_iter,
            random_state=42,
            warm_start=False
        )
        self.scaler = StandardScaler(with_mean=False)

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]