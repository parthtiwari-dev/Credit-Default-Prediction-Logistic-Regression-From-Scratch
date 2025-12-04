# src/scratch_logistic.py
import numpy as np
import pandas as pd
from typing import Tuple
import joblib

class ScratchLogistic:
    def __init__(self, lr=0.01, epochs=3000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.w = None
        self.b = None
        self.loss_history = []

    @staticmethod
    def sigmoid(z):
        # stable sigmoid
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def log_loss(y_true, y_prob, eps=1e-12):
        y_prob = np.clip(y_prob, eps, 1 - eps)
        m = y_true.shape[0]
        loss = - (1.0 / m) * (np.sum(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))
        return loss

    def initialize(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        m, n = X.shape
        self.initialize(n)
        for epoch in range(1, self.epochs + 1):
            z = X.dot(self.w) + self.b
            h = self.sigmoid(z)
            if sample_weight is None:
                dw = (1.0 / m) * (X.T @ (h - y))
                db = (1.0 / m) * np.sum(h - y)
            else:
                # apply sample weights (vector of length m)
                sw = sample_weight.reshape(-1, 1)
                dw = (1.0 / m) * (X.T @ ((h - y) * sw.flatten()))
                db = (1.0 / m) * np.sum((h - y) * sw.flatten())
            self.w -= self.lr * dw
            self.b -= self.lr * db
            if epoch % 50 == 0 or epoch == 1:
                loss = self.log_loss(y, h)
                self.loss_history.append((epoch, loss))
                if self.verbose:
                    print(f"Epoch {epoch} loss {loss:.6f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X.dot(self.w) + self.b
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)

    def save(self, path):
        joblib.dump({'w': self.w, 'b': self.b, 'loss_history': self.loss_history}, path)

    def load(self, path):
        data = joblib.load(path)
        self.w = data['w']
        self.b = data['b']
        self.loss_history = data.get('loss_history', [])
        return self
