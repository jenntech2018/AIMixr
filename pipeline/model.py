# src/model.py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np

def train_mood(X, Y):
    # Ensure numeric labels
    Y = Y.astype(int)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X, Y)
    return clf


def evaluate_mood(clf, X_val, Y_val):
    Y_val = Y_val.astype(int)
    preds = clf.predict(X_val).astype(int)
    return f1_score(Y_val, preds, average="macro")
