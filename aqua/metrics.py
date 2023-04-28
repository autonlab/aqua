import numpy as np
import sklearn.metrics as skm

def f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    return round(skm.f1_score(y_true, y_pred, average='weighted'), 6)