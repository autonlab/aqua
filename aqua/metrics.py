import numpy as np
import sklearn.metrics as skm

SUPPORTED_METRICS = [
    'f1',
    'weighted_f1',
    'accuracy',
    'precision',
    'recall',
    'error_rate'
]

def f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    return round(skm.f1_score(y_true, y_pred, average='weighted'), 6)

def get_metrics(y_true, y_pred):
    f1 = round(skm.f1_score(y_true, y_pred, average='micro'), 6)
    weighted_f1 = round(skm.f1_score(y_true, y_pred, average='weighted'), 6)
    accuracy = round(skm.accuracy_score(y_true, y_pred), 6)
    precision = round(skm.precision_score(y_true, y_pred, average='micro'), 6)
    recall = round(skm.recall_score(y_true, y_pred, average='micro'), 6)
    #roc_auc = round(skm.roc_auc_score(y_true, y_pred, average='micro'), 6)
    #pr_auc = round(skm.average_precision_score(y_true, y_pred, average='micro'), 6)
    error_rate = round((1 - skm.zero_one_loss(y_true, y_pred)), 6)
    return f1, weighted_f1, accuracy, precision, recall, error_rate