# MIT License

# Copyright (c) 2023 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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