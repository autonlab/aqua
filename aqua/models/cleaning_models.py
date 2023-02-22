import os, torch
import numpy as np
import pandas as pd

# AUM imports
from aum import AUMCalculator

# CINCER imports
from torchmetrics import CalibrationError
from scipy.spatial.distance import pdist
from sklearn.utils import check_random_state, Bunch
from sklearn.metrics import precision_recall_fscore_support as prfs
from aqua.models.modules.cincer.negsup.negotiation import get_suspiciousness, find_counterexample

from aqua.data import Aqdata, TestAqdata
from torch.utils.data import DataLoader

"""
Simple wrapper around a base ML model for the 
AUM cleaning model.
"""
class AUM:
    def __init__(self, model):
        self.model = model
        self.dthr = None
        self.orig_dim = self.model.output_dim

        self.model.reinit_model(self.model.model_type, self.model.output_dim+1)


    def _fit_get_aum(self, thr_inds, alpha=0.99):
        self._aum_calculator = AUMCalculator(os.getcwd())
        train_metrics = self.model.get_training_metrics()
        for i in range(len(train_metrics['output'])):
            self._aum_calculator.update(train_metrics['output'][i],
                                        train_metrics['target'][i],
                                        train_metrics['sample_id'][i])
        self._aum_calculator.finalize()
        aum_file = pd.read_csv(os.path.join(os.getcwd(), 'aum_values.csv'))
        thresh = np.percentile(aum_file.iloc[thr_inds]['aum'].values, 99)
        #d_thresh = aum_file.iloc[thr_inds]['aum'].values
        mask = np.array([True]*aum_file.shape[0])  # Selects train indices only, discards THR indices
        mask[thr_inds] = False

        return np.array(aum_file.index)[mask][aum_file['aum'].values[mask] < thresh]
        

    def find_label_issues(self, data, labels):
        # Refer to paper for training strategy

        # Randomly assign N/(c+1) data as the (c+1)th class
        label_val = np.unique(labels).max()+1
        N, c = data.shape[0], np.unique(labels).shape[0]+1
        
        rand_inds = np.random.randint(0, N, size=2*(N//c))

        # Pass 1
        labels_step_1 = labels.copy()
        labels_step_1[rand_inds[:(N//c)]] = label_val
        self.fit(data, labels_step_1)
        incorrect_labels_idx = self._fit_get_aum(rand_inds[:(N//c)])
        #print(incorrect_labels_idx.shape)

        # Pass 2
        self.model.reinit_model(self.model.model_type, self.model.output_dim)
        labels_step_2 = labels.copy()
        labels_step_2[rand_inds[(N//c):]] = label_val
        self.fit(data, labels_step_2)
        incorrect_labels_idx_thresh = self._fit_get_aum(rand_inds[(N//c):])
        total_incorrect_labels = np.union1d(incorrect_labels_idx, incorrect_labels_idx_thresh)
        #print(incorrect_labels_idx_thresh.shape)
        #print(total_incorrect_labels.shape)

        mask = np.array([False]*data.shape[0])  # Selects train indices only, discards THR indices
        mask[total_incorrect_labels] = True

        # Re-instantiate the model with correct number of output neurons
        self.model.reinit_model(self.model.model_type, self.orig_dim)
        return mask
    
    def fit(self, data, labels):        
        return self.model.fit(data, labels, 
                              lr_tune=True,
                              early_stop=True)

    def predict(self, data):
        return self.model.predict(data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)


class CINCER:
    def __init__(self, model):
        self.model = model

    def _prf(self, p, phat):
        ecefunc = CalibrationError(n_bins=3, task='binary')
        """Computes precision, recall, F1."""
        y, yhat = p, np.argmax(phat, axis=1)
        pr, rc, f1, _ = prfs(y, yhat, average='weighted')
        # expected_calibration_error
        log_pred = torch.log(torch.from_numpy(np.max(phat, axis=1)))
        #label = tf.cast(y == yhat, dtype=tf.bool)
        label = torch.from_numpy(y == yhat).type(dtype=torch.bool)
        ecefunc.update(log_pred, label)
        ece = ecefunc.compute()
        return pr, rc, f1, ece.numpy()

    def _negotiate(self, data, labels,
                    kn, inds,
                    inspector="random",
                    threshold=0,
                    rng=None,
                    no_ce=True,
                    negotiator='random',
                    nfisher_radius = 0.1,
                    return_suspiciousness=True):

        if_config = {}
        rng = check_random_state(rng)

        radius = None
        if negotiator == 'nearest_fisher':
            dist = pdist(data.ravel().reshape(data.shape[0], -1))
            max_dist = np.max(dist)
            assert 0 < nfisher_radius < 1
            radius = max_dist * nfisher_radius


        pr, rc, f1, ece = self._prf(labels[kn], self.model.predict_proba(data[kn]))

        # Negotiate

        trace = pd.DataFrame()
        stat = Bunch(n_queried=0,
                    n_mistakes_seen=0,
                    n_cleaned=0,
                    n_cleaned_ce=0,
                    n_cleaned_ex=0,
                    precision=pr,
                    recall=rc,
                    f1=f1,
                    ece=ece,
                    zs_value=0,
                    noisy_ce=0,
                    suspiciousnesses=0,
                    case1=0,
                    case2=0,
                    case3=0,
                    case4=0,
                    case5=0,
                    case6=0,
                    case7=0,
                    case8=0,
                    case9=0,
                    case10=0,
                    case11=0,
                    case12=0,
                    case13=0,
                    case14=0,
                    ce_pr_at_5=np.nan,
                    ce_pr_at_10=np.nan,
                    ce_pr_at_25=np.nan)

        trace = trace.append(stat, ignore_index=True)

        mistake_inds = []
        nlabels = np.unique(labels).shape[0]
        for idx in range(inds.shape[0]):
            #print(idx)
            i = inds[idx]
            kn = np.append(kn, [i])
            if inspector == 'random':
                suspiciousness = None
                suspicious = rng.binomial(1, threshold)
            else:
                suspiciousness = get_suspiciousness(self.model,
                                                    data, labels,
                                                    kn, i,
                                                    n_labels=nlabels,
                                                    inspector=inspector)
                suspicious = suspiciousness > threshold
            stat.suspiciousnesses = suspiciousness

            # stat.n_mistakes_seen += int(i in noisy)

            stat.ce_pr_at_5 = np.nan
            stat.ce_pr_at_10 = np.nan
            stat.ce_pr_at_25 = np.nan
            stat.noisy_ce, stat.zs_value = 0, 0

            candidates = []
            if suspicious:
                in_shape = (1,) + data.shape[1:]
                xi = data[i].reshape(in_shape)
                phati = self.model.predict_proba(xi)
                yhati = np.argmax(phati, axis=1)[0]

                # Identify examples to be cleaned
                if no_ce or return_suspiciousness:
                    if yhati != np.argmax(labels[i]):
                        stat.n_queried += int(suspicious)
                        candidates = [i]
                else:
                    # user and machine don't agree
                    if yhati != np.argmax(labels[i]):
                        stat.n_queried += int(suspicious)
                        j, stat.zs_value, ordered_candidates = find_counterexample(
                                                                self.model,
                                                                data,
                                                                kn, i,
                                                                negotiator,
                                                                if_config,
                                                                radius,
                                                                rng=rng)
                        assert j in kn and j != i
                        if 'ce_removal' == negotiator:
                            candidates = [i]
                        else:
                            candidates = [i, j]

                #raise KeyboardInterrupt
            mistake_inds.extend(candidates)

        return mistake_inds
        

    def find_label_issues(self, data, labels):
        # Two splits of data to discover label issues
        N = data.shape[0]
        rand_inds = np.random.randint(0, N, size=N)
        
        # Pass 1 
        noisy_inds, test_inds = rand_inds[:N//2], rand_inds[N//2:]
        data_train, labels_train = data.copy()[noisy_inds], labels.copy()[noisy_inds]
        self.fit(data_train, labels_train)
        #data_test, labels_test = data.copy()[test_inds], labels.copy()[test_inds]
        te_lbl_issue_inds = self._negotiate(data.copy(), labels.copy(), 
                                            noisy_inds, test_inds,
                                            inspector="margin")
        
        # Pass 2 
        self.model.reinit_model(self.model.model_type, self.model.output_dim)
        data_train, labels_train = data.copy()[test_inds], labels.copy()[test_inds]
        self.fit(data_train, labels_train)
        #data_test, labels_test = data.copy()[test_inds], labels.copy()[test_inds]
        te_lbl_issue_inds += self._negotiate(data.copy(), labels.copy(), 
                                            test_inds, noisy_inds,
                                            inspector="margin")


        mask = np.array([False]*data.shape[0])
        mask[te_lbl_issue_inds] = True

        return mask

    def fit(self, data, labels):
        return self.model.fit(data, labels,
                              lr_tune=True,
                              early_stop=True)

    def predict(self, data):
        return self.model.predict(data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)