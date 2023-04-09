import torch, copy
import numpy as np

# CINCER imports
import pandas as pd
from torchmetrics import CalibrationError
from scipy.spatial.distance import pdist
from sklearn.utils import check_random_state, Bunch
from sklearn.metrics import precision_recall_fscore_support as prfs
from .negsup.negotiation import get_suspiciousness, find_counterexample


class CINCER:
    def __init__(self, model,
                       optimizer):
        self.model = model
        self.optimizer = optimizer

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

    def _negotiate(self, data_aq,
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
        data, labels = data_aq.data, data_aq.labels
        if negotiator == 'nearest_fisher':
            dist = pdist(data.ravel().reshape(data.shape[0], -1))
            max_dist = np.max(dist)
            assert 0 < nfisher_radius < 1
            radius = max_dist * nfisher_radius


        #pr, rc, f1, ece = self._prf(labels[kn], self.model.predict_proba(data[kn]))
        pr = 0
        rc = 0
        f1 = 0
        ece = 0
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
                                                    data_aq,
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
        

    def find_label_issues(self, data_aq, **kwargs):
        # Save initial states of models before training 
        orig_model = copy.deepcopy(self.model.model)
        orig_optim = type(self.optimizer)(orig_model.parameters(), lr=self.optimizer.defaults['lr'])
        orig_optim.load_state_dict(self.optimizer.state_dict())

        data, labels = data_aq.data, data_aq.labels
        inspector = kwargs["inspector"]
        threshold = kwargs["threshold"]
        rng = kwargs["rng"]
        no_ce = kwargs["no_ce"]
        negotiator = kwargs["negotiator"]
        nfisher_radius = kwargs["nfisher_radius"]
        return_suspiciousness = kwargs["return_suspiciousness"]

        # Two splits of data to discover label issues
        N = data.shape[0]
        rand_inds = np.random.randint(0, N, size=N)
        
        # Pass 1 
        noisy_inds, test_inds = rand_inds[:N//2], rand_inds[N//2:]
        temp_data_aq = copy.deepcopy(data_aq)
        self.fit(temp_data_aq)
        te_lbl_issue_inds = self._negotiate(temp_data_aq, 
                                            noisy_inds, test_inds,
                                            inspector=inspector,
                                            threshold=threshold,
                                            rng=rng,
                                            no_ce=no_ce,
                                            negotiator=negotiator,
                                            nfisher_radius=nfisher_radius,
                                            return_suspiciousness=return_suspiciousness)
        
        # Pass 2 
        temp_data_aq = copy.deepcopy(data_aq)
        self.model.reinit_model(orig_model, orig_optim)
        self.fit(temp_data_aq)
        te_lbl_issue_inds += self._negotiate(temp_data_aq, 
                                            test_inds, noisy_inds,
                                            inspector=inspector,
                                            threshold=threshold,
                                            rng=rng,
                                            no_ce=no_ce,
                                            negotiator=negotiator,
                                            nfisher_radius=nfisher_radius,
                                            return_suspiciousness=return_suspiciousness)


        mask = np.array([False]*data.shape[0])
        mask[te_lbl_issue_inds] = True

        return mask

    def fit(self, data_aq):
        return self.model.fit(data_aq,
                              lr_tune=True,
                              early_stop=True)

    def predict(self, data_aq):
        return self.model.predict(data_aq)

    def predict_proba(self, data_aq):
        return self.model.predict_proba(data_aq)