import os, torch, copy
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

# Active Label Cleaning imports
from aqua.models.modules.active_label_cleaning.InnerEye_DataQuality.InnerEyeDataQuality.selection.data_curation_utils import get_user_specified_selectors, update_trainer_for_simulation
from aqua.models.modules.active_label_cleaning.InnerEye_DataQuality.InnerEyeDataQuality.selection.selectors.bald import BaldSelector
from aqua.models.modules.active_label_cleaning.InnerEye_DataQuality.InnerEyeDataQuality.selection.selectors.random_selector import RandomSelector
from aqua.models.modules.active_label_cleaning.InnerEye_DataQuality.InnerEyeDataQuality.selection.selectors.label_based import LabelBasedDecisionRule, LabelDistributionBasedSampler, PosteriorBasedSelector
from aqua.models.modules.active_label_cleaning.InnerEye_DataQuality.InnerEyeDataQuality.selection.selectors.base import SampleSelector
from aqua.models.modules.active_label_cleaning.InnerEye_DataQuality.InnerEyeDataQuality.datasets.label_distribution import LabelDistribution
from aqua.models.modules.active_label_cleaning.InnerEye_DataQuality.InnerEyeDataQuality.selection.simulation import DataCurationSimulator

# SimiFeat imports
from aqua.models.modules.simifeat.sim_utils import noniterate_detection
import aqua.models.modules.simifeat.global_var as global_var


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



class ActiveLabelCleaning:
    def __init__(self, model, selector='Oracle',
                 temperature=8.0,
                 noise_offset=0.0):
        self.model = model
        self.selector = selector
        self.temperature = temperature
        self.noise_offset = noise_offset
        #if config is None:

    def find_label_issues(self, data, labels):
        n_samples = data.shape[0]
        n_classes = np.unique(labels).shape[0]
        if labels.ndim == 2:
            label_counts = labels
        else:
            label_counts = np.zeros((labels.size, n_classes), dtype=np.int64)
            label_counts[np.arange(labels.size), labels] = 1
            
            count = 0
            while count < 5:
                new_labels = labels.copy()
                np.random.shuffle(new_labels)
                count += 1
                new_label_counts = np.zeros((labels.size, n_classes), dtype=np.int64)
                new_label_counts[np.arange(labels.size), new_labels] = 1

                label_counts += new_label_counts

        #print(label_counts)
        #print(label_counts.sum())
        #raise KeyboardInterrupt

        true_distribution = LabelDistribution(seed=0,
                                               label_counts=label_counts,
                                               temperature=self.temperature,
                                               offset=self.noise_offset)

        sample_selectors = {
            'Random': RandomSelector(n_samples, n_classes, name='Random'),
            'Oracle': PosteriorBasedSelector(true_distribution.distribution, n_samples,
                                            num_classes=n_classes,
                                            name='Oracle',
                                            allow_repeat_samples=True,
                                            decision_rule=LabelBasedDecisionRule.INV)}
        
        selector = sample_selectors[self.selector]
        targets = true_distribution.sample_initial_labels_for_all()
        expected_noise_rate = np.mean(np.argmax(true_distribution.distribution, -1) != targets[:n_samples])
        relabel_budget = int(min(n_samples * expected_noise_rate, n_samples) * 0.35)

        #raise KeyboardInterrupt
        # We need to have a look at relabel budget because this might be dataset specific

        simulator = DataCurationSimulator(initial_labels=copy.deepcopy(label_counts),
                                          label_distribution=copy.deepcopy(true_distribution),
                                          relabel_budget=relabel_budget,
                                          seed=0,
                                          sample_selector=copy.deepcopy(selector)) # TODO (vedant) : change this to global seed

        simulator.run_simulation()
        m1, m2 = simulator.global_stats.get_noisy_and_ambiguous_cases(label_counts)
        for arr in m1:
            print(arr)
            print(float(arr))
            
            #raise KeyboardInterrupt
        #print(m1)
        #print(simulator.global_stats.mislabelled_ambiguous_sample_ids)
        #print(simulator.global_stats.mislabelled_not_ambiguous_sample_ids)


class SimiFeat:
    def __init__(self, model, 
                 noise_type="instance",
                 k=10, 
                 noise_rate=0.4, 
                 seed=1,
                 G=10,
                 cnt=15000,
                 max_iter=400,
                 local=False,
                 loss='fw',
                 num_epoch=1,
                 min_similarity=0.0,
                 Tii_offset=1.0,
                 method='rank1'):
        
        self.model = model
        self.config = global_var.SimiArgs(noise_rate=noise_rate,
                                        noise_type=noise_type,
                                        Tii_offset=Tii_offset,
                                        k=k,
                                        G=G,
                                        seed=seed,
                                        cnt=cnt,
                                        max_iter=max_iter,
                                        local=local,
                                        loss=loss,
                                        num_epoch=num_epoch,
                                        min_similarity=min_similarity,
                                        method=method)


    def find_label_issues(self, data, labels):
        num_classes = np.unique(labels).shape[0]
        N = labels.shape[0]

        self.config.num_classes = num_classes
        dataset = Aqdata(data, labels)
        trainloader = DataLoader(dataset,
                                 batch_size=self.model.batch_size,
                                 shuffle=True,
                                 num_workers=4)
        
        sel_noisy_rec = []
        sel_clean_rec = np.zeros((self.config.num_epoch, N))
        sel_times_rec = np.zeros(N)
        global_var._init()
        global_var.set_value('T_init', None)
        global_var.set_value('p_init', None)
        for epoch in range(self.config.num_epoch):
            record = [[] for _ in range(num_classes)]

            for i_batch, (feature, label, index) in enumerate(trainloader):
                feature, label = feature.to(self.device), label.to(self.device)
                with torch.no_grad():    
                    _, extracted_feat = self.model.model(feature, return_feats=True)
                for i in range(extracted_feat.shape[0]):
                    record[label[i]].append({'feature':extracted_feat[i].detach().cpu(), 'index':index[i]})
                if i_batch > 200:
                    break

            if self.config.method == 'both':
                #rank1 + mv
                self.config.method = 'rank1'
                sel_noisy, sel_clean, sel_idx = noniterate_detection(self.config, record, dataset, 
                                                                    sel_noisy=sel_noisy_rec.copy())

                sel_clean_rec[epoch][np.array(sel_clean)] += 0.5
                sel_times_rec[np.array(sel_idx)] += 0.5

                self.config.method = 'mv'
                sel_noisy, sel_clean, sel_idx = noniterate_detection(self.config, record, dataset, 
                                                                    sel_noisy=sel_noisy_rec.copy())
                sel_clean_rec[epoch][np.array(sel_idx)] += 0.5

                self.config.method = 'both'
                sel_times_rec[np.array(sel_idx)] += 0.5
            else:
                # use one method
                sel_noisy, sel_clean, sel_idx = noniterate_detection(self.config, record, dataset, 
                                                                    sel_noisy=sel_noisy_rec.copy())
                if self.config.num_epoch > 1:
                    sel_clean_rec[epoch][np.array(sel_clean)] = 1
                    sel_times_rec[np.array(sel_idx)] += 1

            aa = np.sum(sel_clean_rec[:epoch + 1], 0) / sel_times_rec
            nan_flag = np.isnan(aa)
            aa[nan_flag] = 0