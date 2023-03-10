import copy
import numpy as np

# Active Label Cleaning imports
from .InnerEye_DataQuality.InnerEyeDataQuality.selection.data_curation_utils import get_user_specified_selectors, update_trainer_for_simulation
from .InnerEye_DataQuality.InnerEyeDataQuality.selection.selectors.bald import BaldSelector
from .InnerEye_DataQuality.InnerEyeDataQuality.selection.selectors.random_selector import RandomSelector
from .InnerEye_DataQuality.InnerEyeDataQuality.selection.selectors.label_based import LabelBasedDecisionRule, LabelDistributionBasedSampler, PosteriorBasedSelector
from .InnerEye_DataQuality.InnerEyeDataQuality.selection.selectors.base import SampleSelector
from .InnerEye_DataQuality.InnerEyeDataQuality.datasets.label_distribution import LabelDistribution
from .InnerEye_DataQuality.InnerEyeDataQuality.selection.simulation import DataCurationSimulator


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