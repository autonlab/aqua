import os, json
import torch 
import numpy as np

from aqua.data import load_cifar10_test, load_cifar10_train, load_cifar10H_softlabels, load_cifar10N_softlabels, Aqdata
from aqua.models import TrainAqModel, TestAqModel

from sklearn.metrics import f1_score

def load_cifar(cfg):
    # Load train data
    data_cifar, label_cifar = load_cifar10_train(cfg['cifar10'])
    labels_annot = load_cifar10N_softlabels(os.path.join(cfg['cifar10N'], 'CIFAR-10_human.pt'))

    # Load test data
    data_cifar_test, label_cifar_test = load_cifar10_test(cfg['cifar10'])
    labels_annot_test = load_cifar10H_softlabels(os.path.join(cfg['cifar10H'], 'data/cifar10h-raw.csv'), agreement_threshold=0.9)

    return Aqdata(data_cifar, label_cifar, labels_annot), Aqdata(data_cifar_test, label_cifar_test, labels_annot_test)


# Load config 
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')) as wb:
    cfg = json.load(wb)


data_dict = {
    'cifar10' : load_cifar(cfg['data'])
}

def run_experiment_1(data_aq, 
                     data_aq_test, 
                     modality,
                     dataset, 
                     method,
                     device='cuda:5',
                     file=None):
    # Refer to doc for an exact defintion of experiment 1

    # Define two different base models:
    # 1. B_1 that will be trained on noisy data
    # 2. B_2 that will be trained on data cleaned by a label cleaning method
    noisy_base_model = TrainAqModel(modality, 'noisy', dataset, device)
    clean_base_model = TrainAqModel(modality, 'noisy', dataset, device)

    # Define the cleaning method
    cleaning_method = TrainAqModel(modality, method, dataset, device)
    clean_data, clean_labels = cleaning_method.get_cleaned_labels(data_aq.data, data_aq.labels)

    # TODO : convert fit_predicts to fits
    noisy_base_model.fit_predict(data_aq.data, data_aq.labels)
    clean_base_model.fit_predict(clean_data, clean_labels)

    noisy_test_labels, clean_test_labels = noisy_base_model.predict(data_aq_test.data), clean_base_model.predict(data_aq_test.data)

    print(f"Cleaning method: {method}, Uncleaned Model's F1 Score:", round(f1_score(noisy_test_labels, data_aq_test.labels, average='weighted'), 6), "Cleaned Model's F1 Score:", round(f1_score(clean_test_labels, data_aq_test.labels, average='weighted'), 6), file=file)

def generate_report(file=None):
    print("Generating report... \n\n", file=file)

    print("Experiment 1: \n", file=file)

    for dataset in ['cifar10']:
        modality = None
        if dataset in ['cifar10', 'noisycxt']:
            modality = 'image'

        # TODO : ensure every data loading module returns a test dataset. Q : how to deal with datasets that dont have a test dataset
        data_aq, data_aq_test = data_dict[dataset]

        for method in ['cleanlab']:
            run_experiment_1(data_aq, data_aq_test, modality, dataset, method)

