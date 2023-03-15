import os, json
import torch 
import numpy as np
import pandas as pd

from aqua.models import TrainAqModel, TestAqModel
from aqua.configs import main_config, data_configs
import aqua.data.preset_dataloaders as presets

from sklearn.metrics import f1_score


# TODO : (vedant) : remove this because it is redundant: config.json already has this
# data_dict = {
#     'cifar10' : load_cifar(cfg['data'])
# }

def run_experiment_1(data_aq, 
                     data_aq_test, 
                     architecture,
                     modality,
                     dataset, 
                     method,
                     device='cuda:0',
                     file=None):
    # Refer to doc for an exact defintion of experiment 1

    # Define two different base models:
    # 1. B_1 that will be trained on noisy data
    # 2. B_2 that will be trained on data cleaned by a label cleaning method
    noisy_base_model = TrainAqModel(modality, architecture, 'noisy', dataset, device)
    clean_base_model = TrainAqModel(modality, architecture, 'noisy', dataset, device)

    # Define the cleaning method
    cleaning_method = TrainAqModel(modality, architecture, method, dataset, device)
    clean_data, clean_labels, label_issues = cleaning_method.get_cleaned_labels(data_aq.data, data_aq.labels)
    
    # TODO : convert fit_predicts to fits
    noisy_base_model.fit_predict(data_aq.data, data_aq.labels)
    clean_base_model.fit_predict(clean_data, clean_labels)

    noisy_test_labels, clean_test_labels = noisy_base_model.predict(data_aq_test.data), clean_base_model.predict(data_aq_test.data)

    print(f"Cleaning method: {method}, Uncleaned Model's F1 Score:", round(f1_score(noisy_test_labels, data_aq_test.labels, average='weighted'), 6), "Cleaned Model's F1 Score:", round(f1_score(clean_test_labels, data_aq_test.labels, average='weighted'), 6), file=file)

    return label_issues

def generate_report(file=None):
    print("Generating report... \n\n", file=file)

    print("Experiment 1: \n", file=file)

    for dataset in main_config['datasets']:
        modality = data_configs[dataset]['modality']
        architecture = main_config['architecture'][modality]

        print(f"Modality: {modality},      Base Model's Architecture: {architecture}\n", file=file)
        data_results_dict = {}

        # TODO : ensure every data loading module returns a test dataset. Q : how to deal with datasets that dont have a test dataset
        data_aq, data_aq_test = getattr(presets, f'load_{dataset}')(data_configs[dataset])

        for method in main_config['methods']:
            label_issues = run_experiment_1(data_aq, 
                                            data_aq_test, 
                                            architecture,
                                            modality, 
                                            dataset, 
                                            method,
                                            device=main_config['device'],
                                            file=file)
            
            data_results_dict[method] = label_issues.tolist()

        # Check if human annotated labels are available
        
        data_results_df = pd.DataFrame.from_dict(data_results_dict)
        data_results_df.to_csv(f'results/{dataset}_label_issues.csv')

