import os, json, copy, logging, sys
from dill import dumps
import torch 
import numpy as np
import pandas as pd
import traceback

from aqua.models import TrainAqModel
from aqua.utils import get_optimizer
from aqua.configs import main_config, data_configs, model_configs
import aqua.data.preset_dataloaders as presets
from aqua.models.base_architectures import ConvNet, BertNet, TabularNet, TimeSeriesNet
from aqua.data.process_data import Aqdata

from aqua.metrics import *

from pprint import pformat


model_dict = {
    'image': ConvNet,
    'text': BertNet,
    'timeseries': TimeSeriesNet,
    'tabular': TabularNet
}

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

def run_experiment_1(data_aq: Aqdata, 
                     data_aq_test: Aqdata, 
                     architecture: str,
                     modality: str,
                     dataset: str, 
                     method: str,
                     device: str='cuda:0',
                     timestring: str=None,
                     file=None):
    # Refer to doc for an exact defintion of experiment 1

    # Define two different base models:
    # 1. B_1 that will be trained on noisy data
    # 2. B_2 that will be trained on data cleaned by a label cleaning method
    
    # TODO (vedant) : Optimizers need to be defined outside the class otherwise
    # they share training for some reason. This behaviour needs to be verified

    # Define a noisy model and train it
    noisy_base_model = model_dict[modality](main_config['architecture'][modality], 
                         output_dim=data_configs[dataset]['out_classes'],
                         **model_configs['base'][architecture])
    noisy_optim = get_optimizer(noisy_base_model, architecture)
    noisy_base_model = TrainAqModel(noisy_base_model, 
                                    architecture, 
                                    'noisy', 
                                    dataset, 
                                    device,
                                    noisy_optim)
    noisy_base_model.fit_predict(copy.deepcopy(data_aq))
    noisy_test_labels = noisy_base_model.predict(copy.deepcopy(data_aq_test))
    logging.debug("Base model trained on noisy data")

    del noisy_base_model
    del noisy_optim

    # Define the cleaning method that will detect label issues
    extra_dim = 1 if method == 'aum' else 0
    cleaning_base_model = model_dict[modality](main_config['architecture'][modality], 
                         output_dim=data_configs[dataset]['out_classes']+extra_dim,
                         **model_configs['base'][architecture])
    cleaning_optim = get_optimizer(cleaning_base_model, architecture)
    cleaning_base_model = TrainAqModel(cleaning_base_model, 
                                        architecture, 
                                        method, 
                                        dataset, 
                                        device,
                                        cleaning_optim)
    label_issues = cleaning_base_model.find_label_issues(copy.deepcopy(data_aq))
    
    del cleaning_optim
    del cleaning_base_model
    logging.debug("Label issues detected, number of label issues found: ", np.sum(label_issues))


    # Clean the data 
    data_aq_clean = copy.deepcopy(data_aq)
    data_aq_clean.clean_data(label_issues) 
    logging.debug("Data cleaned using detected label issues")

    # Train a new model on cleaned data
    clean_base_model = model_dict[modality](main_config['architecture'][modality], 
                         output_dim=data_configs[dataset]['out_classes'],
                         **model_configs['base'][architecture])
    clean_optim = get_optimizer(clean_base_model, architecture)
    clean_base_model = TrainAqModel(clean_base_model, 
                                    architecture, 
                                    'noisy', 
                                    dataset, 
                                    device,
                                    clean_optim)
    clean_base_model.fit_predict(data_aq_clean)
    clean_test_labels = clean_base_model.predict(copy.deepcopy(data_aq_test))

    del clean_optim
    del clean_base_model
    logging.debug("Base model trained on cleaned data")

    print(f"Cleaning method: {method}, Uncleaned Model's F1 Score: {f1_score(noisy_test_labels, data_aq_test.labels)}", f"Cleaned Model's F1 Score: {f1_score(clean_test_labels, data_aq_test.labels)}\n", file=file)

    return label_issues


def run_experiment_2(data_aq: Aqdata,
                     architecture: str,
                     modality: str,
                     dataset: str, 
                     method: str,
                     device: str='cuda:0',
                     timestring: str=None,
                     file=None) -> dict:
    # Define the cleaning method that will detect label issues
    label_issue_dict = {}
    for noise_rate in [0.1, 0.2]:
        noisy_data_aq = copy.deepcopy(data_aq)
        noisy_data_aq.noise_rate = noise_rate

        logging.debug(f"Number of labels corrupted: {noisy_data_aq.noise_or_not.sum()}")

        extra_dim = 1 if method == 'aum' else 0
        cleaning_base_model = model_dict[modality](main_config['architecture'][modality], 
                            output_dim=data_configs[dataset]['out_classes']+extra_dim,
                            **model_configs['base'][architecture])
        cleaning_optim = get_optimizer(cleaning_base_model, architecture)
        cleaning_base_model = TrainAqModel(cleaning_base_model, 
                                            architecture, 
                                            method, 
                                            dataset, 
                                            device,
                                            cleaning_optim)
        label_issues = cleaning_base_model.find_label_issues(noisy_data_aq)

        label_issue_dict[noise_rate] = label_issues

        print(f"F1 Score for noise_rate {noise_rate}: ", f1_score(label_issues, noisy_data_aq.noise_or_not), file=file)
        
        if timestring is not None:
            with open(os.path.join(main_config['results_dir'], f'results/results_{timestring}/cleaning_model_noiserate_{noise_rate}.pkl'), 'wb') as file:
                dumps(cleaning_base_model, file)

            # TODO : (vedant) save a model base classification model trained on base noisy data

        del noisy_data_aq
        del cleaning_optim
        del cleaning_base_model

    return label_issue_dict


def generate_report(timestring=None, file=None, experiment_num=1):
    print("Generating report... \n\n", file=file)

    print(f"Experiment {experiment_num}: \n", file=file)

    for dataset in main_config['datasets']:
        modality = data_configs[dataset]['modality']
        architecture = main_config['architecture'][modality]

        print(42*"=", file=file)
        print(f"Modality: {modality} | Base Model's Architecture: {architecture} | Dataset: {dataset}", file=file)
        data_results_dict = {}
        data_aq, data_aq_test = getattr(presets, f'load_{dataset}')(data_configs[dataset])

        logging.info(f"Running on dataset: {dataset}")
        dataset_config = pformat(data_configs[dataset])
        logging.info(f"Dataset config: \n{dataset_config} \n\n\n")
        for method in main_config['methods']:
            logging.info(f"Running {method} on dataset {dataset} with a base architecture {architecture}")
            curr_model_config = pformat(model_configs['base'][architecture])
            curr_cleaning_config = pformat(model_configs['cleaning'][method])
            logging.info(f"Config for base architecture {architecture}: \n{curr_model_config}\n")
            logging.info(f"Config for cleaning method {method}: \n{curr_cleaning_config}\n")

            if experiment_num == 1:
                try:
                    label_issues = run_experiment_1(data_aq, 
                                                    data_aq_test, 
                                                    architecture,
                                                    modality, 
                                                    dataset, 
                                                    method,
                                                    device=main_config['device'],
                                                    timestring=None,
                                                    file=file)
                    data_results_dict[method] = label_issues.tolist()
                
                except Exception:
                    logging.info(f"{method} on dataset {dataset} with a base architecture {architecture} failed to run. Stack trace:")
                    logging.exception("Exception")
                    continue

            elif experiment_num == 2:
                try:
                    label_issue_dict = run_experiment_2(data_aq, 
                                                    architecture,
                                                    modality, 
                                                    dataset, 
                                                    method,
                                                    device=main_config['device'],
                                                    timestring=None,
                                                    file=file)
                    
                    # TODO : (vedant) : add predicted class and ground truth label
                    for key, value in label_issue_dict.items():
                        if key not in data_results_dict:
                            data_results_dict[key] = {}

                        data_results_dict[key][method] = value.tolist()
                
                except Exception:
                    logging.info(f"{method} on dataset {dataset} with a base architecture {architecture} failed to run. Stack trace:")
                    logging.exception("Exception")
                    continue
            
        print(42*"=", file=file)
        # Check if human annotated labels are available

        if experiment_num == 1:
            if data_aq.corrected_labels is not None:
                data_results_dict['Human Annotated Labels'] = (data_aq.corrected_labels != data_aq.labels).tolist()
        
        if timestring is not None:
            if experiment_num == 1:
                data_results_df = pd.DataFrame.from_dict(data_results_dict)
                data_results_df.to_csv(os.path.join(main_config['results_dir'], f'results/results_{timestring}/{dataset}_label_issues.csv'))
            else:
                for key, value in data_results_dict.items():
                    data_results_df = pd.DataFrame.from_dict(data_results_dict)
                    data_results_df.to_csv(os.path.join(main_config['results_dir'], f'results/results_{timestring}/{dataset}_noiserate_{key}_label_issues.csv'))

