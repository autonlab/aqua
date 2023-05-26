import os, json, copy, logging, sys
from dill import dump, load
import torch 
import numpy as np
import pandas as pd
import traceback, warnings

from aqua.models import TrainAqModel
from aqua.utils import get_optimizer
from aqua.configs import main_config, data_configs, model_configs
import aqua.data.preset_dataloaders as presets
from aqua.models.base_architectures import ConvNet, BertNet, TabularNet, TimeSeriesNet
from aqua.data.process_data import Aqdata

from aqua.metrics import *

from pprint import pformat

import pdb


model_dict = {
    'image': ConvNet,
    'text': BertNet,
    'timeseries': TimeSeriesNet,
    'tabular': TabularNet
}

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

def train_base_model(data_aq: Aqdata,
                     architecture:str,
                     modality: str,
                     dataset: str,
                     device: str='cuda:0'):
    model_path = os.path.join(main_config['results_dir'], 'model_factory')
    os.makedirs(model_path, exist_ok=True)

    model_name = os.path.join(model_path, f"base_model_{architecture}_randomseed_{main_config['random_seed']}.pkl")
    if os.path.exists(model_name):
        with open(model_name, 'rb') as f:
            noisy_base_model = load(f)
        return noisy_base_model

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

    with open(model_name, 'wb') as f:
        dump(noisy_base_model.model, f)

    return noisy_base_model.model


def run_experiment_1(data_aq: Aqdata, 
                     data_aq_test: Aqdata, 
                     architecture: str,
                     modality: str,
                     dataset: str, 
                     method: str,
                     device: str='cuda:0',
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
                     noise_type:str,
                     device: str='cuda:0',
                     timestring: str=None,
                     file=None) -> dict:
    
    # This corresponds to exp 3 in the docs
    
    # Define the cleaning method that will detect label issues

    print(f"Cleaning Method: {method}", file=file)
    logging.debug(f"Number of labels corrupted: {data_aq.noise_or_not.sum()}")

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

    if '0.0' not in noise_type:
        print(f"F1 Score for noise {noise_type}: ", f1_score(label_issues, data_aq.noise_or_not), file=file)
    
    if timestring is not None:
        with open(os.path.join(main_config['results_dir'], f'results/results_{timestring}/cleaning_model_{method}_noiserate_{noise_type}.pkl'), 'wb') as mf:
            dump(cleaning_base_model, mf)

        # TODO : (vedant) save a model base classification model trained on base noisy data

    #del noisy_data_aq
    del cleaning_optim
    del cleaning_base_model

    # Train a new model on cleaned data then generate class predictions on noisy data
    data_aq_clean = copy.deepcopy(data_aq)
    data_aq_clean.clean_data(label_issues) 

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
    noisy_train_preds = clean_base_model.predict(copy.deepcopy(data_aq))

    del data_aq_clean
    del clean_optim
    del clean_base_model

    return label_issues, noisy_train_preds


def run_experiment_3(data_aq: Aqdata,
                     data_aq_test: Aqdata,
                     architecture: str,
                     modality: str,
                     dataset: str,
                     device:str ="cuda:0",
                     timestring:str = None,
                     file=None) -> None:
    
    # Train a model on base training data
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

    print(f"Uncleaned Model's F1 Score: {f1_score(noisy_test_labels, data_aq_test.labels)}", file=file)

    if timestring is not None:
        with open(os.path.join(main_config['results_dir'], f'results/results_{timestring}/noisy_model.pkl'), 'wb') as file:
            dump(noisy_base_model, file)

    del noisy_base_model
    del noisy_optim


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

        if experiment_num == 3:
            run_experiment_3(data_aq,
                            data_aq_test,
                            architecture,
                            modality,
                            dataset,
                            device=main_config['device'],
                            timestring=timestring,
                            file=file)
            
            print(42*"=", file=file)
            continue
        
        base_trained_model = train_base_model(data_aq, architecture, modality, dataset, main_config['device'])
        batch_size = model_configs['base'][main_config['architecture'][modality]]['batch_size']
        noise_kwargs = {'model': base_trained_model, 'batch_size': batch_size, 'data': data_aq}

        for noise in main_config["noise"]:
            noise_type = noise
            if noise == 'no-noise':
                noise_rate = [0.0]
                noise_type = 'uniform'
            elif noise in ['asymmetric', 'uniform', 'instance_dependent']:
                noise_rate = main_config['noise_rates']
            elif noise in ['class_dependent']:
                noise_rate = [0.0]
            else:
                warnings.warn(f'Noise type {noise} not valid', RuntimeWarning)
                continue
            
            for nr in noise_rate:
                noisy_aq_data = copy.deepcopy(data_aq)
                noisy_aq_data.noise_type = noise_type

                # TODO : (vedant) : this is a little messy, try to clean this up?
                if noise_type in ['uniform', 'asymmetric', 'instance_dependent']: noise_name = f"{noise}_{nr}"
                else: noise_name = f"{noise}"

                if noise_type in ['uniform', 'asymmetric']: noisy_aq_data.noise_rate = nr
                else: noisy_aq_data.noise_rate = [nr, noise_kwargs]

                if noise_name not in data_results_dict:
                    data_results_dict[noise_name] = {}

                data_results_dict[noise_name]['is_injected_noise'] = noisy_aq_data.noise_or_not
                data_results_dict[noise_name]['noisy_label'] = noisy_aq_data.labels

                for method in main_config['methods']:
                    logging.info(f"Running {method} on dataset {dataset} with a base architecture {architecture}")
                    curr_model_config = pformat(model_configs['base'][architecture])
                    curr_cleaning_config = pformat(model_configs['cleaning'][method])
                    logging.info(f"Config for base architecture {architecture}: \n{curr_model_config}\n")
                    logging.info(f"Config for cleaning method {method}: \n{curr_cleaning_config}\n")

                    if experiment_num == 1:
                        try:
                            label_issues = run_experiment_1(noisy_aq_data, 
                                                            data_aq_test, 
                                                            architecture,
                                                            modality, 
                                                            dataset, 
                                                            method,
                                                            device=main_config['device'],
                                                            file=file)
                            data_results_dict[method] = label_issues.tolist()
                        
                        except Exception:
                            logging.info(f"{method} on dataset {dataset} with a base architecture {architecture} failed to run. Stack trace:")
                            logging.exception("Exception")
                            continue

                    elif experiment_num == 2:
                        try:
                            label_issues, noisy_preds = run_experiment_2(noisy_aq_data, 
                                                                            architecture,
                                                                            modality, 
                                                                            dataset, 
                                                                            method,
                                                                            noise_type=noise_name,
                                                                            device=main_config['device'],
                                                                            timestring=timestring,
                                                                            file=file)

                            data_results_dict[noise_name][f'label_issues_{method}'] = label_issues
                            data_results_dict[noise_name][f'preds_cleaned_{method}'] = noisy_preds
                        
                        except Exception:
                            logging.info(f"{method} on dataset {dataset} with a base architecture {architecture} failed to run. Stack trace:")
                            logging.exception("Exception")
                            continue
                    
                print(42*"=", file=file)
                # Check if human annotated labels are available

        if experiment_num in [1, 2]:
            if data_aq.corrected_labels is not None:
                data_results_dict['Human Annotated Labels'] = (data_aq.corrected_labels != data_aq.labels).tolist()
        
        if timestring is not None:
            if experiment_num == 1:
                data_results_df = pd.DataFrame.from_dict(data_results_dict)
                data_results_df.to_csv(os.path.join(main_config['results_dir'], f'results/results_{timestring}/{dataset}_label_issues.csv'))
            elif experiment_num == 2:
                for key, value in data_results_dict.items():
                    data_results_df = pd.DataFrame.from_dict(value)
                    data_results_df['observed_labels'] = data_aq.labels.tolist()
                    data_results_df.to_csv(os.path.join(main_config['results_dir'], f'results/results_{timestring}/{dataset}_noisename_{key}_label_issues.csv'))

