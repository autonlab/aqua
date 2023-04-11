import os, json, copy, logging
import torch 
import numpy as np
import pandas as pd

from aqua.models import TrainAqModel
from aqua.utils import get_optimizer
from aqua.configs import main_config, data_configs, model_configs
import aqua.data.preset_dataloaders as presets
from aqua.models.base_architectures import ConvNet, BertNet, TabularNet, TimeSeriesNet

from sklearn.metrics import f1_score


# TODO : (vedant) : remove this because it is redundant: config.json already has this
# data_dict = {
#     'cifar10' : load_cifar(cfg['data'])
# }

model_dict = {
    'image': ConvNet,
    'text': BertNet,
    'timeseries': TimeSeriesNet,
    'tabular': TabularNet
}

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



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
    logger.debug("Label issues detected")


    # Clean the data 
    data_aq_clean = copy.deepcopy(data_aq)
    data_aq_clean.clean_data(label_issues)
    logger.debug("Data cleaned using detected label issues")

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
    logger.debug("Base model trained on cleaned data")

    print(f"Cleaning method: {method}, Uncleaned Model's F1 Score:", round(f1_score(noisy_test_labels, data_aq_test.labels, average='weighted'), 6), "Cleaned Model's F1 Score:", round(f1_score(clean_test_labels, data_aq_test.labels, average='weighted'), 6), file=file)

    return label_issues



def generate_report(file=None):
    print("Generating report... \n\n", file=file)

    print("Experiment 1: \n", file=file)

    for dataset in main_config['datasets']:
        modality = data_configs[dataset]['modality']
        architecture = main_config['architecture'][modality]

        print(f"Modality: {modality},      Base Model's Architecture: {architecture},         Dataset: {dataset}\n", file=file)
        data_results_dict = {}

        data_aq, data_aq_test = getattr(presets, f'load_{dataset}')(data_configs[dataset])

        for method in main_config['methods']:
            logging.info(f"\n\nRunning {method} on dataset {dataset} with a base architecture {architecture}")
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

