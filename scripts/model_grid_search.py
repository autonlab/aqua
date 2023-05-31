import sys, os, warnings, torch, numpy as np, argparse, pandas as pd, json, copy
sys.path.append('../')
warnings.filterwarnings("ignore")
from aqua.utils import seed_everything, config_sanity_checks, get_available_gpus
import torch
from dill import load, dump

torch.multiprocessing.set_sharing_strategy('file_system')

from aqua.models import TrainAqModel
from aqua.utils import get_optimizer
from aqua.configs import main_config, data_configs, model_configs
import aqua.data.preset_dataloaders as presets
from aqua.models.base_architectures import ConvNet, BertNet, TabularNet, TimeSeriesNet
from aqua.data.process_data import Aqdata

from aqua.evaluation.eval_utils import get_hyperparam_dict

from aqua.metrics import SUPPORTED_METRICS, get_metrics, f1_score


model_dict = {
    'image': ConvNet,
    'text': BertNet,
    'timeseries': TimeSeriesNet,
    'tabular': TabularNet
}


def train_base_model(data_aq: Aqdata,
                     data_aq_test: Aqdata,
                     architecture:str,
                     modality: str,
                     dataset: str,
                     device: str='cuda:0',
                     force_reload: bool = False):
    model_path = os.path.join(main_config['results_dir'], 'model_factory')
    os.makedirs(model_path, exist_ok=True)

    model_name = os.path.join(model_path, f"base_model_{architecture}_randomseed_{main_config['random_seed']}_{dataset}.pkl")
    if os.path.exists(model_name) and not force_reload:
        with open(model_name, 'rb') as f:
            noisy_base_model = load(f)
    else:
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
    
    train_predictions = noisy_base_model.predict(copy.deepcopy(data_aq))
    test_predictions = noisy_base_model.predict(copy.deepcopy(data_aq_test))

    # Save model
    if force_reload or not os.path.exists(model_name):
        with open(model_name, 'wb') as f:
            dump(noisy_base_model, f)

    return train_predictions, test_predictions


def main(force_reload=False):
    config_sanity_checks()
    print(f"Setting random seed: {main_config['random_seed']}")
    seed_everything(int(main_config['random_seed']))

    train_results_dict, test_results_dict = dict(), dict()

    for dataset in main_config['datasets']:
        modality = data_configs[dataset]['modality']
        architecture = main_config['architecture'][modality]

        hyperparams = get_hyperparam_dict(architecture, None)

        train_results_dict[dataset] = dict()
        test_results_dict[dataset] = dict()


        for hyperparam in hyperparams:

            hyperparam_dict = hyperparam[0]

            for key, value in hyperparam_dict.items():
                model_configs['base'][architecture][key] = value

            print(42*"=")
            data_aq, data_aq_test = getattr(presets, f'load_{dataset}')(data_configs[dataset])

            train_preds, test_preds = train_base_model(data_aq,
                                                        data_aq_test,
                                                        architecture,
                                                        modality,
                                                        dataset,
                                                        device=main_config['device'],
                                                        force_reload=force_reload)
            
            train_f1 = f1_score(data_aq.labels, train_preds)
            test_f1 = f1_score(data_aq_test.labels, test_preds)

            model_hp = ";".join([key+"-"+str(value) for key, value in hyperparam_dict.items()])

            train_results_dict[dataset][model_hp] = train_f1
            test_results_dict[dataset][model_hp] = test_f1

            print(42*"=", "\n\n")

    train_results_df = pd.DataFrame.from_dict(train_results_dict, orient='index')
    test_results_df = pd.DataFrame.from_dict(test_results_dict, orient='index')

    model_path = os.path.join(main_config['results_dir'], 'model_hp_grid')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    train_results_df.to_csv(os.path.join(model_path, f'{architecture}_train_results.csv'))
    test_results_df.to_csv(os.path.join(model_path, f'{architecture}_test_results.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_reload', action='store_true', help='Force reload all models')
    args = parser.parse_args()
    main(args.force_reload)

