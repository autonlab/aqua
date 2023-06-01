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

from aqua.metrics import SUPPORTED_METRICS, get_metrics


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

    train_results_dict, test_results_dict = {m: [] for m in SUPPORTED_METRICS}, {m: [] for m in SUPPORTED_METRICS}

    for dataset in main_config['datasets']:
        modality = data_configs[dataset]['modality']
        architecture = main_config['architecture'][modality]

        print(42*"=")
        data_aq, data_aq_test = getattr(presets, f'load_{dataset}')(data_configs[dataset])

        train_preds, test_preds = train_base_model(data_aq,
                                                    data_aq_test,
                                                    architecture,
                                                    modality,
                                                    dataset,
                                                    device=main_config['device'],
                                                    force_reload=force_reload)
        
        train_metrics = get_metrics(data_aq.labels, train_preds)
        test_metrics = get_metrics(data_aq_test.labels, test_preds)

        for idx, m in enumerate(SUPPORTED_METRICS):
            train_results_dict[m].append(train_metrics[idx])
            test_results_dict[m].append(test_metrics[idx])

        print(42*"=", "\n\n")

    train_results_df = pd.DataFrame.from_dict(train_results_dict)
    test_results_df = pd.DataFrame.from_dict(test_results_dict)

    train_results_df.index = main_config['datasets']
    test_results_df.index = main_config['datasets']
    
    model_path = os.path.join(main_config['results_dir'], 'model_factory')
    train_results_df.to_csv(os.path.join(model_path, 'train_results.csv'))
    test_results_df.to_csv(os.path.join(model_path, 'test_results.csv'))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_reload', action='store_true', help='Force reload all models')
    args = parser.parse_args()
    main(args.force_reload)

