# MIT License

# Copyright (c) 2023 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys, os, warnings, torch, numpy as np, argparse, pandas as pd, json, copy
sys.path.append('../')
warnings.filterwarnings("ignore")
from aqua.utils import seed_everything, config_sanity_checks, get_available_gpus
import torch
from dill import load, dump
import re

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

def get_datacard(results_dict, dataset, architecture, noise_type, random_seed):
    import pdb
    pdb.set_trace()
    if dataset in results_dict:
        if architecture in results_dict[dataset]:
            if noise_type in results_dict[dataset][architecture]:
                if str(random_seed) in results_dict[dataset][architecture][noise_type]:
                    datacard = results_dict[dataset][architecture][noise_type][str(random_seed)]["datacard"]
                    if len(datacard.columns) == 11:
                        return datacard
    return None

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

        del noisy_optim
    
    test_predictions = noisy_base_model.predict(copy.deepcopy(data_aq_test))

    del noisy_base_model

    return test_predictions

def get_results_dict():
    import pdb
    BASE_PATH = "/zfsauton/data/public/vsanil/aqua_results"

    FOLDER_PATTERN = "results_(?P<timestamp>.*)_randomseed_(?P<randomseed>.*)_(?P<basemodel>.*)"
    FILE_PATTERN = "(?P<dataset>.*)_(?P<noisetype>.*)_label_issues.csv"

    results_dict = dict()

    result_dirs = os.listdir(BASE_PATH)
    result_dirs.sort()
    for result_dir in result_dirs:
        re_folder = re.match(FOLDER_PATTERN, result_dir)
        time_stamp = re_folder.group("timestamp")
        random_seed = re_folder.group("randomseed")
        base_model = re_folder.group("basemodel")
        result_dir_path = os.path.join(BASE_PATH, result_dir)
        for filename in os.listdir(result_dir_path):
            re_file = re.match(FILE_PATTERN, filename)
            if not re_file:
                continue
    
            dataset = re_file.group("dataset")
            noise_type = re_file.group("noisetype")
            if dataset not in results_dict:
                results_dict[dataset] = dict()
            if base_model not in results_dict[dataset]:
                if base_model == 'mobilenet-v2':
                    base_model = 'mobilenet_v2'
                results_dict[dataset][base_model] = dict()
            if noise_type not in results_dict[dataset][base_model]:
                results_dict[dataset][base_model][noise_type] = dict()
            if random_seed not in results_dict[dataset][base_model][noise_type]:
                results_dict[dataset][base_model][noise_type][random_seed] = dict()
            data_path = os.path.join(result_dir_path, filename)
            if dataset == 'clothing100k' and base_model == 'mobilenet_v2':
                print(noise_type)
                pdb.set_trace()
            results_dict[dataset][base_model][noise_type][random_seed]["datacard"] = pd.read_csv(data_path, index_col=0)

    pdb.set_trace()
    return results_dict

def main(base_dir, force_retrain=False, train_original=False):
    config_sanity_checks()

    print(f"Setting random seed: {main_config['random_seed']}")
    random_seed = int(main_config['random_seed'])
    seed_everything(random_seed)

    print("Reading results_dict")
    results_dict = get_results_dict()
    print("Done!")

    train_results_dict, test_results_dict = dict(), dict()

    for dataset in main_config['datasets']:
        modality = data_configs[dataset]['modality']
        architecture = main_config['architecture'][modality]
        for noise in main_config['noise']:
            for noise_rate in main_config['noise_rates']:
                noise_type = f"{noise}-{noise_rate}" if (noise != "classdependent") else noise

                datacard = get_datacard(results_dict, dataset, architecture, noise_type, random_seed)

                if datacard is None:
                    print(42*"=")
                    print(f"{dataset}_{architecture}_{noise_type}_{random_seed}")
                    print("Datacard doesn't exist")
                    continue

                for cleaning_method in main_config['methods']:
                    
                    print(42*"=")

                    run_name = f"{dataset}_{architecture}_{noise_type}_{cleaning_method}_{random_seed}"
                    print(run_name)
                    run_file_path = os.path.join(base_dir, run_name+"_test_preds.csv")

                    if os.path.exists(run_file_path) and (not force_retrain):
                        print("Model predictions already exist")
                        continue

                    data_aq, data_aq_test = getattr(presets, f'load_{dataset}')(data_configs[dataset])
                    
                    noise_column = f"label_issues_{cleaning_method}"
                    label_issues = datacard[noise_column]
                    
                    clean_data_aq = copy.deepcopy(data_aq)
                    clean_data_aq.labels = np.array(datacard["noisy_label"], dtype=np.int64)
                    clean_data_aq.clean_data(label_issues)

                    test_preds = train_base_model(clean_data_aq,
                                                    data_aq_test,
                                                    architecture,
                                                    modality,
                                                    dataset,
                                                    device=main_config['device'],
                                                    force_reload=True)
                    
                    test_f1 = f1_score(data_aq_test.labels, test_preds)
                    print(f"F1: {test_f1}")

                    pd.DataFrame(test_preds).to_csv(run_file_path, header=False, index=False)

                print(42*"=")

                cleaning_method = "nocleaning"

                run_name = f"{dataset}_{architecture}_{noise_type}_{cleaning_method}_{random_seed}"
                print(run_name)
                run_file_path = os.path.join(base_dir, run_name+"_test_preds.csv")

                if os.path.exists(run_file_path) and (not force_retrain):
                    print("Model predictions already exist")
                    continue
            
                data_aq, data_aq_test = getattr(presets, f'load_{dataset}')(data_configs[dataset])
                
                clean_data_aq = copy.deepcopy(data_aq)
                clean_data_aq.labels = np.array(datacard["noisy_label"], dtype=np.int64)

                test_preds = train_base_model(clean_data_aq,
                                                data_aq_test,
                                                architecture,
                                                modality,
                                                dataset,
                                                device=main_config['device'],
                                                force_reload=True)
                
                test_f1 = f1_score(data_aq_test.labels, test_preds)
                print(f"F1: {test_f1}")

                pd.DataFrame(test_preds).to_csv(run_file_path, header=False, index=False)


        if train_original:
            print(42*"=")

            cleaning_method = "nocleaning"
            noise_type = "nonoise"

            run_name = f"{dataset}_{architecture}_{noise_type}_{cleaning_method}_{random_seed}"
            print(run_name)
            run_file_path = os.path.join(base_dir, run_name+"_test_preds.csv")

            if os.path.exists(run_file_path) and (not force_retrain):
                print("Model predictions already exist")
                continue
        
            data_aq, data_aq_test = getattr(presets, f'load_{dataset}')(data_configs[dataset])
            
            test_preds = train_base_model(data_aq,
                                            data_aq_test,
                                            architecture,
                                            modality,
                                            dataset,
                                            device=main_config['device'],
                                            force_reload=True)
            
            test_f1 = f1_score(data_aq_test.labels, test_preds)
            print(f"F1: {test_f1}")

            pd.DataFrame(test_preds).to_csv(run_file_path, header=False, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="/zfsauton/data/public/vsanil/aqua/aqua_downstream_models")
    parser.add_argument('--force_retrain', action='store_true', help='Force retraining of models')
    parser.add_argument('--train_original', action='store_true', help='Train on original dataset')
    args = parser.parse_args()
    main(args.base_dir, args.force_retrain, args.train_original)

