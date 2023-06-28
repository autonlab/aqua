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

import sys, os, warnings, torch, numpy as np, random, logging, json
sys.path.append('../')
warnings.filterwarnings("ignore")
from aqua.utils import seed_everything, config_sanity_checks, get_available_gpus
from aqua.configs import main_config, data_configs, model_configs
import datetime
import torch

from joblib import Parallel, delayed

from aqua.report import generate_report #, check_config
from aqua.evaluation.eval_utils import get_hyperparam_dict

torch.multiprocessing.set_sharing_strategy('file_system')

# SET ALL ENVIRONMENT VARIABLES HERE
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

DEBUG = main_config['debug']

def run_single_config(respath):
    # Logging
    if not DEBUG:
        logging.basicConfig(
            format='%(message)s',
            handlers=[
                #logging.StreamHandler(sys),
                logging.FileHandler(os.path.join(os.path.join(main_config['results_dir'], f'results/results_{respath}', 'run_info.log')))
            ],
            level=logging.INFO
        )

    if not DEBUG:
        with open(os.path.join(main_config['results_dir'], f'results/results_{respath}/report.txt'), 'a') as f:
            generate_report(respath, f, main_config['experiment'])
    else:
        generate_report(experiment_num=main_config['experiment'])

def to_str(value):
    if type(value) == list:
        return ",".join(map(str, value))
    return str(value)

def run_single_grid_config(timestring, gpus, run_id, config):
    base_config, clean_config = config[0], config[1]
    architecture = main_config['architecture'][data_configs[main_config['datasets'][0]]['modality']]

    base_config_name, clean_config_name = '', ''
    if base_config:
        base_config_name = architecture+'_'+'_'.join([key+'_'+to_str(value) for key, value in base_config.items()])
    if clean_config:
        clean_config_name = main_config['methods'][0]+'_'+'_'.join([key+'_'+to_str(value) for key, value in clean_config.items()])

    device = gpus[run_id % len(gpus)]
    random_seed = main_config['random_seed']

    timestring = timestring + f'/{base_config_name}/{clean_config_name}'
    if not DEBUG:
        os.makedirs(os.path.join(main_config['results_dir'], f'results/results_{timestring}'), exist_ok=True)
        logging.basicConfig(
            format='%(message)s',
            handlers=[
                #logging.StreamHandler(sys),
                logging.FileHandler(os.path.join(os.path.join(main_config['results_dir'], f'results/results_{timestring}', 'run_info.log')))
            ],
            level=logging.INFO
        )

    for key, value in base_config.items():
        model_configs['base'][architecture][key] = value

    for key, value in clean_config.items():
        model_configs['cleaning'][main_config['methods'][0]][key] = value

    if not DEBUG:
        with open(os.path.join(main_config['results_dir'], f'results/results_{timestring}/report.txt'), 'a') as f:
            generate_report(timestring, f, main_config['experiment'])
    else:
        generate_report(experiment_num=main_config['experiment'])

def main():
    config_sanity_checks()
    print(f"Setting random seed: {main_config['random_seed']}")
    seed_everything(int(main_config['random_seed']))

    timestring = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    models = '-'.join(list(set([main_config['architecture'][data_configs[d]['modality']] for d in main_config['datasets']])))
    timestring = f"{timestring}_randomseed_{main_config['random_seed']}_{models}"
    if not DEBUG:
        if not os.path.exists(os.path.join(main_config['results_dir'], 'results')):
            os.makedirs(os.path.join(main_config['results_dir'], 'results'))
        os.makedirs(os.path.join(main_config['results_dir'], f'results/results_{timestring}'))

    if not main_config['grid_search']: 
        run_single_config(timestring)
    else:
        avail_gpus = get_available_gpus()
        logging.debug(f"Using GPUs: {avail_gpus}")
        cleaning_methods = main_config["methods"]
        datasets = main_config["datasets"]

        # TODO : (vedant) : this does make the code a little unclean, since we are iterating over dataset and method inside generate_report too. fix??
        for dataset in datasets:
            main_config['datasets'] = [dataset]
            if main_config['experiment'] == 3:
                hyperparams = get_hyperparam_dict(main_config['architecture'][data_configs[dataset]['modality']], None)
                for idx, params in enumerate(hyperparams):
                    run_single_grid_config(timestring, avail_gpus, idx, params)
                continue

            for method in cleaning_methods:
                main_config['methods'] = [method]
                hyperparams = get_hyperparam_dict(main_config['architecture'][data_configs[dataset]['modality']], method)
                
                results = Parallel(n_jobs=1)(delayed(run_single_grid_config)(timestring, avail_gpus, idx, params) for idx, params in enumerate(hyperparams))

if __name__ == '__main__':
    main()
