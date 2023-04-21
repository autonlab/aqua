import sys, os, warnings, torch, numpy as np, random, logging
sys.path.append('../')
warnings.filterwarnings("ignore")
from aqua.utils import seed_everything, config_sanity_checks
from aqua.configs import main_config
import datetime

from aqua.report import generate_report

# SET ALL ENVIRONMENT VARIABLES HERE
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

DEBUG = main_config['debug']

def main():
    config_sanity_checks()
    print(f"Setting random seed: {main_config['random_seed']}")
    seed_everything(int(main_config['random_seed']))

    if not DEBUG:
        timestring = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if not os.path.exists(main_config['results_dir']):
            os.makedirs(main_config['results_dir'], 'results')
        os.makedirs(os.path.join(main_config['results_dir'], f'results/results_{timestring}'))

    # Logging
    if not DEBUG:
        logging.basicConfig(
            format='%(message)s',
            handlers=[
                #logging.StreamHandler(sys),
                logging.FileHandler(os.path.join(os.path.join(main_config['results_dir'], f'results/results_{timestring}', 'run_info.log')))
            ],
            level=logging.INFO
        )

    if not DEBUG:
        with open(os.path.join(main_config['results_dir'], f'results/results_{timestring}/report.txt'), 'w') as f:
            generate_report(timestring, f)
    else:
        generate_report()

if __name__ == '__main__':
    main()
