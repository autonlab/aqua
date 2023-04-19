import sys, os, warnings, torch, numpy as np, random, logging
sys.path.append('../')
warnings.filterwarnings("ignore")
from aqua.utils import seed_everything
from aqua.configs import main_config
import datetime

# SET ALL ENVIRONMENT VARIABLES HERE
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from aqua.report import generate_report

def main():
    print(f"Setting random seed: {main_config['random_seed']}")
    seed_everything(int(main_config['random_seed']))

    timestring = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists(main_config['results_dir']):
        os.makedirs(main_config['results_dir'], 'results')
    os.makedirs(os.path.join(main_config['results_dir'], f'results/results_{timestring}'))

    # Logging
    logging.basicConfig(
        format='%(message)s',
        handlers=[
            #logging.StreamHandler(sys),
            logging.FileHandler(os.path.join(os.path.join(main_config['results_dir'], f'results/results_{timestring}', 'run_info.log')))
        ],
        level=logging.INFO
    )

    with open(os.path.join(main_config['results_dir'], f'results/results_{timestring}/report.txt'), 'w') as f:
        generate_report(timestring, f)

if __name__ == '__main__':
    main()