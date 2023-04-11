import sys, os, warnings, torch, numpy as np, random, logging
sys.path.append('../')
warnings.filterwarnings("ignore")

from aqua.utils import seed_everything

import datetime

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# SET ALL ENVIRONMENT VARIABLES HERE
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from aqua.report import generate_report

timestring = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.makedirs(f'results/results_{timestring}')

# Logging
logging.basicConfig(
    format='%(message)s',
    handlers=[
        #logging.StreamHandler(sys),
        logging.FileHandler(os.path.join(f'results/results_{timestring}', 'run_info.log'))
    ],
    level=logging.INFO
)

with open(f'results/results_{timestring}/report.txt', 'w') as f:
    generate_report(timestring, f)