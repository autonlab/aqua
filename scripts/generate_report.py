import sys, os, warnings, torch, numpy as np, random
sys.path.append('../')
warnings.filterwarnings("ignore")

from aqua.utils import seed_everything

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

if not os.path.exists('results'):
    os.mkdir('results')

with open('results/report.txt', 'w') as f:
    generate_report(f)