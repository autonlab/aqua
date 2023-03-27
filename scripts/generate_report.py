import sys, os, warnings
sys.path.append('../')
warnings.filterwarnings("ignore")

from aqua.utils import seed_everything

seed_everything(42)

# SET ALL ENVIRONMENT VARIABLES HERE
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from aqua.report import generate_report

if not os.path.exists('results'):
    os.mkdir('results')

with open('results/report.txt', 'w') as f:
    generate_report(f)