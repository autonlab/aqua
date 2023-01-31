import sys
sys.path.append('../')

from aqua.report import generate_report

with open('output.txt', 'w') as f:
    generate_report(f)