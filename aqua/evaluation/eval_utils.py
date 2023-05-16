import json, os
from sklearn.model_selection import ParameterGrid as param
from itertools import product

dirname = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dirname, 'grid_search_params.json'), 'r') as f:
    param_dict = json.load(f)

def get_hyperparam_dict(base_architecture_name: str, 
                        cleaning_method_name: str) -> list:
    base_arch_params = list(param(param_dict[base_architecture_name]))
    clean_method_params = list(param(param_dict[cleaning_method_name]))

    return [p for p in product(base_arch_params, clean_method_params)]
