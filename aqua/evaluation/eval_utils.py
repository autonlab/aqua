import json, os
from sklearn.model_selection import ParameterGrid as param
from itertools import product

dirname = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dirname, 'grid_search_params.json'), 'r') as f:
    param_dict = json.load(f)

def get_hyperparam_dict(base_architecture_name: str, 
                        cleaning_method_name: str) -> list:
    base_arch_params, clean_method_params = [], []

    if base_architecture_name:
        base_arch_params = list(param(param_dict[base_architecture_name]))
    if cleaning_method_name:
        clean_method_params = list(param(param_dict[cleaning_method_name]))

    if len(base_arch_params) != 0 and len(clean_method_params) == 0:
        return [(p, {}) for p in base_arch_params]
    elif len(base_arch_params) == 0 and len(clean_method_params) != 0:
        return [({}, p) for p in base_arch_params]
    else:
        return [p for p in product(base_arch_params, clean_method_params)]
