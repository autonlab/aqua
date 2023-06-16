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
