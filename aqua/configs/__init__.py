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

import os, warnings
import json

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'main_config.json')) as __wb:
    main_config = json.load(__wb)

data_configs = {}
model_configs = {'cleaning':{}, 'base':{}}

# Load all dataset configs
__data_keys = ['train', 'out_classes', 'val', 'test', 'modality', 'noise_type', 'noise_rate'] # TODO (vedant, mononito) : what else do we want added to dataset config?
for __filename in os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')):
    if __filename.endswith('.json'):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets', __filename)) as __wb:
            __single_data_config = json.load(__wb)

        if all(k in __single_data_config for k in __data_keys):
            data_configs[__filename.replace('.json','')] = __single_data_config
        else:
            __missing_keys = [d for d in __data_keys if d not in __single_data_config]
            warnings.warn(f'Dataset config: {__filename} detected with missing keys. Following keys are missing: {__missing_keys}. This dataset has not been loaded')

# Load all cleaning model configs
for __filename in os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "cleaning")):
     if __filename.endswith('.json'):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'cleaning', __filename)) as __wb:
            __single_model_config = json.load(__wb)
        
        model_configs['cleaning'][__filename.replace('.json', '')] = __single_model_config

# Load all base model configs
for __filename in os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "base")):
    if __filename.endswith('.json'):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'base', __filename)) as __wb:
            __single_model_config = json.load(__wb)
        
        model_configs['base'][__filename.replace('.json', '')] = __single_model_config