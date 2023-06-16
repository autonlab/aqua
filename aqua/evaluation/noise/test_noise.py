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

# Script to test noise injection

import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')
import numpy as np
from aqua.evaluation.noise.uniform_noise import UniformNoise
from aqua.evaluation.noise.dissenting_label_noise import DissentingLabelNoise
from aqua.evaluation.noise.dissenting_worker_noise import DissentingWorkerNoise
from aqua.evaluation.noise.feature_noise import FeatureNoise
from aqua.configs import data_configs
from aqua.data.preset_dataloaders import load_cifar10, load_mitbih

def create_synthetic_data(n=10000, k=5, f=64):
    """Generate synthetic data

    n: Number of data point 
    k: Number of classes
    f: Number of features
    """
    X = np.random.random((n, 3, f, f))
    y = np.argmax(np.random.multinomial(n=1, size=n, pvals=k*[1/k]), axis=1)
    annotator_y = np.random.randint(0, k, size=(k, n))
    
    return X, y, annotator_y

def test_synthetic(noise_obj, n=10000, k=5, f=64):
    """Generate synthetic data and check noise injection

    n: Number of data point 
    k: Number of classes
    f: Number of features
    noise_rate: Noise rate for injection
    """
    # Create synthetic data
    X, y, annotator_y = create_synthetic_data(n, k, f)

    # Noise injection
    if noise_obj.multi_annotator:
        noisy_X, noisy_y = noise_obj.add_noise(X=X, y=y, annotator_y=annotator_y)
    else:
        noisy_X, noisy_y = noise_obj.add_noise(X=X, y=y)

    print("\n---- Synthetic ---")
    print(f'Test 1 (Features should not change): {np.allclose(X, noisy_X)}')
    
    estimated_noise_rate = noise_obj.estimate_noise_rate(y=y, noisy_y=noisy_y)
    print(f'Estimated noise rate: {estimated_noise_rate}')
    print(f'Test 2 (Added and estimate noise rate should be close): {np.abs(estimated_noise_rate - noise_obj.p) < 0.01}')

    if not noise_obj.multi_annotator:
        empirical_noise_transition_matrix = noise_obj.estimate_noise_transition_matrix(y=y.astype(int), noisy_y=noisy_y)
        #print(f'Estimated noise transition matrix:\n {empirical_noise_transition_matrix}')
        print(f'Test 3 (Added and estimated noise transition matrices should be close): {np.allclose(empirical_noise_transition_matrix , noise_obj.noise_transition_matrix, atol=0.05)}')
        
def test_cifar10(noise_obj):
    train_data, test_data = load_cifar10(cfg=data_configs['cifar10'])
    n_classes = data_configs['cifar10']['out_classes']
    X, y, annotator_y = train_data.data, train_data.labels, train_data.annotator_labels

    # Noise injection
    if noise_obj.multi_annotator:
        noisy_X, noisy_y = noise_obj.add_noise(X=X, y=y, annotator_y=annotator_y)
    else:
        noisy_X, noisy_y = noise_obj.add_noise(X=X, y=y)
    
    print("\n---- CIFAR10 ---")
    print(f'Test 1 (Features should not change): {np.allclose(X, noisy_X)}')
    
    estimated_noise_rate = noise_obj.estimate_noise_rate(y=y, noisy_y=noisy_y)
    print(f'Estimated noise rate: {estimated_noise_rate}')
    print(f'Test 2 (Added and estimate noise rate should be close): {np.abs(estimated_noise_rate - noise_obj.p) < 0.01}')

    if not noise_obj.multi_annotator:
        empirical_noise_transition_matrix = noise_obj.estimate_noise_transition_matrix(y=y.astype(int), noisy_y=noisy_y)
        #print(f'Estimated noise transition matrix:\n {empirical_noise_transition_matrix}')
        print(f'Test 3 (Added and estimated noise transition matrices should be close): {np.allclose(empirical_noise_transition_matrix , noise_obj.noise_transition_matrix, atol=0.05)}')
        

def test_synthetic_feature_noise(noise_obj, n=10000, k=5, f=64):

    # Create synthetic data
    X, y, annotator_y = create_synthetic_data(n, k, f)

    # Noise injection
    if noise_obj.multi_annotator:
        noisy_X, noisy_y = noise_obj.add_noise(X=X, y=y, annotator_y=annotator_y)
    else:
        noisy_X, noisy_y = noise_obj.add_noise(X=X, y=y)

    print("\n---- Synthetic ---")
    print(f'Test 1 (Features should change): {not np.allclose(X, noisy_X)}')
    f = X.shape[1] * X.shape[2] * X.shape[3]
    print(f'L1-dist between original features and noisy features: {np.mean(np.linalg.norm(X-noisy_X, ord=1, axis=0)/(f*f))}')

    print(f'Test 2 (Labels should not change): {np.allclose(y, noisy_y)}')


def test_cifar10_feature_noise(noise_obj):

    train_data, test_data = load_cifar10(cfg=data_configs['cifar10'])
    n_classes = data_configs['cifar10']['out_classes']
    X, y, annotator_y = train_data.data, train_data.labels, train_data.annotator_labels

    # Noise injection
    if noise_obj.multi_annotator:
        noisy_X, noisy_y = noise_obj.add_noise(X=X, y=y, annotator_y=annotator_y)
    else:
        noisy_X, noisy_y = noise_obj.add_noise(X=X, y=y)

    print("\n---- CIFAR10 ---")
    print(f'Test 1 (Features should change): {not np.allclose(X, noisy_X)}')
    f = X.shape[1] * X.shape[2] * X.shape[3]
    print(f'L1-dist between original features and noisy features: {np.mean(np.linalg.norm(X-noisy_X, ord=1, axis=0)/(f))}')

    print(f'Test 2 (Labels should not change): {np.allclose(y, noisy_y)}')


def test_mitbih_feature_noise(noise_obj):

    train_data, test_data = load_mitbih(cfg=data_configs['mitbih'])
    X, y, annotator_y = train_data.data, train_data.labels, train_data.annotator_labels

    noisy_X_list, noisy_y_list = [], []
    # Noise injection
    if noise_obj.multi_annotator:
        noisy_X, noisy_y = noise_obj.add_noise(X=X, y=y, annotator_y=annotator_y)
    else:
        noisy_X, noisy_y = noise_obj.add_noise(X=X, y=y)

    print("\n---- MITBIH ---")
    print(f'Test 1 (Features should change): {not np.allclose(X, noisy_X)}')
    f = X.shape[1] * X.shape[2]
    print(f'L1-dist between original features and noisy features: {np.mean(np.linalg.norm(X-noisy_X, ord=1, axis=0)/(f))}')

    print(f'Test 2 (Labels should not change): {np.allclose(y, noisy_y)}')

    return X, y

def main():

    print("\n---- Testing Uniform Noise ---")
    noise_args = {"n_classes":10, "noise_rate":0.2}
    noise_obj = UniformNoise(**noise_args)
    test_synthetic(noise_obj, n=10000, k=10, f=64)
    test_cifar10(noise_obj)

    print("\n---- Testing Dissenting Label Noise ---")
    noise_args = {"n_classes":10, "noise_rate":0.2}
    noise_obj = DissentingLabelNoise(**noise_args)
    test_synthetic(noise_obj, n=10000, k=10, f=64)
    test_cifar10(noise_obj)

    print("\n---- Testing Dissenting Worker Noise ---")
    noise_args = {"n_classes":10, "noise_rate":0.2}
    noise_obj = DissentingWorkerNoise(**noise_args)
    test_synthetic(noise_obj, n=10000, k=10, f=64)
    test_cifar10(noise_obj)

    print("\n---- Testing Image Feature Noise ---")
    noise_args = {"modality":"image"}
    noise_obj = FeatureNoise(**noise_args)
    test_synthetic_feature_noise(noise_obj, n=10000, k=10, f=64)
    test_cifar10_feature_noise(noise_obj)

    print("\n---- Testing Time Series Feature Noise ---")
    noise_args = {"modality":"timeseries", "anomaly_type":"noise"}
    noise_obj = FeatureNoise(**noise_args)
    test_mitbih_feature_noise(noise_obj)

if __name__ == '__main__':
    main()