# Script to test noise injection

import sys
sys.path.append('../../')
import numpy as np
from aqua.evaluation.noise.uniform_noise import UniformNoise
from aqua.evaluation.noise.dissenting_label_noise import DissentingLabelNoise
from aqua.evaluation.noise.dissenting_worker_noise import DissentingWorkerNoise
from aqua.configs import data_configs
from aqua.data.preset_dataloaders import load_cifar10


def create_synthetic_data(n=10000, k=5, f=64):
    """Generate synthetic data

    n: Number of data point 
    k: Number of classes
    f: Number of features
    """
    X = np.random.random((n, f))
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

    print("\n---- Testing Dissenting Label Noise ---")
    noise_args = {"n_classes":10, "noise_rate":0.2}
    noise_obj = DissentingWorkerNoise(**noise_args)
    test_synthetic(noise_obj, n=10000, k=10, f=64)
    test_cifar10(noise_obj)

if __name__ == '__main__':
    main()