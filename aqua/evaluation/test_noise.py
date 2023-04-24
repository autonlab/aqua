# Script to test noise injection

import sys
sys.path.append('../../')
import numpy as np
from aqua.evaluation.uniform_noise import UniformNoise

def test_synthetic(n=10000, k=5, f=64, noise_rate=0.2):
    """Generate synthetic data and check noise injection

    n: Number of data point 
    k: Number of classes
    f: Number of features
    noise_rate: Noise rate for injection
    """
    X = np.random.random((n, f))
    y = np.argmax(np.random.multinomial(n=1, size=n, pvals=k*[1/k]), axis=1)

    # Noise injection
    noise_obj = UniformNoise(n_classes=k, noise_rate=noise_rate)
    print('Noise transition matrix:\n', noise_obj.noise_transition_matrix)

    # Add noise
    noisy_X, noisy_y = noise_obj.add_noise(X=X, y=y)

    print(f'Test 1 (Features should not change) {np.allclose(X, noisy_X)}')
    
    estimated_noise_rate = noise_obj.estimate_noise_rate(y=y, noisy_y=noisy_y)
    print(f'Estimated noise rate: {estimated_noise_rate}')
    print(f'Test 2 (Added and estimate noise rate should be close) {np.abs(estimated_noise_rate - noise_rate) < 0.05}')

    empirical_noise_transition_matrix = noise_obj.estimate_noise_transition_matrix(y=y.astype(int), noisy_y=noisy_y)
    print(f'Estimated noise transition matrix:\n {empirical_noise_transition_matrix}')
    print(f'Test 3 (Added and estimated noise transition matrices should be close) {np.allclose(empirical_noise_transition_matrix , noise_obj.noise_transition_matrix, atol=0.05)}')
    
def test_real(noise_rate=0.2):
    # Load a dataset
    pass

def main():
    test_synthetic(n=10000, k=5, f=64, noise_rate=0.2)

if __name__ == '__main__':
    main()