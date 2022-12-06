import os, pickle
import numpy as np
import pandas as pd
import torch

# Loads CIFAR 10 train
def load_cifar10_train(data_path):
    data, labels = [], []
    for filename in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']:
        with open(os.path.join(data_path, filename), 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')

            labels.append(data_dict[b'labels'])
            data.append(data_dict[b'data'])

    data, labels =  np.vstack(data), np.hstack(labels)
    data = data.reshape((data.shape[0], 3, 32, 32))
    return data, labels

# Loads CIFAR 10 test
def load_cifar10_test(data_path):
    # Load Cifar 10 data
    with open(data_path, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')

    return data_dict[b'data'], data_dict[b'labels']

def load_cifar10H_softlabels(label_path, agreement_threshold=0.5):
    # Load Cifar 10 soft labels
    labels = pd.read_csv(label_path)
    labels = labels[labels['is_attn_check'] == 0]

    # If more than 50% annotators agree then the true label on cifar 10 is correct else it's incorrect
    # Here correct_guess == 1 represents that label was guessed correctly
    anot_1 = pd.pivot_table(labels, values=['correct_guess'], index='cifar10_test_test_idx', aggfunc=lambda x: 1 if x.mean() >= agreement_threshold else 0)
    anot_1 = anot_1.rename(columns={'correct_guess':'labels'})
    return anot_1.sort_index().reset_index()


def load_cifar10N_softlabels(label_path):
    labels = torch.load(label_path)
    return labels['aggre_label']