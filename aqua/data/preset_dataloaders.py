import os, pickle
import numpy as np
import pandas as pd
import torch
import nltk
from tqdm import tqdm
from wfdb import rdrecord, rdann
from sklearn import preprocessing
from scipy.signal import find_peaks
nltk.download('punkt')

from transformers import AutoTokenizer, RobertaTokenizer

from aqua.data.process_data import Aqdata, TestAqdata
from aqua.configs import main_config

# Loads CIFAR 10 train
def __load_cifar10_train(data_path):
    data, labels = [], []
    for filename in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']:
        with open(os.path.join(data_path, filename), 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')

            labels.append(data_dict[b'labels'])
            data.append(data_dict[b'data'])

    data, labels =  np.vstack(data).astype(np.float32), np.hstack(labels).astype(np.int64)
    data = data.reshape((data.shape[0], 3, 32, 32))
    return data, labels

# Loads CIFAR 10 test
def __load_cifar10_test(data_path):
    if os.path.isfile(data_path):
        raise NotADirectoryError("Given path is a file. Parent path to CIFAR10 dataset must be provided with the testing batch. Please refer to CIFAR10's official website on instructions to download: https://www.cs.toronto.edu/~kriz/cifar.html")
    
    data_path = os.path.join(data_path, 'test_batch')
    # Load Cifar 10 data
    with open(data_path, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    
    data, labels = np.array(data_dict[b'data']).astype(np.float32), np.array(data_dict[b'labels']).astype(np.int64)
    data = data.reshape((data.shape[0], 3, 32, 32))
    return data, labels

def __load_cifar10H_softlabels(label_path, agreement_threshold=0.5):
    # Load Cifar 10 soft labels
    labels = pd.read_csv(label_path)
    labels = labels[labels['is_attn_check'] == 0]

    # If more than 50% annotators agree then the true label on cifar 10 is correct else it's incorrect
    # Here correct_guess == 1 represents that label was guessed correctly
    anot_1 = pd.pivot_table(labels, values=['correct_guess'], index='cifar10_test_test_idx', aggfunc=lambda x: 1 if x.mean() >= agreement_threshold else 0)
    anot_1 = anot_1.rename(columns={'correct_guess':'labels'})
    return anot_1.sort_index().reset_index()['labels'].values


def __load_cifar10N_softlabels(label_path):
    labels = torch.load(label_path)
    return labels['aggre_label']


def __load_cxr_train(data_path):
    filedir = '/home/extra_scratch/vsanil/aqua/datasets/cxr'


def __load_imdb(data_path):
    data_dict = {'text':[], 'target':[]}

    # Load pos labels
    pos_path = os.path.join(data_path, 'pos')
    for filename in os.listdir(pos_path):
        filepath = os.path.join(pos_path, filename)
        with open(filepath, 'r') as f:
            data_dict['text'].append(f.read().replace('\n','').strip())
            data_dict['target'].append(1)

    # Load neg labels
    neg_path = os.path.join(data_path, 'neg')
    for filename in os.listdir(neg_path):
        filepath = os.path.join(neg_path, filename)
        with open(filepath, 'r') as f:
            data_dict['text'].append(f.read().replace('\n','').strip())
            data_dict['target'].append(0)

    return pd.DataFrame.from_dict(data_dict)

def __preprocess(text_csv, tokenizer):
    max_len = max([len(text) for text in text_csv.text])
    #texts = [nltk.word_tokenize(text, language='english') for text in text_csv.text]
    texts = [text for text in text_csv.text]
    return [tokenizer(text, 
                      padding='max_length', 
                      max_length=514, 
                      truncation=True, 
                      return_tensors='np',
                      is_split_into_words=False) for text in texts]

def __load_wfdb_waveform(data_path, filelist, input_size, classes):
    labels, data = [], []
    for idx, filename in enumerate(filelist):
        record = rdrecord(os.path.join(data_path, filename), smooth_frames=True)
        signals0 = preprocessing.scale(np.nan_to_num(record.p_signal[:,0])).tolist()
        signals1 = preprocessing.scale(np.nan_to_num(record.p_signal[:,1])).tolist()
        peaks, _ = find_peaks(signals0, distance=150)

        for peak in tqdm(peaks[1:-1], desc=f"File: {idx}/{len(filelist)}"):
            start, end = peak-input_size//2, peak+input_size//2
            ann = rdann(os.path.join(data_path, filename), extension='atr', sampfrom=start, sampto=end, return_label_elements=['symbol'])
            annSymbol = ann.symbol

            # Remove some N which breaks the balance of dataset
            if len(annSymbol) == 1 and (annSymbol[0] in classes) and (annSymbol[0] != "N" or np.random.random()<0.15):
                labels.append(classes.index(annSymbol[0]))
                data.append([signals0[start:end], signals1[start:end]])

    return np.array(data), np.array(labels)

#######################  LOAD FUNCTIONS ########################
def load_cifar10(cfg):
    # Load train data
    data_cifar, label_cifar = __load_cifar10_train(cfg['train']['data'])
    labels_annot = __load_cifar10N_softlabels(cfg['train']['annot_labels'])

    # Load test data
    data_cifar_test, label_cifar_test = __load_cifar10_test(cfg['test']['data'])
    labels_annot_test = __load_cifar10H_softlabels(cfg['test']['annot_labels'], agreement_threshold=0.9)

    return Aqdata(data_cifar, label_cifar, labels_annot), Aqdata(data_cifar_test, label_cifar_test, labels_annot_test)
    

def load_imdb(cfg):
    tokenizer = AutoTokenizer.from_pretrained(main_config['architecture']['text'], model_max_length=514)
    # Load train data
    csv_path = os.path.join(cfg['train']['data'], 'train_csv.csv')
    if not os.path.exists(csv_path):
        train_csv = __load_imdb(cfg['train']['data'])
        train_csv.to_csv(csv_path, index=False)
    else:
        train_csv = pd.read_csv(csv_path)

    train_csv = pd.concat([train_csv[train_csv.target == 0].iloc[:10], train_csv[train_csv.target == 1].iloc[:10]])
    feat_texts, train_labels = __preprocess(train_csv.dropna(), tokenizer=tokenizer), train_csv.dropna().target.values
    train_tokens = np.concatenate([f['input_ids'] for f in feat_texts], axis=0)
    train_attention_masks = np.concatenate([f['attention_mask'] for f in feat_texts], axis=0)

    # Load test data
    csv_path = os.path.join(cfg['test']['data'], 'test_csv.csv')
    if not os.path.exists(csv_path):
        test_csv = __load_imdb(cfg['test']['data'])
        test_csv.to_csv(csv_path, index=False)
    else:
        test_csv = pd.read_csv(csv_path)
    test_csv = pd.concat([test_csv[test_csv.target == 0].iloc[:10], test_csv[test_csv.target == 1].iloc[:10]])
    feat_texts, test_labels = __preprocess(test_csv.dropna(), tokenizer=tokenizer), test_csv.dropna().target.values
    test_tokens = np.concatenate([f['input_ids'] for f in feat_texts], axis=0)
    test_attention_masks = np.concatenate([f['attention_mask'] for f in feat_texts], axis=0)

    return Aqdata(train_tokens, train_labels, attention_mask=train_attention_masks), Aqdata(test_tokens, test_labels, attention_mask=test_attention_masks)

def load_mitbih(cfg):
    classes = ['N','V','/','A','F','~']
    input_size = cfg["input_size"]
    data_path = cfg['train']['data']
    filelist = [f.replace('.hea','') for f in os.listdir(data_path) if f.endswith('.hea')]
    testlist = ['101', '105','114','118', '124', '201', '210' , '217']
    trainlist = [x for x in filelist if x not in testlist]

    # Load training data
    if os.path.exists(os.path.join(data_path, 'train_data.npy')):
        train_data, train_labels = np.load(os.path.join(data_path, 'train_data.npy')),\
                                   np.load(os.path.join(data_path, 'train_labels.npy'))
    else:
        train_data, train_labels = __load_wfdb_waveform(data_path, trainlist, input_size, classes)
        np.save(os.path.join(data_path, 'train_data.npy'), train_data)
        np.save(os.path.join(data_path, 'train_labels.npy'), train_labels)

    # Load testing data
    if os.path.exists(os.path.join(data_path, 'test_data.npy')):
        test_data, test_labels = np.load(os.path.join(data_path, 'test_data.npy')),\
                                 np.load(os.path.join(data_path, 'test_labels.npy'))
    else:
        test_data, test_labels = __load_wfdb_waveform(data_path, testlist, input_size, classes)
        np.save(os.path.join(data_path, 'test_data.npy'), test_data)
        np.save(os.path.join(data_path, 'test_labels.npy'), test_labels)
    
    # Convert dtypes
    train_data, train_labels = train_data.astype(np.float32), train_labels.astype(np.int64)
    test_data, test_labels = test_data.astype(np.float32), test_labels.astype(np.int64)
    return Aqdata(train_data, train_labels), Aqdata(test_data, test_labels)