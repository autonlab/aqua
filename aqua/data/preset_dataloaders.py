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

import os, pickle, json, logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import nltk
from sktime.datasets import load_from_tsfile_to_dataframe
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from wfdb import rdrecord, rdann
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from datasets import load_dataset
from scipy.io import arff
nltk.download('punkt')

from transformers import AutoTokenizer, RobertaTokenizer

from aqua.data.process_data import Aqdata, TestAqdata
from aqua.utils import load_single_datapoint
from aqua.configs import main_config, model_configs, data_configs

SENTENCE_TRANSFORMERS = ["all-distilroberta-v1", "all-MiniLM-L6-v2"]

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

def __load_cifar10H_softlabels(label_path):
    # Load Cifar 10 soft labels
    return np.load(label_path)


def __load_cifar10N_softlabels(label_path):
    labels = torch.load(label_path)

     # Shape: (A, N) - A: # Annotators, N: # Data Points
    annotator_labels = np.stack([labels['random_label'+str(i)] for i in range(1, 4)])  

    return labels['aggre_label'], annotator_labels


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

def __preprocess(text_csv, tokenizer, type='train'):
    #max_len = max([len(text) for text in text_csv.text])
    #texts = [nltk.word_tokenize(text, language='english') for text in text_csv.text]
    return [tokenizer(text, 
                      add_special_tokens=True,
                      max_length=512, 
                      pad_to_max_length=True, 
                      return_tensors='np',
                      is_split_into_words=False) for text in tqdm(text_csv.text, desc=f'Tokenizing {type} data')]

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

def __detect_and_process_categorical(df: pd.DataFrame, 
                                     thresh=0.05):
    categorical_df_columns = df.select_dtypes(include=[object]).columns
    for column_name in categorical_df_columns:
        column = df[column_name]
        unique_count = column.unique().shape[0]
        total_count = column.shape[0]
        if unique_count / total_count < thresh:
            le = preprocessing.LabelEncoder()
            df[column_name] = le.fit_transform(column)
    return df


def __load_tensorflow_format_dataset(par_path: str):
    labels, filenames = [], []

    for label in os.listdir(par_path):
        label_dir = os.path.join(par_path, label)
        for files in os.listdir(label_dir):
            labels.append(label)
            filenames.append(os.path.join(label_dir, files))

    return np.array(filenames), np.array(labels)


def __channelwise_minmax_scaler(X_train, X_test):
    channels_train, channels_test = [], []
    for channel_num in range(X_train.shape[1]):
        channel_tr, channel_te = X_train[:, channel_num, :], X_test[:, channel_num, :]
        scaler = MinMaxScaler()
        scaler.fit(channel_tr)

        channels_train.append(scaler.transform(channel_tr)[:, np.newaxis, :])
        channels_test.append(scaler.transform(channel_te)[:, np.newaxis, :])

    return np.concatenate(channels_train, axis=1), np.concatenate(channels_test, axis=1)


#######################  LOAD FUNCTIONS ########################
def load_cifar10(cfg):
    # Load train data
    data_cifar, label_cifar = __load_cifar10_train(cfg['train']['data'])
    labels_annot, annotator_labels = __load_cifar10N_softlabels(cfg['train']['annot_labels'])
    
    logging.info(f"Total number of human annotated label issues: {(labels_annot != label_cifar).sum()}")
    # Load test data
    data_cifar_test, label_cifar_test = __load_cifar10_test(cfg['test']['data'])
    labels_annot_test = __load_cifar10H_softlabels(cfg['test']['annot_labels'])

    return Aqdata(data_cifar, label_cifar, corrected_labels=labels_annot, annotator_labels=annotator_labels), Aqdata(data_cifar_test, label_cifar_test, corrected_labels=labels_annot_test)
    

def load_imdb(cfg):
    modelname = main_config['architecture']['text']
    if modelname in SENTENCE_TRANSFORMERS:
        modelname = f"sentence-transformers/{modelname}"
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    # Load train data
    csv_path = os.path.join(cfg['train']['data'], 'train_csv.csv')
    if not os.path.exists(csv_path):
        train_csv = __load_imdb(cfg['train']['data'])
        train_csv.to_csv(csv_path, index=False)
    else:
        train_csv = pd.read_csv(csv_path)

    #train_csv = pd.concat([train_csv[train_csv.target == 0].iloc[:10], train_csv[train_csv.target == 1].iloc[:10]])
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
    #test_csv = pd.concat([test_csv[test_csv.target == 0].iloc[:10], test_csv[test_csv.target == 1].iloc[:10]])
    feat_texts, test_labels = __preprocess(test_csv.dropna(), tokenizer=tokenizer, type='test'), test_csv.dropna().target.values
    test_tokens = np.concatenate([f['input_ids'] for f in feat_texts], axis=0)
    test_attention_masks = np.concatenate([f['attention_mask'] for f in feat_texts], axis=0)

    train_tokens, train_attention_masks, train_labels = train_tokens.astype(np.int64), train_attention_masks.astype(np.int64), train_labels.astype(np.int64)
    test_tokens, test_attention_masks, test_labels = test_tokens.astype(np.int64), test_attention_masks.astype(np.int64), test_labels.astype(np.int64)

    #model_configs['base'][main_config['architecture']['text']]['epochs'] = 1

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
    model_configs['base'][main_config['architecture']['timeseries']]['input_length'] = train_data.shape[-1]
    model_configs['base'][main_config['architecture']['timeseries']]['in_channels'] = train_data.shape[-2]
    train_data, train_labels = train_data.astype(np.float32), train_labels.astype(np.int64)
    test_data, test_labels = test_data.astype(np.float32), test_labels.astype(np.int64)

    #train_data, test_data = __channelwise_minmax_scaler(train_data, test_data)

    return Aqdata(train_data, train_labels), Aqdata(test_data, test_labels)


def load_credit_fraud(cfg):
    filename = os.path.join(cfg['train']['data'], 'creditcard.csv')
    file_df = pd.read_csv(filename)
    labels = file_df['Class'].values 
    feat_df = file_df.drop(columns=['Time', 'Class'])
    features = feat_df.values
    model_configs['base'][main_config['architecture']['tabular']]['input_dim'] = features.shape[1]  # ALL TABULAR DATASETS MUST SET INPUT FEATURE DIM

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=0.15,
                                                                                random_state=1,
                                                                                shuffle=True,
                                                                                stratify=labels)
    
    train_features, train_labels = train_features.astype(np.float32), train_labels.astype(np.int64)
    test_features, test_labels = test_features.astype(np.float32), test_labels.astype(np.int64)

    scaler = StandardScaler()
    scaler.fit(train_features)

    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    return Aqdata(train_features, train_labels), Aqdata(test_features, test_labels)

def load_adult(cfg):
    train_data_path = os.path.join(cfg['train']['data'], 'adult.data')
    test_data_path = os.path.join(cfg['test']['data'], 'adult.test')
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country', 'target']
    feat_csv = pd.read_csv(train_data_path, sep=',', names=columns)
    test_feat_csv = pd.read_csv(test_data_path, sep=',', header=1, names=columns)
    
    feat_csv, test_feat_csv = __detect_and_process_categorical(feat_csv),\
                              __detect_and_process_categorical(test_feat_csv)

    train_features, train_labels = feat_csv.values[:,:-1], feat_csv['target'].values 
    test_features, test_labels = test_feat_csv.values[:,:-1], test_feat_csv['target'].values

    train_features, train_labels = train_features.astype(np.float32), train_labels.astype(np.int64)
    test_features, test_labels = test_features.astype(np.float32), test_labels.astype(np.int64)

    model_configs['base'][main_config['architecture']['tabular']]['input_dim'] = train_features.shape[1]

    scaler = StandardScaler()
    scaler.fit(train_features)

    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    return Aqdata(train_features, train_labels), Aqdata(test_features, test_labels)


def load_dry_bean(cfg):
    filename = os.path.join(cfg['train']['data'], 'Dry_Bean_Dataset.xlsx')
    file_df = pd.read_excel(filename)
    file_df = __detect_and_process_categorical(file_df)
    labels = file_df['Class'].values 
    feat_df = file_df.drop(columns=['Class'])
    features = feat_df.values
    model_configs['base'][main_config['architecture']['tabular']]['input_dim'] = features.shape[1]  # ALL TABULAR DATASETS MUST SET INPUT FEATURE DIM

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=0.15,
                                                                                random_state=1,
                                                                                shuffle=True,
                                                                                stratify=labels)
    
    train_features, train_labels = train_features.astype(np.float32), train_labels.astype(np.int64)
    test_features, test_labels = test_features.astype(np.float32), test_labels.astype(np.int64)
    
    scaler = StandardScaler()
    scaler.fit(train_features)

    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    
    return Aqdata(train_features, train_labels), Aqdata(test_features, test_labels)



def load_car_evaluation(cfg):
    filename = os.path.join(cfg['train']['data'], 'car.data')
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'target']
    feat_csv = pd.read_csv(filename, sep=',', names=columns)
    
    feat_csv = __detect_and_process_categorical(feat_csv)

    features, labels = feat_csv.values[:,:-1], feat_csv['target'].values 
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=0.15,
                                                                                random_state=1,
                                                                                shuffle=True,
                                                                                stratify=labels)
    
    train_features, train_labels = train_features.astype(np.float32), train_labels.astype(np.int64)
    test_features, test_labels = test_features.astype(np.float32), test_labels.astype(np.int64)
    model_configs['base'][main_config['architecture']['tabular']]['input_dim'] = train_features.shape[1]

    scaler = StandardScaler()
    scaler.fit(train_features)

    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    model_configs['base'][main_config['architecture']['tabular']]['epochs'] = 200
    model_configs['base'][main_config['architecture']['tabular']]['layers'] = [10]

    return Aqdata(train_features, train_labels), Aqdata(test_features, test_labels)


def load_mushrooms(cfg):
    filename = os.path.join(cfg['train']['data'], 'agaricus-lepiota.data')
    columns = ['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
               'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
               'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
               'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
               'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
    feat_csv = pd.read_csv(filename, sep=',', names=columns)
    feat_csv = __detect_and_process_categorical(feat_csv)

    features, labels = feat_csv.values[:,1:], feat_csv['target'].values 
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=0.15,
                                                                                random_state=1,
                                                                                shuffle=True,
                                                                                stratify=labels)
    
    train_features, train_labels = train_features.astype(np.float32), train_labels.astype(np.int64)
    test_features, test_labels = test_features.astype(np.float32), test_labels.astype(np.int64)
    model_configs['base'][main_config['architecture']['tabular']]['input_dim'] = train_features.shape[1]

    scaler = StandardScaler()
    scaler.fit(train_features)

    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    return Aqdata(train_features, train_labels), Aqdata(test_features, test_labels)

def load_compas(cfg):
    filename = os.path.join(cfg['train']['data'], 'propublicaCompassRecividism_data_fairml.csv', 'propublica_data_for_fairml.csv')
    feat_csv = pd.read_csv(filename)

    features, labels = feat_csv.values[:,1:], feat_csv['Two_yr_Recidivism'].values 
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=0.15,
                                                                                random_state=1,
                                                                                shuffle=True,
                                                                                stratify=labels)
    
    train_features, train_labels = train_features.astype(np.float32), train_labels.astype(np.int64)
    test_features, test_labels = test_features.astype(np.float32), test_labels.astype(np.int64)
    model_configs['base'][main_config['architecture']['tabular']]['input_dim'] = train_features.shape[1]

    scaler = StandardScaler()
    scaler.fit(train_features)

    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    return Aqdata(train_features, train_labels), Aqdata(test_features, test_labels)


def load_cxr(cfg):
    train_file_dir = cfg['train']['data']
    test_file_dir = cfg['test']['data']
    mappings = os.path.join(cfg['train']['labels'], 'pneumonia-challenge-dataset-mappings_2018.json')
    data_df = pd.read_csv(os.path.join(cfg['train']['labels'], 'RSNA_pneumonia_all_probs.csv'))
    with open(mappings) as f:
        mapping_dict = json.load(f)

    data, labels = [], []
    preprocess_data_path = os.path.join(cfg['train']['labels'], 'data.npy')
    preprocess_label_path = os.path.join(cfg['train']['labels'], 'labels.npy')

    if not os.path.exists(preprocess_label_path):
        for item in tqdm(mapping_dict, desc="CXR Labels"):
            if os.path.exists(os.path.join(train_file_dir, item['subset_img_id']+'.dcm')):
                filepath = os.path.join(train_file_dir, item['subset_img_id']+'.dcm')
            elif os.path.exists(os.path.join(test_file_dir, item['subset_img_id']+'.dcm')):
                filepath = os.path.join(test_file_dir, item['subset_img_id']+'.dcm')
            else:
                continue
            orig_label = str(item['orig_labels']).lower()
            if 'pneumonia' in orig_label:
                labels.append(1)
            elif "infiltration" in orig_label or "consolidation" in orig_label:
                row_df = data_df[(data_df['SeriesInstanceUID'] == item['SeriesInstanceUID']) & (data_df['SOPInstanceUID'] == item['SOPInstanceUID']) & (data_df['StudyInstanceUID'] == item['StudyInstanceUID'])]
                if 'Lung Opacity (High Prob)' in row_df['labelName'].values.tolist():
                    labels.append(1)
                else:
                    labels.append(0)
            elif "no finding" in orig_label:
                labels.append(0)
            else:
                labels.append(0)

        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(labels).astype(np.int64)
        np.save(preprocess_label_path, np.array(labels))

    if not os.path.exists(preprocess_data_path):
        for item in mapping_dict:
            if os.path.exists(os.path.join(train_file_dir, item['subset_img_id']+'.dcm')):
                filepath = os.path.join(train_file_dir, item['subset_img_id']+'.dcm')
            elif os.path.exists(os.path.join(test_file_dir, item['subset_img_id']+'.dcm')):
                filepath = os.path.join(test_file_dir, item['subset_img_id']+'.dcm')
            else:
                continue
            #ds = np.repeat(dicom.dcmread(filepath).pixel_array[np.newaxis, :, :], 3, axis=0)
            data.append(filepath)

        im_arrs = []
        for filename in tqdm(data, desc='CXR'):
            im_arrs.append(load_single_datapoint(filename)[np.newaxis, :])
        im_arrs = np.concatenate(im_arrs)

        np.save(preprocess_data_path, im_arrs)

        del im_arrs

    # Load data and labels
    data = np.load(preprocess_data_path)
    labels = np.load(preprocess_label_path)

    train_inds, test_inds = train_test_split(np.arange(data.shape[0]),
                                            test_size=0.4,
                                            random_state=1,
                                            shuffle=True,
                                            stratify=labels)
    train_data, train_labels = data[train_inds], labels[train_inds]
    test_data, test_labels = data[test_inds], labels[test_inds]

    return Aqdata(train_data, train_labels), Aqdata(test_data, test_labels)


def load_clothing100k(cfg):
    train_file_dir = cfg['train']['data']
    test_file_dir = cfg['test']['data']

    # Define data paths
    preprocess_traindata_path = os.path.join(os.path.abspath(train_file_dir), 'train_data.npy')
    preprocess_trainlabel_path = os.path.join(os.path.abspath(train_file_dir), 'train_labels.npy')
    preprocess_testdata_path = os.path.join(os.path.abspath(train_file_dir), 'test_data.npy')
    preprocess_testlabel_path = os.path.join(os.path.abspath(train_file_dir), 'test_labels.npy')
    
    train_data, train_labels = __load_tensorflow_format_dataset(train_file_dir)
    test_data, test_labels = __load_tensorflow_format_dataset(test_file_dir)
    
    le = preprocessing.LabelEncoder()
    train_labels, test_labels = le.fit_transform(train_labels), le.fit_transform(test_labels)

    return Aqdata(train_data, train_labels, lazy_load=True), Aqdata(test_data, test_labels, lazy_load=True)



def load_whalecalls(cfg):
    train_arr = pd.DataFrame(arff.loadarff(cfg['train']['data'])[0]).values
    test_arr = pd.DataFrame(arff.loadarff(cfg['test']['data'])[0]).values
    
    train_features, train_labels = train_arr[:, :-1], train_arr[:,-1]
    test_features, test_labels = test_arr[:, :-1], test_arr[:,-1]

    le = preprocessing.LabelEncoder()
    train_labels = le.fit_transform(train_labels).astype(np.int64)
    test_labels = le.fit_transform(test_labels).astype(np.int64)

    train_features, train_labels = train_features[:, np.newaxis, :].astype(np.float32), train_labels.astype(np.int64)
    test_features, test_labels = test_features[:, np.newaxis, :].astype(np.float32), test_labels.astype(np.int64)

    model_configs['base'][main_config['architecture']['timeseries']]['in_channels'] = train_features.shape[-2]
    model_configs['base'][main_config['architecture']['timeseries']]['input_length'] = train_features.shape[-1]

    #train_features, test_features = __channelwise_minmax_scaler(train_features, test_features)

    return Aqdata(train_features, train_labels), Aqdata(test_features, test_labels)


def load_pendigits(cfg):
    train_X, train_y = load_from_tsfile_to_dataframe(cfg['train']['data'])
    test_X, test_y = load_from_tsfile_to_dataframe(cfg['test']['data'])
    
    train_X = train_X.to_numpy()
    test_X = test_X.to_numpy()

    inp_len_train = train_X.shape[0]
    inp_len_test = test_X.shape[0]
    #train_y = np.hstack([l]*train_X.shape[-1] for l in train_y)
    #test_y = np.hstack([l]*test_X.shape[-1] for l in test_y)
    train_X = np.stack([np.stack([channel.values for channel in batch], axis=0) for batch in train_X], axis=0).reshape((inp_len_train, -1))[:, np.newaxis, :].astype(np.float32)
    test_X = np.stack([np.stack([channel.values for channel in batch], axis=0) for batch in test_X], axis=0).reshape((inp_len_test, -1))[:, np.newaxis, :].astype(np.float32)

    model_configs['base'][main_config['architecture']['timeseries']]['in_channels'] = train_X.shape[-2]
    model_configs['base'][main_config['architecture']['timeseries']]['input_length'] = train_X.shape[-1]

    le = preprocessing.LabelEncoder()
    train_y = le.fit_transform(train_y).astype(np.int64)
    test_y = le.fit_transform(test_y).astype(np.int64)

    #train_X, test_X = __channelwise_minmax_scaler(train_X, test_X)

    return Aqdata(train_X, train_y), Aqdata(test_X, test_y)


def load_eigenworms(cfg):
    train_X, train_y = load_from_tsfile_to_dataframe(cfg['train']['data'])
    test_X, test_y = load_from_tsfile_to_dataframe(cfg['test']['data'])

    train_feats, test_feats = [], []
    for i in range(6):
        train_feats.append([train_X.values[batch,i].values for batch in range(train_X.shape[0])])
        test_feats.append([test_X.values[batch,i].values for batch in range(test_X.shape[0])])

    train_X = np.array(train_feats).transpose(1,0,2).astype(np.float32)
    test_X = np.array(test_feats).transpose(1,0,2).astype(np.float32)

    model_configs['base'][main_config['architecture']['timeseries']]['in_channels'] = train_X.shape[-2]
    model_configs['base'][main_config['architecture']['timeseries']]['input_length'] = train_X.shape[-1]

    le = preprocessing.LabelEncoder()
    train_y = le.fit_transform(train_y).astype(np.int64)
    test_y = le.fit_transform(test_y).astype(np.int64)

    return Aqdata(train_X, train_y), Aqdata(test_X, test_y)

def load_crop(cfg):
    train_df = pd.read_csv(cfg['train']['data'], sep='\t').values
    test_df = pd.read_csv(cfg['test']['data'], sep='\t').values

    train_features, train_labels = train_df[:, 1:], train_df[:,0]-1
    test_features, test_labels = test_df[:, 1:], test_df[:,0]-1
    
    train_features, train_labels = train_features[:, np.newaxis, :].astype(np.float32), train_labels.astype(np.int64)
    test_features, test_labels = test_features[:, np.newaxis, :].astype(np.float32), test_labels.astype(np.int64)
    
    model_configs['base'][main_config['architecture']['timeseries']]['in_channels'] = train_features.shape[-2]
    model_configs['base'][main_config['architecture']['timeseries']]['input_length'] = train_features.shape[-1]
    
    #train_features, test_features = __channelwise_minmax_scaler(train_features, test_features)

    return Aqdata(train_features, train_labels), Aqdata(test_features, test_labels)


def load_insectwingbeat(cfg):
    train_df = pd.read_csv(cfg['train']['data'], sep='\t').values
    test_df = pd.read_csv(cfg['test']['data'], sep='\t').values

    train_features, train_labels = train_df[:, 1:], train_df[:,0]-1
    test_features, test_labels = test_df[:, 1:], test_df[:,0]-1
    
    train_features, train_labels = train_features[:, np.newaxis, :].astype(np.float32), train_labels.astype(np.int64)
    test_features, test_labels = test_features[:, np.newaxis, :].astype(np.float32), test_labels.astype(np.int64)
    
    model_configs['base'][main_config['architecture']['timeseries']]['in_channels'] = train_features.shape[-2]
    model_configs['base'][main_config['architecture']['timeseries']]['input_length'] = train_features.shape[-1]
    
    return Aqdata(train_features, train_labels), Aqdata(test_features, test_labels)


def load_electricdevices(cfg):
    train_df = pd.read_csv(cfg['train']['data'], sep='\t').values
    test_df = pd.read_csv(cfg['test']['data'], sep='\t').values

    train_features, train_labels = train_df[:, 1:], train_df[:,0]-1
    test_features, test_labels = test_df[:, 1:], test_df[:,0]-1
    
    train_features, train_labels = train_features[:, np.newaxis, :].astype(np.float32), train_labels.astype(np.int64)
    test_features, test_labels = test_features[:, np.newaxis, :].astype(np.float32), test_labels.astype(np.int64)
    
    model_configs['base'][main_config['architecture']['timeseries']]['in_channels'] = train_features.shape[-2]
    model_configs['base'][main_config['architecture']['timeseries']]['input_length'] = train_features.shape[-1]
    
    #train_features, test_features = __channelwise_minmax_scaler(train_features, test_features)

    return Aqdata(train_features, train_labels), Aqdata(test_features, test_labels)


def load_tweeteval(cfg):
    modelname = main_config['architecture']['text']
    if modelname in SENTENCE_TRANSFORMERS:
        modelname = f"sentence-transformers/{modelname}"
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    data_load = load_dataset(path=os.path.join(cfg["train"]["data"], 'tweet_eval.py'), name=cfg['type'], cache_dir=cfg["train"]["data"]) # Downloads the dataset if it already doesnt exist
    train_data, test_data = data_load['train'], data_load['test']
    train_data, test_data = [batch for batch in train_data], [batch for batch in test_data]
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    cfg['out_classes'] = len(pd.unique(train_df.label))

    # Tokenize train data
    feat_texts, train_labels = __preprocess(train_df.dropna(), tokenizer=tokenizer), train_df.dropna().label.values
    train_tokens = np.concatenate([f['input_ids'] for f in feat_texts], axis=0)
    train_attention_masks = np.concatenate([f['attention_mask'] for f in feat_texts], axis=0)
    
    # Tokenize test data
    feat_texts, test_labels = __preprocess(test_df.dropna(), tokenizer=tokenizer, type='test'), test_df.dropna().label.values
    test_tokens = np.concatenate([f['input_ids'] for f in feat_texts], axis=0)
    test_attention_masks = np.concatenate([f['attention_mask'] for f in feat_texts], axis=0)

    train_tokens, train_attention_masks, train_labels = train_tokens.astype(np.int64), train_attention_masks.astype(np.int64), train_labels.astype(np.int64)
    test_tokens, test_attention_masks, test_labels = test_tokens.astype(np.int64), test_attention_masks.astype(np.int64), test_labels.astype(np.int64)

    return Aqdata(train_tokens, train_labels, attention_mask=train_attention_masks), Aqdata(test_tokens, test_labels, attention_mask=test_attention_masks)

def load_reuters(cfg):
    tokenizer = AutoTokenizer.from_pretrained(main_config['architecture']['text'], model_max_length=514)
    data_load = load_dataset(path=os.path.join(cfg["train"]["data"], 'reuters21578.py'), name=cfg['type'], cache_dir=cfg["train"]["data"]) # Downloads the dataset if it already doesnt exist
    train_data, test_data = data_load['train'], data_load['test']
    train_data, test_data = [batch for batch in train_data], [batch for batch in test_data]
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    cfg['out_classes'] = len(pd.unique(train_df.label))