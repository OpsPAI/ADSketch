import os
import pickle
import numpy as np
import pandas as pd

base_dir = './data/'


def load_industry_data(metric_name):
    metric_dir = os.path.join(base_dir, 'industry')
    with open(os.path.join(metric_dir, 'industry_data_dict.pkl'), 'rb') as pickle_reader:
        metrics = pickle.load(pickle_reader)

    return metrics[metric_name]['values'], metrics[metric_name]['labels']


def load_aiops18_data(metric_name):
    metric_dir = os.path.join(base_dir, 'aiops18/')
    with open(os.path.join(metric_dir, 'train_data_dict.pkl'), 'rb') as pickle_reader:
        train = pickle.load(pickle_reader)
    with open(os.path.join(metric_dir, 'test_data_dict.pkl'), 'rb') as pickle_reader:
        test = pickle.load(pickle_reader)

    train_metric = train[metric_name]
    train_metric_values = train_metric[0]
    train_metric_labels = train_metric[1]

    test_metric = test[metric_name]
    test_metric_values = test_metric[0]
    test_metric_labels = test_metric[1]

    return train_metric_values, train_metric_labels, test_metric_values, test_metric_labels


def load_yahoo_data(benchmark):
    metric_dir = os.path.join(base_dir, f'yahoo/{benchmark}')
    metric_files = [metric_file for metric_file in os.listdir(metric_dir) if metric_file.endswith('.csv')]
    metric_files = sorted(metric_files)
    metric_values, metric_labels = [], []
    for i in range(len(metric_files)):
        df = pd.read_csv(os.path.join(metric_dir, metric_files[i]))
        df_values = df.values.transpose()
        metric_values.append(df_values[1])
        metric_labels.append(df_values[2])

    return metric_values, metric_labels
