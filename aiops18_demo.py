import os
import json
import logging
import argparse

from dataset_loader import *
from adsketch.motif_operations import *


# Setup the logging file name
seed_everything(seed=1234)
os.makedirs('./logs', exist_ok=True)
init_logging(f'./logs/aiops18_demo.log')

parser = argparse.ArgumentParser()
parser.add_argument("--adaptive", type=bool, default=False, help="Adaptive pattern learning")
parser.add_argument("--res_dir", type=str, default='./res/aiops18/',
                    help="The directory to save experimental figures")
parser.add_argument("--pattern_dir", type=str, default='./offline_metrics/aiops18/', 
                    help="The directory to save the learned metric patterns and other necessary info")
args = vars(parser.parse_args())

os.makedirs(args['res_dir'], exist_ok=True)
os.makedirs(args['pattern_dir'], exist_ok=True)

with open('params.json', 'r') as json_reader:
    params = json.load(json_reader)


def offline_aiops18_data():
    logging.info('{}{}{}'.format('^' * 15, ' Offline anomaly detection for AIOps18 ', '^' * 15))
    aiops18_params = params['aiops18']
    # Model parameters
    metric_name = 'f0932edd'
    m, p = aiops18_params[metric_name]["m"], aiops18_params[metric_name]["p"]
    start, end = aiops18_params[metric_name]["seg"]  # Get the selected anomaly-free metric time series
    fig_dir = os.path.join(args['res_dir'], f'{metric_name}_{m}_{p}_offline.png')
    offline_pattern_dir = os.path.join(args['pattern_dir'], f'{metric_name}_{m}_{p}.pkl')

    train_metric_values, train_metric_labels, test_metric_values, test_metric_labels = load_aiops18_data(metric_name)
    # Select particular segments of data
    train_metric_values, test_metric_values = train_metric_values[start: end], train_metric_values[end:]
    train_metric_labels, test_metric_labels = train_metric_labels[start: end], train_metric_labels[end:]

    logging.info('=' * 60)
    logging.info(f'Dataset: aiops18 ({metric_name}), m: {m}, p: {p}')

    offline_anomaly_detection(m, p, 
                              train_metric_values, test_metric_values, test_metric_labels,
                              offline_pattern_dir, fig_dir)


def online_aiops18_data():
    logging.info('{}{}{}'.format('^' * 15, ' Online anomaly detection for AIOps18 ', '^' * 15))
    aiops18_params = params['aiops18']
    # Whether to conduct updating when performing offline evaluation
    adaptive_learning = args['adaptive']

    # The tuned max cluster size for the following metrics
    max_graph_size_dict = {'43115f2a': 400, '431a8542': 1400, '4d2af31a': 1400, '6a757df4': 2000, 'a8c06b47': 10,
                           'adb2fde9': 600, 'ba5f3328': 100, 'f0932edd': 30, 'ffb82d38': 4000}

    metric_names = list(aiops18_params.keys())
    for metric_name in metric_names:
        # Model parameters
        m, p = aiops18_params[metric_name]["m"], aiops18_params[metric_name]["p"]
        start, end = aiops18_params[metric_name]["seg"]

        logging.info('{}{}{}'.format('^' * 10, f' Dataset: aiops18 ({metric_name}), m: {m}, p: {p} ', '^' * 10))
        logging.info('Parameter settings:')
        logging.info(f'Anomaly-free metric segment: {start}-{end}')
        logging.info(f'Adaptive pattern learning: {adaptive_learning}')

        fig_dir = os.path.join(args['res_dir'], f'{metric_name}_{m}_{p}_offline.png')
        offline_pattern_dir = os.path.join(args['pattern_dir'], f'{metric_name}_{m}_{p}.pkl')

        # Data preparation
        train_metric_values, train_metric_labels, test_metric_values, test_metric_labels = load_aiops18_data(metric_name)
        online_test_metric_values, online_test_metric_labels = test_metric_values, test_metric_labels
        # Select a particular segment of train_data as the training data, while the remaining train_data as testing data
        # Use the entire test_data for model evaluation in offline mode
        train_metric_values, test_metric_values = train_metric_values[start: end], train_metric_values[end:]
        train_metric_labels, test_metric_labels = train_metric_labels[start: end], train_metric_labels[end:]

        max_anomalous_cluster_size = None
        if metric_name in max_graph_size_dict:
            max_anomalous_cluster_size = max_graph_size_dict[metric_name]

        online_anomaly_detection(adaptive_learning, m, p, max_anomalous_cluster_size,
                                 train_metric_values, test_metric_values, test_metric_labels, online_test_metric_values, online_test_metric_labels,
                                 offline_pattern_dir, fig_dir)


if __name__ == '__main__':
    # offline_aiops18_data()
    online_aiops18_data()
