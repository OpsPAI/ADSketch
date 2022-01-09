import os
import json
import logging
import argparse

from dataset_loader import *
from adsketch.motif_operations import *


# Setup the logging file name
seed_everything(seed=1234)
os.makedirs('./logs', exist_ok=True)
init_logging(f'./logs/industry_demo.log')

parser = argparse.ArgumentParser()
parser.add_argument("--adaptive", type=bool, default=False, help="Adaptive pattern learning")
parser.add_argument("--res_dir", type=str, default='./res/industry/',
                    help="The directory to save experimental figures")
parser.add_argument("--pattern_dir", type=str, default='./offline_metrics/industry/',
                    help="The directory to save the learned metric patterns and other necessary info")
args = vars(parser.parse_args())

os.makedirs(args['res_dir'], exist_ok=True)
os.makedirs(args['pattern_dir'], exist_ok=True)

with open('params.json', 'r') as json_reader:
    params = json.load(json_reader)


def offline_industry_data():
    logging.info('{}{}{}'.format('^' * 15, ' Offline anomaly detection for Industry ', '^' * 15))
    industry_params = params['industry']
    normal_point, offline_point = 1440, 5760

    # Model parameters
    metric_name = 'e59c1d14'
    m, p = industry_params[metric_name]["m"], industry_params[metric_name]["p"]
    fig_dir = os.path.join(args['res_dir'], f'{metric_name}_{m}_{p}_offline.png')
    offline_pattern_dir = os.path.join(args['pattern_dir'], f'{metric_name}_{m}_{p}.pkl')

    metric_values, metric_labels = load_industry_data(metric_name)
    # Select particular segments of data
    train_metric_values, test_metric_values = metric_values[:normal_point], metric_values[normal_point:offline_point]
    train_metric_labels, test_metric_labels = metric_labels[:normal_point], metric_labels[normal_point:offline_point]

    logging.info('=' * 60)
    logging.info(f'Dataset: industry ({metric_name}), m: {m}, p: {p}')

    offline_anomaly_detection(m, p,
                              train_metric_values, test_metric_values, test_metric_labels,
                              offline_pattern_dir, fig_dir)


def online_industry_data():
    logging.info('{}{}{}'.format('^' * 15, ' Online anomaly detection for Industry ', '^' * 15))
    industry_params = params['industry']
    # Whether to conduct updating when performing offline evaluation
    adaptive_learning = args['adaptive']
    normal_point, offline_point = 1440, 5760

    metric_names = list(industry_params.keys())
    for metric_name in metric_names:
        # Model parameters
        m, p = industry_params[metric_name]["m"], industry_params[metric_name]["p"]

        logging.info('{}{}{}'.format('^' * 10, f' Dataset: industry ({metric_name}), m: {m}, p: {p} ', '^' * 10))
        logging.info('Parameter settings:')
        logging.info(f'Adaptive pattern learning: {adaptive_learning}')

        fig_dir = os.path.join(args['res_dir'], f'{metric_name}_{m}_{p}_offline.png')
        offline_pattern_dir = os.path.join(args['pattern_dir'], f'{metric_name}_{m}_{p}.pkl')

        # Data preparation
        metric_values, metric_labels = load_industry_data(metric_name)
        # Select particular segments of data
        train_metric_values, test_metric_values = metric_values[:normal_point], \
                                                  metric_values[normal_point:offline_point]
        train_metric_labels, test_metric_labels = metric_labels[:normal_point], \
                                                  metric_labels[normal_point:offline_point]
        online_test_metric_values, online_test_metric_labels = metric_values[offline_point:], \
                                                               metric_labels[offline_point:]

        online_anomaly_detection(adaptive_learning, m, p, None,
                                 train_metric_values, test_metric_values, test_metric_labels, online_test_metric_values, online_test_metric_labels,
                                 offline_pattern_dir, fig_dir)


if __name__ == '__main__':
    # offline_industry_data()
    online_industry_data()
