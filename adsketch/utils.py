import os
import random
import logging
import numpy as np
import multiprocessing
from sklearn.preprocessing import MinMaxScaler


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def init_logging(log_path):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    log.addHandler(consoleHandler)


def scale_two_metrics(train_metrics, test_metrics):
    est = MinMaxScaler()
    scaled_train_metrics = est.fit_transform(train_metrics.reshape(-1, 1)).reshape(-1)
    scaled_test_metrics = est.transform(test_metrics.reshape(-1, 1)).reshape(-1)

    return scaled_train_metrics, scaled_test_metrics


def init_para(scaled_test_metrics, m, graph_centers, batch_size, batch_num, stride):
    global scaled_test_metrics_global, m_global, graph_centers_global, batch_size_global, batch_num_global, stride_global
    scaled_test_metrics_global, m_global, graph_centers_global, batch_size_global, batch_num_global, stride_global = \
        scaled_test_metrics, m, graph_centers, batch_size, batch_num, stride


def unit_operation(index_of_batch):
    segs = get_batch_data(index_of_batch, m_global, batch_size_global,
                          batch_num_global, stride_global, scaled_test_metrics_global)

    dist_matrix = []
    for graph_center in graph_centers_global:
        dists = np.linalg.norm(segs - graph_center, axis=1)
        dist_matrix.append(dists)

    return [np.argmin(dist_matrix, axis=0).tolist(), np.min(dist_matrix, axis=0).tolist()]


def get_batch_data(idx, m, batch_size, batch_num, stride, metrics):
    batch_data = []

    end_batch = (idx+1) * batch_size
    if idx == batch_num - 1:
        end_batch = len(metrics) - m + 1

    for i in range(idx * batch_size, end_batch, stride):
        batch_data.append(metrics[i: i+m])

    return batch_data


def find_nearest_pattern(scaled_test_metrics, m, graph_centers, stride=1):
    batch_size = 10000
    subseq_num = len(scaled_test_metrics) - m + 1
    batch_num = int(subseq_num / batch_size) + 1

    process_num = 8
    pool = multiprocessing.Pool(process_num, initializer=init_para, initargs=(scaled_test_metrics, m, graph_centers,
                                                                              batch_size, batch_num, stride))

    idx = np.arange(batch_num)
    res_wrapper = pool.map(unit_operation, idx)

    nearest_patterns, nearest_dists = [], []
    for unit in res_wrapper:
        nearest_patterns.extend(unit[0])
        nearest_dists.extend(unit[1])

    pool.close()

    return nearest_patterns, nearest_dists
