import os
import copy
import pickle
import stumpy
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import precision_recall_fscore_support

from .utils import *

plt.rcParams["figure.figsize"] = [26, 4]


def get_train_graph(mp, mp_idx, p):
    G = nx.Graph()
    for i in range(len(mp)):
        if mp[i] <= p:
            G.add_edge(i, mp_idx[i])
        else:
            G.add_nodes_from([i, mp_idx[i]])
    return G


# Combine Ina and Inn to get graph G
def combine_graphs(G, test_mp, test_mp_idx, p):
    train_node_num = G.number_of_nodes()

    for i in range(len(test_mp)):
        if test_mp[i] <= p:
            # i + train_node_num to avoid index collision
            G.add_edge(test_mp_idx[i], i + train_node_num)
        else:
            G.add_node(i + train_node_num)

    return G


def get_cluster_centers(m, train_node_num, graphs, train_metrics, test_metrics):
    cluster_centers = []
    cluster_radii = []

    for graph in graphs:
        seqs = []
        for node in graph:
            if node < train_node_num:
                seqs.append(train_metrics[node: node + m])
            else:
                idx = node - train_node_num
                seqs.append(test_metrics[idx: idx + m])

        # Get the graph center for each cluster
        cluster_centers.append(np.mean(seqs, axis=0))
        # Get the radius for each cluster
        dists = np.linalg.norm(seqs - cluster_centers[-1], axis=1)
        cluster_radii.append(np.max(dists))

    return cluster_centers, cluster_radii


def ap_clustering(cluster_centers, damping):
    clustering = AffinityPropagation(random_state=5, damping=damping).fit(cluster_centers)
    return clustering.labels_


def draw_anomalous_subseqs(m, anomalous_subseqs, scaled_test_metrics, test_labels, fig_dir):
    plt.plot(scaled_test_metrics)
    test_labels = np.where(test_labels == 1)[0]
    plt.plot(test_labels, np.zeros(len(test_labels)), 'g.')

    for pattern in anomalous_subseqs:
        seg = np.arange(pattern, pattern+m)
        plt.plot(seg, scaled_test_metrics[seg], 'r')

    logging.info('Experiment result plotted!')
    plt.savefig(fig_dir)
    plt.cla()
    # plt.show()


def evaluate(m, anomalous_subseqs, test_metric_labels):
    y_pred_tmp = []
    for pattern in anomalous_subseqs:
        y_pred_tmp.extend(list(np.arange(pattern, pattern+m)))

    y_pred_tmp = sorted(list(set(y_pred_tmp)))
    y_pred = np.zeros(len(test_metric_labels))
    y_pred[y_pred_tmp] = 1
    y_true = test_metric_labels

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return [precision, recall, f1]


def evaluate_predictions(m, anomalous_subseqs, scaled_test_metrics, test_metric_labels, fig_dir):
    if len(anomalous_subseqs) == 0:
        logging.info('No anomalous patterns detected!')
        if sum(test_metric_labels) == 0:
            res = [1.] * 3  # The algorithm indeed recognizes such a situation
        else:
            res = [0.] * 3
    else:
        y_pred_tmp = []
        for pattern in anomalous_subseqs:
            y_pred_tmp.extend(list(np.arange(pattern, pattern+m)))

        y_pred_tmp = sorted(list(set(y_pred_tmp)))
        y_pred = np.zeros(len(test_metric_labels))
        y_pred[y_pred_tmp] = 1
        y_true = test_metric_labels

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        res = [precision, recall, f1]
    
    precision, recall, f1 = res
    logging.info('precision: {:.3f}, recall: {:.3f}, f1: {:.3f}'.format(precision, recall, f1))
    # Plot anomaly detection results
    draw_anomalous_subseqs(m, anomalous_subseqs, scaled_test_metrics, test_metric_labels, fig_dir)

    return res


# ADSketch Algorithm 1
def anomaly_pattern_discovery(scaled_train_metrics, scaled_test_metrics, m, p=99, noise_p=100, damping=0.9):
    train_mp = stumpy.stump(scaled_train_metrics, m, normalize=False)
    # The threshold over which the subsequences will be considered as anomalies in the training data.
    # The default setting is that the training data are anomaly-free, i.e., noise_p=100.
    # A smaller noise_p means the training data are suspected to contain more anomalies
    train_mp_p = np.percentile(train_mp[:, 0], noise_p)
    train_anomalies_idxes = np.where(train_mp[:, 0] > train_mp_p)[0]

    G = get_train_graph(train_mp[:, 0], train_mp[:, 1], train_mp_p)
    train_node_num = G.number_of_nodes()

    test_mp = stumpy.stump(scaled_test_metrics, m, scaled_train_metrics, ignore_trivial=False, normalize=False)
    G = combine_graphs(G, test_mp[:, 0], test_mp[:, 1], np.percentile(test_mp[:, 0], p))
    G.remove_nodes_from(train_anomalies_idxes)  # Remove the suspicious anomalies in the training data

    # Get connected subgraphs
    subgraphs = list(nx.connected_components(G))
    subgraphs = [list(subgraph) for subgraph in subgraphs]

    isolated_nodes = [subgraph[0] for subgraph in subgraphs if len(subgraph) == 1]
    isolated_nodes = np.array(isolated_nodes) - train_node_num
    logging.info(f'The number of anomalous seqs: {len(isolated_nodes)}')

    # Group graph centers using Affinity Propagation
    subcluster_centers, _ = get_cluster_centers(m, train_node_num, subgraphs, scaled_train_metrics, scaled_test_metrics)
    labels = ap_clustering(subcluster_centers, damping)

    label_dict = {}
    for i in range(len(labels)):
        label = labels[i]
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].extend(subgraphs[i])
    clusters = [label_dict[label] for label in label_dict]

    anomalous_subseqs = []
    anomalous_clusters = []
    for i in range(len(clusters)):
        nodes = np.array(clusters[i]) - train_node_num
        if all([node in isolated_nodes for node in nodes]):
            anomalous_subseqs.extend(nodes)
            # anomalous_clusters.extend(label_dict[label])
            anomalous_clusters.append(i)

    logging.info(f'The number of anomalous seqs after ap clustering: {len(anomalous_subseqs)}')
    logging.info(f'The number of anomalous clusters: {len(anomalous_clusters)}')

    cluster_sizes = [len(cluster) for cluster in clusters]
    cluster_centers, cluster_radii = get_cluster_centers(m, train_node_num, clusters, scaled_train_metrics, scaled_test_metrics)

    return anomalous_subseqs, anomalous_clusters, cluster_sizes, cluster_centers, cluster_radii


def offline_anomaly_detection(m, p,
                              train_metric_values, test_metric_values, test_metric_labels,
                              offline_pattern_dir, fig_dir):
    scaled_train_metrics, scaled_test_metrics = scale_two_metrics(train_metric_values, test_metric_values)

    # anomalous_clusters: the id of the clusters that are identified as anomalous
    anomalous_subseqs, anomalous_clusters, cluster_sizes, cluster_centers, cluster_radii = anomaly_pattern_discovery(
    scaled_train_metrics, scaled_test_metrics, m, p)

    with open(offline_pattern_dir, 'wb') as pickle_writer:
        pickle.dump([anomalous_subseqs, anomalous_clusters, cluster_sizes, cluster_centers, cluster_radii],
                    pickle_writer)
    logging.info('Metric patterns dumped.')

    res = evaluate_predictions(m, anomalous_subseqs, scaled_test_metrics, test_metric_labels, fig_dir)

    return res


def online_anomaly_detection(adaptive_learning, m, p, max_anomalous_cluster_size,
                             train_metric_values, test_metric_values, test_metric_labels, online_test_metric_values, online_test_metric_labels,
                             offline_pattern_dir, fig_dir, stride=1):
    _, online_scaled_test_metrics = scale_two_metrics(train_metric_values, online_test_metric_values)

    # Check if the metric patterns learned from the offline phase exist
    if not os.path.exists(offline_pattern_dir):
        logging.info('Metric patterns not found, conduct offline anomaly detection first')
        offline_anomaly_detection(m, p, train_metric_values, test_metric_values, test_metric_labels, 
                                  offline_pattern_dir, fig_dir+'_offline.png')

    logging.info('Loading metric patterns...')
    with open(offline_pattern_dir, 'rb') as pickle_reader:
        anomalous_subseqs, anomalous_clusters, cluster_sizes, cluster_centers, cluster_radii = pickle.load(
            pickle_reader)

    if os.path.exists(offline_pattern_dir):
        _, scaled_test_metrics = scale_two_metrics(train_metric_values, test_metric_values)
        evaluate_predictions(m, anomalous_subseqs, scaled_test_metrics,
                             test_metric_labels, fig_dir + '_offline.png')

    # The number of metric subsequences to predict normality
    online_subseq_num = len(online_scaled_test_metrics) - m + 1
    anomalous_clusters_origin = copy.deepcopy(anomalous_clusters)
    # An anomalous graph with a size larger than max_anomalous_cluster_size will be regarded as normal.
    if max_anomalous_cluster_size is None:
        if len(anomalous_clusters) == 0:
            max_anomalous_cluster_size = np.min(np.array(cluster_sizes))
        else:
            max_anomalous_cluster_size = np.max(np.array(cluster_sizes)[anomalous_clusters])

    if adaptive_learning:
        logging.info('Online mode with adaptive pattern learning...')

        if len(anomalous_clusters) == 0:
            anomalous_max_dist = benign_max_dist = np.max(cluster_radii)
        else:
            cluster_radii_tmp = np.array(cluster_radii)
            anomalous_max_dist = np.max(cluster_radii_tmp[anomalous_clusters])
            cluster_radii_tmp[anomalous_clusters] = 0
            benign_max_dist = np.max(cluster_radii_tmp)

        logging.info('Interesting parameters (before updating)')
        logging.info('anomalous_max_dist: {:.3f}, benign_max_dist: {:.3f}'.format(anomalous_max_dist, benign_max_dist))
        logging.info(f'anomalous cluster num: {len(anomalous_clusters)}, '
                     f'benign cluster num: {len(cluster_radii) - len(anomalous_clusters)}')
        logging.info(f'max_anomalous_cluster_size: {max_anomalous_cluster_size}, '
                     f'max_benign_cluster_size: {np.max(cluster_sizes)}')

        # The prediction results in online mode
        online_anomalous_subseqs = []
        progress_bar = tqdm(range(0, online_subseq_num, stride))
        for i in progress_bar:
            subseq = online_scaled_test_metrics[np.arange(i, i+m)]
            dists = np.linalg.norm(cluster_centers - subseq, axis=1)
            nearest_pattern = np.argmin(dists)
            nearest_dist = dists[nearest_pattern]

            progress_bar.set_description('#anomalous graph {} ({:.3f}/{:.3f})'.format(len(anomalous_clusters),
                                                                                      anomalous_max_dist,
                                                                                      nearest_dist))

            # If combine subseq to the nearest graph, we have
            cluster_center, cluster_size = cluster_centers[nearest_pattern], cluster_sizes[nearest_pattern]
            updated_center = (cluster_center * cluster_size + subseq) / (cluster_size + 1)
            # The distance between the updated center and subsequence
            subseq_dist = np.linalg.norm(updated_center - subseq)
            # The distance between the updated center and the farthest node in the worst case
            updated_graph_dist = np.linalg.norm(updated_center - cluster_center) + cluster_radii[nearest_pattern]
            max_dist = subseq_dist if subseq_dist > updated_graph_dist else updated_graph_dist

            d_prime = benign_max_dist
            if nearest_pattern in anomalous_clusters:
                online_anomalous_subseqs.append(i)  # Predicted as an anomaly
                d_prime = anomalous_max_dist

            # Create a new anomalous cluster
            if d_prime < nearest_dist:
                # len(cluster_radii) is the id of the new anomalous cluster
                anomalous_clusters.append(len(cluster_radii))
                cluster_centers.append(subseq)
                cluster_sizes.append(1)
                cluster_radii.append(0.0)

            else:
                # Update the radius, cluster center, and cluster size
                cluster_radii[nearest_pattern] = max_dist
                cluster_centers[nearest_pattern] = updated_center
                cluster_sizes[nearest_pattern] += 1

                if nearest_pattern in anomalous_clusters:
                    # If the size of a new anomalous cluster is too large,
                    # it is removed for suspiciously being a benign cluster
                    if cluster_sizes[nearest_pattern] > max_anomalous_cluster_size and \
                            nearest_pattern not in anomalous_clusters_origin:
                            anomalous_clusters.remove(int(nearest_pattern))
                    else:
                        # Update anomalous_max_dist
                        if max_dist > anomalous_max_dist:
                            anomalous_max_dist = max_dist

                else:
                    # Update benign_max_dist
                    if max_dist > benign_max_dist:
                        benign_max_dist = max_dist

        logging.info('Interesting parameters (after updating)')
        logging.info('anomalous_max_dist: {:.3f}, benign_max_dist: {:.3f}'.format(anomalous_max_dist, benign_max_dist))
        logging.info(f'anomalous cluster num: {len(anomalous_clusters)}, '
                     f'benign cluster num: {len(cluster_radii) - len(anomalous_clusters)}')
        logging.info(f'max_anomalous_cluster_size: {np.max(np.array(cluster_sizes)[anomalous_clusters])}, '
                     f'max_benign_cluster_size: {np.max(cluster_sizes)}')

    else:
        logging.info('Online mode without adaptive pattern learning...')
        logging.info(f'The number of subsequences to predict normality: {online_subseq_num}')
        nearest_patterns, nearest_dists = find_nearest_pattern(online_scaled_test_metrics, m, cluster_centers, stride)

        online_anomalous_subseqs = [i for i in range(0, online_subseq_num, stride) if
                                    nearest_patterns[int(i/stride)] in anomalous_clusters]

    online_fig = f'{fig_dir}_adaptive_online.png' if adaptive_learning else f'{fig_dir}_online.png'
    evaluate_predictions(m, online_anomalous_subseqs, online_scaled_test_metrics, 
                         online_test_metric_labels, online_fig)
