from sklearn.cluster import KMeans
import torch
import pandas as pd
import numpy as np
from typing import Sequence, Tuple
import logging
import anomalytransfer as at
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _load_data(path: str) -> Tuple[np.ndarray, Sequence]:
    file_list = at.utils.file_list(path)
    progbar = at.utils.ProgBar(len(file_list), interval=0.5, unit_name='file')
    values = []
    names = []
    for file in file_list:
        filename = at.utils.filename(file)
        names.append(filename)
        values.append(at.clustering.preprocessing.down_sampling(
            [pd.read_csv(file).value.to_numpy()], step=DOWN_SAMPLING_STEP)[0])
        progbar.add(1)
    values = np.expand_dims(np.asarray(values), -1).astype(np.float32)
    return values, names


def _get_latent_vectors(x: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(x)
    seq_length = x.shape[1]
    input_dim = x.shape[2]

    model = at.clustering.LatentTransformer(
        seq_length=seq_length, input_dim=input_dim)
    model.fit(x, epochs=EPOCHS)
    model.save(os.path.join(OUTPUT, 'model.pt'))
    return model.transform(x)


def _get_clustering_result(labels: Sequence, names: Sequence) -> Tuple[Sequence, Sequence]:
    class_count = {}
    base_names = []
    classes = []

    for i in range(len(names)):
        base_name = names[i][:-4]
        if base_name not in class_count.keys():
            class_count[base_name] = [0] * N_CLUSTERS
        class_count[base_name][labels[i]] += 1

    for k, v in class_count.items():
        base_names.append(k)
        classes.append(np.argmax(v))

    return base_names, classes


def _sse_get_best_cluster_num(latent):
    if not os.path.exists("SSE (best cluster num).png"):
        distance_centroid = []
        max_clusters = 50
        for i in range(1, max_clusters):
            km = KMeans(n_clusters=i)
            km.fit(latent)
            distance_centroid.append(km.inertia_)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(1, max_clusters), distance_centroid, marker="o")
        plt.xlabel("The num of clusters")
        plt.ylabel("SSE")
        plt.savefig("SSE (best cluster num).png")


def _get_distance_centroid(features: np.ndarray, centroid: np.ndarray, labels: np.ndarray):
    """
    return: (N_samples, order_in_each_cluster)
    """
    distance_order = np.zeros([features.shape[0]])
    for label in range(centroid.shape[0]):
        center: np.ndarray = centroid[label]
        feature_idx = np.where(labels == label)[0]
        feature_with_label: np.ndarray = features[feature_idx]
        distance = np.sqrt(
            np.power((feature_with_label-center), 2).sum(axis=1))
        order_idx = np.argsort(distance)
        order = np.zeros_like(order_idx)
        order[order_idx] = range(1, order.shape[0]+1)
        distance_order[feature_idx] = order
    return distance_order


def _save_top_k_daily_kpi(order: np.ndarray, labels: np.ndarray, names: Sequence):
    """
    save the KPIs from the average stage
    """
    # save the entrire daily-kpi cluster result
    output_root = os.path.join(OUTPUT, "daily_cluster")
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
    for i, (label, name) in enumerate(zip(labels, names)):
        save_path = os.path.join(output_root, f"cluster-{label}", "data")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"{name}.csv")
        src_file = os.path.join(AVERAGE_OUTPUT, f"{name}.csv")
        shutil.copyfile(src_file, save_file)

    top_k_idx = np.where(order <= TOP_K)[0]

    top_k_labels = labels[top_k_idx]
    top_k_name = np.asarray(names)[top_k_idx]
    top_k_order = order[top_k_idx]

    output_root = os.path.join(OUTPUT, "top_k_daily_cluster")
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    label_based = set()
    for i, (label, name, order) in enumerate(zip(top_k_labels, top_k_name, top_k_order)):
        if order == 1 and label not in label_based:  # generate base
            label_based.add(label)
            save_path = os.path.join(output_root, f"cluster-{label}")
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, "base.csv")
        else:
            save_path = os.path.join(output_root, f"cluster-{label}", "data")
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, f"{name}.csv")

        src_file = os.path.join(AVERAGE_OUTPUT, f"{name}.csv")
        shutil.copyfile(src_file, save_file)


def _save_base_kpi(base_names: Sequence, classes: Sequence):
    """
    save the KPIs from the preprocess stage
    """
    output_root = os.path.join(OUTPUT, "base_cluster")
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    tag = np.zeros([len(classes)])
    for base, cls in zip(base_names, classes):
        if tag[cls] == 0:  # generate base
            save_root = os.path.join(output_root, f"cluster-{cls}")
            save_file = os.path.join(save_root, "base.csv")
            tag[cls] = 1
        else:
            save_root = os.path.join(output_root, f"cluster-{cls}", "data")
            save_file = os.path.join(save_root, f"{base}.csv")
        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)
        src_file = os.path.join(RAW_INPUT, f"{base}.csv")
        shutil.copy(src_file, save_file)


def main():
    at.utils.mkdirs(OUTPUT)
    step_progress = at.utils.ProgLog(4)

    step_progress.log(step='Preparing data...')
    values, names = _load_data(INPUT)

    step_progress.log(step='Getting latent vectors...')
    latent = _get_latent_vectors(values)  # (n_samples, 20)

    step_progress.log(
        step='Performing K-means clustering on latent vectors...')

    _sse_get_best_cluster_num(latent)

    k_means = KMeans(n_clusters=N_CLUSTERS)
    k_means.fit(latent)

    # get the distance of samples to their closest cluster center
    labels = k_means.labels_  # (n_samples, )
    cluster_centers = k_means.cluster_centers_  # (n_clusters, n_features)
    order = _get_distance_centroid(latent, cluster_centers, labels)

    # get TOP K kpi (with the shortest distance from centroid)
    _save_top_k_daily_kpi(order, labels, names)

    step_progress.log(step='Computing clustering result...')
    base_names, classes = _get_clustering_result(labels=labels, names=names)
    _save_base_kpi(base_names, classes)
    df = pd.DataFrame({'name': base_names, 'cluster': classes})
    df.to_csv(os.path.join(OUTPUT, 'result.csv'), index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s [%(levelname)s]] %(message)s')
    config = at.utils.config()

    INPUT = config.get('CLUSTERING', 'input')
    OUTPUT = config.get('CLUSTERING', 'output')
    EPOCHS = config.getint('CLUSTERING', 'epochs')
    AVERAGE_OUTPUT = config.get("CLUSTERING_AVERAGE", "output_daily")
    RAW_INPUT = config.get("CLUSTERING_PREPROCESSING", "input")
    DOWN_SAMPLING_STEP = config.getint(
        'CLUSTERING_PREPROCESSING', 'down_sampling_step')
    try:
        N_CLUSTERS = config.getint('CLUSTERING', 'n_clusters')
    except:
        # see `"SSE (best cluster num).png"` to set best cluster number.
        N_CLUSTERS = 1

    TOP_K = 50
    main()
