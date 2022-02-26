import os
from multiprocessing import Pool

from torch.cuda import is_available
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging
import anomalytransfer as at
from glob import glob
from utils import run_time

from typing import Sequence, Tuple, Dict, Optional
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='[%(asctime)s [%(levelname)s]] %(message)s')

config = at.utils.config()
CLUSTER_OUTPUT = config.get("CLUSTERING", "output")
EPOCHS = config.getint("CLUSTERING", "epochs")
BASE_EPOCHS = config.getint('TRANSFER_LEARNING', 'base_epochs')
DATA_EPOCHS = config.getint('TRANSFER_LEARNING', 'data_epochs')
INPUT = config.get('TRANSFER_LEARNING', 'input')
OUTPUT = config.get('TRANSFER_LEARNING', 'output')
MODEL_PATH = config.get('TRANSFER_LEARNING', 'model_path')
RATIO = config.getfloat('TRANSFER_LEARNING', 'ratio')

RAW_INPUT = config.get("CLUSTERING_PREPROCESSING", "input")


def _get_latent_vectors(x: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(x)
    seq_length = x.shape[1]
    input_dim = x.shape[2]

    model = at.clustering.LatentTransformer(
        seq_length=seq_length, input_dim=input_dim)
    model.fit(x, epochs=EPOCHS, verbose=0)
    model.save(os.path.join(OUTPUT, 'model.pt'))
    return model.transform(x)


def cluster_data(path: str) -> Tuple[str, str]:
    base = None
    data = None
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            data = item_path
        else:
            base = item_path
    if base is None or data is None:
        raise ValueError('Base path or data path not found')
    return base, data

def make_base_model(kpi: at.transfer.data.KPI, epochs: int):
    kpi.complete_timestamp()
    kpi, _, _ = kpi.standardize()
    model = at.transfer.models.AnomalyDetector()
    model.fit(kpi=kpi.no_labels(), epochs=epochs, verbose=0)
    return model

def train_test(train_kpi: at.transfer.data.KPI,
               epochs: int,
               test_kpi: at.transfer.data.KPI = None,
               mask: Optional[Sequence] = None,
               **kwargs) -> float:
    model = at.transfer.models.AnomalyDetector()
    if mask is not None:
        model.load_partial(path=kwargs.get('model_path'), name=kwargs.get('base_kpi').name, mask=mask)
        model.freeze(mask)
        model.fit(kpi=train_kpi.no_labels(), epochs=epochs, verbose=0)
        model.unfreeze(mask)
    model.fit(kpi=train_kpi.no_labels(), epochs=epochs, verbose=0)
    if test_kpi is not None and test_kpi.labels is not None:
        anomaly_scores = model.predict(test_kpi, verbose=0)
        results = at.utils.get_test_results(labels=test_kpi.labels,
                                            scores=anomaly_scores,
                                            missing=test_kpi.missing,
                                            use_spot=False)
        at.utils.log_test_results(name=test_kpi.name, results=results)
        return results['f1score']
    else:
        return None


def _ignore_missing(series_list: Sequence, missing: np.ndarray) -> Tuple[np.ndarray, ...]:
    ret = []
    for series in series_list:
        series = np.copy(series)
        ret.append(series[missing != 1])
    return tuple(ret)


def get_test_results(
        timestamps: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        missing: np.ndarray,
        values: np.ndarray,
        window_size: int = 120,
        **kwargs) -> Dict:
    timestamps = timestamps[window_size - 1:]
    labels = labels[window_size - 1:]
    scores = scores[window_size - 1:]
    missing = missing[window_size - 1:]
    values = values[window_size - 1:]
    adjusted_timestamps, adjusted_labels, adjusted_scores, adjusted_values = _ignore_missing(
        [timestamps, labels, scores, values], missing=missing
    )

    adjusted_scores = at.utils.adjust_scores(
            labels=adjusted_labels, scores=adjusted_scores)
    precision, recall, th = precision_recall_curve(adjusted_labels, adjusted_scores, pos_label=1)

    f1_score = 2 * precision * recall / (precision + recall + 1e-6)

    arg_max = np.argmax(f1_score)

    best_precision, best_recall, best_f1_score = precision[arg_max], recall[arg_max], f1_score[arg_max]
    threshold = th[arg_max]
    return best_f1_score


def main(finetune_num=200):
    print(finetune_num)
    # with torch.cuda.device(torch.device(f"cuda:{finetune_num//200%2}")):
    clusters = os.listdir(INPUT)
    base_values = []
    base_models = []
    for cluster in tqdm(clusters, total=len(clusters)):
        base, data = cluster_data(os.path.join(INPUT, cluster))
        base_kpi = at.utils.load_kpi(base)
        base_kpi.complete_timestamp()
        base_kpi, _, _ = base_kpi.standardize()
        base_model = make_base_model(base_kpi, BASE_EPOCHS)
        base_models.append(base_model)

        dt = pd.read_csv(base)
        base_values.append(dt["value"])

    file_list = at.utils.file_list(RAW_INPUT)
    cluster_values = []
    finetune_values = []
    test_kpis = []
    names = []
    for file in file_list:
        data_kpi = at.utils.load_kpi(file)
        data_kpi.complete_timestamp()
        data_kpi, _, _ = data_kpi.standardize()
        filename = at.utils.filename(file)
        names.append(filename)

        # split idx
        ts = data_kpi.timestamps
        ts = ts % (60 * 60 * 24)
        split_idx = np.where(ts <= 60)[0]
        _, data_kpi = data_kpi.split_by_idx(split_idx[0], window_size=1)

        # split to [for cluster] and [for finetune]
        ts = data_kpi.timestamps
        ts = ts % (60 * 60 * 24)
        split_idx = np.where(ts <= 60)[0]
        cluster_value, finetune_value = data_kpi.split_by_idx(split_idx[1], window_size=1)
        finetune_value, test_value = finetune_value.split_by_idx(finetune_num, window_size=1)

        cluster_values.append(cluster_value.values)
        finetune_values.append(finetune_value)
        test_kpis.append(test_value)
    

    # get latent var
    base_values = np.asarray(base_values, dtype=np.float32)[..., None]
    base_feature = _get_latent_vectors(base_values)
    
    cluster_values = np.asarray(cluster_values, dtype=np.float32)[..., None]
    cluster_feature = _get_latent_vectors(cluster_values)
    
    tmp_result = {name: 0 for name in names}
    tmp_result["num_of_points"] = finetune_num
    for i, (ft, finetune, test_kpi, name) in enumerate(zip(cluster_feature, finetune_values, test_kpis, names)):
        cluster_idx = np.argmin(np.sum((ft - base_feature)**2, axis=1))
        base_model = base_models[cluster_idx]
        # base_model.fit(kpi=finetune.no_labels(), epochs=DATA_EPOCHS, verbose=0)
        anomaly_scores = base_model.predict(test_kpi, verbose=1)
        f1_score = get_test_results(
                    timestamps=test_kpi.timestamps,
                    labels=test_kpi.labels,
                    scores=anomaly_scores,
                    missing=test_kpi.missing,
                    values=test_kpi.values
                )
        tmp_result[name] = f1_score
        print(f"{i} - {name}")

    return tmp_result

if __name__ == '__main__':

    # for num in range(200, 5000, 200):
    #     main(num)
    with Pool(1) as pool:
        results = pool.map(main, range(200, 201, 200))
        # results = pool.map(main, range(200, 201, 200))
    final_result = pd.DataFrame(columns=list(results[0].keys()))
    for res in results:
        final_result = final_result.append(res, ignore_index=True)
    
    final_result = final_result.sort_values("num_of_points")
    final_result.to_csv("result.csv", index=False)
