from sample.scripts.transfer.utils import run_time
import torch
import pandas as pd
from anomalytransfer.transfer.data import KPI
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import anomalytransfer as at
import numpy as np
from glob import glob
from typing import Sequence, Tuple, Dict, Optional, cast
np.seterr(divide='ignore', invalid='ignore')


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
        [timestamps, labels, scores, values], missing=missing)

    return {
        "timestamp": adjusted_timestamps,
        "scores": adjusted_scores,
        "labels": adjusted_labels,
        "values": adjusted_values
    }


def main(TH: int):
    raw_csvs = glob(os.path.join(INPUT, "*.csv"))
    assert len(raw_csvs) > 0

    models = glob(os.path.join(MODEL_PATH, str(TH), "cluster-*"))
    assert len(models) > 0

    time_map = {}
    for raw_csv in raw_csvs:
        print(f"The KPI: {raw_csv}")
        raw_kpi_name = os.path.splitext(os.path.basename(raw_csv))[0]
        time_map[raw_kpi_name] = 0
        raw_kpi = at.utils.load_kpi(raw_csv)
        raw_kpi, _, _ = raw_kpi.standardize()
        raw_kpi.complete_timestamp()

        total_timestamps = []
        total_scores = []
        total_labels = []
        total_values = []

        # get daily KPI
        train_week_day_map, test_week_day_map, test_kpi = raw_kpi.split_days(days=7)

        # get cluster map
        cluster_map = {}   # weekday -> cluster_name
        for cluster in os.listdir(DAILY_OUTPUT):
            data_path = os.path.join(DAILY_OUTPUT, cluster, "data")
            raw_csv_daily = glob(os.path.join(
                data_path, f"{raw_kpi_name}*.csv"))
            raw_csv_daily = [int(os.path.splitext(os.path.basename(csv))[
                                 0][-1]) for csv in raw_csv_daily]
            for daily in raw_csv_daily:
                assert daily not in cluster_map
                cluster_map[daily] = cluster

        # fine-tune with train_kpi
        for weekday, kpi_seq in train_week_day_map.items():
            dst_cluster_name = cluster_map[weekday]
            cluster_model_path = os.path.join(MODEL_PATH, str(TH), dst_cluster_name)
            model = at.transfer.models.AnomalyDetector()
            if os.path.exists(os.path.join(cluster_model_path, "finetune")):
                model.load(cluster_model_path, "finetune")
            else:
                model.load(cluster_model_path, "base")

            for kpi in kpi_seq:
                with run_time() as t:
                    model.fit(kpi, epochs=DATA_EPOCHS, verbose=1)
                time_map[raw_kpi_name] += t.get_time()
            if len(kpi_seq) > 0:
                model.save(cluster_model_path, "finetune")

        # test
        for weekday, kpi_seq in test_week_day_map.items():
            dst_cluster_name = cluster_map[weekday]
            cluster_model_path = os.path.join(MODEL_PATH, str(TH), dst_cluster_name)
            assert os.path.exists(os.path.join(
                cluster_model_path, "finetune")), f"the train stage of {dst_cluster_name} is missed..."

            model = at.transfer.models.AnomalyDetector()
            model.load(cluster_model_path, "finetune")
            for kpi in kpi_seq:
                kpi = cast(KPI, kpi)
                anomaly_scores = model.predict(kpi, verbose=1)
                try:
                    results = get_test_results(
                        timestamps=kpi.timestamps,
                        labels=kpi.labels,
                        scores=anomaly_scores,
                        missing=kpi.missing,
                        values=kpi.values
                    )
                    # results = results['0.0001']['0.98']

                    total_timestamps.extend(results["timestamp"])
                    total_scores.extend(results["scores"])
                    total_labels.extend(results["labels"])
                    total_values.extend(results["values"])
                except:
                    import traceback
                    traceback.print_exc()
                    exit(-1)

        total_timestamps = np.asarray(total_timestamps)
        total_scores = np.asarray(total_scores)
        total_labels = np.asarray(total_labels)
        total_values = np.asarray(total_values)

        sort_idx = np.argsort(total_timestamps)
        total_timestamps = total_timestamps[sort_idx]
        total_scores = total_scores[sort_idx]
        total_values = total_values[sort_idx]
        total_labels = total_labels[sort_idx]

        # # adjust after concatenate
        adjusted_scores = at.utils.adjust_scores(
            labels=total_labels, scores=total_scores)

        dt = pd.DataFrame({
            "ts": total_timestamps,
            "scores": adjusted_scores,
            "values": total_values,
            "label": total_labels,
        })
        # if not os.path.exists(os.path.join(OUTPUT, "transfer")):
        #     os.makedirs(os.path.join(OUTPUT, "transfer"), exist_ok=True)
        if not os.path.exists(os.path.join(OUTPUT, f"transfer_{TH / 10}")):
            os.makedirs(os.path.join(OUTPUT, f"transfer_{TH / 10}"))
        dt.to_csv(os.path.join(OUTPUT, f"transfer_{TH / 10}",
                  f"{raw_kpi_name}.csv"), index=False)

    import json
    json.dump(time_map, open("test_time.json", "w"), indent=4)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s [%(levelname)s]] %(message)s')

    config = at.utils.config()
    CLUSTER_OUTPUT = config.get("CLUSTERING", "output")
    DAILY_OUTPUT = os.path.join(CLUSTER_OUTPUT, "daily_cluster")

    INPUT = config.get('BAGEL', 'input')
    OUTPUT = config.get('TRANSFER_LEARNING', 'output')
    MODEL_PATH = config.get('TRANSFER_LEARNING', 'model_path')
    DATA_EPOCHS = config.getint('TRANSFER_LEARNING', 'data_epochs')

    for th in range(18, 21, 1):
        main(th)
