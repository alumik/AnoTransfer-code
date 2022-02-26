import torch
import pandas as pd
from anomalytransfer.transfer.data import KPI
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import logging
import anomalytransfer as at
import numpy as np
from glob import glob
from typing import Sequence, Tuple, Dict, Optional, cast
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd


def main():
    raw_csvs = glob(os.path.join(INPUT, "*.csv"))[0:1]
    assert len(raw_csvs) > 0

    time_map = {}
    result = {}
    for raw_csv in raw_csvs:
        for exp in range(10):
            print(f"The KPI: {raw_csv}")
            raw_kpi_name = os.path.splitext(os.path.basename(raw_csv))[0]
            time_map[raw_kpi_name] = 0
            raw_kpi = at.utils.load_kpi(raw_csv)
            raw_kpi, _, _ = raw_kpi.standardize()
            raw_kpi.complete_timestamp()

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
                cluster_model_path = os.path.join(MODEL_PATH, dst_cluster_name)
                model = at.transfer.models.AnomalyDetector()
                if os.path.exists(os.path.join(cluster_model_path, "finetune")):
                    model.load(cluster_model_path, "finetune")
                else:
                    model.load(cluster_model_path, "base")

                for kpi in kpi_seq:
                    history = model.fit(kpi, epochs=DATA_EPOCHS, verbose=1)
                    result[f"ts_{exp}"] = history['ts']
                    result[f"loss_{exp}"] = history['loss']
                if len(kpi_seq) > 0:
                    model.save(cluster_model_path, "finetune")
    dt = pd.DataFrame(result)
    dt.to_csv("at.csv", index=False)


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

    main()
