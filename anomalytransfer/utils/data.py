import os
import pandas as pd
import anomalytransfer as at

from typing import Sequence, Tuple


def filename(file: str) -> str:
    return os.path.splitext(os.path.basename(file))[0]
 

def mkdirs(*dir_list):
    for directory in dir_list:
        os.makedirs(directory, exist_ok=True)


def file_list(path: str) -> Sequence:
    if os.path.isdir(path):
        return [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".csv")]
    else:
        return [path]


def load_kpi(file: str, **kwargs) -> at.transfer.data.KPI:
    df = pd.read_csv(file, **kwargs)
    df.dropna(0, inplace=True)
    return at.transfer.data.KPI(timestamps=df.timestamp,
                                values=df.value,
                                labels=df.get('label', None),
                                name=filename(file))


class KPIStats:

    def __init__(self, kpi: at.transfer.data.KPI):
        self.num_points = len(kpi.values)
        self.num_missing = len(kpi.missing[kpi.missing == 1])
        self.num_anomaly = len(kpi.labels[kpi.labels == 1])
        self.missing_rate = self.num_missing / self.num_points
        self.anomaly_rate = self.num_anomaly / self.num_points


def get_kpi_stats(*kpis: at.transfer.data.KPI) -> Tuple[KPIStats, ...]:
    ret = []
    for kpi in kpis:
        ret.append(KPIStats(kpi))
    return tuple(ret)
