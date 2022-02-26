import datetime
import numpy as np

from typing import Sequence, Tuple


def _get_weekday(timestamps: Sequence) -> Sequence:
    return [datetime.datetime.fromtimestamp(t).weekday() for t in timestamps]


def group_data_by_weekday(timestamps: Sequence, values: Sequence) -> Tuple[Sequence, Sequence]:
    timestamps = np.asarray(timestamps, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    weekday = _get_weekday(timestamps)
    grouped_data = [[], [], [], [], [], [], []]
    grouped_ts = [[], [], [], [], [], [], []]
    current_weekday = weekday[0]
    current_index = 0
    for i in range(len(weekday)):
        if weekday[i] != current_weekday:
            if current_index != 0:
                # ! Add more 119 points (Bagel ignore the first 119 points!)
                grouped_data[current_weekday].append(values[(current_index-119):i])
                grouped_ts[current_weekday].append(timestamps[(current_index-119):i])
            current_weekday = weekday[i]
            current_index = i
    return grouped_data, grouped_ts


def get_daily_average(grouped_data: Sequence, grouped_ts: Sequence) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
    daily_average = []
    ts_average = []
    for weekday, ts in zip(grouped_data, grouped_ts):
        daily_average.append(np.mean(weekday, axis=0))
        ts_average.append(ts[0])
    return daily_average, ts_average


def get_weekly_average(daily_average: Sequence, ts_average: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    return np.concatenate(daily_average), np.concatenate(ts_average)
