import numpy as np
import anomalytransfer as at

from typing import Sequence, Tuple


def smoothing_extreme_values(values: Sequence) -> np.ndarray:
    values = np.asarray(values, np.float32)
    if len(values.shape) != 1:
        raise ValueError('`values` must be a 1-D array')

    abnormal_portion = 0.05
    values_deviation = np.abs(values)

    abnormal_max = np.max(values_deviation)
    abnormal_index = np.argwhere(values_deviation >= abnormal_max * (1 - abnormal_portion))
    abnormal = abnormal_index.reshape(len(abnormal_index))
    normal_index = np.argwhere(values_deviation < abnormal_max * (1 - abnormal_portion))
    normal = normal_index.reshape(len(normal_index))
    normal_values = values[normal]
    abnormal_values = np.interp(abnormal, normal, normal_values)
    values[abnormal] = abnormal_values

    return values


def extract_baseline(values: Sequence, window_size: int) -> Tuple[np.ndarray, float, float]:
    baseline = np.convolve(values, np.ones((window_size,)) / window_size, mode='valid')
    return at.clustering.preprocessing.standardize(baseline)
