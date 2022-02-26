import numpy as np

from typing import Sequence, Tuple


def linear_interpolation(timestamp: Sequence, arrays: Sequence[Sequence]) \
        -> Tuple[np.ndarray, np.ndarray, Sequence[np.ndarray]]:
    timestamp = np.asarray(timestamp, np.int64)
    if len(timestamp.shape) != 1:
        raise ValueError('`timestamp` must be a 1-D array')

    arrays = [np.asarray(array) for array in arrays]
    for i, array in enumerate(arrays):
        if array.shape != timestamp.shape:
            raise ValueError(f'The shape of ``arrays[{i}]`` does not agree with '
                             f'the shape of `timestamp` ({array.shape} vs {timestamp.shape})')

    src_index = np.argsort(timestamp)
    timestamp_sorted = timestamp[src_index]
    intervals = np.unique(np.diff(timestamp_sorted))
    interval = np.min(intervals)
    if interval == 0:
        raise ValueError('Duplicated values in `timestamp`')
    for itv in intervals:
        if itv % interval != 0:
            raise ValueError('Not all intervals in `timestamp` are multiples of the minimum interval')

    length = (timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1
    ret_timestamp = np.arange(timestamp_sorted[0], timestamp_sorted[-1] + interval, interval, dtype=np.int64)
    ret_missing = np.ones([length], dtype=np.int32)
    ret_arrays = [np.zeros([length], dtype=array.dtype) for array in arrays]
    dst_index = np.asarray((timestamp_sorted - timestamp_sorted[0]) // interval, dtype=np.int)
    ret_missing[dst_index] = 0
    miss_index = np.argwhere(ret_missing == 1)
    for ret_array, array in zip(ret_arrays, arrays):
        ret_array[dst_index] = array[src_index]

    for ret_array in ret_arrays:
        if len(miss_index) > 0:
            neg = miss_index.reshape(len(miss_index))
            pos_index = np.argwhere(ret_missing == 0)
            pos = pos_index.reshape(len(pos_index))
            pos_values = ret_array[pos]
            neg_values = np.interp(neg, pos, pos_values)
            ret_array[neg] = neg_values

    return ret_timestamp, ret_missing, ret_arrays


def standardize(values: Sequence, mean: float = None, std: float = None) -> Tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float32)
    if len(values.shape) != 1:
        raise ValueError('`values` must be a 1-D array')
    if (mean is None) != (std is None):
        raise ValueError('`mean` and `std` must be both None or not None')

    if mean is None:
        val = values
        mean = val.mean()
        std = val.std()

    return (values - mean) / std, mean, std


def down_sampling(arrays: Sequence[Sequence], step: int) -> Tuple[Sequence, ...]:
    ret_arrays = []
    for array in arrays:
        array = array[::step]
        ret_arrays.append(array)
    return tuple(ret_arrays)
