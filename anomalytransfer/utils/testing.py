import torch
import numpy as np
import anomalytransfer as at

from typing import Sequence, Dict, Tuple, Optional
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support


def adjust_scores(labels: np.ndarray,
                   scores: np.ndarray,
                   delay: Optional[int] = None,
                   inplace: bool = False) -> np.ndarray:
    if np.shape(scores) != np.shape(labels):
        raise ValueError('`labels` and `scores` must have same shape')
    if delay is None:
        delay = len(scores)
    splits = np.where(labels[1:] != labels[:-1])[0] + 1
    is_anomaly = labels[0] == 1
    adjusted_scores = np.copy(scores) if not inplace else scores
    pos = 0
    for part in splits:
        if is_anomaly:
            ptr = min(pos + delay + 1, part)
            adjusted_scores[pos: ptr] = np.max(adjusted_scores[pos: ptr])
            adjusted_scores[ptr: part] = np.maximum(adjusted_scores[ptr: part], adjusted_scores[pos])
        is_anomaly = not is_anomaly
        pos = part
    part = len(labels)
    if is_anomaly:
        ptr = min(pos + delay + 1, part)
        adjusted_scores[pos: part] = np.max(adjusted_scores[pos: ptr])
    return adjusted_scores


def _ignore_missing(series_list: Sequence, missing: np.ndarray) -> Tuple[np.ndarray, ...]:
    ret = []
    for series in series_list:
        series = np.copy(series)
        ret.append(series[missing != 1])
    return tuple(ret)


def _best_f1score(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true=labels, probas_pred=scores, pos_label=1.0)
    f1score = 2 * precision * recall / np.clip(precision + recall, a_min=1e-8, a_max=None)

    best_threshold = thresholds[np.argmax(f1score)]
    best_precision = precision[np.argmax(f1score)]
    best_recall = recall[np.argmax(f1score)]

    return best_threshold, best_precision, best_recall, np.max(f1score)


def _f1score_given_alarms(labels: Sequence, alarms: Sequence) -> Tuple[float, float, float, float]:
    pred = np.zeros(len(labels))
    pred[alarms] = 1
    precision, recall, f1score, _ = precision_recall_fscore_support(y_true=labels,
                                                                    y_pred=pred,
                                                                    average='binary',
                                                                    pos_label=1)
    return np.nan, precision, recall, f1score


def set_num_threads(num_threads: int):
    torch.set_num_threads(num_threads)


def get_test_results(labels: np.ndarray,
                     scores: np.ndarray,
                     missing: np.ndarray,
                     window_size: int = 120,
                     use_spot: bool = False,
                     **kwargs) -> Dict:
    labels = labels[window_size - 1:]
    scores = scores[window_size - 1:]
    missing = missing[window_size - 1:]
    scores = adjust_scores(labels=labels, scores=scores)
    adjusted_labels, adjusted_scores = _ignore_missing([labels, scores], missing=missing)

    if use_spot:
        n_init = 1000
        init_data = adjusted_scores[:n_init]
        data = adjusted_scores[n_init:]
        labels = adjusted_labels[n_init:]

        result = {}
        for risk in kwargs.get('risks', [0.0001]):
            risk_result = {}
            for level in kwargs.get('levels', [0.98]):
                threshold, precision, recall, f1score = -1, -1, -1, -1
                try:
                    spot = at.transfer.SPOT(q=risk)
                    spot.fit(init_data, data)
                    spot.initialize(level=level)
                    r = spot.run()
                    alarms = r['alarms']
                    threshold, precision, recall, f1score = _f1score_given_alarms(labels=labels, alarms=alarms)
                except Exception:
                    pass
                    # import traceback
                    # traceback.print_exc()
                finally:
                    level_result = {
                        'threshold': threshold,
                        'precision': precision,
                        'recall': recall,
                        'f1score': f1score
                    }
                    risk_result[f'{level}'] = level_result
            result[f'{risk}'] = risk_result
        return result
    else:
        try:
            threshold, precision, recall, f1score = _best_f1score(labels=adjusted_labels, scores=adjusted_scores)
            return {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1score': f1score,
                "scores": adjusted_scores,
                "labels": adjusted_labels
            }
        except:
            import traceback
            traceback.print_exc()
            return {
                "scores": adjusted_scores,
                "labels": adjusted_labels
            }
