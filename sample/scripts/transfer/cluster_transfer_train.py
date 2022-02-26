import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
import anomalytransfer as at
from glob import glob
from utils import run_time

from typing import Sequence, Tuple, Dict, Optional


def make_base_model(kpi: at.transfer.data.KPI, model_path: str, epochs: int):
    kpi.complete_timestamp()
    kpi, _, _ = kpi.standardize()
    at.utils.mkdirs(os.path.join(model_path, kpi.name))
    model = at.transfer.models.AnomalyDetector()
    model.fit(kpi=kpi.no_labels(), epochs=epochs, verbose=1)
    model.save(path=model_path, name=kpi.name)


def train_test(train_kpi: at.transfer.data.KPI,
               epochs: int,
               test_kpi: at.transfer.data.KPI = None,
               mask: Optional[Sequence] = None,
               **kwargs) -> float:
    model = at.transfer.models.AnomalyDetector()
    if mask is not None:
        model.load_partial(path=kwargs.get('model_path'), name=kwargs.get('base_kpi').name, mask=mask)
        model.freeze(mask)
        model.fit(kpi=train_kpi.no_labels(), epochs=epochs, verbose=1)
        model.unfreeze(mask)
    model.fit(kpi=train_kpi.no_labels(), epochs=epochs, verbose=1)
    if test_kpi is not None and test_kpi.labels is not None:
        anomaly_scores = model.predict(test_kpi, verbose=1)
        results = at.utils.get_test_results(labels=test_kpi.labels,
                                            scores=anomaly_scores,
                                            missing=test_kpi.missing,
                                            use_spot=False)
        at.utils.log_test_results(name=test_kpi.name, results=results)
        return results['f1score']
    else:
        return None

def transfer_learning(base_kpi: at.transfer.data.KPI,
                      data_kpi: at.transfer.data.KPI,
                      train_ratio: float,
                      model_path: str,
                      epochs: int,
                      TH: int) -> Optional[Dict]:
    result = {}
    progress = at.utils.ProgLog(3, indent=3)

    progress.log(step='Preparing KPI...')
    data_kpi.complete_timestamp()
    train_kpi = data_kpi
    train_kpi, mean, std = train_kpi.standardize()

    progress.log(step='Training and testing before transfer...')
    result['f1score_pre_transfer'] = train_test(train_kpi=train_kpi,
                                                epochs=epochs)

    progress.log(step='Training and testing after transfer...')
    sbd = at.transfer.models.sbd_(base_kpi, data_kpi)
    mask = at.transfer.models.find_optimal_mask(sbd,
                                                threshold=args.th / 10,
                                                # threshold=0.3,
                                                less_mask=((1, 1, 1), (1, 1, 1)),
                                                greater_mask=((1, 1, 0), (0, 1, 1)))
    result['f1score_post_transfer'] = train_test(train_kpi=train_kpi,
                                                 epochs=epochs,
                                                 mask=mask,
                                                 model_path=model_path,
                                                 base_kpi=base_kpi)

    return result


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


def main(TH: int):
    at.utils.mkdirs(OUTPUT, MODEL_PATH)
    clusters = os.listdir(INPUT)

    cluster_prog = at.utils.ProgLog(len(clusters))
    time_map = {}
    for cluster in clusters:
        cluster_model_path = os.path.join(MODEL_PATH, str(TH), cluster)
        cluster_name = os.path.basename(cluster_model_path)
        time_map[cluster_name] = 0
        cluster_prog.log(cluster=cluster)

        base, data = cluster_data(os.path.join(INPUT, cluster))
        file_list = at.utils.file_list(data)
        step_progress = at.utils.ProgLog(2, indent=1)

        step_progress.log(step='Making base model...', cluster=cluster)
        base_kpi = at.utils.load_kpi(base)
        make_base_model(kpi=base_kpi, model_path=cluster_model_path, epochs=BASE_EPOCHS)

    #     step_progress.log(step='Performing transfer learning...', cluster=cluster)
    #     output_path = os.path.join(OUTPUT, f'{cluster}.csv')
    #     with open(output_path, 'w') as output:
    #         output.write('kpi_name,f1score_pre_transfer,f1score_post_transfer\n')

    #     file_progress = at.utils.ProgLog(len(file_list), indent=2)
    #     for file in file_list:
    #         data_kpi = at.utils.load_kpi(file)
    #         file_progress.log(kpi=data_kpi.name, cluster=cluster)
    #         with run_time() as t:
    #             result = transfer_learning(base_kpi=base_kpi,
    #                                     data_kpi=data_kpi,
    #                                     train_ratio=RATIO,
    #                                     model_path=cluster_model_path,
    #                                     epochs=DATA_EPOCHS)
    #         time_map[cluster_name] += t.get_time()
    #         if result is not None and \
    #             result.get("f1score_pre_transfer") is not None and \
    #             result.get("f1score_post_transfer") is not None:
    #             with open(output_path, 'a') as output:
    #                 output.write(f'{data_kpi.name},'
    #                              f'{result.get("f1score_pre_transfer"):.3f},'
    #                              f'{result.get("f1score_post_transfer"):.3f}\n')
    # import json
    # json.dump(time_map, open("train_cluster_time.json", "w"), indent=4)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s [%(levelname)s]] %(message)s')

    config = at.utils.config()
    CLUSTER_OUTPUT = config.get("CLUSTERING", "output")
    BASE_EPOCHS = config.getint('TRANSFER_LEARNING', 'base_epochs')
    DATA_EPOCHS = config.getint('TRANSFER_LEARNING', 'data_epochs')
    INPUT = config.get('TRANSFER_LEARNING', 'input')
    OUTPUT = config.get('TRANSFER_LEARNING', 'output')
    MODEL_PATH = config.get('TRANSFER_LEARNING', 'model_path')
    RATIO = config.getfloat('TRANSFER_LEARNING', 'ratio')

    for th in range(0, 21, 1):
        main(th)
