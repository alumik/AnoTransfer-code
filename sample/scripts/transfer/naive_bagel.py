import os
import logging
import anomalytransfer as at
os.environ["CUDA_VISIBLE_DEVICES"] = '1' 

def main():
    at.utils.mkdirs(OUTPUT)
    file_list = at.utils.file_list(INPUT)
    proglog = at.utils.ProgLog(len(file_list))

    for file in file_list:
        kpi = at.utils.load_kpi(file)
        proglog.log(kpi=kpi.name)
        kpi.complete_timestamp()
        total_minutes = 7 * 24 * 60
        interval = kpi.interval / 60
        num_of_point = int(total_minutes / interval)
        train_kpi, test_kpi = kpi.split_by_idx(num_of_point)

        train_kpi, mean, std = train_kpi.standardize()
        test_kpi, _, _ = test_kpi.standardize(mean=mean, std=std)

        model = at.transfer.AnomalyDetector()
        model.fit(kpi=train_kpi.no_labels(), epochs=EPOCHS)
        anomaly_scores = model.predict(test_kpi)

        results = at.utils.get_test_results(labels=test_kpi.labels,
                                            scores=anomaly_scores,
                                            missing=test_kpi.missing)
        stats = at.utils.get_kpi_stats(kpi, test_kpi)
        at.utils.log_test_results(kpi.name, results=results)

        with open(f'{os.path.join(OUTPUT, kpi.name)}.txt', 'w') as output:
            output.write(f'[result]\n'
                         f'threshold={results.get("threshold")}\n'
                         f'precision={results.get("precision"):.3f}\n'
                         f'recall={results.get("recall"):.3f}\n'
                         f'f1_score={results.get("f1score"):.3f}\n\n'

                         '[overall]\n'
                         f'num_points={stats[0].num_points}\n'
                         f'num_missing_points={stats[0].num_missing}\n'
                         f'missing_rate={stats[0].missing_rate:.6f}\n'
                         f'num_anomaly_points={stats[0].num_anomaly}\n'
                         f'anomaly_rate={stats[0].anomaly_rate:.6f}\n\n'

                         '[test]\n'
                         f'num_points={stats[1].num_points}\n'
                         f'num_missing_points={stats[1].num_missing}\n'
                         f'missing_rate={stats[1].missing_rate:.6f}\n'
                         f'num_anomaly_points={stats[1].num_anomaly}\n'
                         f'anomaly_rate={stats[1].anomaly_rate:.6f}\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s [%(levelname)s]] %(message)s')

    config = at.utils.config()
    NUM_THREADS = config.getint('COMMON', 'num_threads')
    EPOCHS = config.getint('BAGEL', 'epochs')
    INPUT = config.get('BAGEL', 'input')
    OUTPUT = config.get('BAGEL', 'output')

    # at.utils.set_num_threads(NUM_THREADS)
    main()
