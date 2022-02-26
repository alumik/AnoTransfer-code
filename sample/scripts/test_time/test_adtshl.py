import os
import logging
import anomalytransfer as at
os.environ["CUDA_VISIBLE_DEVICES"] = '' 
import pandas as pd

def main():
    at.utils.mkdirs(OUTPUT)
    file_list = at.utils.file_list(INPUT)[0:1]  # only one
    proglog = at.utils.ProgLog(len(file_list))

    result = {}
    for file in file_list:
        for exp in range(10):
            kpi = at.utils.load_kpi(file)
            proglog.log(kpi=kpi.name)
            kpi.complete_timestamp()
            total_minutes = 24 * 60
            interval = kpi.interval / 60
            num_of_point = int(total_minutes / interval)
            train_kpi, test_kpi = kpi.split_by_idx(num_of_point)

            train_kpi, mean, std = train_kpi.standardize()
            test_kpi, _, _ = test_kpi.standardize(mean=mean, std=std)

            model = at.transfer.AnomalyDetector()
            history = model.fit(kpi=train_kpi.no_labels(), epochs=EPOCHS)
            result[f"ts_{exp}"] = history['ts']
            result[f"loss_{exp}"] = history['loss']
    dt = pd.DataFrame(result)
    # dt.to_csv("adtshl.csv", index=False)
    dt.to_csv("bagel.csv", index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s [%(levelname)s]] %(message)s')

    config = at.utils.config()
    NUM_THREADS = config.getint('COMMON', 'num_threads')
    EPOCHS = config.getint('BAGEL', 'epochs')
    INPUT = config.get('BAGEL', 'input')
    OUTPUT = config.get('BAGEL', 'output')

    # at.utils.set_num_threads(NUM_THREADS)
    main()
