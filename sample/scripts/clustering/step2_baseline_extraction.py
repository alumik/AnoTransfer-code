import os
import pandas as pd
import anomalytransfer as at


def main():
    at.utils.mkdirs(OUTPUT)
    file_list = at.utils.file_list(INPUT)
    progbar = at.utils.ProgBar(len(file_list), interval=0.5, unit_name='file')
    print('Extracting baselines...')

    for file in file_list:
        filename = at.utils.filename(file)
        df = pd.read_csv(file)
        values = at.clustering.baseline_extraction.smoothing_extreme_values(df.value)
        standardized = at.clustering.baseline_extraction.extract_baseline(values, window_size=WINDOW_SIZE)
        df = pd.DataFrame({'timestamp': df.timestamp.iloc[WINDOW_SIZE - 1:], 'value': standardized[0]})
        df.to_csv(os.path.join(OUTPUT, filename + '.csv'), index=False)
        progbar.add(1)


if __name__ == '__main__':
    config = at.utils.config()

    INPUT = config.get('CLUSTERING_BASELINE_EXTRACTION', 'input')
    OUTPUT = config.get('CLUSTERING_BASELINE_EXTRACTION', 'output')
    WINDOW_SIZE = config.getint('CLUSTERING_BASELINE_EXTRACTION', 'window_size')

    main()
