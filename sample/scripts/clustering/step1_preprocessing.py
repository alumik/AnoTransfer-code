import os
import pandas as pd
import anomalytransfer as at


def main():
    at.utils.mkdirs(OUTPUT)
    file_list = at.utils.file_list(INPUT)
    progbar = at.utils.ProgBar(len(file_list), interval=0.5, unit_name='file')
    print('Preprocessing...')

    for file in file_list:
        filename = at.utils.filename(file)
        df = pd.read_csv(file)
        timestamps, _, ret_arrays = at.clustering.preprocessing.linear_interpolation(df.timestamp, [df.value])
        # ! Don't downsample before train bagel
        # timestamps, values = at.clustering.preprocessing.down_sampling([timestamps, ret_arrays[0]],
        #                                                                step=DOWN_SAMPLING_STEP)
        values, _, _ = at.clustering.preprocessing.standardize(ret_arrays[0])
        df = pd.DataFrame({'timestamp': timestamps, 'value': values})
        df.to_csv(os.path.join(OUTPUT, filename + '.csv'), index=False)
        progbar.add(1)


if __name__ == '__main__':
    config = at.utils.config()

    INPUT = config.get('CLUSTERING_PREPROCESSING', 'input')
    OUTPUT = config.get('CLUSTERING_PREPROCESSING', 'output')
    DOWN_SAMPLING_STEP = config.getint('CLUSTERING_PREPROCESSING', 'down_sampling_step')

    main()
