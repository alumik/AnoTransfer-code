import os
import pandas as pd
import anomalytransfer as at


def main():
    at.utils.mkdirs(OUTPUT_DAILY, OUTPUT_WEEKLY)
    file_list = at.utils.file_list(INPUT)
    progbar = at.utils.ProgBar(len(file_list), interval=0.5, unit_name='file')
    print('Extracting sub-curves...')

    for file in file_list:
        filename = at.utils.filename(file)
        df = pd.read_csv(file)
        daily_average, ts_average = at.clustering.average.get_daily_average(
            *at.clustering.average.group_data_by_weekday(timestamps=df.timestamp, values=df.value))
        for i in range(len(daily_average)):
            df = pd.DataFrame({
                'timestamp': ts_average[i], 
                'value': daily_average[i]
            })
            
            df.to_csv(os.path.join(OUTPUT_DAILY, filename + f'_wd{i}.csv'), index=False)
        weekly_average, ts_average = at.clustering.average.get_weekly_average(daily_average, ts_average)
        df = pd.DataFrame({
            'timestamp': ts_average,
            'value': weekly_average
        })
        df.to_csv(os.path.join(OUTPUT_WEEKLY, filename + '.csv'), index=False)
        progbar.add(1)


if __name__ == '__main__':
    config = at.utils.config()

    INPUT = config.get('CLUSTERING_AVERAGE', 'input')
    OUTPUT_DAILY = config.get('CLUSTERING_AVERAGE', 'output_daily')
    OUTPUT_WEEKLY = config.get('CLUSTERING_AVERAGE', 'output_weekly')

    main()
