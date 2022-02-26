import os
import datetime
import numpy as np
import anomalytransfer as at
import matplotlib.pyplot as plt


def _expand(a: np.ndarray) -> np.ndarray:
    ret = np.copy(a)
    for i in range(length := len(a)):
        if a[i] == 1:
            if i - 1 >= 0:
                ret[i - 1] = 1
            if i + 1 < length:
                ret[i + 1] = 1
    return ret


def _plot_kpi(kpi: at.transfer.data.KPI):
    x = [datetime.datetime.fromtimestamp(timestamp) for timestamp in kpi.timestamps]
    y_anomaly, y_missing = np.copy(kpi.values), np.copy(kpi.values)
    y_anomaly[_expand(kpi.labels) == 0] = np.inf
    y_missing[_expand(kpi.missing) == 0] = np.inf
    plt.plot(x, kpi.values)
    plt.plot(x, y_anomaly, color='red')
    plt.plot(x, y_missing, color='orange')
    plt.title(kpi.name)
    plt.ylim(-7.5, 7.5)


def main():
    at.utils.mkdirs(OUTPUT)
    file_list = at.utils.file_list(INPUT)

    plt.figure(figsize=(FIG_W, FIG_H), dpi=FIG_DPI)
    progbar = at.utils.ProgBar(len(file_list), interval=0.5, unit_name='file')
    print('Plotting...')

    for file in file_list:
        kpi = at.utils.load_kpi(file)
        kpi, _, _ = kpi.standardize()
        kpi.complete_timestamp()
        _plot_kpi(kpi)
        plt.savefig(os.path.join(OUTPUT, f'{kpi.name}.png'))
        plt.clf()
        progbar.add(1)


if __name__ == '__main__':
    config = at.utils.config()

    INPUT = config.get('PLOT_KPI', 'input')
    OUTPUT = config.get('PLOT_KPI', 'output')
    FIG_W = config.getfloat('PLOT_KPI', 'fig_width')
    FIG_H = config.getfloat('PLOT_KPI', 'fig_height')
    FIG_DPI = config.getint('PLOT_KPI', 'fig_dpi')

    main()
