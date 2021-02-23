# This is a sample Python script.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import savgol_filter

plt.style.use('seaborn-poster')


def savitzky_golay_filtering(timeseries, wnds=[11, 7], orders=[2, 4], debug=True):
    interp_ts = pd.Series(timeseries)
    interp_ts = interp_ts.interpolate(method='linear', limit=14)
    smooth_ts = interp_ts
    wnd, order = wnds[0], orders[0]
    F = 1e8
    W = None
    it = 0
    while True:
        smoother_ts = savgol_filter(smooth_ts, window_length=wnd, polyorder=order)
        diff = smoother_ts - interp_ts
        sign = diff > 0
        if W is None:
            W = 1 - np.abs(diff) / np.max(np.abs(diff)) * sign
            wnd, order = wnds[1], orders[1]
        fitting_score = np.sum(np.abs(diff) * W)
        print(it, ' : ', fitting_score)
        if fitting_score > F:
            break
        else:
            F = fitting_score
            it += 1
        smooth_ts = smoother_ts * sign + interp_ts * (1 - sign)
    if debug:
        return smooth_ts, interp_ts
    return smooth_ts


def savitzky_golay_piecewise(xvals, data, kernel=11, order=4):
    dx = xvals[0] - xvals[1]
    turnpoint = 0
    last = len(xvals)
    if xvals[1] > xvals[0]:  # x is increasing?
        for i in range(1, last):  # yes
            if xvals[i] < xvals[i - 1]:  # search where x starts to fall
                turnpoint = i
                break
    else:  # no, x is decreasing
        for i in range(1, last):  # search where it starts to rise
            if xvals[i] > xvals[i - 1]:
                turnpoint = i
                break
    if turnpoint == 0:  # no change in direction of x
        return savgol_filter(data, kernel, order, delta=dx)
    else:
        # smooth the first piece
        firstpart = savgol_filter(data[0:turnpoint], kernel, order, delta=dx)
        # recursively smooth the rest
        rest = savitzky_golay_piecewise(xvals[turnpoint:], data[turnpoint:], kernel, order)
        return np.concatenate((firstpart, rest))


def read_data(file):
    arr = []
    with open(file) as file_data:
        for line in file_data.readlines():
            arr.append(line.replace('\n', '').split(' '))
    return arr


def read_label(file):
    arr = []
    with open(file) as file_data:
        for line in file_data.readlines()[0:2]:
            arr.append(line.replace('\n', ''))
        print('\n'.join(arr))
    return '|'.join(arr)


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def get_median_filtered(signal, threshold=3):
    signal = signal.copy()
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal


def replace_outliers(data, m=2, u=10, debug=False):
    data = np.copy(data)
    for i in range(m, len(data) - m):
        meanlim = np.mean(data[i - m:i + m])
        stdlim = np.std(data[i - m:i + m])

        if abs(data[i] - meanlim) < u * stdlim:

            if debug:
                print(data[i], 'changed to ', meanlim)
            data[i] = meanlim
    return data


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def spline(y_new):
    from scipy.interpolate import make_interp_spline, BSpline

    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(0, 1100, 300)
    spl = make_interp_spline(np.linspace(0, 1100, len(y_new)), y_new, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    return xnew, power_smooth


def zipData(x, y, start, step):
    stop = x[np.argmax(y[start:])]
    mask = np.where(np.logical_and(x > start, x < stop))
    end = np.where(x >= stop)
    x1 = x.copy()
    x1[mask] = start + (x[mask] - start) / (abs(start - stop) / step)
    x1[end] = x[end] - (abs(start - stop)) + (abs(start - stop) / step)
    # print('\n'.join([f'{str(i)} {str(j)}' for i, j in zip(x, temp)]))
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(xtemp, temp, 'r-', label='shrinked')
    # ax.plot(x[np.where(xtemp)], y[np.where(xtemp)], label='masked data')
    # ax.plot(x, y, 'y--', label='raw data')
    # ax.plot(x1, y, 'r-', label='zipped data')
    # ax.plot(x[int(stop):], y[stop:], label='col data')
    # plt.legend()
    # plt.show()
    return x1


def dataalign(path, lim=-1, sep=' '):
    data = np.loadtxt(path, delimiter=sep, dtype=np.float, skiprows=2)
    lab = read_label(path)
    x, y = data.T
    x = x - min(x)

    x = x[np.where(x < 300)]
    ax.plot(x, y[:len(x)], label=lab.split("|")[1])


def dataRes(path, lim=-1, sep=' '):
    data = np.loadtxt(path, delimiter=sep, dtype=np.float, skiprows=2)
    lab = read_label(path)
    x, y = data.T
    x = x - min(x)
    x = x[np.where(x < 1100)]
    t300, max_t = y[300], max(y)
    return abs(t300 - max_t)


def data_normal(path):
    data = np.loadtxt(path, delimiter=' ', dtype=np.float, skiprows=2)
    lab = read_label(path)

    x, y = data.T
    ax.plot(x, y, color="y", label="Raw curve")

    ax.set_title(lab)
    ax.ylabel('Current(V)', fontweight='bold')
    ax.xlabel('Time(s)', fontweight='bold')

    plt.legend(loc="lower right")


def dataprocess(path):
    data = np.loadtxt(path, delimiter=' ', dtype=np.float, skiprows=2)
    lab = read_label(path)

    x, y = data.T
    # data_norm = reject_outliers(data[::100], m=10)
    t_x = np.arange(x[0], x[-1], 10)
    # n_x, n_y = data.T
    # x1 = smoothify(x)  # windoxw size 51, polynomial order 3
    print(y)
    # yhat = scipy.signal.savgol_filter(y, 51, 3)
    print(y.shape)
    # yhat = savgol_filter(y[::100], 11, 3)
    y_savgol = savgol_filter(get_median_filtered(y[::100]), 3, 2)
    # yhat = y
    print(y_savgol.shape)

    # Smoothing here
    # xnew = np.linspace(x[0], x[-1], num=35, endpoint=True)
    # ynew = f(xnew)

    y_savgol_median = get_median_filtered(y_savgol, 2)
    y_moving_average = moving_average(y, int(len(y) / 10))
    f = interpolate.interp1d(x, y, kind='previous')
    x_int = np.linspace(x[0], x[-1], 10)
    y_int = f(x_int)
    plt.grid()
    f_x, f_y = spline(y_moving_average)
    print(len(f_x), len(f_y))
    f_x1 = zipData(f_x, f_y, 80, 20)
    ax.plot(f_x1 + 200, (f_y) - min(f_y[80:]), label=lab.split("|")[1])
    # ax.plot(*spline(y_savgol_median), color='tomato', label='Savgol Median')

    # ax.plot(*spline(y_new), label='Smoothed curve')
    ax.set_title(lab)

    plt.ylim(bottom=-1E-5, top=2.5E-5)
    plt.xlim(left=200, right=600)
    plt.legend(loc="lower right")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import os

    text_size = 14
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.xticks(weight='bold', size=text_size)
    plt.yticks(weight='bold', size=text_size)
    plt.ylabel('Current(V)', fontweight='bold', size=text_size)
    plt.xlabel('Time(s)', fontweight='bold', size=text_size)
    filetype = '.txt'
    for root, dirs, files in os.walk('data/kıymetliler'):
        for file in sorted(files, reverse=True):
            if file.lower().endswith(filetype.lower()):
                dataprocess(os.path.join(root, file))

    plt.title('Calibration Graph', weight='bold', size=text_size)
    plt.legend(loc=0, prop={'size': 9})
    # plt.gca().invert_yaxis()
    plt.savefig('data/calibration.png', dpi=300)
    fig.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
