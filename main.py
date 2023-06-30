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


def read_spike_title(file):
    return int(open(file).readlines()[0].replace('\n', '').replace('Title: ', ''))


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


def moving_average(x, y, w):
    y = np.convolve(y, np.ones(w), 'valid') / w
    return np.linspace(x[0], x[-1], len(y)), y


def spline_b(y_new, w=60):
    from scipy.interpolate import make_interp_spline, BSpline
    y_new = y_new[::w]
    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(0, 300, 300)
    spl = make_interp_spline(np.linspace(0, 300, len(y_new)), y_new, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    return xnew, power_smooth


def spline(y_new):
    from scipy.interpolate import make_interp_spline, BSpline

    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(0, 1100, 300)
    spl = make_interp_spline(np.linspace(0, 1100, len(y_new)), y_new, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    return xnew, power_smooth


def spline_w_spike(y_new, spike):
    from scipy.interpolate import make_interp_spline, BSpline
    # 300 represents number of points to make between T.min and T.max
    x_start = np.linspace(-300, 0, 300)
    x_stop = np.linspace(0, 800, 300)
    spl_start = make_interp_spline(np.linspace(-300, 0, len(y_new[:spike])), y_new[:spike], k=3)  # type: BSpline
    spl_stop = make_interp_spline(np.linspace(0, 800, len(y_new[spike:])), y_new[spike:], k=3)  # type: BSpline
    power_smooth_start = spl_start(x_start)
    power_smooth_stop = spl_stop(x_stop)
    x = [*x_start, *x_stop]
    y = [*power_smooth_start, *power_smooth_stop]

    return x, y


def zipData(x, y, start, step):
    stop = x[start + np.argmax(y[start:])]
    mask = np.where(np.logical_and(x > start, x < stop))
    end = np.where(x >= stop)
    # print('stop=', stop)
    x1 = x.copy()
    x1[mask] = start + (x[mask] - start) / (abs(start - stop) / step)
    x1[end] = x[end] - (abs(start - stop)) + (abs(start - stop) / step)
    # y -= y[start]
    # y = y - y[start]
    # # y /= (y[end][0] - y[start])
    # x1 -= x[start]
    return x1, y


def dataalign(path, lim=-1, sep=' '):
    data = np.loadtxt(path, delimiter=sep, dtype=np.float, skiprows=2)
    lab = read_label(path)
    x, y = data.T
    x = x - min(x)

    # x = x[np.where(x < 300)]
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
    ax.ylabel('-Z\'\'(Ω)', fontweight='bold')
    ax.xlabel('Z\'(Ω)', fontweight='bold')
    plt.legend(loc="lower right")


def dataPipe_test(path):
    data = np.loadtxt(path, delimiter=' ', dtype=np.float, skiprows=2)
    lab = read_label(path)
    spike = read_spike_title(path)
    x, y = data.T
    x, y = zipData(x, y, spike, 20)

    ax.plot(x, y, label=lab.split("|")[1])
    ax.plot(x[spike], y[spike], 'b^', label='spike')


def sigmoid(x, mi, mx): return mi + (mx - mi) * (lambda t: (1 + 200 ** (-t + 0.5)) ** (-1))((x - mi) / (mx - mi))


def divide_sample(x, y, spike):
    y_0 = y[np.argwhere(x <= spike)]
    x_0 = x[np.argwhere(x <= spike)]

    stop = x[spike + np.argmax(y[spike:])]
    mask = np.where(np.logical_and(x > spike, x < stop))
    x_1 = x[mask]
    y_1 = y[mask]
    x_2 = x[x >= stop]
    y_2 = y[x >= stop]

    return (x_0, y_0), (x_1, y_1), (x_2, y_2)


def bspline(x, y, w=40):
    from scipy.interpolate import splrep, splev
    bspl = splrep(x[::w], y[::w], s=0)
    # values for the x axis
    x_smooth = np.linspace(min(x), max(x), 1000)
    # get y values from interpolated curve
    bspl_y = splev(x_smooth, bspl)
    return x_smooth, bspl_y


def dataprocess(path):
    data = np.loadtxt(path, delimiter=' ', dtype=np.float, skiprows=2)
    lab = read_label(path)
    spike = read_spike_title(path)
    x, y = data.T
    initial_val = np.argwhere(x > spike)[0][0]
    p0, p1, p2 = divide_sample(x, y, spike)
    # x, y = spline_w_spike(y, initial_val)
    # # initial_val -= initial_val
    # plt.grid()
    # # y=savitzky_golay_piecewise(x,y)
    # # x, y = zipData(np.array(x), np.array(y), 0, 20)
    # # print(initial_val)
    # # y -= y[initial_val]
    # # x -= x[initial_val]
    # # y = savgol_filter(y, 51, 1)  # window size 51, polynomial order 3
    # # x-=x[300]
    # x = np.array(x)
    # y = np.array(y)

    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2
    # y1 = savgol_filter(y1, 3, 1)  # window size 51, polynomial order 3
    # y1 = sigmoid(y1, -1, 2)
    # x1, y1 = spline(y1)

    # x3, y3 = moving_average(x1, y1, int(len(x1) / 20))
    from scipy.ndimage import uniform_filter1d
    y2 = savgol_filter(y2, 51, 1)
    # y3 = savgol_filter(y3, 89, 1)
    x1, y1 = spline_b(y1)
    print(len(x1), len(y2))
    import scipy.signal as signal
    y = signal.detrend(y, type='linear')
    y = signal.wiener(y, 41)
    # y = signal.gauss_spline(y, 0)
    # x2,y2 = bspline(x1, y1)
    # y = [*y[:300], *y_1]
    # # y-=y[300]
    # x -= x[300]
    yy = [*y1, *y2, *y3]
    xx = [*x1, *x2, *x3]
    # xx -= xx[300]
    # yy -= yy[300]
    ax.plot(x, y)
    print(y[100])
    # ax.plot(x2, y2, ',', alpha=0.5, label=lab.split("|")[1])
    # ax.plot(x1, y, '-', alpha=1, label=lab.split("|")[1])
    # ax.plot(x1, y1, '-', alpha=1, label=lab.split("|")[1])
    # ax.plot(*p0, '-', alpha=0.5, label=lab.split("|")[1])
    # ax.plot(*p1, '-', alpha=0.5, label=lab.split("|")[1])
    # ax.plot(*p2, '-', alpha=0.5, label=lab.split("|")[1])
    # ax.plot(x[np.argmax(y)], np.max(y), 'k^')
    # ax.plot(x[np.argwhere(x >= 0)], y[np.argwhere(x >= 0)], 'k.')
    # ax.plot(x[initial_val], y[initial_val], 'k.')
    # print(y[initial_val] - max(y))
    # ax.set_title(lab)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import os

    text_size = 14
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.xticks(weight='bold', size=text_size)
    plt.yticks(weight='bold', size=text_size)
    plt.ylabel('-Z\'\'(Ω)', fontweight='bold', size=text_size)
    plt.xlabel('Z\'(Ω)', fontweight='bold', size=text_size)
    # plt.ylim(top=1.38195,bottom=1.381999)
    # plt.xlim(left=100, right=800)
    filetype = '.txt'
    for root, dirs, files in os.walk('data/sh_var/'):
        for file in sorted(files, reverse=True):
            if file.lower().endswith(filetype.lower()):
                dataalign(os.path.join(root, file))

    # plt.title('Calibration Graph', weight='bold', size=text_size)
    plt.legend(loc=0, prop={'size': 12})
    # plt.gca().invert_yaxis()
    # plt.savefig('data/calibration2.png', dpi=300)
    fig.show()
    # plt.gca().invert_yaxis()
    # plt.savefig('data/calibration_reverse2.png', dpi=300)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
