import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


def read_spike_title(file):
    return int(open(file).readlines()[0].replace('\n', '').replace('Title: ', ''))


# path = 'data/ekstra_kiymetliler/0.02mM(1).txt'
path = 'data/optimization/nanoaldolaz/5 uL.txt'
data = np.loadtxt(path, delimiter=' ', dtype=np.float, skiprows=2)

x, y = data.T
#
# choice = lambda x: np.random.choice(y[np.where(x < 600)])
# tt = lambda r: choice(y) if x[np.where(y == r)] > 600 else r
# np.array(x)
# temp = []
# ran = np.random.choice(y[np.where(np.logical_and(x < 600, x > 500))])
# ss = []
# xtemp = []
start = 300
# stop = 900
step = 10


# print(start, stop)
# print(x[np.argmax(y)])
# print(y[np.where(min(y))], x[np.where(min(y))])
# num_len = (abs(start - stop) / step)
# print(num_len)


# for ind, i in enumerate(np.array(y)):
#     if ind > start and ind < stop:
#         if ind % step == 0:
#             xtemp.append(start + ((ind - start) / step))
#             if len(ss) > 0:
#                 temp.append(sum(ss) / len(ss))
#             else:
#                 temp.append(i)
#             ss = []
#         else:
#             ss.append(i)
#     else:
#         if ind >= stop:
#             xtemp.append(x[ind]-abs(abs(start - stop) - num_len))
#             print(abs(abs(start - stop) - num_len))
#         else:
#             xtemp.append(x[ind])
#         temp.append(i)

def read_tt(file):
    return ''.join(open(file).readlines()[0:2])


# print(y[np.where(np.logical_and(x < 600, x > 400))])
# print(y)
def zippedPlot(path, start, step, stop=None):
    data = np.loadtxt(path, delimiter=' ', dtype=np.float, skiprows=2)
    x, y = data.T
    if stop is None:
        stop = x[start + np.argmax(y[start:])]
    print('max=', np.argmax(y[start:]))
    print('stop=', stop)
    mask = np.where(np.logical_and(x > start, x < stop))
    end = np.where(x >= stop)
    x1 = x.copy()
    x1[mask] = start + (x[mask] - start) / (abs(start - stop) / step)
    x1[end] = x[end] - (abs(start - stop)) + (abs(start - stop) / step)
    # print('\n'.join([f'{str(i)} {str(j)}' for i, j in zip(x, temp)]))
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(xtemp, temp, 'r-', label='shrinked')
    # ax.plot(x[np.where(xtemp)], y[np.where(xtemp)], label='masked data')
    # ax.plot(x, y, 'y--', label='raw data')
    ax.plot(x1, y, 'r-', label='zipped data')
    ax.plot(stop, np.max(y[start:]), 'gx', label='zipped data')
    ax.plot(start, y[start], 'b^', label='spike')
    # ax.plot(x[int(stop):], y[stop:], label='col data')
    plt.legend()
    plt.show()
    print('val=', y[start] - y[end][0])


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def spline_w_spike(y_new, spike):
    from scipy.interpolate import make_interp_spline, BSpline

    # 300 represents number of points to make between T.min and T.max
    x_start = np.linspace(0, spike, 300)
    x_stop = np.linspace(spike, len(y_new), 800)
    spl_start = make_interp_spline(np.linspace(0, 300, len(y_new[:spike])), y_new[:spike], k=1)  # type: BSpline
    spl_stop = make_interp_spline(np.linspace(300, 1100, len(y_new[spike:])), y_new[spike:], k=2)  # type: BSpline
    power_smooth_start = spl_start(x_start)
    power_smooth_stop = spl_stop(x_stop)
    x = [*x_start, *x_stop]
    y = [*power_smooth_start, *power_smooth_stop]
    x -= x[300]
    y -= y[300]
    return x, y


def zipData(x, y, start, step):
    stop = x[start + np.argmax(y[start:])]
    mask = np.where(np.logical_and(x > start, x < stop))
    end = np.where(x >= stop)
    print('stop=', stop)
    x1 = x.copy()
    x1[mask] = start + (x[mask] - start) / (abs(start - stop) / step)
    x1[end] = x[end] - (abs(start - stop)) + (abs(start - stop) / step)
    # y -= y[start]
    # y = y - y[start]
    # # y /= (y[end][0] - y[start])
    # x1 -= x[start]
    return x1, y


def plot(x, y):
    # fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(x, y, 'r-', label='zipped data')
    plt.plot(x[np.argmax(y[300:])], np.max(y[300:]), 'gx')
    # print(x[np.argmax(y[300:])], np.max(y[300:]))
    print(y[find_nearest(x, 300)], np.max(y[300:]))
    plt.show()




spike = read_spike_title(path)
print(read_tt(path))
print(spike)
data = np.loadtxt(path, delimiter=' ', dtype=np.float, skiprows=2)
x, y = data.T
mask = np.argwhere(y < 15e-1)
x, y = spline_w_spike(y, spike)
plt.plot(x, y, 'y--')
x, y = zipData(x, y, 0, 10)
y = savgol_filter(y, 89, 1)
plt.plot(x, y, 'g-')

plt.show()
print(np.min(y), np.max(y))
