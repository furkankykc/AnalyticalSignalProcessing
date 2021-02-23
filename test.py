import os

import numpy as np
from matplotlib import pyplot as plt

path = 'data/kıymetliler/0.1mM.txt'
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


# print(y[np.where(np.logical_and(x < 600, x > 400))])
# print(y)
def zippedPlot(path, start, step):
    data = np.loadtxt(path, delimiter=' ', dtype=np.float, skiprows=2)
    x, y = data.T
    stop = x[np.argmax(y[start:])]
    mask = np.where(np.logical_and(x > start, x < stop))
    end = np.where(x >= stop)
    x1 = x.copy()
    x1[mask] = start + (x[mask] - start) / (abs(start - stop) / step)
    x1[end] = x[end] - (abs(start - stop)) + (abs(start - stop) / step)
    # print('\n'.join([f'{str(i)} {str(j)}' for i, j in zip(x, temp)]))
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(xtemp, temp, 'r-', label='shrinked')
    # ax.plot(x[np.where(xtemp)], y[np.where(xtemp)], label='masked data')
    ax.plot(x, y, 'y--', label='raw data')
    ax.plot(x1, y, 'r-', label='zipped data')
    # ax.plot(x[int(stop):], y[stop:], label='col data')
    plt.legend()
    plt.show()
zippedPlot(path,start,1)

# for root, dirs, files in os.walk('data/nanoaldolaz'):
#     for file in files:
#         if file.lower().endswith('txt'.lower()):
#             zippedPlot(os.path.join(root, file), start, step)
