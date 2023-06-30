import numpy as np
import matplotlib.pyplot as plt
import os
from main import read_label

text_size = 14


def plot(path):
    data = np.loadtxt(path, delimiter=' ', dtype=np.float, skiprows=2)
    lab = read_label(path)
    # spike = read_spike_title(path)
    x, y = data.T

    plt.xticks(weight='bold', size=text_size)
    plt.yticks(weight='bold', size=text_size)
    plt.ylabel('Current(A)', fontweight='bold', size=text_size)
    plt.xlabel('Time(s)', fontweight='bold', size=text_size)
    plt.title(path, fontweight='bold', size=text_size)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    filetype = '.txt'
    for root, dirs, files in os.walk('data/Cost Deneyler /calibration/19.02.2021'):
        for file in sorted(files, reverse=True):
            if file.lower().endswith(filetype.lower()):
                plot(os.path.join(root, file))
                # print(os.path.join(root, file))
