import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv("data/nanopartikul/nano.tsv", delimiter='\t', index_col="x")
    go = pd.read_csv("data/nanopartikul/GO_01.csv")
    g = pd.read_csv("data/nanopartikul/Grafen_01.csv")
    dd = data[data.columns[1:2]]
    go['Unsubtracted Weight'] = go['Unsubtracted Weight'] / max(go['Unsubtracted Weight']) * 100
    g['Unsubtracted Weight'] = g['Unsubtracted Weight'] / max(g['Unsubtracted Weight']) * 100
    data['2'] = data['2'] / max(data['2']) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data['1'], data['2'], color="r", label="Altin Platin")
    ax.plot(go['Program Temperature'], go['Unsubtracted Weight'], color="g", label="Grafen Oksit")
    # ax.plot(g['Program Temperature'], g['Unsubtracted Weight'], color="b", label="Grafen")
    plt.legend(loc="best")
    plt.ylabel("Weight(%)")
    plt.xlabel("Temperature (C)")

    plt.show()
