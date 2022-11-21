#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from tqdm import trange

if __name__ == '__main__':
    x_0 = 0
    sigma = 1

    N = 10_000
    T = 100_000
    L = 100
    dt = 0.01

    dt_red = np.sqrt(dt)
    time_range = int(T/dt)

    x = np.zeros((N,1))
    x[:,0] = x_0

    BINS = 31
    LWR_BND = -L/2
    UPR_BND = L/2
    
    plot_times = np.array([10, 100, 1_000, 10_000, 100_000])

    for t in trange(time_range):
        diff = (np.round(np.random.rand(N,1)) * 2 - 1) * sigma * dt_red
        x += diff

        x[(x < -L / 2)] = - L - x[(x < -L / 2)]
        x[(x > L / 2)] = L - x[(x > L / 2)]

        if t+1 in (plot_times/dt).astype(int):
            counts, bins = np.histogram(x[:], bins=BINS, range=(LWR_BND,UPR_BND))
            print(f't={t+1}')
            print(f'  Avg:{np.mean(x[:])} Std:{np.std(x[:]) / L}')
            plt.stairs(counts,bins, fill=True, alpha=0.2, color='#590995', label=f'$t={t+1}$')
            plt.stairs(counts,bins, color='#590995')
    
    plt.xlabel('$x_j$')
    plt.ylabel('Count')
    plt.legend()
    plt.grid()

    plt.show()


