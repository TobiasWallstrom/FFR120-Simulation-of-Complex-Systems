#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

if __name__ == '__main__':
    x_0 = 0
    sigma_0 = 1
    ds = 1.8  # delta sigma

    N = 10_000
    T = 100_000  # 100_000 # seconds
    L = 100
    dt = 0.01

    dt_red = np.sqrt(dt)
    time_range = int(T/dt)

    x = np.zeros((N, 1))
    x[:] = x_0

    BINS = 100
    LWR_BND = -L / 2
    UPR_BND = L / 2

    plot_colors = ['#590995', '#d22b2b', '#ffa500', '#0bda51', '#1434a4']
    plot_times = (np.array([10, 100, 1_000, 10_000, 100_000])/dt).astype(int)
    avgs = np.zeros(plot_times.shape)
    stds = np.zeros(plot_times.shape)
    idx = 0

    for t in trange(time_range):
        diff = (sigma_0 + ds * x / L) * (np.round(np.random.rand(N, 1)) * 2 - 1) * dt_red
        x += diff

        x[(x < LWR_BND)] = - L - x[(x < LWR_BND)]
        x[(x > UPR_BND)] = L - x[(x > UPR_BND)]

        if t + 1 in plot_times:
            counts, bins = np.histogram(x[:], bins=BINS, range=(LWR_BND, UPR_BND))
            avgs[idx] = np.mean(x[:])
            stds[idx] = np.std(x[:]) / L
            plt.stairs(counts, bins, fill=True, alpha=0.2, color=plot_colors[idx], label=f'$t={int((t + 1) * dt)}$')
            plt.stairs(counts, bins, color=plot_colors[idx])
            idx += 1

    for i in range(idx):
        print(f't = {plot_times[i]}')
        print(f'  Avg:{avgs[i]} Std:{stds[i]}')

    plt.xlabel('$x$')
    plt.ylabel('Count')
    plt.legend()
    plt.grid()

    plt.show()
