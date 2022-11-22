#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

if __name__ == '__main__':
    x_0 = 0
    sigma_0 = 1
    ds = 0.9  # delta sigma

    N = 10_000
    T = 100_000  # 100_000 # seconds
    L = 100
    dt = 0.01

    dt_red = np.sqrt(dt)
    time_range = int(T/dt)

    x = np.zeros((time_range+1, N))
    x[:, 0] = x_0

    BINS = 31
    LWR_BND = -L / 2
    UPR_BND = L / 2

    plot_colors = ['#590995', '#d22b2b', '#ffa500', '#0bda51', '#1434a4']
    plot_times = (np.array([10, 100, 1_000, 10_000, 100_000])/dt).astype(int)
    avgs = np.zeros(plot_times.shape)
    stds = np.zeros(plot_times.shape)

    for t in trange(time_range):
        x[t+1] = x[t, :] + (sigma_0+ds*x[t, :]/L)*ds/L*dt + \
            (sigma_0+ds*x[t, :]/L)*np.random.randn(N)*np.sqrt(dt)

        x[t+1, (x[t+1, :] < LWR_BND)] = - L - x[t+1, (x[t+1, :] < LWR_BND)]
        x[t+1, (x[t+1, :] > UPR_BND)] = L - x[t+1, (x[t+1, :] > UPR_BND)]

    for i in range(len(plot_times)):
        # be able to quickly change the end time without needing to add/remove plot times
        if plot_times[i] > time_range:
            break
        counts, bins = np.histogram(
            x[i, :], bins=BINS, range=(LWR_BND, UPR_BND))
        avgs[i] = np.mean(x[plot_times[i], :])
        stds[i] = np.std(x[plot_times[i], :]) / L
        plt.stairs(counts, bins, fill=True, alpha=0.2, color=plot_colors[i], label=f'$t={i}$')
        plt.stairs(counts, bins, color=plot_colors[i])
        print(f't = {plot_times[i]}')
        print(f'  Avg:{avgs[i]} Std:{stds[i]}')

    plt.xlabel('$x$')
    plt.ylabel('Count')
    plt.legend()
    plt.grid()

    plt.show()
