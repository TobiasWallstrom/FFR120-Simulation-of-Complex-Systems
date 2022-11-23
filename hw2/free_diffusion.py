#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from tqdm import trange

if __name__ == '__main__':
    x_0 = 0
    sigma_0 = 1
    dt = 1
    dt_red = np.sqrt(dt)

    N = 10_000
    T = 1_000
    time_range = int(T / dt)

    x = np.zeros((N, 1))
    x[:] = x_0

    BINS = 100
    LWR_BND = 0
    UPR_BND = 0

    plot_colors = ['#590995', '#d22b2b', '#ffa500', '#0bda51', '#1434a4']
    plot_times = (np.array([10, 100, 1_000, 10_000, 100_000]) / dt).astype(int)
    avgs = np.zeros(plot_times.shape)
    stds = np.zeros(plot_times.shape)
    expected_stds = np.zeros(plot_times.shape)
    idx = 0

    assert len(plot_colors) <= len(plot_times), 'Not enough plot colors, add more'

    for t in trange(time_range):
        diff = (np.round(np.random.random_sample((N, 1))) * 2 - 1) * sigma_0 * dt_red
        x += diff

        if t + 1 in plot_times:
            counts, bins = np.histogram(x[:], bins=BINS)
            avgs[idx] = np.mean(x[:])
            stds[idx] = np.std(x[:])
            expected_stds[idx] = sigma_0 * np.sqrt(t)
            plt.stairs(counts, bins, fill=True, alpha=0.2, color=plot_colors[idx], label=f'$t={int((t + 1) * dt)}$ s')
            plt.stairs(counts, bins, color=plot_colors[idx])
            idx += 1

    for i in range(idx):
        print(f't = {plot_times[i]}')
        print(f'  Avg:{avgs[i]} Std:{stds[i]} Expected std:{expected_stds[i]}')

    plt.title('Free diffusion')
    plt.xlabel('$x$')
    plt.ylabel('Count')
    plt.legend()
    plt.grid()

    plt.show()
