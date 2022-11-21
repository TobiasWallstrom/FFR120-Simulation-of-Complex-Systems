#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from tqdm import trange

if __name__ == '__main__':
    x_0 = 0
    sigma_0 = 1
    ds = 1.8 # delta sigma

    use_multiplicative_noise = False
    ds = 1.8 # delta sigma

    N = 10_000
    T =  10_000 # 100_000 # seconds
    L = 100
    dt = 0.01

    dt_red = np.sqrt(dt)
    time_range = int(T/dt)

    x = np.zeros((N,1))
    x[:,0] = x_0

    BINS = 31
    LWR_BND = -L/2
    UPR_BND = L/2

    for t in trange(time_range):
        sigma = sigma_0 + ds / L * x
        diff = (np.round(np.random.rand(N,1)) * 2 - 1) * sigma * dt_red
        x += diff

        x[(x < -L / 2)] = - L - x[(x < -L / 2)]
        x[(x > L / 2)] = L - x[(x > L / 2)]

        if t == int(10 / dt) or t == int(100 / dt) or t == int(1_000 / dt) or t == int(10_000 / dt) or t == int(100_000 / dt):
            counts, bins = np.histogram(x[:], bins=BINS, range=(LWR_BND,UPR_BND))
            # print(f'j={j}')
            # print(f'  Avg:{np.mean(x[:])} Std:{np.std(x[:]) / L}')
            plt.stairs(counts,bins, fill=True, alpha=0.2, color='#590995', label=f'$t={t}$')
            plt.stairs(counts,bins, color='#590995')
    
    plt.xlabel('$x_j$')
    plt.ylabel('Count')
    plt.legend()
    plt.grid()

    plt.show()
