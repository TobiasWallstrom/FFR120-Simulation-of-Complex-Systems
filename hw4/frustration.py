#!/usr/bin/env python

import numpy as np
import numpy.typing as npt
from scipy.signal import fftconvolve
from tqdm import trange
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def get_frustration_happiness(town: npt.NDArray[npt.NDArray[int]]) -> npt.NDArray[npt.NDArray[int]]:
    size = 3
    convolution_kernel = np.ones((size, size))
    convolution = fftconvolve(town, convolution_kernel, mode='same')
    return (convolution < 0).astype(int) - (convolution > 2).astype(int)


def calculate_happiness(town: npt.NDArray[npt.NDArray[int]]) -> (float, float, float):
    happiness = get_frustration_happiness(town) * town

    A = np.count_nonzero((happiness > 0) & (town > 0)) / np.count_nonzero(town > 0)
    B = np.count_nonzero((happiness > 0) & (town < 0)) / np.count_nonzero(town < 0)
    total = (A + B) / 2

    return A, B, total


def main():
    N = 50
    f = 0.1  # free houses
    population_equal = (1 - f) / 2  # equal distribution between A and B in the occupied housing

    rng = np.random.default_rng()
    town = rng.permuted(np.concatenate((np.zeros(int(N ** 2 * f)), np.ones(int(N ** 2 * population_equal)), -
                                        np.ones(int(N ** 2 * population_equal))))).reshape((N, N))

    num_rounds = 100_000
    plot_rounds = list(filter(lambda x: x <= num_rounds, [0, 10_000, 100_000]))
    town_at_time = np.zeros((len(plot_rounds), N, N))
    town_at_time[0] = town  # since we check if t+1 is in plot range, we need to manually add first town

    happiness_a = np.zeros(num_rounds)
    happiness_b = np.zeros(num_rounds)
    avg_happiness = np.zeros(num_rounds)

    move_happened = np.zeros(num_rounds, dtype=bool)

    for t in trange(num_rounds):
        happiness_a[t], happiness_b[t], avg_happiness[t] = calculate_happiness(town)

        happiness = get_frustration_happiness(town)

        occupied = np.argwhere(town != 0)
        i, j = occupied[np.random.randint(len(occupied))]

        if happiness[i, j]:
            move_happened[t] = True

            free = np.argwhere(town == 0)
            i_new, j_new = free[np.random.randint(len(free))]

            town[i_new, j_new] = town[i, j]
            town[i, j] = 0

        if t+1 in plot_rounds:
            town_at_time[plot_rounds.index(t+1)] = town

    plotlen = len(town_at_time)
    colors = [(230/256, 108/256, 0), (1, 1, 1), (241/256, 211/256, 19/256)]

    cmap = LinearSegmentedColormap.from_list('', colors, N=3)
    plt.suptitle('Frustration')

    for i in range(plotlen):
        plt.subplot(1, plotlen, i + 1)
        plt.imshow(town_at_time[i], cmap=cmap)
        plt.title(f'$t={plot_rounds[i]}$')

    plt.figure()
    rge = np.arange(num_rounds)
    plt.suptitle('Frustration')

    plt.plot(rge, happiness_a, label='Happiness (A)')
    plt.plot(rge, happiness_b, label='Happiness (B)')
    plt.plot(rge, avg_happiness, label='Happiness (total)')
    window_size = 1000
    plt.plot(rge[window_size:-window_size], (fftconvolve(move_happened, np.ones(window_size), mode='same') /
             window_size)[window_size:-window_size], label=f'Moves (moving average, window={window_size})')
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('Happiness')
    plt.ylim((0, 1))

    plt.show()


if __name__ == '__main__':
    main()
