#!/usr/bin/env python

import numpy as np
import numpy.typing as npt
import networkx as nx
from scipy.special import comb

from matplotlib import pyplot as plt


def gen_graph(n: int, p: float):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            A[j, i] = A[i, j] = np.random.rand() < p
    return A


def degree_prob(n: int, k: int, p: float):
    return comb(n - 1, k) * p ** k * (1 - p) ** (n - k - 1)


def get_degrees(A: npt.NDArray[npt.NDArray[bool]], return_connections: bool = False):
    n = A.shape[0]
    connections = np.zeros(n)
    for j in range(n):
        connections[j] = np.count_nonzero(A[j, :])

    i, x = np.unique(connections, return_counts=True)

    # Wanted single line return but it raised exceptions due to different tuple sizes
    if return_connections:
        return i, x, connections
    else:
        return i, x


def normal_dist(x: npt.NDArray[float], mu: float = 0.0, sigma: float = 1.0):
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)


def averages(n: int):
    avg_length = np.zeros(100)
    cluster = np.zeros(100)
    len_range = np.arange(0, 100) / 100.0
    for i, p in enumerate(len_range):
        A = gen_graph(n, p)
        A3 = A @ A @ A

        degrees = np.sum(A, axis=0)  # if degree is 1, cluster will give warning
        cluster[i] = np.sum(np.trace(A3)) / np.sum(degrees * (degrees - 1))

    plt.title(f'Average Erdős-Rényi, $n$: {n}')
    plt.plot(len_range, cluster)
    plt.show()


def main():
    for i, n, p in [(0, 100, 0.05), (1, 400, 0.01), (2, 200, 0.05)]:
        A = gen_graph(n, p)

        G = nx.from_numpy_array(A)
        plt.subplot(2, 3, i + 1)

        plt.title(f'$n$: {n}, $p$: {p}')
        nx.draw(G, pos=nx.circular_layout(G), node_size=20)

        plt.subplot(2, 3, i + 4)
        plt.title(f'$n$: {n}, $p$: {p}')

        k, k_count = get_degrees(A)
        plt.bar(k, k_count / n)
        plt.plot(np.arange(np.max(k)), degree_prob(n, np.arange(np.max(k)), p), color='#ffa500')

    plt.figure()

    for i, N in enumerate([100, 1000, 10000]):
        A = gen_graph(N, 0.01)

        x = np.linspace(-10, 10, num=N)

        k, k_count, connections = get_degrees(A, True)
        plt.subplot(1, 3, i + 1)
        plt.plot(k, normal_dist(k, mu=np.mean(connections), sigma=np.std(connections)), label='Gaussian dist.')
        plt.plot(k, k_count/N, label='Data points')
        plt.title(f'$N = {N}$')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

    n = 500
    averages(n)
