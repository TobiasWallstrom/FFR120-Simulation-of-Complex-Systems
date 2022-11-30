#!/usr/bin/env python3

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
    if return_connections:
        return (i, x, connections)
    return (i, x)


def normal_dist(x: npt.NDArray[float], mu: float = 0.0, sigma: float = 1.0):
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)


if __name__ == '__main__':
    for (i, n, p) in [(1, 100, 0.05), (2, 400, 0.01), (3, 200, 0.05)]:
        A = gen_graph(n, p)

        G = nx.from_numpy_array(A)
        plt.subplot(2, 3, i)

        plt.title(f"$n$: {n}, $p$: {p}")
        pos = nx.circular_layout(G)
        nx.draw(G, pos=pos, node_size=30)

        plt.subplot(2, 3, i + 3)
        plt.title(f"$n$: {n}, $p$: {p}")

        k, k_count = get_degrees(A)
        plt.bar(k, k_count / n)
        plt.plot(np.arange(np.max(k)), degree_prob(n, np.arange(np.max(k)), p), color="#E66C00")

    plt.figure()

    for (i, N) in enumerate([100, 1000, 10000]):
        A = gen_graph(N, 0.01)

        x = np.linspace(-10, 10, num=N)

        (k, k_count, connections) = get_degrees(A, True)
        plt.subplot(1, 3, i+1)
        plt.plot(k, normal_dist(k, mu=np.mean(connections), sigma=np.std(connections)), label="Gaussian")
        plt.plot(k, k_count/N, label="Data points")
        plt.title(f"$N = {N}$")

    plt.legend()
    plt.show()
