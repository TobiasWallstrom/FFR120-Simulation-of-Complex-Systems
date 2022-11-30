#!/usr/bin/env python3

import numpy as np
import networkx as nx

from matplotlib import pyplot as plt


def gen_matrix(n: int, c: int, p: float):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = j - i

            dx = n - dx if dx > n / 2 else dx

            if dx % n <= c / 2:
                A[j, i] = A[i, j] = 1

            if np.random.rand() < p:
                A[j, i] = A[i, j] = np.abs(A[i, j] - 1)
    return A


if __name__ == '__main__':
    A = gen_matrix(10, 4, 0.2)

    G = nx.from_numpy_array(A)
    # plt.subplot(2, 3, i)

    # plt.title(f"$n$: {n}, $p$: {p}")
    pos = nx.circular_layout(G)
    nx.draw(G, pos=pos, node_size=30)

    plt.show()
