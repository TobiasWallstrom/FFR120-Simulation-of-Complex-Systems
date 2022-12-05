#!/usr/bin/env python

import numpy as np
import numpy.typing as npt
import networkx as nx

from matplotlib import pyplot as plt


# This one incorrectly adds and removes connections, there are too many connections for a given c
def gen_matrix(n: int, c: int, p: float) -> npt.NDArray[int]:
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = j - i  # |i - j|, but j > i always in this loop
            dx = n - dx if dx > n / 2 else dx

            if dx % n <= c / 2:
                A[j, i] = A[i, j] = 1

            if np.random.rand() < p:
                A[j, i] = A[i, j] = np.abs(A[i, j] - 1)
    return A


if __name__ == '__main__':
    i = 0
    for n, c, p in [(20, 2, 0), (20, 4, 0), (20, 8, 0), (20, 2, 0.2), (20, 4, 0.2), (20, 8, 0.2)]:
        A = gen_matrix(n, c, p)

        G = nx.from_numpy_array(A)
        plt.subplot(2, 3, i + 1)

        plt.title(f'$n$: {n}, $c$: {c}, $p$: {p}')
        nx.draw(G, pos=nx.circular_layout(G), node_size=30)
        i += 1

    plt.show()
