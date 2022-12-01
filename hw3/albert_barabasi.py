#!/usr/bin/env python

import numpy as np
import networkx as nx

from matplotlib import pyplot as plt
from random import sample


def preferential_growth(n, n_0, m):
    A = np.zeros((n, n))
    for i in range(n_0):
        for j in range(i + 1, n_0):
            A[j, i] = A[i, j] = 1
    for t in range(n-n_0):
        prob = np.sum(A, axis=0)/np.sum(A)
        edges = np.random.choice(np.arange(n), m, replace=False, p=prob)
        for i in range(m):
            A[edges[i], t + n_0] = A[t + n_0, edges[i]] = 1

    return A


def distribution(n, n_0, m):
    A = preferential_growth(n, n_0, m)
    degrees = np.sum(A, 1)
    max_deg_range = np.arange(m, np.max(degrees))

    plt.loglog(np.sort(degrees)[::-1], np.arange(n)/n, '.')
    plt.loglog(max_deg_range, m ** 2 / max_deg_range ** (2), '--')

    plt.title('Distribution')
    plt.ylabel('$C(k)$')
    plt.xlabel('$k$')

    plt.show()


def main():
    for i, n, n_0, m in ([(0, 100, 5, 3), (1, 100, 15, 3), (2, 100, 10, 8)]):
        A = preferential_growth(n, n_0, m)

        G = nx.from_numpy_array(A)

        plt.subplot(1, 3, i + 1)
        plt.title(f'$n = {n}, n_0 = {n_0}, m = {m}$')
        nx.draw(G, pos=nx.circular_layout(G), node_size=30)

    plt.show()


if __name__ == '__main__':
    main()

    n = 1000
    n_0 = 5
    m = 3
    distribution(n, n_0, m)
