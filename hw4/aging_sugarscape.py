#!/usr/bin/env python

import numpy as np
import numpy.typing as npt
from typing import NewType

from matplotlib import pyplot as plt
from tqdm import trange

Agent = NewType('Agent', npt.NDArray[int])
Grid = NewType('Grid', npt.NDArray[npt.NDArray[int]])


def gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


def lorenz(wealth: npt.NDArray[int]):
    return np.cumsum(np.sort(wealth))/np.sum(wealth)


def place_sugar(grid: Grid, row: int, col: int, radius: float) -> None:
    # no idea in checking values guaranteed outside the radius
    # use min and max to avoid index out of bounds
    for i in range(max(row - int(np.ceil(radius)), 0), min(grid.shape[0], row + int(np.ceil(radius)))):
        for j in range(max(col - int(np.ceil(radius)), 0), min(grid.shape[1], col + int(np.ceil(radius)))):
            if (row - i) ** 2 + (col - j) ** 2 <= radius ** 2:
                grid[i, j] += 1


def place_agents(agents: npt.NDArray[Agent], N: int, regenerate: bool = False) -> npt.NDArray[Agent]:
    agent_grid = np.zeros((N, N))
    if regenerate:
        for agent in agents:
            if not agent[6]:
                continue
            i, j = agent[0:2]
            agent_grid[int(i), int(j)] = 1

    for a in range(agents.shape[0]):
        free_spaces = np.argwhere(agent_grid == 0)
        i, j = free_spaces[np.random.randint(len(free_spaces))]

        agents[a, 0:2] = (i, j)
        agent_grid[i, j] = 1
    return agent_grid


def generate_agents(num_agents: int) -> npt.NDArray[Agent]:
    # One agent: [ pos_x, pos_y, wealth, vision, metabolism, alive, age, max_age ]
    agents = np.zeros((num_agents, 8))

    agents[:, 2] = np.random.randint(5, 26, num_agents)
    agents[:, 3] = np.random.randint(1, 7, num_agents)
    agents[:, 4] = np.random.randint(1, 5, num_agents)
    agents[:, 5] = np.ones(num_agents)
    agents[:, 6] = np.zeros(num_agents)
    agents[:, 7] = np.random.randint(60, 100, num_agents)

    return agents


def regenerate_agent(agents: npt.NDArray[Agent], N: int) -> None:
    reborn_agents = np.argwhere(agents[:, 5] == 0).flatten()

    for agent in reborn_agents:  # do an index replacement of each agent to be reborn
        agents[agent, 2] = np.random.randint(5, 26)
        agents[agent, 3] = np.random.randint(1, 7)
        agents[agent, 4] = np.random.randint(1, 5)
        agents[agent, 5] = 1
        agents[agent, 6] = 0
        agents[agent, 7] = np.random.randint(60, 100)


def main():
    num_agents = 400
    N = 50
    num_rounds = 500
    regrowth_constant = 1
    rng = np.random.default_rng()

    sugar_grid = np.zeros((N, N))

    for radius in [20, 15, 10, 5]:
        place_sugar(sugar_grid, 40, 10, radius)
        place_sugar(sugar_grid, 10, 40, radius)

    max_capacity = sugar_grid.copy()
    agents = generate_agents(num_agents)

    agent_grid = place_agents(agents, N)

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(sugar_grid.copy(), cmap='summer')
    plt.plot(agents[:, 0], agents[:, 1], 'r.')
    plt.title('$t=0$')

    plt.figure(2)
    v_hist, v_bins = np.histogram(agents[:, 3], bins=[1, 2, 3, 4, 5, 6, 7])
    m_hist, m_bins = np.histogram(agents[:, 4], bins=[1, 2, 3, 4, 5])
    plt.subplot(1, 2, 1)
    plt.bar(v_bins[:-1] - 0.2, v_hist, width=0.4, label='Initial')
    plt.xlabel('Vision')
    plt.ylabel('Number of agents')
    plt.subplot(1, 2, 2)
    plt.bar(m_bins[:-1] - 0.2, m_hist, width=0.4, label='Initial')
    plt.xlabel('Metabolism')

    wealth_hists = []
    wealth_bins = []
    timestamps = []

    gini_coeffs = np.zeros(num_rounds)
    lorenz_curves = [lorenz(agents[:, 2])]
    lorenz_times = [0, 125, 250, 375, 500]

    plt.figure(4)
    plt.subplot(1, 2, 1)
    plt.title('Initial')
    wealth_lorenz = lorenz(agents[:, 2])
    plt.plot([0, 1], [0, 1], color='k')
    plt.plot(np.arange(num_agents)/num_agents, wealth_lorenz)

    max_wealth = 0
    for t in trange(num_rounds):
        # Update agents
        agent_order = rng.permuted(np.arange(agents.shape[0]))

        if t % 20 == 0 and t < 80:  # get t = 20, t = 40, t = 60, and t = 80
            alive_agents = agents[agents[:, 5] > 0]
            s_hist, s_bins = np.histogram(alive_agents[:, 2], bins=15)
            wealth_hists.append(s_hist)
            wealth_bins.append(s_bins)
            timestamps.append(f'${t = }$')

        if t + 1 in lorenz_times:
            lorenz_curves.append(lorenz(agents[:, 2]))

        for i in agent_order:
            # no check if dead here since all were regenerated at the end of last round

            j = int(agents[i, 0])
            k = int(agents[i, 1])
            agents[i, 2] += (sugar_grid[j, k] - agents[i, 4]) * agents[i, 5]
            # Kill if out of sugar
            agents[i, 5] = (agents[i, 2] > 0).astype(int)
            sugar_grid[j, k] = 0

            most_sugar_found = 0

            possible_destinations = []

            vision = int(agents[i, 3])

            # Right
            for v in range(1, vision+1):
                if j + v >= N:  # avoid index oob
                    break
                if agent_grid[j + v, k]:  # check next if spot is taken
                    continue
                sugar_found = sugar_grid[j + v, k]
                if sugar_found >= most_sugar_found:  # add to possible destination if at least equal
                    if sugar_found > most_sugar_found:  # clear list first if strictly more
                        possible_destinations.clear()
                    possible_destinations.append((j + v, k))
                    most_sugar_found = sugar_found

            # Left
            for v in range(1, vision+1):
                if j - v < 0:
                    break
                if agent_grid[j - v, k]:
                    continue
                sugar_found = sugar_grid[j - v, k]
                if sugar_found >= most_sugar_found:
                    if sugar_found > most_sugar_found:
                        possible_destinations.clear()
                    possible_destinations.append((j - v, k))
                    most_sugar_found = sugar_found

            # Up
            for v in range(1, vision+1):
                if k - v < 0:
                    break
                if agent_grid[j, k - v]:
                    continue
                sugar_found = sugar_grid[j, k - v]
                if sugar_found >= most_sugar_found:
                    if sugar_found > most_sugar_found:
                        possible_destinations.clear()
                    possible_destinations.append((j, k - v))
                    most_sugar_found = sugar_grid[j, k - v]

            # Down
            for v in range(1, vision+1):
                if k + v >= N:
                    break
                if agent_grid[j, k + v]:
                    continue
                sugar_found = sugar_grid[j, k + v]
                if sugar_found >= most_sugar_found:
                    if sugar_found > most_sugar_found:
                        possible_destinations.clear()
                    possible_destinations.append((j, k + v))
                    most_sugar_found = sugar_grid[j, k + v]

            j_max, k_max = rng.choice(possible_destinations) if len(possible_destinations) else (j, k)

            agent_grid[j, k] = 0
            agent_grid[j_max, k_max] = 1
            agents[i, 0:2] = j_max, k_max

        # Update sugar scape
        sugar_grid += regrowth_constant
        sugar_grid = np.minimum(sugar_grid, max_capacity)
        # Age agent and kill if too old
        agents[:, 6] += 1
        agents[agents[:, 6] >= agents[:, 7], 5] = 0
        gini_coeffs[t] = gini(agents[2])
        if t == num_rounds - 1:  # regenerate unless final generation
            break
        regenerate_agent(agents, N)
        place_agents(agents, N, True)
        max_wealth = np.max(agents[:, 2]) if np.max(agents[:, 2]) > max_wealth else max_wealth

    alive_agents = agents[agents[:, 5] > 0]
    alive_agents = alive_agents[alive_agents[:, 6] < alive_agents[:, 7]]

    print(f'Number of alive agents: {alive_agents.shape[0]}')

    plt.figure(1)
    plt.suptitle('Agent location, aging')
    plt.subplot(1, 2, 2)
    plt.imshow(sugar_grid, cmap='summer')
    plt.plot(alive_agents[:, 0], alive_agents[:, 1], 'r.')
    plt.title(f'$t={num_rounds}$')

    plt.figure(2)
    plt.suptitle('Visual ranges vs metabolism rates, aging')
    v_hist, v_bins = np.histogram(alive_agents[:, 3], bins=[1, 2, 3, 4, 5, 6, 7])
    m_hist, m_bins = np.histogram(alive_agents[:, 4], bins=[1, 2, 3, 4, 5])
    plt.subplot(1, 2, 1)
    plt.bar(v_bins[:-1] + 0.2, v_hist, width=0.4, label='Final')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.bar(m_bins[:-1] + 0.2, m_hist, width=0.4, label='Final')
    plt.legend()

    plt.figure(3)
    plt.suptitle('Wealth distribution, aging')
    wealth_index = wealth_bins[-1:].index(max(wealth_bins[-1:]))
    s_bins = wealth_bins[wealth_index]
    plt.hist(wealth_hists, s_bins, label=timestamps, alpha=0.8, density=True, histtype='bar')
    plt.xlabel('Wealth')
    plt.ylabel('Number of agents')
    plt.legend()

    plt.figure(4)
    plt.suptitle('Gini coefficient, aging')
    plt.subplot(1, 2, 2)
    plt.title('Final')
    wealth_lorenz = lorenz(agents[:, 2])
    plt.plot([0, 1], [0, 1], color='k')
    plt.plot(np.arange(agents.shape[0])/agents.shape[0], wealth_lorenz)

    plt.figure(5)
    plt.suptitle('Gini coefficient per round, aging')
    plt.plot(np.arange(1, num_rounds + 1), gini_coeffs)

    plt.figure(6)
    plt.suptitle('Lorenz curve at different rounds, aging')
    plt.plot([0, 1], [0, 1], color='k')
    for curve in lorenz_curves:
        plt.plot(np.arange(agents.shape[0])/agents.shape[0], curve)
    plt.legend(lorenz_times)

    plt.show()


if __name__ == '__main__':
    main()
