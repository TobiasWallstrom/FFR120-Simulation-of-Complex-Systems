#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from tqdm import trange
from typing import NewType, Callable, Tuple

Vector = NewType('Vector', npt.NDArray[float])

def hooke_force(x: Vector, k: float):
    return -k * x

def euler(pos: Vector, vel: Vector, acc: Vector, dt = 1.0) -> Tuple[Vector, Vector]:
    return pos + vel * dt, vel + acc * dt

def leapfrog(pos: Vector, vel: Vector, acc_func: Callable[[Vector], Vector], dt = 1.0) -> Tuple[Vector, Vector]:
    pos_half = pos + vel * dt/2
    acc_half = acc_func(pos_half)

    vel += acc_half * dt
    pos = pos_half + vel * dt/2
    return (pos, vel)

def analytical_positions(t: Vector, magnitude: float, omega: float, phi=0):
    return magnitude * np.cos(omega * t + phi)

# d/dt of positions
def analytical_velocity(t: Vector, magnitude: float, omega: float, phi=0):
    return -omega * magnitude * np.sin(omega * t + phi)

def potential_energy(k, x):
    return 0.5 * k * x ** 2

def kinetic_energy(v, m):
    return 0.5 * m * v ** 2

def energy(pos: Vector, vel: Vector, k: float, m: float):
    return potential_energy(k, pos) + kinetic_energy(vel, m)

def main():
    m = 0.1
    k = 3.0
    dt_values = [0.0001, 0.002, 0.01]
    start_pos = 0.2
    start_vel = 0.0
    start_acc = 0.0 # should not be changed unless you specifically want to add another force

    use_leapfrog_alg = True

    T = 10 
    for plot in range(len(dt_values)):
        dt = dt_values[plot]
        time_range = int(T / dt)

        position = np.array([start_pos])
        velocity = np.array([start_vel])
        acceleration = np.array([start_acc])

        positions = np.zeros((time_range,1))
        energies = np.zeros((time_range,1))

        for t in trange(time_range):
            positions[t] = position
            energies[t] = energy(position, velocity, k, m)

            if use_leapfrog_alg:
                position, velocity = leapfrog(position, velocity, lambda x: hooke_force(x, k) / m, dt=dt)
            else:
                position, velocity = euler(position, velocity, hooke_force(position, k) / m, dt=dt)
            
        times = np.arange(time_range) * dt
        an_sol = analytical_positions(times, start_pos, np.sqrt(k/m))
        an_vel = analytical_velocity(times, start_pos, np.sqrt(k/m))
        an_energy = energy(an_sol, an_vel, k, m)

        plt.subplot(2, len(dt_values), plot+1)
        plt.plot(times, positions[:,0], '.', markersize=1)
        plt.plot(times, an_sol)
        plt.title(f'$\mathrm{{d}}t = {dt * 1000:.1f}$ ms')
        if plot % len(dt_values) == 0:
            plt.ylabel('$x(t)$')

        plt.subplot(2, len(dt_values), plot+1+len(dt_values))
        plt.plot(times, energies/an_energy[0]-1)
        plt.plot(times, an_energy/an_energy[0]-1)
        plt.xlabel('$t$')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.ylim(-5e-5, 8e-4)
        if plot % len(dt_values) == 0:
            plt.ylabel('Relative energy drift')

    plt.show()

if __name__ == '__main__':
    main()