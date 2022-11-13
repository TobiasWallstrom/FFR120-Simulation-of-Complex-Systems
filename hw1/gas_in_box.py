#!/usr/bin/env python3

import math
import numpy as np
import numpy.typing as npt
from scipy.constants import k
from matplotlib import pyplot as plt
from random import randrange
from typing import NewType
# from sympy import symbols

Vector = NewType('Vector', npt.NDArray[float])
VectorArray = NewType('VectorArray', npt.NDArray[Vector])

# m_0, epsilon_0, sigma_0 = symbols('m_0 epsilon_0 sigma_0')
sigma_0 = 1
m_0 = 0.1
epsilon_0 = 1
v_0 = 1
dt = 1

dim = 2
N = 100
m = m_0
epsilon = epsilon_0
sigma = sigma_0
L = 100 * sigma_0
v_0 = math.sqrt(2 * epsilon / m_0)
t_0 = sigma_0 * math.sqrt(m_0 / (2 * epsilon))

def dist_between_points(x1: Vector, x2: Vector) -> float:
    if x1.size != x2.size:
        raise AssertionError('Error: Different dimensions of vectors given to dist_between_points()')
    return math.sqrt(sum((x1[i] - x2[i]) ** 2 for i in range(len(x1))))

def angle_between_points(x1: Vector, x2: Vector):
    x1[1]

def new_particle_too_close(P: Vector, x: Vector, min_dist: float) -> bool:
    return min([dist_between_points(P[i], x) for i in range(len(P))]) < min_dist

def initialize_particles(N: int, L: float) -> VectorArray:
    x = L * np.random.rand(1,2)
    for i in range(1, N):
        new_point = L * np.random.rand(2)
        while new_particle_too_close(x, new_point, sigma):
            new_point = L * np.random.rand(2)
        x = np.vstack([x, new_point])
    return x

def initialize_velocities(N: int, scale: float) -> VectorArray:
    v = np.empty((0,2), float)
    for i in range(N):
        theta = 2 * math.pi * np.random.random()
        v = np.vstack([v, [scale * math.cos(theta), scale * math.sin(theta)]])
    return v

def lennard_jones_force(positions: VectorArray) -> VectorArray:
    length = positions.shape[0]
    forces = np.zeros_like(positions)

    for i in range(length):
        # "Triangular" iteration avoids calculating force on self and duplicate calculations
        for j in range(i + 1, length):
            r = np.sqrt(np.sum((positions[i,:] - positions[j,:])**2))
            magnitude = 4 * epsilon * ( 12*np.power(sigma,12)*np.power(r, -13) - 6*np.power(sigma,6)*np.power(r,-7) )
            # Direction of force exerted on i (towards j)
            direction = (positions[j,:] - positions[i,:]) / r

            forces[i,:] -= magnitude*direction
            forces[j,:] += magnitude*direction
    return forces

def lennard_jones_potential(positions: VectorArray) -> npt.NDArray[float]:
    length = positions.shape[0]
    potential = np.zeros(length)

    for i in range(length):
        for j in range(i+1, length):
            r = sum((positions[i,:] - positions[j,:]) ** 2)
            mag = 4 * epsilon * (np.power(sigma ** 2 / r, 6) - np.power(sigma**2/r, 3))
            potential[i] += mag
            potential[j] += mag

# Void fn, updates internal values
def update_velocities(particles: VectorArray, velocities: VectorArray) -> None:
    length = len(particles)
    s6 = sigma ** 6
    F = np.zeros_like(particles)
    angles = np.zeros_like(particles)
    for i in range(length):
        for j in range(i + 1, length): # since it's symmetric
            rij = dist_between_points(particles[i], particles[j])
            # angles = 
            F[j,i] = F[i,j] = 6 * s6 * (rij ** 6 - s6) / rij ** 13
            
    # velocities = 

def constrain_particles(particles: VectorArray) -> None:
    pass

# Void fn, updates internal values
def update_particles(particles: VectorArray, velocities: VectorArray) -> None:
    intermediary = [pos + velocities[idx]*dt/2 for idx, pos in enumerate(particles)]
    update_velocities(particles, velocities)
    for i in range(len(particles)):
        particles[i] = intermediary[i] + velocities[i]*dt/2
    for pos in particles:
        if pos[0] > L:
            pos[0] = 2 * L - pos[0]
        elif pos[0] < 0:
            pos[0] *= -1
        if pos[1] > L:
            pos[1] = 2 * L - pos[1]
        elif pos[1] < 0:
            pos[1] *= -1

def kinetic_energy(velocities: VectorArray, scale: float):
    return 0.5 * m * sum((velocities/scale) ** 2)

def potential_energy(positions: VectorArray, epsilon: float, sigma: float) -> float:
    pass

def main():
    particles = initialize_particles(N, L)
    velocities = initialize_velocities(N, 2 * v_0)

    time_range = 50000
    plot_freq = 5
    dt = sigma/(2 * v_0) * 0.02
    
    position_history = np.empty((time_steps//plot_freq,N,2))

    E_k_history = np.empty(time_steps//plot_freq)
    E_p_history = np.empty(time_steps//plot_freq)

    plotting = ENERGIES
    logging = False

    update_particles(particles, velocities)


if __name__ == 'main':
    sigma_0 = 1
    m_0 = 0.1
    epsilon_0 = 1
    v_0 = 1
    dt = 1

    dim = 2
    N = 100
    m = m_0
    epsilon = epsilon_0
    sigma = sigma_0
    L = 100 * sigma_0
    v_0 = math.sqrt(2 * epsilon / m_0)
    t_0 = sigma_0 * math.sqrt(m_0 / (2 * epsilon))

    main()