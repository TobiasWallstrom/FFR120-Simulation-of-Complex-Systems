#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from random import randrange
from tqdm import trange
from typing import Callable, NewType, Tuple
from sys import argv

Vector = NewType('Vector', npt.NDArray[float])
VectorArray = NewType('VectorArray', npt.NDArray[Vector])

SNAPSHOT = 0
ENERGIES = 1
plotting = ENERGIES
logging = False

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
v_0 = np.sqrt(2 * epsilon / m_0)
t_0 = sigma_0 * np.sqrt(m_0 / (2 * epsilon))

def dist_between_points(x1: Vector, x2: Vector) -> float:
    if x1.shape != x2.shape:
        raise AssertionError('Error: Different dimensions of vectors given to dist_between_points()')
    return np.sqrt(sum((x1[i] - x2[i]) ** 2 for i in range(len(x1))))

def leapfrog(pos: VectorArray, vel: VectorArray, acc_func: Callable[[VectorArray], VectorArray], dt = 1.0) -> Tuple[VectorArray, VectorArray]:
    pos_half = pos + vel * dt/2
    acc_half = acc_func(pos_half)

    vel_next = vel + acc_half * dt
    pos_next = pos_half + vel_next * dt/2
    return (pos_next, vel_next)

def new_particle_too_close(P: Vector, x: Vector, min_dist: float) -> bool:
    return min([dist_between_points(P[i], x) for i in range(len(P))]) < min_dist

def initialize_particles(N: int, L: float) -> VectorArray:
    pos = np.zeros((N,2))
    for i in range(1, N):
        new_point = L * np.random.rand(2)
        while new_particle_too_close(pos, new_point, sigma):
            new_point = L * np.random.rand(2)
        pos[i] =  new_point
    return pos

def initialize_velocities(N: int, scale: float) -> VectorArray:
    v = np.empty((0,2), float)
    for i in range(N):
        theta = 2 * np.pi * np.random.random()
        v = np.vstack([v, [scale * np.cos(theta), scale * np.sin(theta)]])
    return v

def lennard_jones_force(particles: VectorArray) -> VectorArray:
    length = particles.shape[0]
    forces = np.zeros_like(particles)

    for i in range(length):
        for j in range(i + 1, length):
            r = np.sqrt(np.sum((particles[i,:] - particles[j,:])**2))
            magnitude = 4 * epsilon * ( 12*np.power(sigma,12)*np.power(r, -13) - 6*np.power(sigma,6)*np.power(r,-7) )
            # Direction of force exerted on i (towards j)
            direction = (particles[j,:] - particles[i,:]) / r

            forces[i,:] -= magnitude*direction
            forces[j,:] += magnitude*direction
    return forces

def lennard_jones_potential(particles: VectorArray) -> npt.NDArray[float]:
    length = particles.shape[0]
    potential = np.zeros(length)

    for i in range(length):
        for j in range(i+1, length):
            r = np.sum((particles[i,:] - particles[j,:]) ** 2)
            mag = 4 * epsilon * (np.power(sigma ** 2 / r, 6) - np.power(sigma**2/r, 3))
            potential[i] += mag
            potential[j] += mag
    return potential

def constrain_particles(particles: VectorArray) -> None:
    for i in range(len(particles)):
        if particles[i,0] > L:
            particles[i,0] = 2 * L - particles[i,0]
        elif particles[i,0] < 0:
            particles[i,0] *= -1
        if particles[i,1] > L:
            particles[i,1] = 2 * L - particles[i,1]
        elif particles[i,1] < 0:
            particles[i,1] *= -1

def kinetic_energy(velocities: VectorArray, scale: float):
    return 0.5 * m * np.sum((velocities/scale) ** 2)

def potential_energy(particles: VectorArray) -> float:
    return np.sum(lennard_jones_potential(particles))

def main():
    particles = initialize_particles(N, L)
    velocity_scale = 2 * v_0
    velocities = initialize_velocities(N, velocity_scale)

    time_range = 50000 
    plot_freq = 5
    dt = sigma/(velocity_scale) * 0.02
    
    position_history = np.empty((time_range//plot_freq,N,2))

    E_k_history = np.empty(time_range//plot_freq)
    E_p_history = np.empty(time_range//plot_freq)

    plotting = ENERGIES
    logging = False

    for t in trange(time_range):
        if logging or plotting == SNAPSHOT:
            position_history[t//plot_freq,:,:] = particles
        if t % plot_freq == 0 and plotting == ENERGIES:
            # print(f'{t//plot_freq} of {time_range//plot_freq}')
            # Something is wrong with the units. Graphs have the correct shape but are not of the same scale
            E_k_history[t//plot_freq] = kinetic_energy(velocities, velocity_scale)
            E_p_history[t//plot_freq] = potential_energy(particles)

        particles, velocities = leapfrog(particles, velocities, lambda x: lennard_jones_force(x), dt = dt)
        constrain_particles(particles)
    
    time = np.arange(time_range//plot_freq) * dt / t_0
    # plot_vals(plotting, time, E_k_history, E_p_history)
    if plotting == SNAPSHOT:
        for i in range(N):
            plt.plot(position_history[:,i,0].squeeze(), position_history[:,i,1].squeeze())
        plt.gca().set_xlim([0, L])
        plt.gca().set_ylim([0, L])
    elif plotting == ENERGIES:
        plt.subplot(3, 1, 1)
        plt.plot(time, E_k_history / epsilon, '', label="Kinetic Energy", markersize=1)
        plt.ylabel("$E_k$")
        plt.subplot(3, 1, 2)
        plt.plot(time, E_p_history / epsilon, '', label="Potential Energy", markersize=1)
        plt.ylabel("$E_p$")
        plt.subplot(3, 1, 3)
        plt.plot(time, (E_k_history + E_p_history) / epsilon, '', label="Total Energy", markersize=1)
        plt.ylabel("$E$")
    np.save('time.npy', time)
    np.save('particles.npy', position_history)
    np.save('E_k.npy', E_k_history)
    np.save('E_p.npy', E_p_history)

    if logging:
        try:
            os.mkdir("./logs")
        except FileExistsError:
            pass
        np.savetxt("./logs/" + datetime.now().strftime("%d-%m-%Y-%H:%M:%S") + ".txt", position_history.reshape(position_history.shape[0],N*2))

    plt.show()

def plot_energies(time:Vector, E_k: Vector, E_p: Vector, pos_hist: Vector = None):
    plt.subplot(3, 1, 1)
    plt.plot(time, E_k/ epsilon, '', label="Kinetic Energy", markersize=1)
    plt.ylabel("$E_k$")
    plt.subplot(3, 1, 2)
    plt.plot(time, E_p / epsilon, '', label="Potential Energy", markersize=1)
    plt.ylabel("$E_p$")
    plt.subplot(3, 1, 3)
    plt.plot(time, (E_k + E_p)/ epsilon, '', label="Potential Energy", markersize=1)
    plt.ylabel("$E$")
    
    plt.show()

def plot_old_values():
    time = np.load('time.npy')
    particle_hist = np.load('particles.npy')
    E_k = np.load('E_k.npy')
    E_p = np.load('E_p.npy')

    print(time)
    print(E_k)
    plot_energies(ENERGIES, time, E_k, E_p)

if __name__ == '__main__':
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
    v_0 = np.sqrt(2 * epsilon / m_0)
    t_0 = sigma_0 * np.sqrt(m_0 / (2 * epsilon))

    if len(argv) > 1 and 'plot' in argv:
        plot_old_values()
    else:
        main()
