#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from tqdm import trange

def main():
    x_0 = 0
    sigma = 1
    dt = 1
    dt_red = np.sqrt(dt)

    N = 10_000
    T = 2000
    time_range = int(T/dt)

    x = np.zeros((N,time_range))
    x[:,0] = x_0

    for t in trange(time_range-1):
        diff = np.round(np.random.random_sample((1,N))) * 2 - 1
        x[:,t+1] = x[:,t] + diff * sigma * dt_red
    
    

if __name__ == '__main__':
    main()