import numpy as np
from itertools import product
import multiprocessing as mp
from numba import jit

@jit(nopython=False,forceobj=True)
def cic_density_2d(pars, ng, h=1):
    '''
    Derive density field from particle positions and masses using CIC scheme.
    Input:
        pars: list of Particle objects
        ng: grid size
        h: grid spacing
    Output:
        rho: density field
    
    '''
    rho = np.zeros((ng, ng)) # initialize density field
    for par in pars:
        pos = np.array(par.pos)
        pos_float = pos / h - 0.5   # floating point index
        pos_floor = np.floor(pos_float).astype(int) # floor of floating point index
        pos_cel = pos_floor + 1 # ceiling of floating point index
        pos_star = pos_float - pos_floor # distance from floor
        # for idx_shift in product(range(2), repeat=2): # Density change: loop over 8 cells
        for idx_shift in np.array([[0, 0], [0, 1], [1, 0], [1, 1]]): # Density change: loop over 8 cells
            rho_idx = pos_cel - idx_shift
            rho_idx = np.where(rho_idx != ng, rho_idx, 0)
            rho[rho_idx[0], rho_idx[1]] += np.multiply.reduce(np.where(idx_shift, 1-pos_star, pos_star)) * par.mass / (h ** 2)
    return rho

def cic_density_2d2(pars, ng, h=1):
    '''
    Derive density field from particle positions and masses using CIC scheme.
    Input:
        pars: list of Particle objects
        ng: grid size
        h: grid spacing
    Output:
        rho: density field
    
    '''
    rho = np.zeros((ng, ng)) # initialize density field
    for par in pars:
        pos = np.array(par.pos)
        pos_float = pos / h - 0.5   # floating point index
        pos_floor = np.floor(pos_float).astype(int) # floor of floating point index
        pos_cel = pos_floor + 1 # ceiling of floating point index
        pos_star = pos_float - pos_floor # distance from floor
        # for idx_shift in product(range(2), repeat=2): # Density change: loop over 8 cells
        for idx_shift in np.array([[0, 0], [0, 1], [1, 0], [1, 1]]): # Density change: loop over 8 cells
            rho_idx = pos_cel - idx_shift
            rho_idx = np.where(rho_idx != ng, rho_idx, 0)
            rho[rho_idx[0], rho_idx[1]] += np.multiply.reduce(np.where(idx_shift, 1-pos_star, pos_star)) * par.mass / (h ** 2)
    return rho
    





def cic_density_3d(pars, ng, h=1):
    '''
    Derive density field from particle positions and masses using CIC scheme.
    Input:
        pars: list of Particle objects
        ng: grid size
        h: grid spacing
    Output:
        rho: density field
    
    '''
    rho = np.zeros((ng, ng, ng)) # initialize density field
    for par in pars:
        pos = par.pos
        pos_float = pos / h - 0.5   # floating point index
        pos_floor = np.floor(pos_float).astype(int) # floor of floating point index
        pos_cel = pos_floor + 1 # ceiling of floating point index
        pos_star = pos_float - pos_floor # distance from floor
        for idx_shift in product(range(2), repeat=3): # Density change: loop over 8 cells
            rho_idx = pos_cel - idx_shift
            rho_idx = np.where(rho_idx != ng, rho_idx, 0)
            rho[rho_idx[0], rho_idx[1], rho_idx[2]] += np.multiply.reduce(np.where(idx_shift, 1-pos_star, pos_star)) * par.mass / (h ** 3)
    return rho
