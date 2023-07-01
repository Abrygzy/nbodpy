import numpy as np
from itertools import product
import multiprocessing as mp

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
        pos = par.pos
        pos_float = pos / h - 0.5   # floating point index
        pos_floor = np.floor(pos_float).astype(int) # floor of floating point index
        pos_cel = pos_floor + 1 # ceiling of floating point index
        pos_star = pos_float - pos_floor # distance from floor
        for idx_shift in product(range(2), repeat=2): # Density change: loop over 8 cells
            rho_idx = pos_cel - idx_shift
            rho_idx = np.where(rho_idx != ng, rho_idx, 0)
            rho[rho_idx[0], rho_idx[1]] += np.multiply.reduce(np.where(idx_shift, 1-pos_star, pos_star)) * par.mass / (h ** 2)
    return rho
    


def assign_mass(par, ng, h=1):
    rho = np.zeros((ng, ng)) # initialize density field
    pos = par.pos
    pos_float = pos / h - 0.5   # floating point index
    pos_floor = np.floor(pos_float).astype(int) # floor of floating point index
    pos_cel = pos_floor + 1 # ceiling of floating point index
    pos_star = pos_float - pos_floor # distance from floor
    for idx_shift in product(range(2), repeat=2): # Density change: loop over 4 cells
        rho_idx = pos_cel - idx_shift
        rho_idx = np.where(rho_idx != ng, rho_idx, 0)
        rho[rho_idx[0], rho_idx[1]] += np.multiply.reduce(np.where(idx_shift, 1-pos_star, pos_star)) * par.mass / (h ** 2)
    return rho

def mass_assign_parallel_2d(pars, ng, h=1, n_processes=None):
    '''
    Assign mass to grid using CIC scheme with multiprocessing.
    Input:
        pars: list of Particle objects
        ng: grid size
        h: grid spacing
        n_processes: number of processes to use (default is number of CPU cores)
    Output:
        rho: density field
    '''
    if n_processes is None:
        n_processes = mp.cpu_count()

    
    rho_list = []
    with mp.Pool() as pool:
        rho_list.append(pool.starmap(assign_mass, [(par, rho, ng, h) for par in pars]))
    for rho_i in rho_list:
        rho += rho_i.get()
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

def mass_assign_parallel_3d(pars, ng, h=1, n_processes=None):
    '''
    Assign mass to grid using CIC scheme with multiprocessing.
    Input:
        pars: list of Particle objects
        ng: grid size
        h: grid spacing
        n_processes: number of processes to use (default is number of CPU cores)
    Output:
        rho: density field
    '''
    if n_processes is None:
        n_processes = mp.cpu_count()

    rho = np.zeros((ng, ng, ng)) # initialize density field

    def assign_mass(par):
        pos = par.pos
        pos_float = pos / h - 0.5   # floating point index
        pos_floor = np.floor(pos_float).astype(int) # floor of floating point index
        pos_cel = pos_floor + 1 # ceiling of floating point index
        pos_star = pos_float - pos_floor # distance from floor
        for idx_shift in product(range(2), repeat=3): # Density change: loop over 8 cells
            rho_idx = pos_cel - idx_shift
            rho_idx = np.where(rho_idx != ng, rho_idx, 0)
            rho[rho_idx[0], rho_idx[1], rho_idx[2]] += np.multiply.reduce(np.where(idx_shift, 1-pos_star, pos_star)) * par.mass / (h ** 3)

    with mp.Pool(n_processes) as pool:
        pool.map(assign_mass, pars)

    return rho