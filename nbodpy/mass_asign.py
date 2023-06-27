import numpy as np

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
        for idx_shift in [[x, y] for x in range(2) for y in range(2)]: # Density change: loop over 8 cells
            rho_idx = pos_cel - idx_shift
            rho_idx = np.where(rho_idx != ng, rho_idx, 0)
            rho[rho_idx[0], rho_idx[1]] += np.multiply.reduce(np.where(idx_shift, 1-pos_star, pos_star)) * par.mass / (h ** 3)
    return rho