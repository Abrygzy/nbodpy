import numpy as np
from numba import jit

@jit(nopython=True) 
def cic_density_2d(pars_pos_mass, ng, h=1):
    dens = np.zeros((ng, ng)) # initialize density field
    pars_pos = pars_pos_mass[:,:2]
    pars_mass = pars_pos_mass[:,2]
    dens = np.zeros((ng, ng)) # initialize density field
    pos_float = pars_pos/h - 0.5 # particle positions in float
    pos_floor = np.floor(pos_float).astype(np.int16) # particle positions in int
    pos_cel = pos_floor + 1 # ceiling of floating point index
    pos_star = pos_float - pos_floor # distance from floor
    for i in range(len(pars_pos)):
        for idx_shift in np.array([[0, 0], [0, 1], [1, 0], [1, 1]]): # Density change: loop over 4 cells
            rho_idx = pos_cel[i] - idx_shift
            rho_idx[rho_idx == ng] = 0            
            dens[rho_idx[0], rho_idx[1]] +=  [pos_star[i], 1-pos_star[i]][idx_shift[0]][0] * [pos_star[i], 1-pos_star[i]][idx_shift[1]][1] * pars_mass[i] / (h ** 2)
    return dens

@jit(nopython=True) 
def cic_density_3d(pars_pos_mass, ng, h=1):
    dens = np.zeros((ng, ng, ng)) # initialize density field
    pars_pos = pars_pos_mass[:,:3]
    pars_mass = pars_pos_mass[:,3]
    pos_float = pars_pos/h - 0.5 # particle positions in float
    pos_floor = np.floor(pos_float).astype(np.int16) # particle positions in int
    pos_cel = pos_floor + 1 # ceiling of floating point index
    pos_star = pos_float - pos_floor # distance from floor
    for i in range(len(pars_pos)):
        # for idx_shift in [[x,y,z] for x in range(2) for y in range(2) for z in range(2)]: # Density change: loop over 8 cells
        for idx_shift in np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]): # Density change: loop over 8 cells
            rho_idx = pos_cel[i] - idx_shift
            rho_idx[rho_idx == ng] = 0            
            dens[rho_idx[0], rho_idx[1], rho_idx[2]] += \
                [pos_star[i], 1-pos_star[i]][idx_shift[0]][0] * \
                [pos_star[i], 1-pos_star[i]][idx_shift[1]][1] * \
                [pos_star[i], 1-pos_star[i]][idx_shift[2]][2] * pars_mass[i] / (h ** 3)
    return dens