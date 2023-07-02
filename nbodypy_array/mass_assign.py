import numpy as np
from multiprocessing import Pool
from numba import jit
from paras import ng, h

@jit(nopython=True) 
def cic_density_2d(pars_pos_mass):
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

# @jit(nopython=True) 
# def cic_density_3d(pars_pos_mass, ng, h=1):
#     dens = np.zeros((ng, ng, ng)) # initialize density field
#     pars_pos = pars_pos_mass[:,:3]
#     pars_mass = pars_pos_mass[:,3]
#     pos_float = pars_pos/h - 0.5 # particle positions in float
#     pos_floor = np.floor(pos_float).astype(np.int16) # particle positions in int
#     pos_cel = pos_floor + 1 # ceiling of floating point index
#     pos_star = pos_float - pos_floor # distance from floor
#     for i in range(len(pars_pos)):
#         # for idx_shift in [[x,y,z] for x in range(2) for y in range(2) for z in range(2)]: # Density change: loop over 8 cells
#         for idx_shift in np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]): # Density change: loop over 8 cells
#             rho_idx = pos_cel[i] - idx_shift
#             rho_idx[rho_idx == ng] = 0            
#             dens[rho_idx[0], rho_idx[1], rho_idx[2]] += \
#                 [pos_star[i], 1-pos_star[i]][idx_shift[0]][0] * \
#                 [pos_star[i], 1-pos_star[i]][idx_shift[1]][1] * \
#                 [pos_star[i], 1-pos_star[i]][idx_shift[2]][2] * pars_mass[i] / (h ** 3)
#     return dens



@jit(nopython=True) 
def cic_density_3d(pars_pos_mass):
    dens = np.zeros((ng, ng, ng)) # initialize density field

    pos_float = pars_pos_mass[:,:3]/h - 0.5 # particle positions in float
    pos_floor = np.floor(pos_float).astype(np.int16) # particle positions in int
    pos_star = pos_float - pos_floor # distance from floor
    for i in range(len(pos_float)):
        for idx_shift in np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]): # Density change: loop over 8 cells
            rho_idx = pos_floor[i] + 1 - idx_shift
            rho_idx[rho_idx == ng] = 0            
            dens[rho_idx[0], rho_idx[1], rho_idx[2]] += \
                [pos_star[i], 1-pos_star[i]][idx_shift[0]][0] * \
                [pos_star[i], 1-pos_star[i]][idx_shift[1]][1] * \
                [pos_star[i], 1-pos_star[i]][idx_shift[2]][2] * pars_pos_mass[i,3] / (h ** 3)
    return dens

# def cic_density_3d_split(pars_pos_mass, ng, h=1):
#     sub_ng = ng // 2
#     sub_pars_pos_mass = []
#     for i in range(2):
#         for j in range(2):
#             for k in range(2):
#                 sub_pars_pos_mass.append(pars_pos_mass[(pars_pos_mass[:,0] >= i*sub_ng*h) & (pars_pos_mass[:,0] < (i+1)*sub_ng*h) & \
#                                                         (pars_pos_mass[:,1] >= j*sub_ng*h) & (pars_pos_mass[:,1] < (j+1)*sub_ng*h) & \
#                                                         (pars_pos_mass[:,2] >= k*sub_ng*h) & (pars_pos_mass[:,2] < (k+1)*sub_ng*h)])
#     with Pool(8) as p:
#         sub_dens = p.starmap(cic_density_3d, [(sub_pars_pos_mass[i], sub_ng, h) for i in range(8)])
#     dens = np.zeros((ng, ng, ng))
#     for i in range(2):
#         for j in range(2):
#             for k in range(2):
#                 dens[i*sub_ng:(i+1)*sub_ng, j*sub_ng:(j+1)*sub_ng, k*sub_ng:(k+1)*sub_ng] = sub_dens[i*4+j*2+k]
#     return dens