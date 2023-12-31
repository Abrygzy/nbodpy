import numpy as np
from itertools import product

def cic_acc_2d(pars, phi, h=1):
    '''
    Acceleration of each particle using CIC scheme.
    Input:
        pars: list of Particle objects
        gravity_pbc: gravitational potential with periodic boundary conditions (0 cell padding for all axis)
        h: grid spacing
    Output:
        Put acceleration into Particle.acc
    '''
    ng = phi.shape[0]
    phi_pdc = np.pad(phi, pad_width=1,mode='wrap')
    acc_x_pdc, acc_y_pdc = np.gradient(phi_pdc, 1, edge_order=1)
    acc_x, acc_y= -1 * acc_x_pdc[1:-1, 1:-1], -1 * acc_y_pdc[1:-1, 1:-1]
    
    for par in pars:
        pos = par.pos
        pos_float = pos / h - 0.5   # floating point index
        pos_floor = np.floor(pos_float).astype(int) # floor of floating point index
        pos_cel = pos_floor + 1 # ceiling of floating point index
        pos_star = pos_float - pos_floor # distance from floor
        par_acc_x, par_acc_y = 0, 0
        for idx_shift in product(range(2), repeat=2): # Density change: loop over 8 cells
            rho_idx = pos_cel - idx_shift
            rho_idx = np.where(rho_idx != ng, rho_idx, 0)
            par_acc_x += np.multiply.reduce(np.where(idx_shift, 1-pos_star, pos_star)) * acc_x[rho_idx[0], rho_idx[1]] * (h ** 2)
            par_acc_y += np.multiply.reduce(np.where(idx_shift, 1-pos_star, pos_star)) * acc_y[rho_idx[0], rho_idx[1]] * (h ** 2)
        par.acc = np.array([par_acc_x, par_acc_y])
    return acc_x, acc_y

def cic_acc_3d(pars, phi, h=1):
    '''
    Acceleration of each particle using CIC scheme.
    Input:
        pars: list of Particle objects
        phi: gravitational potential with periodic boundary conditions (0 cell padding for all axis)
        h: grid spacing
    Output:
        Put acceleration into Particle.acc
    '''
    ng = phi.shape[0]
    phi_pdc = np.pad(phi, pad_width=1, mode='wrap')
    acc_x_pdc, acc_y_pdc, acc_z_pdc = np.gradient(phi_pdc, 1, 1, 1, edge_order=1)
    acc_x, acc_y, acc_z = -1 * acc_x_pdc[1:-1, 1:-1, 1:-1], -1 * acc_y_pdc[1:-1, 1:-1, 1:-1], -1 * acc_z_pdc[1:-1, 1:-1, 1:-1]
    
    for par in pars:
        pos = par.pos
        pos_float = pos / h - 0.5   # floating point index
        pos_floor = np.floor(pos_float).astype(int) # floor of floating point index
        pos_cel = pos_floor + 1 # ceiling of floating point index
        pos_star = pos_float - pos_floor # distance from floor
        par_acc_x, par_acc_y, par_acc_z = 0, 0, 0
        for idx_shift in product(range(2), repeat=3): # Density change: loop over 8 cells
            rho_idx = pos_cel - idx_shift
            rho_idx = np.where(rho_idx != ng, rho_idx, 0)
            par_acc_x += np.multiply.reduce(np.where(idx_shift, 1-pos_star, pos_star)) * acc_x[rho_idx[0], rho_idx[1], rho_idx[2]] * (h ** 3)
            par_acc_y += np.multiply.reduce(np.where(idx_shift, 1-pos_star, pos_star)) * acc_y[rho_idx[0], rho_idx[1], rho_idx[2]] * (h ** 3)
            par_acc_z += np.multiply.reduce(np.where(idx_shift, 1-pos_star, pos_star)) * acc_z[rho_idx[0], rho_idx[1], rho_idx[2]] * (h ** 3)
        par.acc = np.array([par_acc_x, par_acc_y, par_acc_z])
    return acc_x, acc_y, acc_z